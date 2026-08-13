# -*- coding: utf-8 -*-
"""
悪材料の除外による改善検証（ボックス反発）

仮説
----
ボックス反発で「支持線割れ(-7%損切り)」になった負けの中には、
・地合いで一時的に下げただけ（本来は反発する押し目）
・悪材料で下げ始めた（そのまま下落継続する罠）
が混在している。後者を除外できれば勝率が上がるはず。

悪材料の判定（決算データのみ・推測なし）
----------------------------------------
エントリー直前の「直近の決算」を見て、以下のいずれかなら「悪材料あり」：
  B1: 会社予想を下回る    … 営業利益が会社予想の一定割合未満
  B2: 前年同期比で減益    … 営業利益が前年同期より減少
  B3: 前年同期比で減収    … 売上が前年同期より減少

これらは全て決算の数値から計算でき、ニュース解釈（ハルシネーション）は
一切入らない。「直近決算」は、エントリー日以前に開示済みのものだけを使う
（先読み防止）。

検証内容
--------
悪材料フィルターのon/off全組み合わせ(2^3=8通り)で、
ボックス反発の成績がどう変わるかを見る。
「悪材料を除外すると超過リターンが上がる」なら仮説は正しい。
"""

from __future__ import annotations

import datetime as dt
import itertools
import os

import numpy as np
import pandas as pd

import backtest
import config
import technical


# 悪材料の判定パラメータ
FC_MISS_RATIO = 0.90     # 会社予想の90%未満なら「予想未達」


def _latest_fin_before(fin2, code, as_of):
    """as_of 以前に開示された、その銘柄の最新決算1行を返す。"""
    g = fin2[(fin2["code"] == code) &
             (fin2["disclosed_date"] <= pd.Timestamp(as_of))]
    if g.empty:
        return None
    return g.sort_values("disclosed_date").iloc[-1]


def _bad_flags(row):
    """決算1行から、3種の悪材料フラグを返す。"""
    if row is None:
        return {"予想未達": False, "減益": False, "減収": False}
    op = row.get("operating_profit")
    fc = row.get("fc_operating_profit")
    op_prev = row.get("op_prev")
    sales = row.get("net_sales")
    sales_prev = row.get("sales_prev")

    miss = False
    if pd.notna(op) and pd.notna(fc) and fc and fc > 0:
        miss = op < fc * FC_MISS_RATIO

    down_profit = False
    if pd.notna(op) and pd.notna(op_prev) and pd.notna(op_prev):
        down_profit = op < op_prev

    down_sales = False
    if pd.notna(sales) and pd.notna(sales_prev) and pd.notna(sales_prev):
        down_sales = sales < sales_prev

    return {"予想未達": miss, "減益": down_profit, "減収": down_sales}


# 除外フィルター: 名前 → その悪材料があるとエントリーを見送る
BAD_FILTERS = [
    ("予想未達を除外", "予想未達"),
    ("減益を除外",     "減益"),
    ("減収を除外",     "減収"),
]


def _prep_fin_extended(fin):
    """_prep_fin に加え、前年同期の売上も計算する。"""
    f = technical._prep_fin(fin)
    f = f.sort_values(["code", "disclosed_date"])
    if "net_sales" in f.columns:
        f["sales_prev"] = f.groupby(["code", "period_type"])["net_sales"].shift(1)
    else:
        f["sales_prev"] = np.nan
    return f


def run(quotes, fin, listed, step_days=7):
    qi = technical.add_indicators(quotes)
    by_code = technical._build_index(qi)
    all_dates = np.sort(qi["date"].unique())
    fin2 = _prep_fin_extended(fin)
    base_index = {c: (v[0], v[1]) for c, v in by_code.items()}

    # ボックス反発のエントリーを集め、各銘柄の悪材料フラグも記録
    print("  ボックス反発のエントリー候補を収集し、決算の悪材料を判定中...")
    spec = technical.STRATEGIES["B_ボックス反発"]
    horizon = spec["horizon"]
    entries = []
    s_i, e_i = 80, len(all_dates) - horizon
    for as_of in all_dates[s_i:e_i:step_days]:
        snap = technical._snapshot(qi, as_of)
        if snap.empty:
            continue
        codes = technical.entries_box_bottom(snap)
        if not codes:
            continue
        for code in codes:
            path, _ = technical._forward(by_code, code, as_of, horizon)
            if path is None:
                continue
            r, _ = technical._exit_box(path)     # 現行ルール(+15/-7)で決済
            fin_row = _latest_fin_before(fin2, code, as_of)
            flags = _bad_flags(fin_row)
            entries.append({"as_of": pd.Timestamp(as_of), "horizon": horizon,
                            "ret": r, "flags": flags,
                            "has_fin": fin_row is not None})
    print(f"  候補: {len(entries):,} 件")
    with_fin = sum(1 for e in entries if e["has_fin"])
    print(f"  うち決算データあり: {with_fin:,} 件（{with_fin/len(entries)*100:.0f}%）")

    base_cache = {}
    def _base(as_of, horizon):
        k = (as_of, horizon)
        if k not in base_cache:
            base_cache[k] = backtest.market_baseline(base_index, as_of, horizon)
        return base_cache[k]

    results = []
    n = len(BAD_FILTERS)
    for bits in itertools.product([0, 1], repeat=n):
        active_keys = [BAD_FILTERS[i][1] for i in range(n) if bits[i]]
        parts = [BAD_FILTERS[i][0] for i in range(n) if bits[i]]
        label = " + ".join(parts) if parts else "なし(基準)"

        rets, period_map = [], {}
        for e in entries:
            # active_keys のいずれかの悪材料があれば除外
            if any(e["flags"].get(k, False) for k in active_keys):
                continue
            rets.append(e["ret"])
            period_map.setdefault((e["as_of"], e["horizon"]), []).append(e["ret"])

        if len(rets) < 50:
            results.append({"除外条件": label, "取引数": len(rets),
                            "勝率%": None, "平均%": None, "対相場超過%": None})
            continue
        arr = np.array(rets)
        excess = []
        for (as_of, h), rs in period_map.items():
            b = _base(as_of, h)
            if pd.notna(b):
                excess.append(np.median(rs) - b)
        results.append({
            "除外条件": label,
            "取引数": len(arr),
            "勝率%": round((arr > 0).mean() * 100, 1),
            "平均%": round(arr.mean() * 100, 2),
            "対相場超過%": round(np.median(excess) * 100, 2) if excess else None,
        })

    df = pd.DataFrame(results)
    df["_s"] = df["対相場超過%"].fillna(-999)
    return df.sort_values("_s", ascending=False).drop(columns="_s").reset_index(drop=True)


def summarize(df):
    L = ["=" * 74,
         "悪材料の除外による改善検証（ボックス反発・決算データで判定）",
         "=" * 74, ""]
    L.append(df.to_string(index=False))
    L += ["", "=" * 74, "読み方", "=" * 74,
          "・『なし(基準)』より対相場超過が上がった除外条件が、有効な悪材料。",
          "・悪材料を除外すると取引数は減る。減りすぎ＝機会損失も見る。",
          "・勝率と超過の両方が上がっていれば、仮説（悪材料の罠を避ける）が正しい。",
          "・決算データが無い銘柄は除外対象にならない（判定不能なので通す）。",
          "・改善しても、決算だけでは捉えられない悪材料（不祥事・訴訟等）は残る。",
          "・有効なら、別期間での再現確認（分割検証）へ進む。",
          "・手数料・スリッページ未考慮。"]
    return "\n".join(L)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    from sources import JQuants, JST
    jq = JQuants()
    print("キャッシュからデータを読み込み中...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    fin = jq.financials(config.BACKTEST_LOOKBACK_DAYS)
    listed = jq.listed()
    print(f"  株価 {len(quotes):,}行 / 決算 {len(fin):,}行 / 銘柄 {len(listed):,}")

    df = run(quotes, fin, listed)
    report = summarize(df)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    path = os.path.join(config.OUTPUT_DIR, f"{today}_badnews.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n比較表: {path}")


if __name__ == "__main__":
    main()
