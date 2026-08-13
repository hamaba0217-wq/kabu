# -*- coding: utf-8 -*-
"""
④ 需給（業種別空売り比率・投資部門別）× 売られすぎ の検証

機関投資家フレームワークの「④ 投資主体別需給」を2つのデータで検証する。

(A) 業種別空売り比率
    空売り比率が高い業種＝弱気/踏み上げ期待。売られすぎ銘柄がその業種に
    あるとき、反発しやすいか（踏み上げ）逆に弱いかを見る。
    売られすぎ銘柄→その業種→as_of時点の空売り比率、で紐付け。

(B) 投資部門別（海外勢の売買）
    市場全体の週次データ。個別銘柄は絞れないので、
    「海外勢が買い越しの週にエントリー vs 売り越しの週」で
    相場タイミングとして効くかを見る（マーケット需給軸）。

先読み防止：需給データは as_of 以前の直近公表値。値動きは as_of 以降。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import backtest
import config
import technical
from oversold import add_oversold_indicators


HOLD = 15
STOP_LOSS = -0.10
STEP_DAYS = 5
DEV_MIN, DEV_MAX = -20.0, -10.0
LAG_DAYS = 7      # 週次データの公表遅延を考慮


def _simulate(path):
    for px in path:
        if px - 1.0 <= STOP_LOSS:
            return STOP_LOSS
    return path[-1] - 1.0 if len(path) else 0.0


def _latest_before(g, date_col, as_of, lag=LAG_DAYS):
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=lag)
    past = g[g[date_col] <= cutoff]
    if past.empty:
        return None
    return past.iloc[-1]


def _collect(quotes, listed):
    qi = add_oversold_indicators(quotes)
    by_code = technical._build_index(qi)
    all_dates = np.sort(qi["date"].unique())
    base_index = {c: (v[0], v[1]) for c, v in by_code.items()}

    # 銘柄→業種の対応（コードと名前の両方を持つ）
    sec_map = {}       # code -> sector33名（表示・投資部門用）
    sec_code_map = {}  # code -> sector33コード（空売り比率の結合用）
    if "sector33" in listed.columns:
        for _, r in listed.iterrows():
            sec_map[str(r["code"])] = r["sector33"]
    if "sector33_code" in listed.columns:
        for _, r in listed.iterrows():
            sec_code_map[str(r["code"])] = str(r["sector33_code"])

    print("  売られすぎ銘柄を収集中...")
    entries = []
    start_idx, end_idx = 260, len(all_dates) - HOLD
    for as_of in all_dates[start_idx:end_idx:STEP_DAYS]:
        snap = qi[qi["date"] == as_of]
        if snap.empty:
            continue
        sub = snap[(snap["ma25_dev"] >= DEV_MIN) & (snap["ma25_dev"] <= DEV_MAX)]
        for _, s in sub.iterrows():
            code = s["code"]
            arrs = by_code.get(code)
            if arrs is None:
                continue
            close_arr, dates_arr = arrs[0], arrs[1]
            pos = np.searchsorted(dates_arr, np.datetime64(pd.Timestamp(as_of)), side="right")
            if pos == 0:
                continue
            entry = close_arr[pos - 1]
            if not entry or entry <= 0:
                continue
            fut = close_arr[pos:pos + HOLD]
            if len(fut) < HOLD * 0.5:
                continue
            ret = _simulate(fut / entry)
            entries.append({"as_of": pd.Timestamp(as_of), "code": code,
                            "sector33": sec_map.get(str(code)),
                            "sector33_code": sec_code_map.get(str(code)),
                            "ret": ret})
    print(f"  売られすぎサンプル: {len(entries):,} 件")
    return entries, base_index


def _eval(entries, base_index, pred, label):
    base_cache = {}
    def _base(as_of):
        if as_of not in base_cache:
            base_cache[as_of] = backtest.market_baseline(base_index, as_of, HOLD)
        return base_cache[as_of]
    rets, excess = [], []
    for e in entries:
        if not pred(e):
            continue
        rets.append(e["ret"])
        b = _base(e["as_of"])
        if pd.notna(b):
            excess.append(e["ret"] - b)
    if len(rets) < 20:
        return {"条件": label, "件数": len(rets), "勝率%": None,
                "平均%": None, "対相場超過%": None}
    arr = np.array(rets)
    return {"条件": label, "件数": len(arr),
            "勝率%": round((arr > 0).mean() * 100, 1),
            "平均%": round(arr.mean() * 100, 2),
            "対相場超過%": round(np.median(excess) * 100, 2) if excess else None}


def run_short_ratio(entries, base_index, short_ratio):
    """(A) 業種別空売り比率で分ける。"""
    sr = short_ratio.copy()
    sr["date"] = pd.to_datetime(sr["date"])
    sr = sr.sort_values(["sector33", "date"])
    sr_by_sec = {str(s): g for s, g in sr.groupby("sector33")}
    sr_keys = set(sr_by_sec.keys())

    # 結合キーを決める：空売り比率の業種キー（コード）に、
    # エントリー側の sector33_code が一致するか、なければ sector33名で試す。
    def _entry_key(e):
        kc = e.get("sector33_code")
        if kc is not None and str(kc) in sr_keys:
            return str(kc)
        kn = e.get("sector33")
        if kn is not None and str(kn) in sr_keys:
            return str(kn)
        return None

    # 各エントリーに、その業種・as_of時点の空売り比率を付ける
    matched = 0
    for e in entries:
        key = _entry_key(e)
        g = sr_by_sec.get(key) if key is not None else None
        if g is None:
            e["short_ratio"] = np.nan
            continue
        row = _latest_before(g, "date", e["as_of"])
        if row is None:
            e["short_ratio"] = np.nan
        else:
            e["short_ratio"] = row["short_ratio"]
            matched += 1
    print(f"  空売り比率を紐付けできた: {matched:,} 件")

    vals = [e["short_ratio"] for e in entries if pd.notna(e["short_ratio"])]
    if len(vals) < 40:
        print("  [!] 空売り比率の紐付けが少なすぎます。業種コードの対応を確認してください。")
        print(f"      売られすぎ銘柄の業種例: {[e.get('sector33') for e in entries[:3]]}")
        print(f"      空売り比率データの業種例: {list(sr_by_sec.keys())[:3]}")
        return None
    med = float(np.median(vals))
    rows = [
        _eval(entries, base_index, lambda e: True, "売られすぎのみ(基準)"),
        _eval(entries, base_index,
              lambda e: pd.notna(e["short_ratio"]) and e["short_ratio"] >= med,
              f"＋空売り比率≥中央値{med:.2f}(弱気/踏み上げ期待)"),
        _eval(entries, base_index,
              lambda e: pd.notna(e["short_ratio"]) and e["short_ratio"] < med,
              f"＋空売り比率<中央値{med:.2f}(空売り少ない)"),
    ]
    return pd.DataFrame(rows)


def run_investor(entries, base_index, investor):
    """(B) 投資部門別：海外勢の買い越し/売り越し週で分ける。"""
    if investor is None or len(investor) == 0:
        print("  [!] 投資部門別データがありません。")
        return None
    inv = investor.copy()
    # 列名を頑健に解決（実データ: PubDate, FrgnBal 等の略称）
    low = {c: c.lower() for c in inv.columns}
    # 日付は公表日(PubDate)を優先、なければ date を含む列
    date_col = next((c for c in inv.columns if "pubdate" in low[c]), None) \
        or next((c for c in inv.columns if "date" in low[c]), None)
    # 海外勢の買い越し額: frgn/foreign を含み bal を含む列
    bal_col = next((c for c in inv.columns
                    if ("frgn" in low[c] or "foreign" in low[c]) and "bal" in low[c]), None)
    if date_col is None or bal_col is None:
        print(f"  [!] 投資部門別の列を特定できません。実際の列: {list(inv.columns)}")
        return None
    print(f"  投資部門別の列を解決: 日付={date_col} / 海外買い越し={bal_col}")
    inv[date_col] = pd.to_datetime(inv[date_col], errors="coerce")
    inv = inv.dropna(subset=[date_col]).sort_values(date_col)
    inv["foreign_bal"] = pd.to_numeric(inv[bal_col], errors="coerce")

    def _foreign_net_positive(as_of):
        cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=LAG_DAYS)
        past = inv[inv[date_col] <= cutoff]
        if past.empty:
            return None
        return bool(past.iloc[-1]["foreign_bal"] > 0)

    for e in entries:
        e["foreign_pos"] = _foreign_net_positive(e["as_of"])

    n_have = sum(1 for e in entries if e["foreign_pos"] is not None)
    print(f"  海外勢データを紐付けできた: {n_have:,} 件")
    if n_have < 40:
        return None
    rows = [
        _eval(entries, base_index, lambda e: True, "売られすぎのみ(基準)"),
        _eval(entries, base_index, lambda e: e.get("foreign_pos") is True,
              "＋海外勢が買い越しの週"),
        _eval(entries, base_index, lambda e: e.get("foreign_pos") is False,
              "＋海外勢が売り越しの週"),
    ]
    return pd.DataFrame(rows)


def summarize(df_sr, df_inv):
    L = ["=" * 74,
         "④ 需給（業種別空売り比率・投資部門別）× 売られすぎ の検証",
         "=" * 74, ""]
    L.append("土台＝売られすぎ（移動平均-20〜-10乖離・利確なし・保有15日・損切-10%）。")
    L.append("")
    L.append("── (A) 業種別空売り比率 ──")
    if df_sr is not None:
        L.append(df_sr.to_string(index=False))
    else:
        L.append("  紐付け不足のため評価できませんでした（上のログ参照）。")
    L.append("")
    L.append("── (B) 投資部門別（海外勢の買い越し/売り越し週）──")
    if df_inv is not None:
        L.append(df_inv.to_string(index=False))
    else:
        L.append("  データ不足のため評価できませんでした（上のログ参照）。")
    L += ["", "=" * 74, "読み方", "=" * 74,
          "・(A) 空売り比率が高い業種の売られすぎ銘柄が、対相場超過で上回れば、",
          "  『空売りが多い＝踏み上げ余地』が反発の燃料になっている。",
          "・(B) 海外勢が買い越しの週にエントリーした売られすぎ銘柄が上回れば、",
          "  相場全体の資金流入に乗るのが有効（マーケット需給軸）。",
          "・いずれも対相場超過がプラスで初めて意味がある。",
          "  平均リターンがプラスでも超過がマイナスなら相場に乗っただけ。",
          "・週次・公表遅延ありのため7日前までの値を使用（先読み防止）。",
          "・手数料・スリッページ未考慮。"]
    return "\n".join(L)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    from sources import JQuants, JST
    jq = JQuants()
    print("キャッシュからデータを読み込み中...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    listed = jq.listed()
    print(f"  株価 {len(quotes):,}行 / 銘柄 {len(listed):,}")

    entries, base_index = _collect(quotes, listed)

    print("\n(A) 業種別空売り比率を取得します（Standardプラン）...")
    try:
        short_ratio = jq.short_ratio(config.BACKTEST_LOOKBACK_DAYS)
        print(f"  空売り比率 {len(short_ratio):,}行")
    except Exception as e:
        print(f"  [!] 取得失敗: {e}")
        short_ratio = None
    df_sr = run_short_ratio(entries, base_index, short_ratio) if short_ratio is not None else None

    print("\n(B) 投資部門別を取得します（Standardプラン）...")
    try:
        investor = jq.investor_types()
        print(f"  投資部門別 {len(investor):,}行  列: {list(investor.columns)[:8]}")
    except Exception as e:
        print(f"  [!] 取得失敗: {e}")
        investor = None
    df_inv = run_investor(entries, base_index, investor) if investor is not None else None

    report = summarize(df_sr, df_inv)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    with open(os.path.join(config.OUTPUT_DIR, f"{today}_supply_demand2.txt"),
              "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nレポート: output/{today}_supply_demand2.txt")


if __name__ == "__main__":
    main()
