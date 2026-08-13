# -*- coding: utf-8 -*-
"""
① 資本効率（ROE・PBR）による勝率検証

機関投資家フレームワークの「① ガバナンス・資本効率」を、数値で取れる
ROE と PBR で近似し、これまでの売られすぎ戦略の勝率がどう変わるかを見る。

指標の計算（決算データ + 株価）
--------------------------------
  ROE = 当期純利益(Profit) ÷ 純資産(Equity)     … 高いほど効率的
  PBR = 株価 ÷ 一株あたり純資産(BookValuePerShare) … 低いほど割安
      （BPSが無い場合: 時価総額 ÷ 純資産 で代用）

フレームワークの考え方
----------------------
「何を買うか＝長期観点（ROE高・PBR割安）」で銘柄を選び、
「いつ買うか＝短期観点（売られすぎ）」でタイミングを計る。
この掛け算が本当に勝率を上げるかを検証する。

  基準:  売られすぎ（移動平均-20〜-10乖離）のみ
  ＋①:  そこに ROE高・PBR割安 の条件を足す

先読み防止：ROE/PBRは as_of 以前に開示された決算から計算。株価は as_of 時点。
値動きは as_of 以降。
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
TAKE_PROFIT = 0.99
STEP_DAYS = 5
DEV_MIN, DEV_MAX = -20.0, -10.0    # 売られすぎ区分（土台）

# ROE・PBRの閾値（機関投資家が注目する水準）
ROE_HIGH = 0.08      # ROE 8%以上を「効率的」
PBR_LOW = 1.0        # PBR 1倍未満を「割安」（東証が改善要請した水準）


def _prep_quality_fin(fin):
    """決算データから、各開示時点のROE素材とBPSを整える。"""
    f = fin.sort_values(["code", "disclosed_date"]).copy()
    # 数値化（sources側で済んでいるはずだが念のため）
    for c in ("net_profit", "equity", "bps"):
        if c in f.columns:
            f[c] = pd.to_numeric(f[c], errors="coerce")
        else:
            f[c] = np.nan
    # ROE = 純利益 / 純資産（四半期純利益なので簡易にそのまま比率、目安として使う）
    f["roe"] = np.where((f["equity"].notna()) & (f["equity"] > 0),
                        f["net_profit"] / f["equity"], np.nan)
    return f


def _latest_fin_before(fin_by_code, code, as_of):
    g = fin_by_code.get(code)
    if g is None:
        return None
    past = g[g["disclosed_date"] <= pd.Timestamp(as_of)]
    if past.empty:
        return None
    return past.iloc[-1]


def run(quotes, fin, listed, step_days=STEP_DAYS):
    qi = add_oversold_indicators(quotes)
    by_code = technical._build_index(qi)
    all_dates = np.sort(qi["date"].unique())
    base_index = {c: (v[0], v[1]) for c, v in by_code.items()}

    fin2 = _prep_quality_fin(fin)
    fin_by_code = {c: g for c, g in fin2.groupby("code")}

    print("  売られすぎ銘柄を収集し、ROE・PBRを計算中...")
    entries = []
    start_idx, end_idx = 260, len(all_dates) - HOLD
    for as_of in all_dates[start_idx:end_idx:step_days]:
        snap = qi[qi["date"] == as_of]
        if snap.empty:
            continue
        # 売られすぎ区分に該当する銘柄だけ
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
            path = fut / entry
            # 決済（利確なし・損切-10%・保有15日）
            ret = STOP_LOSS
            hit_stop = False
            for px in path:
                r = px - 1.0
                if r <= STOP_LOSS:
                    ret = STOP_LOSS; hit_stop = True; break
            if not hit_stop:
                ret = path[-1] - 1.0

            # ROE・PBR（as_of以前の決算）
            fin_row = _latest_fin_before(fin_by_code, code, as_of)
            roe = fin_row["roe"] if fin_row is not None else np.nan
            bps = fin_row["bps"] if fin_row is not None else np.nan
            pbr = entry / bps if (pd.notna(bps) and bps > 0) else np.nan

            entries.append({"as_of": pd.Timestamp(as_of), "ret": ret,
                            "roe": roe, "pbr": pbr})
    print(f"  サンプル: {len(entries):,} 件")
    have_roe = sum(1 for e in entries if pd.notna(e["roe"]))
    have_pbr = sum(1 for e in entries if pd.notna(e["pbr"]))
    print(f"  ROE計算できた: {have_roe:,} 件 / PBR計算できた: {have_pbr:,} 件")

    base_cache = {}
    def _base(as_of):
        if as_of not in base_cache:
            base_cache[as_of] = backtest.market_baseline(base_index, as_of, HOLD)
        return base_cache[as_of]

    def _eval(pred, label):
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
        return {
            "条件": label, "件数": len(arr),
            "勝率%": round((arr > 0).mean() * 100, 1),
            "平均%": round(arr.mean() * 100, 2),
            "対相場超過%": round(np.median(excess) * 100, 2) if excess else None,
        }

    rows = [
        _eval(lambda e: True, "売られすぎのみ(基準)"),
        _eval(lambda e: pd.notna(e["roe"]) and e["roe"] >= ROE_HIGH,
              f"＋ROE≥{int(ROE_HIGH*100)}%(効率的)"),
        _eval(lambda e: pd.notna(e["pbr"]) and e["pbr"] < PBR_LOW,
              f"＋PBR<{PBR_LOW}(割安)"),
        _eval(lambda e: pd.notna(e["roe"]) and pd.notna(e["pbr"])
              and e["roe"] >= ROE_HIGH and e["pbr"] < PBR_LOW,
              f"＋ROE≥{int(ROE_HIGH*100)}%かつPBR<{PBR_LOW}"),
        # 逆の確認：ROE低・PBR高だと悪化するか
        _eval(lambda e: pd.notna(e["roe"]) and e["roe"] < ROE_HIGH,
              f"＋ROE<{int(ROE_HIGH*100)}%(非効率・参考)"),
        _eval(lambda e: pd.notna(e["pbr"]) and e["pbr"] >= PBR_LOW,
              f"＋PBR≥{PBR_LOW}(割高・参考)"),
    ]
    return pd.DataFrame(rows)


def summarize(df):
    L = ["=" * 74,
         "① 資本効率（ROE・PBR）× 売られすぎ の勝率検証",
         "=" * 74, ""]
    L.append("土台＝売られすぎ（移動平均-20〜-10乖離・利確なし・保有15日・損切-10%）。")
    L.append("そこにROE・PBRの条件を足すと勝率・対相場超過が上がるかを見る。")
    L.append("")
    L.append(df.to_string(index=False))
    L += ["", "=" * 74, "読み方", "=" * 74,
          "・『基準』よりROE高・PBR割安を足して勝率・対相場超過が上がれば、",
          "  ①資本効率は有効。フレームワーク通り『質の高い銘柄を安く買う』が効く。",
          "・逆に『ROE低』『PBR割高』（参考行）で悪化していれば、対照として裏づけ。",
          "・件数も見る。条件を足すと減る。少なすぎ（NaN）は評価不能。",
          "・ROEは四半期純利益÷純資産の簡易値で、年率換算していない点に注意。",
          "・有効なら分割検証へ。手数料・スリッページ未考慮。"]
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

    # 決算データに新項目があるか確認
    for col in ("net_profit", "equity", "bps"):
        n = fin[col].notna().sum() if col in fin.columns else 0
        print(f"  {col}: {n:,} 件に値あり")
    if all((fin[c].notna().sum() if c in fin.columns else 0) == 0
           for c in ("net_profit", "equity", "bps")):
        print("\n  [!] ROE・PBRの元データがキャッシュにありません。")
        print("      決算データの再取得が必要です: del data\\fin.pkl のあと再実行してください。")
        print("      （新しい項目 Profit/Equity/BookValuePerShare を取得し直します）")
        return

    df = run(quotes, fin, listed)
    report = summarize(df)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    path = os.path.join(config.OUTPUT_DIR, f"{today}_quality.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n比較表: {path}")


if __name__ == "__main__":
    main()
