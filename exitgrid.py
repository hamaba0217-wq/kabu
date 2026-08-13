# -*- coding: utf-8 -*-
"""
利確・損切りラインの総当たり最適化（ボックス反発）

利確ラインと損切りラインを +5%〜+20% / -5%〜-20% の範囲で
すべて組み合わせ、勝率・平均リターン・対相場超過がどう変わるかを見る。

対象戦略
--------
ボックス反発（大勝ち率トップ・大負け0%だった戦略）に絞る。
エントリー条件は既存のボックス反発のまま。決済ルールだけを変えて比較する。

注意
----
・利確・損切りを過去データに合わせ込むと過剰最適化になる。
  「一番良い1つ」ではなく「良い値が固まっている範囲」を見ること。
・利確を上げるほど大勝ちは増えるが到達率は下がる。
  損切りを深くするほど1回の負けは大きいが損切り撤退は減る。
  そのトレードオフを表で見る。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import backtest
import config
import technical


# 試す利確・損切りライン
TAKE_PROFITS = [0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
STOP_LOSSES = [-0.05, -0.07, -0.08, -0.10, -0.12, -0.15, -0.18, -0.20]


def _simulate(path, target, stop):
    """目標到達で利確、損切りで撤退、期限まで未達なら成行。"""
    for px in path:
        ret = px - 1.0
        if ret <= stop:
            return stop, "損切り"
        if ret >= target:
            return target, "利確"
    return (path[-1] - 1.0 if len(path) else 0.0), "期限切れ"


def run(quotes, fin, listed, step_days=7):
    qi = technical.add_indicators(quotes)
    by_code = technical._build_index(qi)
    all_dates = np.sort(qi["date"].unique())
    fin2 = technical._prep_fin(fin)
    base_index = {c: (v[0], v[1]) for c, v in by_code.items()}

    # ボックス反発のエントリーだけ集める（決済前のpathを保持）
    print("  ボックス反発のエントリー候補を収集中...")
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
            entries.append({"as_of": pd.Timestamp(as_of), "horizon": horizon,
                            "path": path})
    print(f"  候補: {len(entries):,} 件。{len(TAKE_PROFITS)}×{len(STOP_LOSSES)}通りを評価します。")

    base_cache = {}
    def _base(as_of, horizon):
        k = (as_of, horizon)
        if k not in base_cache:
            base_cache[k] = backtest.market_baseline(base_index, as_of, horizon)
        return base_cache[k]

    results = []
    for tp in TAKE_PROFITS:
        for sl in STOP_LOSSES:
            rets, period_map = [], {}
            for e in entries:
                r, _ = _simulate(e["path"], tp, sl)
                rets.append(r)
                period_map.setdefault((e["as_of"], e["horizon"]), []).append(r)
            arr = np.array(rets)
            excess = []
            for (as_of, h), rs in period_map.items():
                b = _base(as_of, h)
                if pd.notna(b):
                    excess.append(np.median(rs) - b)
            results.append({
                "利確%": int(tp * 100),
                "損切%": int(sl * 100),
                "勝率%": round((arr > 0).mean() * 100, 1),
                "平均%": round(arr.mean() * 100, 2),
                "中央%": round(np.median(arr) * 100, 2),
                "対相場超過%": round(np.median(excess) * 100, 2) if excess else None,
            })
    return pd.DataFrame(results)


def summarize(df):
    L = ["=" * 70,
         "利確×損切り 総当たり（ボックス反発）", "=" * 70, ""]

    # 平均リターンのピボット表
    L.append("【平均リターン% の一覧】（縦=利確 / 横=損切り）")
    piv = df.pivot(index="利確%", columns="損切%", values="平均%")
    L.append(piv.to_string())
    L.append("")

    # 対相場超過のピボット
    L.append("【対相場超過% の一覧】（縦=利確 / 横=損切り）")
    piv2 = df.pivot(index="利確%", columns="損切%", values="対相場超過%")
    L.append(piv2.to_string())
    L.append("")

    # 上位10（対相場超過で）
    L.append("【対相場超過が高い上位10】")
    top = df.sort_values("対相場超過%", ascending=False).head(10)
    L.append(top.to_string(index=False))
    L += ["", "=" * 70, "読み方", "=" * 70,
          "・平均リターンと対相場超過の両方を見る。",
          "・『一番良い1マス』でなく『良い値が固まっている領域』を探す。",
          "  周囲もまとめて良ければ本物、そのマスだけ突出は偶然の可能性。",
          "・利確を上げると大勝ちは増えるが到達率は下がる（期限切れが増える）。",
          "・損切りを深くすると1回の負けは大きいが、途中で切られにくい。",
          "・現状ルール（利確+15/損切-7）が表のどこにあるか確認し、",
          "  改善余地があるか見る。",
          "・ここで見つけた最良値も、別期間での再現確認（分割検証）が必要。",
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
    path = os.path.join(config.OUTPUT_DIR, f"{today}_exitgrid.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n比較表: {path}")


if __name__ == "__main__":
    main()
