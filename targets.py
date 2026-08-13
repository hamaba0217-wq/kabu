# -*- coding: utf-8 -*-
"""
目標倍率×保有期間の比較検証

6パターンを、実戦ルール（目標到達で全利確・損切りあり・期限で手仕舞い）で
検証します。

  期間 × 倍率
  ─────────────
  1か月(20日) × 1.5倍 / 2倍
  2か月(40日) × 1.5倍 / 2倍
  3か月(60日) × 1.5倍 / 2倍

各パターンで損切り幅も変えています
------------------------------------
保有期間が短いほど、値動きのブレも小さいので損切りを浅く。
長いほど深く。値幅を保有期間に合わせないと、短期なのに深い損切りで
無駄に耐えたり、長期なのに浅い損切りでノイズに刈られたりします。

  1か月 → 損切り -10%
  2か月 → 損切り -15%
  3か月 → 損切り -20%

判定の見方
----------
・目標達成率: 期間内に目標倍率へ到達した割合
・平均リターン: 損切り・利確・期限切れをすべて含めた実際の損益
・対相場 超過: 同期間の市場平均をどれだけ上回ったか（最重要）
・期待値がプラスでも、超過がマイナスなら市場に負けている
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import backtest
import config
import screen


# (ラベル, 保有営業日, 目標リターン, 損切り)
PATTERNS = [
    ("1か月×1.5倍", 20, 0.50, -0.10),
    ("1か月×2倍",   20, 1.00, -0.10),
    ("2か月×1.5倍", 40, 0.50, -0.15),
    ("2か月×2倍",   40, 1.00, -0.15),
    ("3か月×1.5倍", 60, 0.50, -0.20),
    ("3か月×2倍",   60, 1.00, -0.20),
]


def _forward_paths(by_code, codes, as_of, horizon):
    """各銘柄の、as_of以降 horizon 営業日の値動き（entry基準化）を返す。"""
    as_of_ts = np.datetime64(pd.Timestamp(as_of))
    out = {}
    for code in codes:
        arrs = by_code.get(code)
        if arrs is None:
            continue
        close_arr, dates_arr = arrs
        pos = np.searchsorted(dates_arr, as_of_ts, side="right")
        if pos == 0:
            continue
        entry = close_arr[pos - 1]
        if not entry or entry <= 0:
            continue
        fut = close_arr[pos:pos + horizon]
        if len(fut) < horizon * 0.6:
            continue
        out[code] = fut / entry
    return out


def run(quotes, fin, listed, step_days=21):
    q = screen.add_rolling(quotes, config.MA_WINDOW)
    by_code = backtest.build_index(q)
    all_dates = np.sort(q["date"].unique())

    results = []
    for label, horizon, target, stop in PATTERNS:
        print(f"\n{'='*56}\n▶ {label}（保有{horizon}日 / 目標+{target:.0%} / 損切り{stop:.0%}）\n{'='*56}")

        start_idx = max(config.MA_WINDOW, 30)
        end_idx = len(all_dates) - horizon
        if end_idx <= start_idx:
            print("  期間不足でスキップ")
            continue
        as_of_list = all_dates[start_idx:end_idx:step_days]

        trade_rets, reasons, hit_target = [], [], []
        excess_per_period = []

        for as_of in as_of_list:
            prices = screen.snapshot_at(q, as_of)
            yoy = screen.yoy_table(fin[fin["disclosed_date"] <= pd.Timestamp(as_of)])
            if yoy.empty:
                continue
            picks, _ = screen.run_screen(prices, yoy, listed, pd.DataFrame())
            if len(picks) == 0:
                continue

            paths = _forward_paths(by_code, picks["code"].tolist(), as_of, horizon)
            if not paths:
                continue

            period_rets = []
            for code, path in paths.items():
                r, reason = backtest.simulate_target(path, target, stop)
                trade_rets.append(r)
                reasons.append(reason)
                hit_target.append(reason == "目標達成")
                period_rets.append(r)

            # 相場ベースライン（同期間の全銘柄・単純保有の中央値）
            base = backtest.market_baseline(by_code, as_of, horizon)
            if period_rets and pd.notna(base):
                excess_per_period.append(np.median(period_rets) - base)

        if not trade_rets:
            results.append({"パターン": label, "取引数": 0})
            continue

        arr = np.array(trade_rets)
        ex = np.array(excess_per_period)
        results.append({
            "パターン": label,
            "取引数": len(arr),
            "目標達成率%": round(np.mean(hit_target) * 100, 1),
            "平均リターン%": round(arr.mean() * 100, 1),
            "中央値%": round(np.median(arr) * 100, 1),
            "勝率%": round((arr > 0).mean() * 100, 1),
            "損切り率%": round(reasons.count("損切り") / len(reasons) * 100, 1),
            "対相場 超過%": round(np.median(ex) * 100, 1) if len(ex) else None,
            "相場に勝った回": f"{int((ex > 0).sum())}/{len(ex)}" if len(ex) else "—",
        })

    return pd.DataFrame(results)


def summarize(df: pd.DataFrame) -> str:
    L = ["=" * 78,
         "目標倍率×保有期間の比較（実戦ルール：目標で全利確・損切りあり）",
         "=" * 78, ""]
    L.append(df.to_string(index=False))
    L += ["", "=" * 78, "読み方", "=" * 78,
          "・「対相場 超過%」が最重要。プラスなら市場平均に勝っている。",
          "・「目標達成率%」が高くても、達成前に損切りされた分で相殺される。",
          "  だから「平均リターン%」と「対相場 超過%」で最終判断する。",
          "・取引数が極端に少ないパターンは、結果が偶然の可能性。",
          "・どのパターンも超過がマイナスなら、スクリーニング条件側の",
          "  見直しが必要（目標倍率を変えても勝てない、という意味）。",
          "・手数料・スリッページ・約定滑りは未考慮。実際はこれより悪い。"]
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
    path = os.path.join(config.OUTPUT_DIR, f"{today}_targets.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n比較表: {path}")


if __name__ == "__main__":
    main()
