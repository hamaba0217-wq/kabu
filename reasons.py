# -*- coding: utf-8 -*-
"""
終了理由の内訳集計

analyze_trades が出力した明細CSV（reason列を含む）から、
各戦略がどんな理由で決済されたかの内訳を出す。

特にボックス反発の「支持線割れ(-7%損切り)」の割合を知るために使う。
「大負け(-10%以下)」とは別に、「-7%で損切りされた率」が分かる。
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import config


def analyze(trades: pd.DataFrame):
    L = []
    A = L.append
    A("=" * 66)
    A("終了理由の内訳（各戦略が、どう決済されたか）")
    A("=" * 66)
    A("")

    for strat, g in trades.groupby("戦略"):
        n = len(g)
        A(f"■ {strat}  (n={n:,})")
        rc = g["reason"].value_counts()
        for reason, cnt in rc.items():
            pct = cnt / n * 100
            # その理由の平均リターン
            avg = g[g["reason"] == reason]["return"].mean() * 100
            bar = "█" * int(pct / 2)
            A(f"    {reason:<10} : {pct:5.1f}%  平均{avg:+6.2f}%  (n={cnt:>5}) {bar}")
        # 勝ち・負けの別
        win = (g["return"] > 0).mean() * 100
        A(f"    ── 勝率 {win:.1f}% / 平均リターン {g['return'].mean()*100:+.2f}%")
        A("")

    A("=" * 66)
    A("読み方")
    A("=" * 66)
    A("・ボックス反発の『支持線割れ』= -7%で損切りした率。")
    A("  これが『負けて撤退した』主な形。")
    A("・『反発利確』= 目標+15%まで戻して利確できた率（大勝ちの源）。")
    A("・『期限切れ』= どちらにも到達せず、保有期限で手仕舞いした率。")
    A("  この最終損益はプラスにもマイナスにもなる。")
    return "\n".join(L)


def main():
    files = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "*_trade_analysis.csv")))
    if not files:
        print("先に  py main.py analyze-trades  を実行してください。")
        return
    latest = files[-1]
    print(f"読み込み: {latest}\n")
    trades = pd.read_csv(latest)
    if "reason" not in trades.columns:
        print("明細に reason 列がありません。analyze-trades を実行し直してください。")
        return
    print(analyze(trades))


if __name__ == "__main__":
    main()
