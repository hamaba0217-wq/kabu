# -*- coding: utf-8 -*-
"""
大勝ち（+10%以上）に効く条件の探索

通常の勝敗分析は「勝つ(>0)か負けるか」を見る。
これは視点を変え、「大きく勝つ(+10%以上)」だけに注目する。

狙い
----
少額でこまめに、ではなく「当たったときに大きく取る」方向。
大勝ちtrade がどんな特徴を持っていたかを、全trade を分母にして
「その特徴帯の大勝ち率」で見る（生存者バイアスを避ける）。

勝率とは別物であることに注意
----------------------------
大勝ち率が高い＝勝率が高い、ではない。
大勝ちを狙う条件は、外れも大きい（大負けも増える）ことが多い。
だから「大勝ち率 − 大負け率」の差（ネット）も併記し、
損小利大が成立しているかを見る。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config


BIG_WIN = 0.10       # +10%以上を「大勝ち」
BIG_LOSE = -0.10     # -10%以下を「大負け」


def analyze(trades: pd.DataFrame):
    """analyze_trades が出力した全trade明細を受け取り、大勝ち要因を分析。"""
    df = trades.copy()
    df["is_bigwin"] = df["return"] >= BIG_WIN
    df["is_biglose"] = df["return"] <= BIG_LOSE

    L = []
    A = L.append
    A("=" * 72)
    A("大勝ち(+10%以上)に効く条件の探索")
    A("=" * 72)
    base_bw = df["is_bigwin"].mean()
    base_bl = df["is_biglose"].mean()
    A(f"総トレード: {len(df):,}")
    A(f"全体の大勝ち率: {base_bw*100:.1f}%  /  大負け率: {base_bl*100:.1f}%")
    A(f"ネット(大勝ち率-大負け率): {(base_bw-base_bl)*100:+.1f}%")
    A("")
    A("※各特徴帯で『大勝ち率』『大負け率』『ネット』を見る。")
    A("  ネットが大きくプラスの帯 = 大勝ちを取りやすく大負けしにくい＝狙い目。")
    A("")

    specs = [
        ("時価総額_億", "時価総額(億円)",
         [0, 50, 100, 300, 1000, 1e9], ["〜50", "50-100", "100-300", "300-1000", "1000〜"]),
        ("株価", "株価(円)",
         [0, 200, 500, 1000, 3000, 1e9], ["〜200", "200-500", "500-1000", "1000-3000", "3000〜"]),
        ("高値からの位置", "52週高値から(%)",
         [-1e9, -50, -30, -15, -5, 1e9], ["〜-50", "-50〜-30", "-30〜-15", "-15〜-5", "-5〜"]),
        ("直近1M騰落", "直近1か月騰落(%)",
         [-1e9, -10, 0, 10, 30, 1e9], ["〜-10", "-10〜0", "0-10", "10-30", "30〜"]),
        ("出来高急増", "出来高急増(20日平均比)",
         [0, 1, 2, 3, 5, 1e9], ["〜1", "1-2", "2-3", "3-5", "5〜"]),
        ("相場環境", "エントリー日の相場(%)",
         [-1e9, -1, 0, 1, 1e9], ["〜-1(下落)", "-1〜0", "0〜1", "1〜(上昇)"]),
    ]

    for col, title, bins, labels in specs:
        if col not in df.columns:
            continue
        sub = df[df[col].notna()].copy()
        if len(sub) < 20:
            continue
        sub["_b"] = pd.cut(sub[col], bins=bins, labels=labels)
        A(f"■ {title}")
        for b, g in sub.groupby("_b", observed=True):
            bw = g["is_bigwin"].mean() * 100
            bl = g["is_biglose"].mean() * 100
            net = bw - bl
            avg = g["return"].mean() * 100
            mark = "◎" if net > (base_bw - base_bl) * 100 + 3 else \
                   ("×" if net < (base_bw - base_bl) * 100 - 3 else " ")
            bar = "█" * int(max(0, bw))
            A(f"  {str(b):>12} : 大勝ち{bw:5.1f}%  大負け{bl:5.1f}%  "
              f"ネット{net:+6.1f}%  平均{avg:+6.2f}%  (n={len(g):>5}) {mark}")
        A("")

    # 業種別の大勝ち率
    if "業種" in df.columns and df["業種"].notna().any():
        A("■ 業種別 大勝ち率（n≥50・ネット上位10）")
        sec = df[df["業種"].notna()].groupby("業種").agg(
            件数=("is_bigwin", "size"),
            大勝ち=("is_bigwin", "mean"),
            大負け=("is_biglose", "mean"),
            平均=("return", "mean")).reset_index()
        sec = sec[sec["件数"] >= 50]
        sec["ネット"] = sec["大勝ち"] - sec["大負け"]
        sec = sec.sort_values("ネット", ascending=False).head(10)
        for _, r in sec.iterrows():
            A(f"  {r['業種']:<14} : 大勝ち{r['大勝ち']*100:5.1f}%  "
              f"大負け{r['大負け']*100:5.1f}%  ネット{r['ネット']*100:+6.1f}%  "
              f"平均{r['平均']*100:+6.2f}%  (n={int(r['件数'])})")
        A("")

    # 戦略別
    A("■ 戦略別 大勝ち率")
    for name, g in df.groupby("戦略"):
        bw = g["is_bigwin"].mean() * 100
        bl = g["is_biglose"].mean() * 100
        A(f"  {name:<16} : 大勝ち{bw:5.1f}%  大負け{bl:5.1f}%  "
          f"ネット{bw-bl:+6.1f}%  (n={len(g):,})")
    A("")

    A("=" * 72)
    A("読み方")
    A("=" * 72)
    A("・『ネット』(大勝ち率−大負け率)がプラスに大きい帯が、大勝ち狙いの候補。")
    A("・ただし大勝ちを狙う条件は取引機会が減りがち。件数(n)も確認。")
    A("・勝率が高い条件と、大勝ちしやすい条件は別物。")
    A("  『こまめに勝つ』か『大きく取る』か、狙いに応じて使い分ける。")
    A("・ここで見えた条件も、別期間での再現確認が必要。")
    return "\n".join(L)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    # analyze_trades の明細CSVを読み込む（無ければ作成を促す）
    import glob
    files = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "*_trade_analysis.csv")))
    if not files:
        print("先に  py main.py analyze-trades  を実行してください。")
        print("（大勝ち分析は、その明細データ（*_trade_analysis.csv）を使います）")
        return
    latest = files[-1]
    print(f"読み込み: {latest}")
    trades = pd.read_csv(latest)
    print(f"  {len(trades):,} トレード\n")

    report = analyze(trades)
    print(report)

    import datetime as dt
    from sources import JST
    today = dt.datetime.now(JST).date().isoformat()
    path = os.path.join(config.OUTPUT_DIR, f"{today}_bigwin.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n保存: {path}")


if __name__ == "__main__":
    main()
