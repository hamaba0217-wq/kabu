# -*- coding: utf-8 -*-
"""
売られすぎ銘柄の +5%到達率 分析

「売られすぎた銘柄が反発して +5%以上上がるか」を検証する。
これまでの「勝率・対相場超過」とは評価軸が違い、
主役は「+5%以上の上昇に到達した割合（到達率）」。

売られすぎの3指標（まとめて網羅）
--------------------------------
  1. 25日移動平均からの下方乖離率  … 平均からどれだけ下に離れたか
  2. 52週安値からの距離           … 安値にどれだけ近いか
  3. RSI(14)                      … 定番の売られすぎ指標（30以下が売られすぎ）

各指標を水準で区切り、その水準の銘柄が
・N営業日以内に +5%以上上がった割合（到達率）
・平均リターン
・+5%到達 vs -5%以上下落 のネット
を見る。生存者バイアスを避け、全該当銘柄を分母にする。

先読み防止
----------
指標はすべて as_of 時点で計算できるもの。将来リターンは as_of 以降で測る。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import backtest
import config
import technical


TARGET_UP = 0.05        # +5%到達を「成功」とする
TARGET_UP2 = 0.10       # +10%到達も併せて見る
HORIZON = 20            # 何営業日以内に到達するかを見る
STEP_DAYS = 5


def add_oversold_indicators(quotes):
    """売られすぎ3指標を計算する。"""
    q = quotes.sort_values(["code", "date"]).copy()
    g = q.groupby("code")

    # 1. 25日移動平均乖離率
    q["ma25"] = g["close"].transform(lambda s: s.rolling(25, min_periods=10).mean())
    q["ma25_dev"] = (q["close"] / q["ma25"] - 1.0) * 100      # %

    # 2. 52週安値からの距離
    q["low_52w"] = g["close"].transform(
        lambda s: s.shift(1).rolling(250, min_periods=20).min())
    q["from_low"] = (q["close"] / q["low_52w"] - 1.0) * 100   # %。0に近いほど安値圏

    # 3. RSI(14)
    def _rsi(s, period=14):
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        # 損失ゼロ（下げなし）はRSI=100、上げゼロ（gain=0）はRSI=0
        rsi = rsi.where(loss != 0, 100.0)
        rsi = rsi.where(~((loss == 0) & (gain == 0)), 50.0)  # 無変動は中立50
        return rsi
    q["rsi"] = g["close"].transform(_rsi)

    return q


def _reaches_target(path):
    """path中に+5%/+10%到達したか。到達フラグと最大・最終リターンを返す。"""
    if len(path) == 0:
        return None
    hit5 = bool((path >= 1 + TARGET_UP).any())
    hit10 = bool((path >= 1 + TARGET_UP2).any())
    max_ret = path.max() - 1.0
    final = path[-1] - 1.0
    return hit5, hit10, max_ret, final


def run(quotes, listed, step_days=STEP_DAYS):
    qi = add_oversold_indicators(quotes)
    by_code = technical._build_index(qi)
    all_dates = np.sort(qi["date"].unique())

    # 全銘柄・全時点のサンプルを集める（売られすぎ指標つき）
    print("  売られすぎ指標つきでサンプルを収集中...")
    samples = []
    start_idx, end_idx = 260, len(all_dates) - HORIZON
    for as_of in all_dates[start_idx:end_idx:step_days]:
        snap = qi[qi["date"] == as_of]
        if snap.empty:
            continue
        for _, s in snap.iterrows():
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
            fut = close_arr[pos:pos + HORIZON]
            if len(fut) < HORIZON * 0.5:
                continue
            res = _reaches_target(fut / entry)
            if res is None:
                continue
            hit5, hit10, max_ret, final = res
            samples.append({
                "as_of": pd.Timestamp(as_of),
                "ma25_dev": s["ma25_dev"],
                "from_low": s["from_low"],
                "rsi": s["rsi"],
                "hit_5": hit5,
                "hit_10": hit10,
                "max_ret": max_ret,
                "final_ret": final,
            })
    df = pd.DataFrame(samples)
    print(f"  サンプル: {len(df):,} 件")
    return df


def _bucket_table(df, col, bins, labels, title):
    sub = df[df[col].notna()].copy()
    if len(sub) < 50:
        return None
    sub["_b"] = pd.cut(sub[col], bins=bins, labels=labels)
    base5 = sub["hit_5"].mean()
    base10 = sub["hit_10"].mean()
    L = [f"■ {title}"]
    L.append(f"  （全体: +5%到達{base5*100:.1f}% / +10%到達{base10*100:.1f}%）")
    for b, g in sub.groupby("_b", observed=True):
        hit5 = g["hit_5"].mean() * 100
        hit10 = g["hit_10"].mean() * 100
        avg = g["final_ret"].mean() * 100
        lift5 = hit5 / (base5 * 100) if base5 > 0 else 0
        lift10 = hit10 / (base10 * 100) if base10 > 0 else 0
        mark = "◎" if lift10 >= 1.15 else ("×" if lift10 <= 0.85 else " ")
        L.append(f"    {str(b):>12} : +5%{hit5:5.1f}%(x{lift5:4.2f})  "
                 f"+10%{hit10:5.1f}%(x{lift10:4.2f})  "
                 f"平均{avg:+6.2f}%  (n={len(g):>6}) {mark}")
    return "\n".join(L)


def summarize(df):
    L = ["=" * 82,
         f"売られすぎ銘柄の +5% / +10%到達率 分析（{HORIZON}営業日以内）",
         "=" * 82, ""]
    L.append(f"総サンプル: {len(df):,} 件")
    L.append(f"全体の +5%到達率: {df['hit_5'].mean()*100:.1f}%  "
             f"/ +10%到達率: {df['hit_10'].mean()*100:.1f}%")
    L.append("")
    L.append("※各指標の水準ごとに、+5%/+10%以上に到達した割合を見る。")
    L.append("  (x1.72)はリフト。◎は+10%到達のリフトが1.15以上の水準。")
    L.append("  +10%を狙うなら+10%のリフトが高い水準を見る。")
    L.append("")

    tables = [
        ("ma25_dev", [-1e9, -30, -20, -10, -5, 0, 1e9],
         ["〜-30", "-30〜-20", "-20〜-10", "-10〜-5", "-5〜0", "0〜(平均超)"],
         "25日移動平均からの乖離率（下に離れ% = 売られすぎ）"),
        ("from_low", [0, 5, 10, 20, 40, 1e9],
         ["安値±5%", "5-10%", "10-20%", "20-40%", "40%〜"],
         "52週安値からの距離（0に近い = 安値圏）"),
        ("rsi", [0, 20, 30, 40, 50, 70, 1e9],
         ["〜20", "20-30", "30-40", "40-50", "50-70", "70〜"],
         "RSI(14)（30以下 = 売られすぎ）"),
    ]
    for col, bins, labels, title in tables:
        t = _bucket_table(df, col, bins, labels, title)
        if t:
            L.append(t); L.append("")

    L += ["=" * 82, "読み方", "=" * 82,
          "・『+5%到達率』が高くリフトが大きい水準 = 売られすぎ反発を取りやすい。",
          "・ただし『平均リターン』も必ず見る。到達率が高くても、",
          "  そのまま下げ続ける銘柄が多ければ平均はマイナスになる。",
          "・『最大平均』= 期間中の高値の平均。+5%到達しても、",
          "  その後戻せば意味がないので、いつ利確するかが別途重要。",
          "・売られすぎ = 反発しやすい とは限らない。下げ続ける『ナイフ』も多い。",
          "  到達率とセットで平均リターンがプラスの水準を探すこと。",
          "・有望な水準が見つかれば、押し目×高勝率業種と組み合わせて再検証。",
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

    df = run(quotes, listed)
    if df.empty:
        print("サンプルがありませんでした。")
        return
    report = summarize(df)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    path = os.path.join(config.OUTPUT_DIR, f"{today}_oversold.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nサンプル明細: {path}")


if __name__ == "__main__":
    main()
