# -*- coding: utf-8 -*-
"""
利確ライン別の到達率分析（運用ルールに直結）

運用イメージ
------------
・エントリー後、指値を +X% に置く（X = 6,7,8,...,15）
・5営業日 または 10営業日 以内に指値に届けば利確、届かなければ売る
そこで「各利確ライン +6〜+15% に、5日/10日以内で到達した割合」を見る。

売られすぎ水準ごとに見る
------------------------
25日移動平均からの乖離率（前回、最も効いた指標）で銘柄を区切り、
各区分で +6%〜+15% の到達率を一覧にする。
「どの売られすぎ水準で、どの利確ラインが、どれくらい届くか」が分かる。

到達率 = 期間内にその利確ラインに一度でも触れた割合。
指値は「触れれば約定」なので、ザラ場高値ベースの到達で判定する
（ただし終値データのみのため、日々の終値が到達したかで近似）。

先読み防止
----------
売られすぎ指標は as_of 時点。到達判定は as_of 以降の値動き。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import config
import technical
from oversold import add_oversold_indicators


# 利確ライン（+6%〜+15%）
TARGETS = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
# 保有期間（営業日）
HOLD_DAYS = [5, 10]
STEP_DAYS = 5

# 売られすぎ区分（25日移動平均乖離率）
DEV_BINS = [-1e9, -30, -20, -10, -5, 0, 1e9]
DEV_LABELS = ["〜-30", "-30〜-20", "-20〜-10", "-10〜-5", "-5〜0", "0〜(平均超)"]


def run(quotes, listed, step_days=STEP_DAYS):
    qi = add_oversold_indicators(quotes)
    by_code = technical._build_index(qi)
    all_dates = np.sort(qi["date"].unique())

    print("  売られすぎ指標つきでサンプルを収集中...")
    max_hold = max(HOLD_DAYS)
    samples = []
    start_idx, end_idx = 260, len(all_dates) - max_hold
    for as_of in all_dates[start_idx:end_idx:step_days]:
        snap = qi[qi["date"] == as_of]
        if snap.empty:
            continue
        for _, s in snap.iterrows():
            if pd.isna(s["ma25_dev"]):
                continue
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
            rec = {"ma25_dev": s["ma25_dev"]}
            # 各保有期間で、期間内の最大リターンを記録
            for hd in HOLD_DAYS:
                fut = close_arr[pos:pos + hd]
                if len(fut) < hd * 0.5:
                    rec[f"max_{hd}"] = np.nan
                else:
                    rec[f"max_{hd}"] = (fut / entry).max() - 1.0
            samples.append(rec)
    df = pd.DataFrame(samples)
    print(f"  サンプル: {len(df):,} 件")
    return df


def _reach_rate(series_max, target):
    """最大リターンが target 以上だった割合（=指値到達率）。"""
    valid = series_max.dropna()
    if len(valid) == 0:
        return None
    return (valid >= target).mean() * 100


def summarize(df):
    L = ["=" * 90,
         "利確ライン別 到達率（+6%〜+15% × 保有5日/10日）／売られすぎ水準ごと",
         "=" * 90, ""]
    L.append(f"総サンプル: {len(df):,} 件")
    L.append("")
    L.append("各セル = その利確ラインに、保有期間内で到達した割合(%)。")
    L.append("指値をそこに置けば約定した割合と考えてよい（終値ベースの近似）。")
    L.append("")

    df = df.copy()
    df["_b"] = pd.cut(df["ma25_dev"], bins=DEV_BINS, labels=DEV_LABELS)

    for hd in HOLD_DAYS:
        L.append("=" * 90)
        L.append(f"【保有 {hd}営業日以内】各利確ラインへの到達率(%)")
        L.append("=" * 90)
        # ヘッダー
        header = "  売られすぎ(移動平均乖離)  " + "".join(f"+{int(t*100):>4}%" for t in TARGETS) + "   件数"
        L.append(header)
        col = f"max_{hd}"
        # 全体
        row = "  " + f"{'全体':<20}"
        for t in TARGETS:
            r = _reach_rate(df[col], t)
            row += f"{r:>5.0f} " if r is not None else "   -- "
        row += f"  {df[col].notna().sum():>6}"
        L.append(row)
        L.append("  " + "-" * 86)
        # 区分ごと
        for b in DEV_LABELS:
            g = df[df["_b"] == b]
            if len(g) < 30:
                continue
            row = "  " + f"{b:<20}"
            for t in TARGETS:
                r = _reach_rate(g[col], t)
                row += f"{r:>5.0f} " if r is not None else "   -- "
            row += f"  {g[col].notna().sum():>6}"
            L.append(row)
        L.append("")

    L += ["=" * 90, "読み方", "=" * 90,
          "・行=売られすぎの度合い、列=利確ライン。セルは到達率(%)。",
          "・例: 『-20〜-10』行の『+8%』列が50なら、移動平均から-10〜-20%乖離した",
          "  銘柄は、保有中に半分が+8%に到達した、という意味。",
          "・利確ラインを上げるほど到達率は下がる（当然）。どこで折り合うかを見る。",
          "・5日と10日を比べ、期間を延ばして到達率がどれだけ上がるか（待つ価値）。",
          "・到達率が高い＝実際に儲かる、ではない。到達しなかった時の下落も別途重要。",
          "  ここでは『指値がどれくらい約定するか』だけを見ている。",
          "・売られすぎが深いほど高い利確ラインに届きやすいが、",
          "  外した時の下落（落ちるナイフ）リスクも上がる点に注意。",
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
    path = os.path.join(config.OUTPUT_DIR, f"{today}_profit_targets.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nサンプル明細: {path}")


if __name__ == "__main__":
    main()
