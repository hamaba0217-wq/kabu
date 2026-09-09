# -*- coding: utf-8 -*-
"""
limitup_streak_buy_optimize.py - 連騰シグナル点灯後の「買値（指値）最適化」実データ検証
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import limitup


def run_buy_optimization(quotes: pd.DataFrame, fin=None, listed=None):
    print("=" * 76)
    print("連騰シグナル銘柄の『買値（指値）最適化』実データ検証")
    print("※ 3連騰以上（または4連騰以上）のシグナル点灯後、翌日にどこで買うべきか")
    print("※ 手数料・スリッページ未考慮 / 投資助言ではありません")
    print("=" * 76)

    qi = quotes.sort_values(["code", "date"]).reset_index(drop=True)
    g = qi.groupby("code")

    # 1. 出来高20日平均比
    qi["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    qi["vol_ratio"] = qi["volume"] / qi["vol_ma20"]

    # 2. 連続上昇日数
    streaks = []
    for _, grp in g:
        c_vals = grp["close"].values
        stk = np.zeros(len(c_vals), dtype=int)
        cnt = 0
        for i in range(len(c_vals) - 1):
            if c_vals[i + 1] > c_vals[i]:
                cnt += 1
            else:
                cnt = 0
            stk[i + 1] = cnt
        streaks.append(stk)
    qi["streak"] = np.concatenate(streaks)

    # 翌営業日の株価
    qi["next_open"] = g["open"].shift(-1)
    qi["next_high"] = g["high"].shift(-1)
    qi["next_low"] = g["low"].shift(-1)
    qi["next_close"] = g["close"].shift(-1)
    qi["next_date"] = g["date"].shift(-1)

    df = qi[qi["next_close"].notna()].copy()
    all_dates = sorted(df["date"].unique())
    mid_date = pd.to_datetime(all_dates[len(all_dates) // 2])

    # 対象：3連騰以上 ＆ 出来高3倍以上（先ほどの検証で最もリフトが高かった母集団）
    sub_base = df[(df["streak"] >= 3) & (df["vol_ratio"] >= 3.0)].copy()
    print(f"検証対象シグナル銘柄（3連騰以上 × 出来高3倍以上）: {len(sub_base):,} 件 / 分割境界: {mid_date.strftime('%Y-%m-%d')}\n")

    if len(sub_base) == 0:
        print("該当サンプルがありません。")
        return

    # 買値バリエーションごとのシミュレーション
    # 約定条件: next_low <= buy_px
    # 決済条件: 翌日大引け(next_close)で決済、または5日以内+10%到達
    rules = [
        ("前日終値成行(引け買い)", sub_base["close"]),
        ("前日終値 -1%指値", sub_base["close"] * 0.99),
        ("前日終値 -2%指値", sub_base["close"] * 0.98),
        ("前日終値 -3%指値", sub_base["close"] * 0.97),
        ("翌日始値成行(寄り買い)", sub_base["next_open"]),
        ("翌日始値 -1%指値", sub_base["next_open"] * 0.99),
        ("翌日始値 -2%指値", sub_base["next_open"] * 0.98),
    ]

    rows = []
    for label, buy_px_series in rules:
        # 約定判定 (翌日安値が指値以下)
        hit_mask = sub_base["next_low"] <= buy_px_series
        matched = sub_base[hit_mask].copy()
        if len(matched) < 50:
            continue
        b_px = buy_px_series[hit_mask]

        # リターン計算 (翌日終値決済)
        ret_close = (matched["next_close"] / b_px - 1.0) * 100.0

        # 分割検証
        h1 = matched[matched["date"] <= mid_date]
        b1 = b_px[matched["date"] <= mid_date]
        r1 = (h1["next_close"] / b1 - 1.0) * 100.0 if len(h1) else np.array([])

        h2 = matched[matched["date"] > mid_date]
        b2 = b_px[matched["date"] > mid_date]
        r2 = (h2["next_close"] / b2 - 1.0) * 100.0 if len(h2) else np.array([])

        rows.append({
            "買値ルール": label,
            "約定件数": len(matched),
            "約定率": f"{len(matched)/len(sub_base)*100:.1f}%",
            "勝率": f"{(ret_close > 0).mean()*100:.1f}%",
            "平均R(%)": round(ret_close.mean(), 2),
            "中央値(%)": round(float(np.median(ret_close)), 2),
            "前半R(%)": round(r1.mean(), 2) if len(r1) else np.nan,
            "後半R(%)": round(r2.mean(), 2) if len(r2) else np.nan,
            "判定": "◎両プラス" if (len(r1) and len(r2) and r1.mean() > 0 and r2.mean() > 0) else ("△片方" if (len(r1) and len(r2) and (r1.mean() > 0 or r2.mean() > 0)) else "×全滅"),
        })

    res = pd.DataFrame(rows).sort_values("平均R(%)", ascending=False)
    print("--- 【3連騰以上 × 出来高3倍以上シグナルの『買値最適化』結果】 ---")
    print(res.to_string(index=False))
