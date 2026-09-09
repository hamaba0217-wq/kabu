# -*- coding: utf-8 -*-
"""
streak_deep_dive.py - 連騰数（3〜7日連続プラス）×出来高×買値の最適化検証
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import limitup

def run_deep_dive(quotes: pd.DataFrame, fin=None, listed=None):
    qi = quotes.sort_values(["code", "date"]).reset_index(drop=True)
    g = qi.groupby("code")

    qi["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    qi["vol_ratio"] = qi["volume"] / qi["vol_ma20"]

    # 連騰数計算
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

    print("=" * 76)
    print("連騰数（3〜6連騰以上）× 翌日エントリー検証")
    print(f"母集団: {len(df):,} 行 / 分割境界: {mid_date.strftime('%Y-%m-%d')}")
    print("=" * 76)

    # 1. 連騰数別の単純リターン（前日引け買い -> 翌日引け売り）
    print("\n--- 【1. 連騰数別の翌日リターン（前日引け買い → 翌日引け売り）】 ---")
    rows1 = []
    for s_min in [3, 4, 5, 6]:
        sub = df[df["streak"] >= s_min]
        ret = (sub["next_close"] / sub["close"] - 1.0) * 100.0
        h1 = sub[sub["date"] <= mid_date]
        h2 = sub[sub["date"] > mid_date]
        r1 = (h1["next_close"] / h1["close"] - 1.0) * 100.0 if len(h1) else np.array([])
        r2 = (h2["next_close"] / h2["close"] - 1.0) * 100.0 if len(h2) else np.array([])

        rows1.append({
            "条件": f"{s_min}連騰以上",
            "件数": len(sub),
            "勝率": f"{(ret > 0).mean()*100:.1f}%",
            "平均R(%)": round(ret.mean(), 2),
            "中央値(%)": round(float(np.median(ret)), 2),
            "前半R(%)": round(r1.mean(), 2) if len(r1) else np.nan,
            "後半R(%)": round(r2.mean(), 2) if len(r2) else np.nan,
            "判定": "◎" if (len(r1) and len(r2) and r1.mean() > 0 and r2.mean() > 0) else ("△" if (len(r1) and len(r2) and (r1.mean() > 0 or r2.mean() > 0)) else "×"),
        })
    print(pd.DataFrame(rows1).to_string(index=False))

    # 2. 5連騰以上 × 出来高急増（1.5倍、2倍、3倍）
    print("\n--- 【2. 5連騰以上 × 出来高条件（前日引け買い → 翌日引け売り）】 ---")
    rows2 = []
    sub5 = df[df["streak"] >= 5]
    for vr_min in [1.0, 1.5, 2.0, 3.0]:
        sub = sub5[sub5["vol_ratio"] >= vr_min]
        if len(sub) < 30: continue
        ret = (sub["next_close"] / sub["close"] - 1.0) * 100.0
        h1 = sub[sub["date"] <= mid_date]
        h2 = sub[sub["date"] > mid_date]
        r1 = (h1["next_close"] / h1["close"] - 1.0) * 100.0 if len(h1) else np.array([])
        r2 = (h2["next_close"] / h2["close"] - 1.0) * 100.0 if len(h2) else np.array([])

        rows2.append({
            "出来高条件": f"出来高{vr_min}倍以上",
            "件数": len(sub),
            "勝率": f"{(ret > 0).mean()*100:.1f}%",
            "平均R(%)": round(ret.mean(), 2),
            "前半R(%)": round(r1.mean(), 2) if len(r1) else np.nan,
            "後半R(%)": round(r2.mean(), 2) if len(r2) else np.nan,
            "判定": "◎" if (len(r1) and len(r2) and r1.mean() > 0 and r2.mean() > 0) else ("△" if (len(r1) and len(r2) and (r1.mean() > 0 or r2.mean() > 0)) else "×"),
        })
    print(pd.DataFrame(rows2).to_string(index=False))

    # 3. 翌日「いくらで買うか」（前日終値比での指値買い検証）
    # 5連騰銘柄を対象に、翌日どこで買うのが最も期待値・勝率が高いか
    # 買い指値: 
    # (a) 前日終値（C0）
    # (b) 前日終値 -1%, -2%, -3%
    # (c) 翌日始値（Open）
    # (d) 翌日始値 -1%, -2%
    print("\n--- 【3. 5連騰銘柄の『どこで買うか（買値最適化）』】（大引け決済） ---")
    rows3 = []
    # 買い位置テスト
    # 約定条件: next_low <= buy_px
    # リターン: next_close / buy_px - 1.0
    for label, buy_px_series in [
        ("前日終値成行(引け買い)", sub5["close"]),
        ("前日終値-1%指値", sub5["close"] * 0.99),
        ("前日終値-2%指値", sub5["close"] * 0.98),
        ("前日終値-3%指値", sub5["close"] * 0.97),
        ("翌日始値成行(寄り買い)", sub5["next_open"]),
        ("翌日始値-1%指値", sub5["next_open"] * 0.99),
        ("翌日始値-2%指値", sub5["next_open"] * 0.98),
    ]:
        hit_mask = sub5["next_low"] <= buy_px_series
        matched = sub5[hit_mask]
        b_px = buy_px_series[hit_mask]
        ret = (matched["next_close"] / b_px - 1.0) * 100.0

        h1 = matched[matched["date"] <= mid_date]
        b_px_h1 = buy_px_series[matched["date"] <= mid_date]
        r1 = (h1["next_close"] / b_px_h1 - 1.0) * 100.0 if len(h1) else np.array([])

        h2 = matched[matched["date"] > mid_date]
        b_px_h2 = buy_px_series[matched["date"] > mid_date]
        r2 = (h2["next_close"] / b_px_h2 - 1.0) * 100.0 if len(h2) else np.array([])

        rows3.append({
            "買値ルール": label,
            "約定件数": len(matched),
            "約定率": f"{len(matched)/len(sub5)*100:.1f}%",
            "勝率": f"{(ret > 0).mean()*100:.1f}%",
            "平均R(%)": round(ret.mean(), 2),
            "前半R(%)": round(r1.mean(), 2) if len(r1) else np.nan,
            "後半R(%)": round(r2.mean(), 2) if len(r2) else np.nan,
            "判定": "◎" if (len(r1) and len(r2) and r1.mean() > 0 and r2.mean() > 0) else "×",
        })
    print(pd.DataFrame(rows3).to_string(index=False))
