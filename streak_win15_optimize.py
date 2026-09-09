# -*- coding: utf-8 -*-
"""
streak_win15_optimize.py - 15%達成率50%以上を狙う「大勝ち条件」の逆算分析
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import refine
from badnews import _bad_flags

def run_win15_opt(quotes: pd.DataFrame, fin=None, listed=None, margin=None):
    print("=" * 76)
    print("5連騰シグナル銘柄 ×『15%達成率50%以上』を叩き出す条件の逆算分析")
    print("※ 手数料・スリッページ未考慮 / 投資助言ではありません")
    print("=" * 76)

    sec_map, mkt_map = {}, {}
    if listed is not None:
        if "sector33" in listed.columns:
            sec_map = dict(zip(listed["code"].astype(str), listed["sector33"]))
        if "market" in listed.columns:
            mkt_map = dict(zip(listed["code"].astype(str), listed["market"]))

    shares_map = {}
    if fin is not None and "shares_outstanding" in fin.columns:
        f_sorted = fin.sort_values("disclosed_date").dropna(subset=["shares_outstanding"])
        shares_map = dict(f_sorted.groupby("code")["shares_outstanding"].last())

    qi = quotes.sort_values(["code", "date"]).reset_index(drop=True)
    g = qi.groupby("code")

    print("指標計算中（ボラティリティ・出来高・高値位置）...")
    qi["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    qi["vol_ratio"] = qi["volume"] / qi["vol_ma20"]

    qi["high_52w"] = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / qi["high_52w"] - 1.0
    qi["day_range"] = (qi["high"] - qi["low"]) / qi["open"] * 100.0

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

    # 未来10営業日以内の最高値 (+15%到達を判定)
    HORIZON = 10
    THRESHOLD = 15.0
    qi["fut_high_10d"] = g["high"].transform(lambda s: s.shift(-1).rolling(HORIZON, min_periods=HORIZON).max())
    qi["max_up_10d"] = (qi["fut_high_10d"] / qi["close"] - 1.0) * 100.0

    df = qi[qi["max_up_10d"].notna()].copy()

    print("属性・観点の付与中...")
    codes_str = df["code"].astype(str)
    
    shares_series = codes_str.map(shares_map).fillna(0)
    df["mcap_oku"] = df["close"] * shares_series / 1e8
    df["is_small_cap"] = (df["mcap_oku"] > 0) & (df["mcap_oku"] < 100)
    df["is_bottom"] = df["pct_from_high"] <= -0.30
    df["is_vol_surge"] = df["vol_ratio"] >= 2.0
    df["is_vol_extreme"] = df["vol_ratio"] >= 4.0
    df["is_range_wide"] = df["day_range"] >= 4.0

    all_dates = sorted(df["date"].unique())
    mid_date = pd.to_datetime(all_dates[len(all_dates) // 2])

    sub5 = df[df["streak"] >= 5].copy()
    print(f"5連騰シグナル対象: {len(sub5):,} 件 / 分割境界: {mid_date.strftime('%Y-%m-%d')}\n")

    # 15%達成率を検証する複合パターン
    patterns = [
        ("基準: 5連騰のみ", sub5["streak"] >= 5),
        ("P1: 5連騰 × 小型株(<100億)", (sub5["streak"] >= 5) & sub5["is_small_cap"]),
        ("P2: 5連騰 × 底値圏(-30%↓)", (sub5["streak"] >= 5) & sub5["is_bottom"]),
        ("P3: 5連騰 × 底値圏 × 小型株", (sub5["streak"] >= 5) & sub5["is_bottom"] & sub5["is_small_cap"]),
        ("P4: 5連騰 × 底値圏 × 小型株 × 出来高2倍以上", (sub5["streak"] >= 5) & sub5["is_bottom"] & sub5["is_small_cap"] & sub5["is_vol_surge"]),
        ("P5: 5連騰 × 底値圏 × 小型株 × 出来高4倍以上", (sub5["streak"] >= 5) & sub5["is_bottom"] & sub5["is_small_cap"] & sub5["is_vol_extreme"]),
        ("P6: 5連騰 × 底値圏 × 小型株 × 日中レンジ4%超", (sub5["streak"] >= 5) & sub5["is_bottom"] & sub5["is_small_cap"] & sub5["is_range_wide"]),
        ("P7: 【特選】5連騰 × 底値圏 × 小型株 × 出来高2倍 × レンジ4%超", (sub5["streak"] >= 5) & sub5["is_bottom"] & sub5["is_small_cap"] & sub5["is_vol_surge"] & sub5["is_range_wide"]),
    ]

    rows = []
    for label, mask in patterns:
        sub = sub5[mask]
        n = len(sub)
        if n < 10: continue
        
        hit_15 = (sub["max_up_10d"] >= THRESHOLD).astype(int)
        win_rate_15 = hit_15.mean() * 100.0

        h1 = sub[sub["date"] <= mid_date]
        h2 = sub[sub["date"] > mid_date]
        r1 = (h1["max_up_10d"] >= THRESHOLD).mean() * 100.0 if len(h1) else 0.0
        r2 = (h2["max_up_10d"] >= THRESHOLD).mean() * 100.0 if len(h2) else 0.0

        rows.append({
            "検証パターン": label,
            "件数": n,
            "10日内+15%達成率": f"{win_rate_15:.1f}%",
            "前半達成率": f"{r1:.1f}%",
            "後半達成率": f"{r2:.1f}%",
            "判定": "◎50%超・両立" if (r1 >= 45.0 and r2 >= 45.0) else ("△要確認" if (r1 >= 40.0 or r2 >= 40.0) else "×未達"),
        })

    res_df = pd.DataFrame(rows).sort_values("10日内+15%達成率", ascending=False)
    print("--- 【10日以内に +15%到達する確率（達成率）の最適化結果】 ---")
    print(res_df.to_string(index=False))
