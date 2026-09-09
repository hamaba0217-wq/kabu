# -*- coding: utf-8 -*-
"""
streak_bottom_explosion.py - 底値圏×出来高急増×レンジ拡大×小型株の初動爆発戦略の検証
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import refine
from badnews import _bad_flags

def run_explosion(quotes: pd.DataFrame, fin=None, listed=None, margin=None):
    print("=" * 76)
    print("『底値圏 × 出来高急増 × レンジ拡大 × 小型株』初動爆発戦略の検証")
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

    # 未来10営業日以内の最高値 (+15%到達を判定)
    HORIZON = 10
    THRESHOLD = 15.0
    qi["fut_high_10d"] = g["high"].transform(lambda s: s.shift(-1).rolling(HORIZON, min_periods=HORIZON).max())
    qi["max_up_10d"] = (qi["fut_high_10d"] / qi["close"] - 1.0) * 100.0

    # 翌日終値決済リターン
    qi["next_close"] = g["close"].shift(-1)
    qi["next_ret"] = (qi["next_close"] / qi["close"] - 1.0) * 100.0

    df = qi[qi["max_up_10d"].notna()].copy()

    print("属性・観点の付与中...")
    codes_str = df["code"].astype(str)
    
    shares_series = codes_str.map(shares_map).fillna(0)
    df["mcap_oku"] = df["close"] * shares_series / 1e8
    df["is_small_cap"] = (df["mcap_oku"] > 0) & (df["mcap_oku"] < 100)
    df["is_bottom"] = df["pct_from_high"] <= -0.30
    df["is_vol_surge"] = df["vol_ratio"] >= 2.0
    df["is_range_wide"] = df["day_range"] >= 5.0

    bad_set = _bad_flags(fin) if fin is not None else set()
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code_str"] = codes_str
    df["is_clean"] = [1 if (c, d) not in bad_set else 0 for c, d in zip(df["code_str"], df["date_str"])]

    all_dates = sorted(df["date"].unique())
    mid_date = pd.to_datetime(all_dates[len(all_dates) // 2])

    print(f"全体サンプル数: {len(df):,} 件 / 分割境界: {mid_date.strftime('%Y-%m-%d')}\n")

    patterns = [
        ("基準: 全体ベース", df["close"] > 0),
        ("B1: 底値圏(-30%↓)", df["is_bottom"]),
        ("B2: 底値圏 × 小型株(<100億)", df["is_bottom"] & df["is_small_cap"]),
        ("B3: 底値圏 × 小型株 × 出来高急増(2倍以上)", df["is_bottom"] & df["is_small_cap"] & df["is_vol_surge"]),
        ("B4: 底値圏 × 小型株 × レンジ拡大(5%超)", df["is_bottom"] & df["is_small_cap"] & df["is_range_wide"]),
        ("B5: 底値圏 × 小型株 × 出来高2倍 × レンジ5%超", df["is_bottom"] & df["is_small_cap"] & df["is_vol_surge"] & df["is_range_wide"]),
        ("B6: 【厳選最強】底値圏 × 小型株 × 出来高2倍 × レンジ5%超 × 悪材料なし", df["is_bottom"] & df["is_small_cap"] & df["is_vol_surge"] & df["is_range_wide"] & (df["is_clean"] == 1)),
    ]

    rows = []
    for label, mask in patterns:
        sub = df[mask]
        n = len(sub)
        if n < 10: continue
        
        hit_15 = (sub["max_up_10d"] >= THRESHOLD).astype(int)
        win_rate_15 = hit_15.mean() * 100.0
        avg_ret = sub["next_ret"].mean()
        win_rate_next = (sub["next_ret"] > 0).mean() * 100.0

        h1 = sub[sub["date"] <= mid_date]
        h2 = sub[sub["date"] > mid_date]
        r1 = (h1["max_up_10d"] >= THRESHOLD).mean() * 100.0 if len(h1) else 0.0
        r2 = (h2["max_up_10d"] >= THRESHOLD).mean() * 100.0 if len(h2) else 0.0

        rows.append({
            "検証パターン": label,
            "母数(件)": n,
            "10日内+15%達成率": f"{win_rate_15:.1f}%",
            "翌日勝率": f"{win_rate_next:.1f}%",
            "翌日平均R(%)": round(avg_ret, 2),
            "前半達成率": f"{r1:.1f}%",
            "後半達成率": f"{r2:.1f}%",
            "判定": "◎最強・両立" if (r1 >= 30.0 and r2 >= 30.0) else ("△要確認" if (r1 >= 25.0 or r2 >= 25.0) else "—"),
        })

    res_df = pd.DataFrame(rows).sort_values("10日内+15%達成率", ascending=False)
    print("--- 【底値圏・初動爆発戦略の検証結果（詳細データ）】 ---")
    print(res_df.to_string(index=False))
