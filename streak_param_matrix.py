# -*- coding: utf-8 -*-
"""
streak_param_matrix.py - 「一日一銘柄（一番おすすめ）」ルールを加えた最適化マトリックス検証
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import datetime as dt

from sources import JQuants
import config
from badnews import _bad_flags
import refine

def run_matrix(quotes: pd.DataFrame, fin=None, listed=None, margin=None):
    print("=" * 76)
    print("『一日一銘柄（一番おすすめ）』× パラメータ最適化検証")
    print("※ 該当候補の中で出来高倍率（vol_ratio）が最大の銘柄を1日1つ選定")
    print("=" * 76)

    shares_map = {}
    if fin is not None and "shares_outstanding" in fin.columns:
        f_sorted = fin.sort_values("disclosed_date").dropna(subset=["shares_outstanding"])
        shares_map = dict(f_sorted.groupby("code")["shares_outstanding"].last())

    qi = quotes.sort_values(["code", "date"]).reset_index(drop=True)
    g = qi.groupby("code")

    print("指標計算中...")
    qi["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    qi["vol_ratio"] = qi["volume"] / qi["vol_ma20"]
    qi["high_52w"] = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / qi["high_52w"] - 1.0
    qi["day_range"] = (qi["high"] - qi["low"]) / qi["open"] * 100.0

    for i in range(1, 11):
        qi[f"fwd_ret_{i}"] = (g["close"].shift(-i) / qi["close"] - 1.0) * 100.0
        qi[f"fwd_max_{i}"] = (g["high"].shift(-1).rolling(i, min_periods=1).max() / qi["close"] - 1.0) * 100.0
        qi[f"fwd_min_{i}"] = (g["low"].shift(-1).rolling(i, min_periods=1).min() / qi["close"] - 1.0) * 100.0

    codes_str = qi["code"].astype(str)
    shares_series = codes_str.map(shares_map).fillna(0)
    qi["mcap_oku"] = qi["close"] * shares_series / 1e8
    qi["is_small_cap"] = (qi["mcap_oku"] > 0) & (qi["mcap_oku"] < 100)
    qi["is_bottom"] = qi["pct_from_high"] <= -0.30
    qi["is_vol_surge"] = qi["vol_ratio"] >= 2.0
    qi["is_range_wide"] = qi["day_range"] >= 5.0

    bad_set = _bad_flags(fin) if fin is not None else set()
    qi["date_str"] = pd.to_datetime(qi["date"]).dt.strftime("%Y-%m-%d")
    qi["is_clean"] = [1 if (str(c), d) not in bad_set else 0 for c, d in zip(codes_str, qi["date_str"])]

    b6_mask = qi["is_bottom"] & qi["is_small_cap"] & qi["is_vol_surge"] & qi["is_range_wide"] & (qi["is_clean"] == 1)
    qi["date_dt"] = pd.to_datetime(qi["date"])
    sub_p = qi[(qi["date_dt"] >= "2026-01-01") & (qi["date_dt"] <= "2026-07-01") & b6_mask].copy()

    # 一番おすすめ（出来高倍率が最大の銘柄を1日1つ選択）
    best_per_day = sub_p.sort_values(["date", "vol_ratio"], ascending=[True, False]).groupby("date").first().reset_index()

    print(f"厳選された「毎日一番おすすめの1銘柄」のトレード数 (2026/1/1〜7/1): {len(best_per_day)} 件\n")
    if len(best_per_day) == 0:
        print("該当シグナルがありません。")
        return

    take_profits = [5.0, 8.0, 10.0, 15.0] # %
    stop_losses = [-3.0, -5.0, -8.0, None]
    horizons = [5, 10]

    results = []

    for tp in take_profits:
        for sl in stop_losses:
            for hz in horizons:
                capital = 1000000.0
                rets = []
                wins = 0
                
                for _, row in best_per_day.iterrows():
                    max_h = row[f"fwd_max_{hz}"]
                    min_l = row[f"fwd_min_{hz}"]
                    final_r = row[f"fwd_ret_{hz}"]
                    
                    exit_r = final_r
                    if sl is not None and min_l <= sl:
                        exit_r = sl
                    elif max_h >= tp:
                        exit_r = tp
                        
                    net_r = exit_r - 0.15 # スリッページ
                    capital *= (1.0 + net_r / 100.0)
                    rets.append(net_r)
                    if net_r > 0: wins += 1
                
                win_rate = (wins / len(best_per_day)) * 100.0 if len(best_per_day) > 0 else 0
                mean_ret = np.mean(rets) if len(rets) > 0 else 0
                results.append({
                    "利食い": f"+{int(tp)}%",
                    "損切り": f"{int(sl)}%" if sl is not None else "なし",
                    "保有日数": f"{hz}日",
                    "勝率": f"{win_rate:.1f}%",
                    "平均R": f"{mean_ret:.2f}%",
                    "最終資産": f"{capital:,.0f}円"
                })

    res_df = pd.DataFrame(results)
    print("--- 【「一日一銘柄（一番おすすめ）」パラメータ最適化マトリックス】 ---")
    print(res_df.to_string(index=False))
