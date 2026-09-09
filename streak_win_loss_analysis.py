# -*- coding: utf-8 -*-
"""
streak_win_loss_analysis.py - 日中レンジ7%以上の高勝率フィルターを正しく直結させた分析モジュール
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import datetime as dt

from sources import JQuants
import config
from badnews import _bad_flags
import refine

def run_analysis(quotes: pd.DataFrame, fin=None, listed=None, margin=None):
    print("=" * 76)
    print("『日中レンジ7%以上・高勝率フィルター直結』検証分析")
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

    for i in range(1, 6):
        qi[f"fwd_max_{i}"] = (g["high"].shift(-1).rolling(i, min_periods=1).max() / qi["close"] - 1.0) * 100.0
        qi[f"fwd_min_{i}"] = (g["low"].shift(-1).rolling(i, min_periods=1).min() / qi["close"] - 1.0) * 100.0
        qi[f"fwd_ret_{i}"] = (g["close"].shift(-i) / qi["close"] - 1.0) * 100.0

    codes_str = qi["code"].astype(str)
    shares_series = codes_str.map(shares_map).fillna(0)
    qi["mcap_oku"] = qi["close"] * shares_series / 1e8
    qi["is_small_cap"] = (qi["mcap_oku"] > 0) & (qi["mcap_oku"] < 100)
    qi["is_bottom"] = qi["pct_from_high"] <= -0.30
    qi["is_vol_surge"] = qi["vol_ratio"] >= 2.0
    # ここで日中レンジを明確に 7.0% 以上に直結
    qi["is_range_wide"] = qi["day_range"] >= 7.0

    bad_set = _bad_flags(fin) if fin is not None else set()
    qi["date_str"] = pd.to_datetime(qi["date"]).dt.strftime("%Y-%m-%d")
    qi["is_clean"] = [1 if (str(c), d) not in bad_set else 0 for c, d in zip(codes_str, qi["date_str"])]

    b6_mask = qi["is_bottom"] & qi["is_small_cap"] & qi["is_vol_surge"] & qi["is_range_wide"] & (qi["is_clean"] == 1)
    qi["date_dt"] = pd.to_datetime(qi["date"])
    all_signals = qi[b6_mask].copy()

    best_per_day = all_signals.sort_values(["date", "vol_ratio"], ascending=[True, False]).groupby("date").first().reset_index()
    sub_best = best_per_day[best_per_day["date_dt"] >= "2025-01-01"].copy()
    
    tp = 10.0 # +10%
    hz = 5  # 5日
    
    outcomes = []
    for _, row in sub_best.iterrows():
        max_h = row[f"fwd_max_{hz}"]
        min_l = row[f"fwd_min_{hz}"]
        
        won = 1 if max_h >= tp else 0
        outcomes.append({
            "date": row["date_dt"],
            "code": row["code"],
            "vol_ratio": row["vol_ratio"],
            "day_range": row["day_range"],
            "pct_from_high": row["pct_from_high"] * 100.0,
            "won": won
        })
        
    df_out = pd.DataFrame(outcomes)
    print(f"新フィルター(レンジ7%以上直結)適用後の総シグナル数: {len(df_out)} 件")
    if len(df_out) > 0:
        win_cnt = df_out["won"].sum()
        print(f"強化後勝率 (+10%到達率): {win_cnt / len(df_out) * 100.0:.1f}%")
