# -*- coding: utf-8 -*-
"""
streak_market_sector_analysis.py - 地合いフィルター＆セクターモメンタム分析モジュール（sector33対応版）
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import datetime as dt

from sources import JQuants
import config
from badnews import _bad_flags
import refine

def run_market_sector_analysis(quotes: pd.DataFrame, fin=None, listed=None, margin=None):
    print("=" * 76)
    print("『地合いフィルター ＆ セクターモメンタム』複合検証分析（33業種対応版）")
    print("=" * 76)

    shares_map = {}
    if fin is not None and "shares_outstanding" in fin.columns:
        f_sorted = fin.sort_values("disclosed_date").dropna(subset=["shares_outstanding"])
        shares_map = dict(f_sorted.groupby("code")["shares_outstanding"].last())

    sector_map = {}
    if listed is not None:
        code_col = "Code" if "Code" in listed.columns else ("code" if "code" in listed.columns else None)
        sec_col = None
        for c in ["sector33", "Sector33CodeName", "sector33_code", "Sector17CodeName"]:
            if c in listed.columns:
                sec_col = c
                break
        if code_col and sec_col:
            sector_map = dict(zip(listed[code_col].astype(str).str.slice(0, 4), listed[sec_col]))

    qi = quotes.sort_values(["code", "date"]).reset_index(drop=True)
    g = qi.groupby("code")

    print("指標および市場平均（地合い）の計算中...")
    qi["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    qi["vol_ratio"] = qi["volume"] / qi["vol_ma20"]
    qi["high_52w"] = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / qi["high_52w"] - 1.0
    qi["day_range"] = (qi["high"] - qi["low"]) / qi["open"] * 100.0

    qi["ret_1d"] = g["close"].pct_change()
    daily_market = qi.groupby("date")["ret_1d"].mean().reset_index(name="market_ret")
    daily_market["market_ma5"] = daily_market["market_ret"].rolling(5, min_periods=1).mean()
    
    qi = pd.merge(qi, daily_market[["date", "market_ma5"]], on="date", how="left")

    for i in range(1, 6):
        qi[f"fwd_max_{i}"] = (g["high"].shift(-1).rolling(i, min_periods=1).max() / qi["close"] - 1.0) * 100.0
        qi[f"fwd_min_{i}"] = (g["low"].shift(-1).rolling(i, min_periods=1).min() / qi["close"] - 1.0) * 100.0

    codes_str = qi["code"].astype(str)
    shares_series = codes_str.map(shares_map).fillna(0)
    qi["mcap_oku"] = qi["close"] * shares_series / 1e8
    qi["is_small_cap"] = (qi["mcap_oku"] > 0) & (qi["mcap_oku"] < 100)
    qi["is_bottom"] = qi["pct_from_high"] <= -0.30
    qi["is_vol_surge"] = qi["vol_ratio"] >= 2.0
    qi["is_range_wide"] = qi["day_range"] >= 7.0
    
    qi["sector"] = codes_str.str.slice(0, 4).map(sector_map).fillna("その他")

    bad_set = _bad_flags(fin) if fin is not None else set()
    qi["date_str"] = pd.to_datetime(qi["date"]).dt.strftime("%Y-%m-%d")
    qi["is_clean"] = [1 if (str(c), d) not in bad_set else 0 for c, d in zip(codes_str, qi["date_str"])]

    qi["is_good_regime"] = qi["market_ma5"] >= 0.0

    b6_mask = qi["is_bottom"] & qi["is_small_cap"] & qi["is_vol_surge"] & qi["is_range_wide"] & (qi["is_clean"] == 1) & qi["is_good_regime"]
    qi["date_dt"] = pd.to_datetime(qi["date"])
    all_signals = qi[b6_mask].copy()

    best_per_day = all_signals.sort_values(["date", "vol_ratio"], ascending=[True, False]).groupby("date").first().reset_index()
    sub_best = best_per_day[best_per_day["date_dt"] >= "2025-01-01"].copy()
    
    tp = 10.0 # +10%
    hz = 5  # 5日
    
    outcomes = []
    for _, row in sub_best.iterrows():
        max_h = row[f"fwd_max_{hz}"]
        won = 1 if max_h >= tp else 0
        outcomes.append({
            "date": row["date_dt"],
            "code": row["code"],
            "sector": row["sector"],
            "won": won
        })
        
    df_out = pd.DataFrame(outcomes)
    print(f"\n【検証結果】地合いフィルター（市場MA5 >= 0）適用後の総シグナル数: {len(df_out)} 件")
    if len(df_out) > 0:
        win_cnt = df_out["won"].sum()
        print(f"地合いフィルター適用後 勝率 (+10%到達率): {win_cnt / len(df_out) * 100.0:.1f}%")

    print("\n--- 【セクター別 勝率・件数】 ---")
    if len(df_out) > 0:
        res = []
        for name, group in df_out.groupby("sector", observed=False):
            cnt = len(group)
            win_p = (group["won"].sum() / cnt * 100.0) if cnt > 0 else 0.0
            res.append({"セクター": name, "件数": cnt, "勝率": f"{win_p:.1f}%"})
        res_df = pd.DataFrame(res).sort_values("件数", ascending=False)
        print(res_df.to_string(index=False))
