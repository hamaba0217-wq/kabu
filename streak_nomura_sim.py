# -*- coding: utf-8 -*-
"""
streak_nomura_sim.py - 野村証券手数料・元手250万・50%上限・1ポジション制の15パターン検証
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import datetime as dt

from sources import JQuants
import config
from badnews import _bad_flags
import refine

def nomura_fee(amount):
    # 野村証券現物手数料の標準モデル (約定代金の0.8% + 最低2,750円)
    fee = amount * 0.008
    return max(fee, 2750.0)

def run_nomura(quotes: pd.DataFrame, fin=None, listed=None, margin=None):
    print("=" * 76)
    print("『野村証券手数料 × 250万元手 × 50%上限 × 売り切るまで次不可』15パターン検証")
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
    all_signals = qi[b6_mask].copy()

    best_per_day = all_signals.sort_values(["date", "vol_ratio"], ascending=[True, False]).groupby("date").first().reset_index()
    best_per_day = best_per_day.sort_values("date_dt").reset_index(drop=True)

    start_dates = pd.date_range(start="2025-01-01", periods=15, freq="MS")
    matrix_results = []

    tp = 0.10
    hz = 5
    initial_cash = 2500000.0

    all_dates_q = sorted(qi["date_dt"].unique())
    sig_dict = {row["date_dt"]: row for _, row in best_per_day.iterrows()}

    for start_dt in start_dates:
        end_dt = start_dt + pd.DateOffset(months=6)
        
        cash = initial_cash
        trades = 0
        wins = 0
        
        in_position = False
        holding_code = None
        holding_entry_price = 0.0
        holding_shares = 0
        holding_invest_amount = 0.0
        days_held = 0
        
        dates_in_sim = [d for d in all_dates_q if start_dt <= d <= end_dt]
        
        for d in dates_in_sim:
            if in_position:
                days_held += 1
                day_data = qi[(qi["code"] == holding_code) & (qi["date_dt"] == d)]
                if not day_data.empty:
                    row_d = day_data.iloc[0]
                    h = row_d["high"]
                    l = row_d["low"]
                    c = row_d["close"]
                    
                    hit = False
                    exit_price = c
                    
                    if (h / holding_entry_price - 1.0) >= tp:
                        exit_price = holding_entry_price * (1.0 + tp)
                        hit = True
                    elif days_held >= hz:
                        exit_price = c
                        hit = True
                    
                    if hit:
                        gross_proceeds = holding_shares * exit_price
                        fee_sell = nomura_fee(gross_proceeds)
                        net_proceeds = gross_proceeds - fee_sell
                        cash += net_proceeds
                        
                        profit = net_proceeds - (holding_invest_amount + nomura_fee(holding_invest_amount))
                        if profit > 0: wins += 1
                        trades += 1
                        
                        in_position = False
                        holding_code = None
                        days_held = 0
            
            if not in_position and d in sig_dict:
                row_sig = sig_dict[d]
                alloc_amt = cash * 0.50
                if alloc_amt > cash:
                    alloc_amt = cash
                
                if alloc_amt >= 50000:
                    entry_price = row_sig["close"]
                    fee_buy = nomura_fee(alloc_amt)
                    net_alloc = alloc_amt - fee_buy
                    if net_alloc > 0:
                        shares = int(net_alloc // entry_price)
                        if shares > 0:
                            actual_invest = shares * entry_price
                            total_cost = actual_invest + fee_buy
                            if total_cost <= cash:
                                cash -= total_cost
                                in_position = True
                                holding_code = row_sig["code"]
                                holding_entry_price = entry_price
                                holding_shares = shares
                                holding_invest_amount = actual_invest
                                days_held = 0

        final_val = cash
        if in_position:
            last_day_data = qi[(qi["code"] == holding_code) & (qi["date_dt"] <= end_dt)]
            if not last_day_data.empty:
                last_c = last_day_data.iloc[-1]["close"]
                final_val += holding_shares * last_c
            else:
                final_val += holding_invest_amount

        win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
        matrix_results.append({
            "検証期間": f"{start_dt.strftime('%Y/%m')}〜{end_dt.strftime('%Y/%m')}",
            "トレード数": trades,
            "勝率": f"{win_rate:.1f}%",
            "最終資産": f"{final_val:,.0f}円",
            "損益": f"{final_val - initial_cash:+,.0f}円"
        })

    res_m_df = pd.DataFrame(matrix_results)
    print("--- 【野村証券手数料・15パターン検証結果】 ---")
    print(res_m_df.to_string(index=False))
