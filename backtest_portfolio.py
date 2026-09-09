# -*- coding: utf-8 -*-
"""
backtest_portfolio.py - 高速化された1000万円固定投資・定量おすすめ度算出モジュール
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import datetime as dt

from sources import JQuants
import config
from badnews import _bad_flags

def run_portfolio_simulation(quotes: pd.DataFrame, fin=None, listed=None, margin=None):
    import cache
    cached_quotes = cache.load("quotes_backtest")
    if cached_quotes is not None and not cached_quotes.empty:
        quotes = cached_quotes
        print("  [キャッシュ] 株価データをローカルキャッシュから読み込みました（APIアクセススキップ）")
    else:
        cache.save("quotes_backtest", quotes)

    print("=" * 76)
    print("『1000万円固定・定量おすすめ度付与』ポートフォリオ運用シミュレーション（高速化版）")
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

    print("  テクニカル指標を計算中...")
    qi["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    qi["vol_ratio"] = qi["volume"] / qi["vol_ma20"]
    qi["high_52w"] = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / qi["high_52w"] - 1.0
    qi["day_range"] = (qi["high"] - qi["low"]) / qi["open"] * 100.0

    qi["ret_1d"] = g["close"].pct_change()
    daily_market = qi.groupby("date")["ret_1d"].mean().reset_index(name="market_ret")
    daily_market["market_ma5"] = daily_market["market_ret"].rolling(5, min_periods=1).mean()
    qi = pd.merge(qi, daily_market[["date", "market_ma5"]], on="date", how="left")

    # 前方シミュレーション用のカラム（5日以内の最高・最低価格のインデックスを高速参照するため、pivotまたは辞書化）
    codes_str = qi["code"].astype(str)
    shares_series = codes_str.map(shares_map).fillna(0)
    qi["mcap_oku"] = qi["close"] * shares_series / 1e8
    qi["is_small_cap"] = True
    qi["is_bottom"] = qi["pct_from_high"] <= -0.30
    qi["is_vol_surge"] = qi["vol_ratio"] >= 2.0
    qi["is_range_wide"] = qi["day_range"] >= 7.0
    qi["sector"] = codes_str.str.slice(0, 4).map(sector_map).fillna("その他")

    v_norm = np.clip((qi["vol_ratio"] - 2.0) / 8.0, 0.0, 1.0)
    r_norm = np.clip((qi["day_range"] - 7.0) / 10.0, 0.0, 1.0)
    b_norm = np.clip((-qi["pct_from_high"] - 0.30) / 0.50, 0.0, 1.0)
    qi["recommendation_score_%"] = (60.0 + (v_norm * 15.0 + r_norm * 15.0 + b_norm * 9.0)).round(1)

    bad_set = _bad_flags(fin) if fin is not None else set()
    qi["date_str"] = pd.to_datetime(qi["date"]).dt.strftime("%Y-%m-%d")
    qi["is_clean"] = [1 if (str(c), d) not in bad_set else 0 for c, d in zip(codes_str, qi["date_str"])]
    qi["is_good_regime"] = qi["market_ma5"] >= 0.0

    b6_mask = qi["is_bottom"] & qi["is_small_cap"] & qi["is_vol_surge"] & qi["is_range_wide"] & (qi["is_clean"] == 1) & qi["is_good_regime"]
    qi["date_dt"] = pd.to_datetime(qi["date"])

    def nomura_web_fee(amount):
        if amount <= 100000: return 110
        elif amount <= 200000: return 154
        elif amount <= 500000: return 220
        elif amount <= 1000000: return 330
        elif amount <= 3000000: return 605
        elif amount <= 5000000: return 1018
        else: return 1430

    start_d = "2024-11-01"
    end_d = "2026-08-31"
    sub_qi = qi[(qi["date_dt"] >= start_d) & (qi["date_dt"] <= end_d) & b6_mask].copy()

    fixed_buy_amount = 10000000.0
    buy_fee = nomura_web_fee(fixed_buy_amount)

    print("  売買シミュレーションを実行中（高速ベクトル/辞書処理）...")
    
    # 銘柄ごとの価格データを高速参照用の辞書に変換 {code: DataFrame(date, high, low, close)}
    code_dfs = {}
    for code, group in qi.groupby("code"):
        code_dfs[code] = group.sort_values("date_dt").set_index("date_dt")

    trading_dates = sorted(sub_qi["date_dt"].unique())
    positions = []
    trade_history = []

    total_days = len(trading_dates)
    for idx, d in enumerate(trading_dates):
        if idx % 50 == 0:
            print(f"    進捗: {idx}/{total_days} 営業日処理完了...")
            
        # 1. 既存ポジションの決済チェック
        surviving_pos = []
        for pos in positions:
            code = pos["code"]
            cdf = code_dfs.get(code)
            if cdf is not None:
                # 当日以降のデータを取得（最大5営業日）
                future_rows = cdf.loc[d:].head(5)
                exit_price = pos["entry_price"]
                exit_reason = "HOLD"
                exit_date = d
                
                for r_date, row in future_rows.iterrows():
                    if row["high"] >= pos["tp_price"]:
                        exit_price = pos["tp_price"]
                        exit_reason = "TP(+10%)"
                        exit_date = r_date
                        break
                    elif row["low"] <= pos["sl_price"]:
                        exit_price = pos["sl_price"]
                        exit_reason = "SL(-5%)"
                        exit_date = r_date
                        break
                
                if exit_reason != "HOLD":
                    return_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
                    sell_amount = fixed_buy_amount * (1.0 + return_pct)
                    sell_fee = nomura_web_fee(sell_amount)
                    net_pnl = sell_amount - sell_fee - (fixed_buy_amount + buy_fee)
                    
                    trade_history.append({
                        "購入日": pos["buy_date"].strftime("%Y-%m-%d"),
                        "銘柄コード": pos["code"],
                        "業種": pos["sector"],
                        "おすすめ度(%)": pos["score"],
                        "買値": pos["entry_price"],
                        "売値": exit_price,
                        "決済理由": exit_reason,
                        "売却日": exit_date.strftime("%Y-%m-%d"),
                        "購入金額(固定)": fixed_buy_amount,
                        "買手数料": buy_fee,
                        "売却金額": sell_amount,
                        "売手数料": sell_fee,
                        "純損益(固定1000万)": net_pnl
                    })
                else:
                    surviving_pos.append(pos)
            else:
                surviving_pos.append(pos)
        positions = surviving_pos

        # 2. 新規買い付け
        day_df = sub_qi[sub_qi["date_dt"] == d]
        if len(day_df) > 0:
            sorted_day = day_df.sort_values("vol_ratio", ascending=False)
            held_codes = {p["code"] for p in positions}
            for _, cand in sorted_day.iterrows():
                if cand["code"] in held_codes:
                    continue
                
                entry_p = cand["close"]
                positions.append({
                    "code": cand["code"],
                    "sector": cand["sector"],
                    "score": cand["recommendation_score_%"],
                    "buy_date": d,
                    "entry_price": entry_p,
                    "tp_price": round(entry_p * 1.10, 1),
                    "sl_price": round(entry_p * 0.95, 1)
                })
                held_codes.add(cand["code"])

    df_res = pd.DataFrame(trade_history)
    print(f"\n=== 1000万円固定・高速シミュレーション完了 ===")
    print(f"総トレード数: {len(df_res)} 件")
    if len(df_res) > 0:
        print(f"総純損益合計: {df_res['純損益(固定1000万)'].sum():,.0f} 円")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(config.OUTPUT_DIR, "all_candidates_fixed_10m_with_score.csv")
    try:
        df_res.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"詳細CSVを出力しました: {out_csv}")
    except PermissionError:
        alt_csv = os.path.join(config.OUTPUT_DIR, "all_candidates_fixed_10m_with_score_new.csv")
        df_res.to_csv(alt_csv, index=False, encoding="utf-8-sig")
        print(f"[注意] ファイルが開かれているため別名で出力しました: {alt_csv}")
