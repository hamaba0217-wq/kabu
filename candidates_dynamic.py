# -*- coding: utf-8 -*-
"""
candidates_dynamic.py - 地合い連動型の動的フィルタリングによる本日の候補抽出モジュール
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import datetime as dt

from sources import JQuants
import config
from badnews import _bad_flags
import refine

def run_dynamic_candidates(quotes: pd.DataFrame, fin=None, listed=None, margin=None):
    print("=" * 76)
    print("『地合い連動型』本日の候補銘柄抽出")
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

    # 指標計算
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

    # 最新日の地合い（market_ma5）を判定
    latest_date = qi["date"].max()
    latest_subset = qi[qi["date"] == latest_date]
    market_ma5_val = latest_subset["market_ma5"].iloc[0] if len(latest_subset) > 0 and "market_ma5" in latest_subset.columns else 0.0

    print(f"\n【本日の相場環境（地合い）判定】")
    print(f"  最新データ日付: {latest_date}")
    print(f"  市場5日平均リターン (market_ma5): {market_ma5_val*100:.3f}%")

    if market_ma5_val > 0.005:
        regime = "強気（積極モード）"
        print(f"  判定: 【 {regime} 】 -> 全セクター対象で積極的にエントリーを検討します。")
        qi["is_good_regime"] = True
    elif market_ma5_val >= 0.0:
        regime = "通常（標準モード）"
        print(f"  判定: 【 {regime} 】 -> 基本フィルター（レンジ7%以上等）を適用して手堅く選別します。")
        qi["is_good_regime"] = True
    else:
        regime = "弱気（安全・見送りモード）"
        print(f"  判定: 【 {regime} 】 -> 地合い逆風のため、高勝率セクター（小売業等）または厳選銘柄以外はスキップ推奨です。")
        qi["is_good_regime"] = False

    b6_mask = qi["is_bottom"] & qi["is_small_cap"] & qi["is_vol_surge"] & qi["is_range_wide"] & (qi["is_clean"] == 1)
    if market_ma5_val < 0:
        b6_mask = b6_mask & qi["sector"].isin(["小売業", "情報･通信業", "サービス業"])

    # 修正された括弧：日付一致とマスク条件を正確に分離
    today_candidates = qi[(qi["date"] == latest_date) & b6_mask].copy()

    print(f"\n--- 【本日の該当候補銘柄 ({latest_date})】 ---")
    print(f"抽出件数: {len(today_candidates)} 件")
    if len(today_candidates) > 0:
        out_df = today_candidates[["code", "sector", "vol_ratio", "day_range", "mcap_oku"]].copy()
        out_df.columns = ["銘柄コード", "セクター", "出来高倍率", "日中レンジ(%)", "時価総額(億)"]
        print(out_df.to_string(index=False))
    else:
        print("本日は条件を満たす銘柄がありませんでした（または安全モードにより見送り）。")
