# -*- coding: utf-8 -*-
"""
daily_morning_routine.py - 毎朝8:30実行：本日の推奨銘柄（勝率55%以上セクター優先 ＆ 高スコア参考）と指値（利確・損切）の算出
"""
from __future__ import annotations
import os
import datetime as dt
import pandas as pd
import numpy as np
from sources import JQuants, JST
import config
from badnews import _bad_flags

def run_morning_routine():
    print("=" * 76)
    print("【毎朝8:30自動更新】本日の推奨銘柄・指値（利確/損切）プラン")
    print("=" * 76)

    jq = JQuants()
    quotes = jq.quotes(250) # 直近データ
    fin = jq.financials(250)
    listed = jq.listed()

    if quotes.empty:
        print("株価データが取得できませんでした。")
        return

    # 最新日のデータを抽出
    latest_date = quotes["date"].max()
    print(f"対象基準日: {latest_date.date()}")

    # テクニカル指標の計算
    qi = quotes.sort_values(["code", "date"]).reset_index(drop=True)
    g = qi.groupby("code")

    qi["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    qi["vol_ratio"] = qi["volume"] / qi["vol_ma20"]
    qi["high_52w"] = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / qi["high_52w"] - 1.0
    qi["day_range"] = (qi["high"] - qi["low"]) / qi["open"] * 100.0

    qi["ret_1d"] = g["close"].pct_change()
    daily_market = qi.groupby("date")["ret_1d"].mean().reset_index(name="market_ret")
    daily_market["market_ma5"] = daily_market["market_ret"].rolling(5, min_periods=1).mean()
    qi = pd.merge(qi, daily_market[["date", "market_ma5"]], on="date", how="left")

    codes_str = qi["code"].astype(str)
    sector_map = {}
    if listed is not None:
        code_col = "Code" if "Code" in listed.columns else "code"
        sec_col = "sector33" if "sector33" in listed.columns else "Sector33CodeName"
        if code_col in listed.columns and sec_col in listed.columns:
            sector_map = dict(zip(listed[code_col].astype(str).str.slice(0, 4), listed[sec_col]))

    qi["sector"] = codes_str.str.slice(0, 4).map(sector_map).fillna("その他")
    qi["is_bottom"] = qi["pct_from_high"] <= -0.30
    qi["is_vol_surge"] = qi["vol_ratio"] >= 2.0
    qi["is_range_wide"] = qi["day_range"] >= 7.0
    qi["is_good_regime"] = qi["market_ma5"] >= 0.0

    # スコア算出
    v_norm = np.clip((qi["vol_ratio"] - 2.0) / 8.0, 0.0, 1.0)
    r_norm = np.clip((qi["day_range"] - 7.0) / 10.0, 0.0, 1.0)
    b_norm = np.clip((-qi["pct_from_high"] - 0.30) / 0.50, 0.0, 1.0)
    qi["recommendation_score_%"] = (60.0 + (v_norm * 15.0 + r_norm * 15.0 + b_norm * 9.0)).round(1)

    # 最新日の候補を抽出
    today_df = qi[qi["date"] == latest_date].copy()
    valid_mask = today_df["is_bottom"] & today_df["is_vol_surge"] & today_df["is_range_wide"] & today_df["is_good_regime"]
    candidates = today_df[valid_mask].sort_values("vol_ratio", ascending=False)

    if len(candidates) == 0:
        print("本日の条件を満たす銘柄はありません。")
        return

    # 勝率55%以上の高勝率セクター
    target_sectors = ['パルプ・紙', '倉庫･運輸関連業', '海運業', '石油･石炭製品', '証券･商品先物取引業', '鉄鋼', '銀行業']

    results = []
    for _, row in candidates.iterrows():
        code = row["code"]
        close_p = row["close"]
        sector = row["sector"]
        score = row["recommendation_score_%"]
        
        # 指値・決済価格の設定
        buy_limit = close_p # 買い指値（現在値ベース）
        tp_price = round(close_p * 1.10, 1) # 利回り +10%
        sl_price = round(close_p * 0.95, 1) # 損切 -5%
        
        is_preferred_sector = sector in target_sectors
        
        results.append({
            "銘柄コード": code,
            "業種": sector,
            "区分": "【本命】高勝率セクター" if is_preferred_sector else "【参考】高スコア銘柄",
            "おすすめ度(%)": score,
            "買い指値": buy_limit,
            "利確目標(TP+10%)": tp_price,
            "損切ライン(SL-5%)": sl_price,
            "出来高倍率": round(row["vol_ratio"], 2)
        })

    df_res = pd.DataFrame(results)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    today_str = dt.datetime.now(JST).date().isoformat()
    out_csv = os.path.join(config.OUTPUT_DIR, f"{today_str}_morning_orders.csv")
    df_res.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n抽出された推奨銘柄数: {len(df_res)} 件")
    print(df_res.to_string(index=False))
    print(f"\n指値プランをCSVに出力しました: {out_csv}")

if __name__ == "__main__":
    run_morning_routine()
