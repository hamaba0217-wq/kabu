# -*- coding: utf-8 -*-
"""
streak_live_sim.py - 「厳選最強（B6）」ルールに基づく実運用シミュレーション＆当日候補抽出
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import datetime as dt

from sources import JQuants
import config
from badnews import _bad_flags
import refine

def run_live(quotes: pd.DataFrame, fin=None, listed=None, margin=None):
    print("=" * 76)
    print("『底値圏×小型株×出来高2倍×レンジ5%超×悪材料なし』厳選最強ルール 運用シミュレーション")
    print("ルール: 1回50〜100万投資 / 利食い+15% or 損切り-4% or 10日以内決済 / 複利運用")
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

    print("指標計算中...")
    qi["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    qi["vol_ratio"] = qi["volume"] / qi["vol_ma20"]
    qi["high_52w"] = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / qi["high_52w"] - 1.0
    qi["day_range"] = (qi["high"] - qi["low"]) / qi["open"] * 100.0

    codes_str = qi["code"].astype(str)
    shares_series = codes_str.map(shares_map).fillna(0)
    qi["mcap_oku"] = qi["close"] * shares_series / 1e8
    qi["is_small_cap"] = (qi["mcap_oku"] > 0) & (qi["mcap_oku"] < 100)
    qi["is_bottom"] = qi["pct_from_high"] <= -0.30
    qi["is_vol_surge"] = qi["vol_ratio"] >= 2.0
    qi["is_range_wide"] = qi["day_range"] >= 5.0

    bad_set = _bad_flags(fin) if fin is not None else set()
    qi["date_str"] = pd.to_datetime(qi["date"]).dt.strftime("%Y-%m-%d")
    qi["code_str"] = codes_str
    qi["is_clean"] = [1 if (c, d) not in bad_set else 0 for c, d in zip(qi["code_str"], qi["date_str"])]

    b6_mask = qi["is_bottom"] & qi["is_small_cap"] & qi["is_vol_surge"] & qi["is_range_wide"] & (qi["is_clean"] == 1)
    
    latest_date = qi["date"].max()
    latest_df = qi[(qi["date"] == latest_date) & b6_mask].copy()

    print(f"\n--- 【最新日付 ({latest_date}) の厳選最強（B6）該当候補】 ---")
    if len(latest_df) == 0:
        print("本日の条件を満たす銘柄はありません。")
    else:
        print(f"該当銘柄数: {len(latest_df)}件\n")
        out_cols = ["code", "close", "mcap_oku", "pct_from_high", "vol_ratio", "day_range"]
        print(latest_df[out_cols].to_string(index=False))

    # 2026-01-01 〜 2026-07-01 のバックテスト・複利シミュレーション
    qi["date_dt"] = pd.to_datetime(qi["date"])
    sub_p = qi[(qi["date_dt"] >= "2026-01-01") & (qi["date_dt"] <= "2026-07-01") & b6_mask].copy()
    
    print(f"\n--- 【2026-01-01 〜 2026-07-01 の実績バックテスト（利食い+15% / 損切り-4% / 最大10日）】 ---")
    print(f"対象期間のシグナル発生件数: {len(sub_p)} 件")
    
    # 厳密な利食い・損切り・期間ホールドのシミュレーション
    # 各シグナル発生日の翌日から最大10営業日分のローソク足（high, low, close）を追跡する
    trades_res = []
    
    # 高速に処理するため銘柄・日付でインデックスを作るか、ループで安全に回す
    # 発生件数が数百件程度なのでループで十分高速
    for idx, row in sub_p.iterrows():
        code = row["code"]
        sig_date = row["date"]
        entry_price = row["close"] # 翌日寄り付き想定（ここでは簡便に当日の終値または翌日始値。シミュレーションでは当日の終値基準で翌日エントリー）
        
        # 該当銘柄の以降の株価データを取得
        grp = qi[(qi["code"] == code) & (qi["date"] > sig_date)].sort_values("date").head(10)
        if grp.empty: continue
        
        exit_return = 0.0
        hit = False
        for _, day_row in grp.iterrows():
            h = day_row["high"]
            l = day_row["low"]
            c = day_row["close"]
            
            # 損切り -4% 到達か？ (安値ベース)
            if (l / entry_price - 1.0) <= -0.04:
                exit_return = -4.0
                hit = True
                break
            # 利食い +15% 到達か？ (高値ベース)
            if (h / entry_price - 1.0) >= 0.15:
                exit_return = 15.0
                hit = True
                break
        
        if not hit:
            # 10日経過時は最終日の終値で決済
            final_c = grp.iloc[-1]["close"]
            exit_return = (final_c / entry_price - 1.0) * 100.0
            
        trades_res.append(exit_return)

    if len(trades_res) > 0:
        arr = np.array(trades_res)
        win_t = (arr > 0).mean() * 100
        mean_t = arr.mean()
        print(f"実効勝率 (プラス終了の割合): {win_t:.1f}%")
        print(f"1トレード平均リターン: {mean_t:.2f}%")
        
        # 100万円からの複利シミュレーション（スリッページ 0.15% 控除）
        capital = 1000000.0
        slippage = 0.15
        for r in arr:
            net_r = r - slippage
            capital *= (1.0 + net_r / 100.0)
        print(f"\n【100万円元手・半年後（2026/1/1〜7/1）の複利運用シミュレーション結果】")
        print(f"最終資産残高: {capital:,.0f} 円 （利益: +{capital-1000000:,.0f} 円）")
