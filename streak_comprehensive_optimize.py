# -*- coding: utf-8 -*-
"""
streak_comprehensive_optimize.py - 小型株×底値圏をベースにした高精度絞り込み検証
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import refine
from badnews import _bad_flags, _prep_fin_extended

def run_comprehensive(quotes: pd.DataFrame, fin=None, listed=None, margin=None):
    print("=" * 76)
    print("5連騰シグナル銘柄 × 小型株・底値圏ベースの『高精度絞り込み』検証")
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

    # 決算データから営業利益の伸びを抽出
    fin_ext = _prep_fin_extended(fin) if fin is not None else pd.DataFrame()
    op_growth_map = {}
    if not fin_ext.empty and "operating_profit" in fin_ext.columns:
        fin_ext = fin_ext.sort_values(["code", "disclosed_date"])
        fin_ext["op_prev"] = fin_ext.groupby(["code", "period_type"])["operating_profit"].shift(1)
        fin_ext["op_yoy"] = (fin_ext["operating_profit"] / fin_ext["op_prev"].abs() - 1.0) * 100.0
        # 直近の増益フラグ（営業利益が前年同期比でプラス、または黒字）
        for _, r in fin_ext.dropna(subset=["op_yoy"]).iterrows():
            op_growth_map[(str(r["code"]), pd.to_datetime(r["disclosed_date"]).strftime("%Y-%m-%d"))] = r["op_yoy"]

    qi = quotes.sort_values(["code", "date"]).reset_index(drop=True)
    g = qi.groupby("code")

    print("指標計算中（ベクトル演算）...")
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

    qi["next_close"] = g["close"].shift(-1)
    df = qi[qi["next_close"].notna()].copy()

    print("属性・観点の付与中...")
    codes_str = df["code"].astype(str)
    
    df["sec"] = codes_str.map(sec_map).fillna("その他")
    df["is_high_win_sec"] = df["sec"].isin(refine.HIGH_WIN_SECTORS)

    shares_series = codes_str.map(shares_map).fillna(0)
    df["mcap_oku"] = df["close"] * shares_series / 1e8
    df["is_small_cap"] = (df["mcap_oku"] > 0) & (df["mcap_oku"] < 100)

    df["is_bottom"] = df["pct_from_high"] <= -0.30
    df["is_vol_moderate"] = (df["vol_ratio"] >= 0.8) & (df["vol_ratio"] < 2.0) # 極端な爆発ではない、適度な出来高

    bad_set = _bad_flags(fin) if fin is not None else set()
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code_str"] = codes_str
    df["is_clean"] = [1 if (c, d) not in bad_set else 0 for c, d in zip(df["code_str"], df["date_str"])]

    all_dates = sorted(df["date"].unique())
    mid_date = pd.to_datetime(all_dates[len(all_dates) // 2])

    sub5 = df[df["streak"] >= 5].copy()
    print(f"5連騰シグナル対象: {len(sub5):,} 件 / 分割境界: {mid_date.strftime('%Y-%m-%d')}\n")

    strategies = [
        ("基準: 5連騰のみ", sub5["streak"] >= 5),
        ("S7: 5連騰 × 底値圏(-30%↓) × 小型株(<100億)", (sub5["streak"] >= 5) & sub5["is_bottom"] & sub5["is_small_cap"]),
        ("S11: 小型株 × 底値圏 × 悪材料なし", (sub5["streak"] >= 5) & sub5["is_bottom"] & sub5["is_small_cap"] & (sub5["is_clean"] == 1)),
        ("S12: 小型株 × 底値圏 × 適度な出来高(0.8〜2倍)", (sub5["streak"] >= 5) & sub5["is_bottom"] & sub5["is_small_cap"] & sub5["is_vol_moderate"]),
        ("S13: 小型株 × 底値圏 × 高勝率業種", (sub5["streak"] >= 5) & sub5["is_bottom"] & sub5["is_small_cap"] & sub5["is_high_win_sec"]),
        ("S14: 超小型株(<50億) × 底値圏", (sub5["streak"] >= 5) & sub5["is_bottom"] & (sub5["mcap_oku"] > 0) & (sub5["mcap_oku"] < 50)),
        ("S15: 【厳選最強】小型株 × 底値圏 × 悪材料なし × 適度な出来高", (sub5["streak"] >= 5) & sub5["is_bottom"] & sub5["is_small_cap"] & (sub5["is_clean"] == 1) & sub5["is_vol_moderate"]),
    ]

    print("戦略ごとの評価中...")
    rows = []
    for label, mask in strategies:
        sub = sub5[mask]
        n = len(sub)
        if n < 10: continue
        ret = (sub["next_close"] / sub["close"] - 1.0) * 100.0

        h1 = sub[sub["date"] <= mid_date]
        h2 = sub[sub["date"] > mid_date]
        r1 = (h1["next_close"] / h1["close"] - 1.0) * 100.0 if len(h1) else np.array([])
        r2 = (h2["next_close"] / h2["close"] - 1.0) * 100.0 if len(h2) else np.array([])

        rows.append({
            "検証戦略パターン": label,
            "件数": n,
            "勝率": f"{(ret > 0).mean()*100:.1f}%",
            "平均R(%)": round(ret.mean(), 2),
            "中央値(%)": round(float(np.median(ret)), 2),
            "前半R(%)": round(r1.mean(), 2) if len(r1) else np.nan,
            "後半R(%)": round(r2.mean(), 2) if len(r2) else np.nan,
            "判定": "◎両プラス" if (len(r1) and len(r2) and r1.mean() > 0 and r2.mean() > 0) else ("△片方" if (len(r1) and len(r2) and (r1.mean() > 0 or r2.mean() > 0)) else "×全滅"),
        })

    res_df = pd.DataFrame(rows).sort_values("平均R(%)", ascending=False)
    print("--- 【小型株×底値圏ベースの高精度絞り込み検証結果】 ---")
    print(res_df.to_string(index=False))
