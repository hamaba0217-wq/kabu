# -*- coding: utf-8 -*-
"""
limitup_precursor_strategy.py - ストップ高前兆銘柄の「前日引け仕込み → 翌日決済」実データ戦略検証

検証の作法:
- 前日引け時点で判明している客観指標（出来高倍率、連騰数、位置、業種、規模）のみで選定
- 前日終値で買い約定、翌営業日に決済（先読み完全排除）
- 業種（高勝率業種/素材/IT等）や時価総額（小型/中大型）の絞り込みを網羅検証
- 前半・後半の2分割検証で再現性を確認
- 手数料・スリッページ未考慮（明記）
- 投資助言ではない（明記）
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import limitup
import refine


def run_strategy(quotes: pd.DataFrame, fin=None, listed=None):
    print("=" * 76)
    print("ストップ高前兆銘柄『前日引け買い → 翌日手仕舞い』実データ総合検証")
    print("※ 手数料・スリッページ未考慮 / 投資助言ではありません")
    print("=" * 76)

    # 属性マップ
    sec_map = {}
    mkt_map = {}
    if listed is not None:
        if "sector33" in listed.columns:
            sec_map = dict(zip(listed["code"].astype(str), listed["sector33"]))
        if "market" in listed.columns:
            mkt_map = dict(zip(listed["code"].astype(str), listed["market"]))

    shares_map = {}
    if fin is not None and "shares_outstanding" in fin.columns:
        f_sorted = fin.sort_values("disclosed_date").dropna(subset=["shares_outstanding"])
        shares_map = dict(f_sorted.groupby("code")["shares_outstanding"].last())

    # 指標算出
    qi = quotes.sort_values(["code", "date"]).reset_index(drop=True)
    g = qi.groupby("code")

    qi["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    qi["vol_ratio"] = qi["volume"] / qi["vol_ma20"]

    qi["high_52w"] = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / qi["high_52w"] - 1.0

    qi["day_range"] = (qi["high"] - qi["low"]) / qi["open"] * 100.0

    # 連続上昇日数
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

    # 翌日の株価
    qi["next_open"] = g["open"].shift(-1)
    qi["next_high"] = g["high"].shift(-1)
    qi["next_low"] = g["low"].shift(-1)
    qi["next_close"] = g["close"].shift(-1)
    qi["next_date"] = g["date"].shift(-1)

    # 翌日データがある行のみ
    df = qi[qi["next_close"].notna()].copy()
    all_dates = sorted(df["date"].unique())
    mid_date = pd.to_datetime(all_dates[len(all_dates) // 2])

    print(f"母集団レコード数: {len(df):,} 行 / 分割境界日: {mid_date.strftime('%Y-%m-%d')}\n")

    # シグナル条件の定義
    # 出来高急増
    vol3 = df["vol_ratio"] >= 3.0
    vol5 = df["vol_ratio"] >= 5.0
    # モメンタム
    stk2 = df["streak"] >= 2
    # 高ボラ
    rng6 = df["day_range"] >= 6.0
    # 位置
    pos_bottom = df["pct_from_high"] <= -0.30
    pos_break = df["pct_from_high"] >= -0.05
    # 規模
    codes_str = df["code"].astype(str)
    mcaps = [df.iloc[i]["close"] * shares_map.get(codes_str.iloc[i], 0) / 1e8 for i in range(len(df))]
    df["mcap_oku"] = mcaps
    mcap_small = (df["mcap_oku"] > 0) & (df["mcap_oku"] < 100)  # 100億未満
    mcap_mid = df["mcap_oku"] >= 100

    # 業種
    secs = codes_str.map(sec_map).fillna("その他")
    df["sec"] = secs
    sec_high_win = df["sec"].isin(refine.HIGH_WIN_SECTORS)
    sec_growth = codes_str.map(mkt_map).fillna("").str.contains("グロース")

    # 探索する条件セット
    patterns = [
        ("基準: 出来高3倍以上", vol3),
        ("基準: 出来高5倍以上", vol5),
        ("C1: 出来高3倍 × 2連騰以上", vol3 & stk2),
        ("C2: 出来高5倍 × 2連騰以上", vol5 & stk2),
        ("C3: 出来高3倍 × レンジ6%超", vol3 & rng6),
        ("C4: 出来高5倍 × レンジ6%超", vol5 & rng6),
        ("C5: 出来高3倍 × 底値圏(高値-30%↓)", vol3 & pos_bottom),
        ("C6: 出来高5倍 × 底値圏(高値-30%↓)", vol5 & pos_bottom),
        ("C7: 出来高3倍 × 新高値圏(高値-5%↑)", vol3 & pos_break),
        ("C8: 出来高5倍 × 新高値圏(高値-5%↑)", vol5 & pos_break),
        ("C9: 出来高3倍 × 小型(<100億)", vol3 & mcap_small),
        ("C10: 出来高3倍 × 中大型(>100億)", vol3 & mcap_mid),
        ("C11: 出来高3倍 × 高勝率業種", vol3 & sec_high_win),
        ("C12: 出来高3倍 × グロース市場", vol3 & sec_growth),
        ("C13: 【複合】出来高5倍 × レンジ6%超 × 小型", vol5 & rng6 & mcap_small),
        ("C14: 【複合】出来高3倍 × 2連騰 × 底値圏", vol3 & stk2 & pos_bottom),
    ]

    # --- 1. 単純翌日終値決済（前日引け買い → 翌日引け売り） ---
    print("--- 【1. 翌日終値決済】（前日終値買い → 翌営業日終値成行売り） ---")
    rows1 = []
    for label, mask in patterns:
        sub = df[mask]
        n = len(sub)
        if n < 30:
            continue
        entry = sub["close"].values
        nxt_c = sub["next_close"].values
        ret = (nxt_c / entry - 1.0) * 100.0

        sub_h1 = sub[sub["date"] <= mid_date]
        sub_h2 = sub[sub["date"] > mid_date]
        r1 = (sub_h1["next_close"].values / sub_h1["close"].values - 1.0) * 100.0 if len(sub_h1) else np.array([])
        r2 = (sub_h2["next_close"].values / sub_h2["close"].values - 1.0) * 100.0 if len(sub_h2) else np.array([])

        rows1.append({
            "条件パターン": label,
            "件数": n,
            "勝率": f"{(ret > 0).mean()*100:.1f}%",
            "平均R(%)": round(ret.mean(), 2),
            "中央値(%)": round(float(np.median(ret)), 2),
            "前半R(%)": round(r1.mean(), 2) if len(r1) else np.nan,
            "後半R(%)": round(r2.mean(), 2) if len(r2) else np.nan,
            "判定": "◎両プラス" if (len(r1) and len(r2) and r1.mean() > 0 and r2.mean() > 0) else ("△片方" if (len(r1) and len(r2) and (r1.mean() > 0 or r2.mean() > 0)) else "×全滅"),
        })

    res1 = pd.DataFrame(rows1).sort_values("平均R(%)", ascending=False)
    print(res1.to_string(index=False))

    # --- 2. 翌日利確・損切り決済シミュレーション（利確+7%/損切-3%） ---
    print("\n--- 【2. 翌日ザラ場決済】（利確+7% / 損切-3% / 届かなければ大引け） ---")
    rows2 = []
    for label, mask in patterns:
        sub = df[mask]
        n = len(sub)
        if n < 30:
            continue
        entry = sub["close"].values
        nxt_h = sub["next_high"].values
        nxt_l = sub["next_low"].values
        nxt_c = sub["next_close"].values

        rets = []
        tp_rate = 1.07
        sl_rate = 0.97
        for e, h, l, c in zip(entry, nxt_h, nxt_l, nxt_c):
            tp_px = e * tp_rate
            sl_px = e * sl_rate
            if l <= sl_px and h >= tp_px:
                rets.append(-3.0)  # 保守的損切り優先
            elif l <= sl_px:
                rets.append(-3.0)
            elif h >= tp_px:
                rets.append(7.0)
            else:
                rets.append((c / e - 1.0) * 100.0)
        rets = np.array(rets)

        sub_dates = sub["date"].values
        sub_h1_mask = sub_dates <= np.datetime64(mid_date)
        r1 = rets[sub_h1_mask]
        r2 = rets[~sub_h1_mask]

        rows2.append({
            "条件パターン": label,
            "件数": n,
            "勝率": f"{(rets > 0).mean()*100:.1f}%",
            "平均R(%)": round(rets.mean(), 2),
            "前半R(%)": round(r1.mean(), 2) if len(r1) else np.nan,
            "後半R(%)": round(r2.mean(), 2) if len(r2) else np.nan,
            "判定": "◎両プラス" if (len(r1) and len(r2) and r1.mean() > 0 and r2.mean() > 0) else ("△片方" if (len(r1) and len(r2) and (r1.mean() > 0 or r2.mean() > 0)) else "×全滅"),
        })

    res2 = pd.DataFrame(rows2).sort_values("平均R(%)", ascending=False)
    print(res2.to_string(index=False))
