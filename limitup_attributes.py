# -*- coding: utf-8 -*-
"""
limitup_attributes.py - ストップ高翌日の値動きを業種・規模・位置・初動等の属性別に分解検証

検証の作法:
- 属性ごとに分類し、プラスで残る領域が存在するかを事実で確認
- 前半・後半の2分割検証で再現性を確認
- 手数料・スリッページ未考慮（明記）
- 投資助言ではない（明記）
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import limitup
import refine


def analyze_attributes(quotes: pd.DataFrame, fin=None, listed=None, margin=None) -> pd.DataFrame:
    st = limitup.detect_limit_up(quotes)
    if st.empty:
        return pd.DataFrame()

    # 銘柄属性マップ
    sec_map = {}
    name_map = {}
    market_map = {}
    if listed is not None:
        if "sector33" in listed.columns:
            sec_map = dict(zip(listed["code"].astype(str), listed["sector33"]))
        if "company_name" in listed.columns:
            name_map = dict(zip(listed["code"].astype(str), listed["company_name"]))
        if "market" in listed.columns:
            market_map = dict(zip(listed["code"].astype(str), listed["market"]))

    # 株式数（時価総額計算用）
    shares_map = {}
    if fin is not None and "shares_outstanding" in fin.columns:
        f_sorted = fin.sort_values("disclosed_date").dropna(subset=["shares_outstanding"])
        shares_map = dict(f_sorted.groupby("code")["shares_outstanding"].last())

    # 信用残マップ
    margin_by_code = {}
    if margin is not None and len(margin):
        mm = margin.copy()
        mm["date"] = pd.to_datetime(mm["date"])
        for c in ("long_margin", "short_margin"):
            if c in mm.columns:
                mm[c] = pd.to_numeric(mm[c], errors="coerce")
        margin_by_code = {str(c): g.sort_values("date") for c, g in mm.groupby("code")}

    # 52週高値計算
    qi = quotes.sort_values(["code", "date"]).copy()
    g = qi.groupby("code")
    qi["high_52w"] = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / qi["high_52w"] - 1.0

    grouped = {str(code): grp.reset_index(drop=True) for code, grp in qi.groupby("code")}

    records = []

    # 初動判定（過去5日間にストップ高があったか）
    st_dates_by_code = {}
    for code, grp in st.groupby("code"):
        st_dates_by_code[str(code)] = set(pd.to_datetime(grp["date"]))

    LAG = 7
    def _latest_margin(g_m, as_of):
        cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=LAG)
        past = g_m[g_m["date"] <= cutoff]
        if past.empty:
            return None, None
        last_row = past.iloc[-1]
        lm = last_row.get("long_margin")
        sm = last_row.get("short_margin")
        return (float(lm) if pd.notna(lm) else None, float(sm) if pd.notna(sm) else None)

    for _, row in st.iterrows():
        code = str(row["code"])
        st_date = pd.to_datetime(row["date"])
        st_close = row["close"]

        if code not in grouped:
            continue
        cqi = grouped[code]
        m = cqi.index[cqi["date"] == row["date"]].tolist()
        if not m or m[0] + 1 >= len(cqi):
            continue

        st_idx = m[0]
        bar_next = cqi.iloc[st_idx + 1]
        o = bar_next["open"]
        l = bar_next["low"]
        c = bar_next["close"]
        if o <= 0:
            continue

        # 1. 業種
        sec = sec_map.get(code, "不明")
        is_high_win_sec = sec in refine.HIGH_WIN_SECTORS

        # 2. 市場区分
        mkt = market_map.get(code, "不明")
        is_growth = "グロース" in mkt

        # 3. 時価総額
        sh = shares_map.get(code)
        mcap_oku = (st_close * sh / 1e8) if (sh and sh > 0) else np.nan
        if pd.notna(mcap_oku):
            if mcap_oku < 50:
                mcap_cat = "超小型(<50億)"
            elif mcap_oku < 150:
                mcap_cat = "小型(50-150億)"
            else:
                mcap_cat = "中大型(>150億)"
        else:
            mcap_cat = "不明"

        # 4. 位置（52週高値に対する位置）
        p_high = cqi.iloc[st_idx].get("pct_from_high")
        if pd.notna(p_high):
            if p_high >= -0.05:
                pos_cat = "新高値圏(高値-5%以内)"
            elif p_high >= -0.20:
                pos_cat = "中間圏(高値-20〜-5%)"
            else:
                pos_cat = "底値圏(高値-20%以下)"
        else:
            pos_cat = "不明"

        # 5. 初動か連続か（過去5営業日以内にストップ高があったか）
        st_history = st_dates_by_code.get(code, set())
        past_5days = [cqi.iloc[i]["date"] for i in range(max(0, st_idx - 5), st_idx)]
        had_recent_st = any(d in st_history for d in past_5days)
        streak_cat = "2日目以降/連続" if had_recent_st else "初動(初日)"

        # 6. 信用需給
        mr_cat = "不明"
        if code in margin_by_code:
            lm, sm = _latest_margin(margin_by_code[code], st_date)
            if lm and sm and sm > 0:
                ratio = lm / sm
                if ratio < 1.0:
                    mr_cat = "売り長(倍率<1)"
                elif ratio < 3.0:
                    mr_cat = "中立(倍率1〜3)"
                else:
                    mr_cat = "買い長(倍率>3)"

        # リターン計算
        # A: 寄り成行買い → 大引け決済
        ret_open_close = (c / o - 1.0) * 100.0

        # B: 寄り-5%指値買い → 大引け決済（約定時のみ）
        dip5_px = o * 0.95
        ret_dip5 = (c / dip5_px - 1.0) * 100.0 if l <= dip5_px else np.nan

        records.append({
            "base_date": st_date,
            "code": code,
            "sec": sec,
            "is_high_win_sec": "高勝率業種" if is_high_win_sec else "一般業種",
            "market": mkt,
            "is_growth": "グロース" if is_growth else "プライム/スタンダード",
            "mcap_cat": mcap_cat,
            "pos_cat": pos_cat,
            "streak_cat": streak_cat,
            "mr_cat": mr_cat,
            "ret_open_close": ret_open_close,
            "ret_dip5": ret_dip5,
            "hit_dip5": 1 if l <= dip5_px else 0,
        })

    return pd.DataFrame(records)


def summarize_cat(df: pd.DataFrame, cat_col: str, mid_date) -> pd.DataFrame:
    rows = []
    for val, g in df.groupby(cat_col):
        n = len(g)
        if n < 30:
            continue
        arr_oc = g["ret_open_close"].values
        g_h1 = g[g["base_date"] <= mid_date]
        g_h2 = g[g["base_date"] > mid_date]

        r1 = g_h1["ret_open_close"].mean() if len(g_h1) else np.nan
        r2 = g_h2["ret_open_close"].mean() if len(g_h2) else np.nan

        # 下ヒゲ(-5%)拾い
        g_dip = g.dropna(subset=["ret_dip5"])
        dip_n = len(g_dip)
        dip_r = g_dip["ret_dip5"].mean() if dip_n else np.nan

        rows.append({
            "分類": val,
            "件数": n,
            "寄り買勝率": f"{(arr_oc > 0).mean()*100:.1f}%",
            "寄り買平均R": round(arr_oc.mean(), 2),
            "前半R": round(r1, 2),
            "後半R": round(r2, 2),
            "-5%拾い件数": dip_n,
            "-5%拾い平均R": round(dip_r, 2) if dip_n else "—",
        })
    res = pd.DataFrame(rows)
    return res.sort_values("寄り買平均R", ascending=False) if not res.empty else res


def run(quotes: pd.DataFrame, fin=None, listed=None, margin=None):
    print("=" * 76)
    print("ストップ高翌日（Day 1） 属性別・特徴別ブレイクダウン検証")
    print("※ 手数料・スリッページ未考慮 / 投資助言ではありません")
    print("=" * 76)

    df = analyze_attributes(quotes, fin=fin, listed=listed, margin=margin)
    if df.empty:
        print("データがありませんでした。")
        return

    unique_dates = sorted(df["base_date"].unique())
    mid_date = unique_dates[len(unique_dates) // 2]
    print(f"総ストップ高数: {len(df):,} 件 / 分割境界: {mid_date.strftime('%Y-%m-%d')}\n")

    # 1. 初動 vs 2日目以降
    print("--- ① 【初動 vs 連続ストップ高】 ---")
    print(summarize_cat(df, "streak_cat", mid_date).to_string(index=False))

    # 2. 高値位置（青天井新高値 vs 中間 vs 底値圏反発）
    print("\n--- ② 【52週高値からの位置】 ---")
    print(summarize_cat(df, "pos_cat", mid_date).to_string(index=False))

    # 3. 企業規模（時価総額）
    print("\n--- ③ 【時価総額（規模）】 ---")
    print(summarize_cat(df, "mcap_cat", mid_date).to_string(index=False))

    # 4. 高勝率業種 vs 一般業種
    print("\n--- ④ 【業種グループ（高勝率業種 vs 一般）】 ---")
    print(summarize_cat(df, "is_high_win_sec", mid_date).to_string(index=False))

    # 5. 市場区分（グロース vs プライム/スタンダード）
    print("\n--- ⑤ 【市場区分】 ---")
    print(summarize_cat(df, "is_growth", mid_date).to_string(index=False))

    # 6. 信用需給（売り長 vs 買い長）
    print("\n--- ⑥ 【信用需給（倍率）】 ---")
    print(summarize_cat(df, "mr_cat", mid_date).to_string(index=False))

    # 7. 複合条件（「初動 × 新高値圏」など最も強そうな組み合わせ）
    print("\n--- ⑦ 【複合条件の探索（有望な組み合わせ）】 ---")
    df["combo1"] = df["streak_cat"] + " × " + df["pos_cat"]
    print(summarize_cat(df, "combo1", mid_date).head(6).to_string(index=False))
