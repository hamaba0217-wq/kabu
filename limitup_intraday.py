# -*- coding: utf-8 -*-
"""
limitup_intraday.py - ストップ高後の日中ローソク足（OHLC）振れ幅・値動きの統計分析

検証の作法:
- 事実にもとづいて決める
- 前半・後半の2分割検証で再現性を確認
- 手数料・スリッページは未考慮（明記）
- 投資助言ではない（明記）
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import limitup
import limitup_analyze


def analyze_intraday(quotes: pd.DataFrame, fin=None, listed=None, max_days: int = 7) -> pd.DataFrame:
    """
    ストップ高イベント後 1〜max_days 営業日の日中OHLC指標を集計する。
    """
    # ストップ高検出
    st = limitup.detect_limit_up(quotes)
    if st.empty:
        return pd.DataFrame()

    # 理由分類
    try:
        reason_df = limitup_analyze.classify_reason(quotes, st, fin=fin, listed=listed)
        reason_df["event_id"] = reason_df["code"].astype(str) + "_" + pd.to_datetime(reason_df["date"]).dt.strftime('%Y%m%d')
        reason_map = dict(zip(reason_df["event_id"], reason_df["reason_type"]))
    except Exception as e:
        print(f"  [!] 理由分類スキップ: {e}")
        reason_map = {}

    quotes_sorted = quotes.sort_values(["code", "date"]).copy()
    grouped = {code: group.reset_index(drop=True) for code, group in quotes_sorted.groupby("code")}

    records = []

    for _, row in st.iterrows():
        code = row["code"]
        st_date = row["date"]
        ev_id = f"{code}_{pd.to_datetime(st_date).strftime('%Y%m%d')}"
        reason = reason_map.get(ev_id, "説明不能")

        if code not in grouped:
            continue
        cqi = grouped[code]
        match_idx = cqi.index[cqi["date"] == st_date].tolist()
        if not match_idx:
            continue
        st_idx = match_idx[0]

        for day_offset in range(1, max_days + 1):
            t_idx = st_idx + day_offset
            if t_idx >= len(cqi):
                break
            t_row = cqi.iloc[t_idx]
            o = t_row["open"]
            h = t_row["high"]
            l = t_row["low"]
            c = t_row["close"]

            if o <= 0 or np.isnan(o):
                continue

            dip_pct = (o - l) / o * 100.0          # 下落幅 (Open -> Low)
            rally_pct = (h - o) / o * 100.0        # 上昇幅 (Open -> High)
            range_pct = (h - l) / o * 100.0        # 全レンジ (High - Low)
            day_ret_pct = (c - o) / o * 100.0      # 日中リターン (Open -> Close)
            is_yang = 1 if c > o else 0             # 陽線フラグ

            records.append({
                "event_id": ev_id,
                "code": code,
                "base_date": pd.to_datetime(st_date),
                "target_date": pd.to_datetime(t_row["date"]),
                "day": day_offset,
                "reason": reason,
                "dip_pct": dip_pct,
                "rally_pct": rally_pct,
                "range_pct": range_pct,
                "day_ret_pct": day_ret_pct,
                "is_yang": is_yang,
            })

    return pd.DataFrame(records)


def summarize_by_day(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, g in df.groupby("day"):
        rows.append({
            "Day": int(day),
            "件数": len(g),
            "下落幅中位(%)": round(g["dip_pct"].median(), 2),
            "上昇幅中位(%)": round(g["rally_pct"].median(), 2),
            "全レンジ中位(%)": round(g["range_pct"].median(), 2),
            "日中終値中位(%)": round(g["day_ret_pct"].median(), 2),
            "陽線率(%)": round(g["is_yang"].mean() * 100.0, 1),
        })
    return pd.DataFrame(rows).sort_values("Day")


def run(quotes: pd.DataFrame, fin=None, listed=None):
    print("=" * 68)
    print("ストップ高後 日中ローソク足特性・ボラティリティ検証 (Day 1〜7)")
    print("※ 手数料・スリッページ未考慮 / 投資助言ではありません")
    print("=" * 68)

    df = analyze_intraday(quotes, fin=fin, listed=listed, max_days=7)
    if df.empty:
        print("ストップ高イベントまたは日中データが見つかりませんでした。")
        return

    print(f"対象ストップ高イベント数: {df['event_id'].nunique():,} 件")
    print(f"日中レコード総数: {len(df):,} 行")

    # 全期間集計
    print("\n--- 【全期間】Day 1〜7 日中ローソク特性 ---")
    s_all = summarize_by_day(df)
    print(s_all.to_string(index=False))

    # 前半・後半の2分割検証
    unique_dates = sorted(df["base_date"].unique())
    mid_date = unique_dates[len(unique_dates) // 2]
    df_h1 = df[df["base_date"] <= mid_date]
    df_h2 = df[df["base_date"] > mid_date]

    print(f"\n--- 【前半（〜{mid_date.strftime('%Y-%m-%d')}）】 ---")
    print(summarize_by_day(df_h1).to_string(index=False))

    print(f"\n--- 【後半（{mid_date.strftime('%Y-%m-%d')}〜）】 ---")
    print(summarize_by_day(df_h2).to_string(index=False))

    # 原因タイプ別（仕手的特徴・決算・説明不能）のDay 1〜3
    print("\n--- 【タイプ別】Day 1〜3 の日中値動き特性 ---")
    cause_rows = []
    for (reason, day), g in df[df["day"].isin([1, 2, 3])].groupby(["reason", "day"]):
        cause_rows.append({
            "タイプ": reason,
            "Day": int(day),
            "件数": len(g),
            "下落幅中位(%)": round(g["dip_pct"].median(), 2),
            "上昇幅中位(%)": round(g["rally_pct"].median(), 2),
            "全レンジ中位(%)": round(g["range_pct"].median(), 2),
            "陽線率(%)": round(g["is_yang"].mean() * 100.0, 1),
        })
    print(pd.DataFrame(cause_rows).to_string(index=False))
