# -*- coding: utf-8 -*-
"""
limitup_14d_search.py - ストップ高後14日以内の「勝ち筋」網羅的探索

探索の切り口:
1. 日柄調整（何日待ってから入るか：Day 3, 5, 7, 10）
2. 押し目・下げ止まりパターン:
   - ST高当日の終値ライン付近まで押したところ（ストップ高ライン支持線）
   - 25日移動平均線まで押したところ（MA25反発）
   - 初めて陽線が出た日（陰線連続からの反転）
   - 出来高急減（過熱感の完全冷却）
3. 保有期間（3日、5日、10日）× 利確(+5%, +10%, +15%) / 損切(-3%, -5%, -7%)
4. 前半・後半の2分割検証で再現性を確認
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import limitup
import technical
import refine
from badnews import _latest_fin_before, _bad_flags, _prep_fin_extended
from preearnings import _next_earnings_date
from preentry import _biz_days_until


def search_14d_patterns(quotes: pd.DataFrame, fin=None, listed=None) -> pd.DataFrame:
    st = limitup.detect_limit_up(quotes)
    if st.empty:
        return pd.DataFrame()

    qi = quotes.sort_values(["code", "date"]).copy()
    g = qi.groupby("code")
    # MA25, MA20出来高
    qi["ma25"] = g["close"].transform(lambda s: s.rolling(25, min_periods=10).mean())
    qi["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())

    grouped = {str(code): grp.reset_index(drop=True) for code, grp in qi.groupby("code")}

    sec_map = {}
    if listed is not None and "sector33" in listed.columns:
        sec_map = dict(zip(listed["code"].astype(str), listed["sector33"]))

    records = []

    for _, row in st.iterrows():
        code = str(row["code"])
        st_date = pd.to_datetime(row["date"])
        st_close = row["close"]

        if code not in grouped:
            continue
        cqi = grouped[code]
        m = cqi.index[cqi["date"] == row["date"]].tolist()
        if not m:
            continue
        st_idx = m[0]

        # 14日間の値動きを見るため、最低14営業日の余裕が必要
        if st_idx + 14 >= len(cqi):
            continue

        sec = sec_map.get(code, "その他")

        # --- パターンA: 日柄待ち（3日後、5日後、7日後、10日後の寄り付き参入） ---
        for wait_days in [3, 5, 7, 10]:
            e_idx = st_idx + wait_days
            if e_idx >= len(cqi):
                continue
            e_bar = cqi.iloc[e_idx]
            e_px = e_bar["open"]
            if e_px <= 0:
                continue

            # 5営業日保有シミュレーション（利確+10%/損切-5%）
            hold = cqi.iloc[e_idx: min(e_idx + 6, len(cqi))]
            if len(hold) < 2:
                continue
            tp_px = e_px * 1.10
            sl_px = e_px * 0.95
            ret = (hold["close"].values[-1] / e_px - 1.0) * 100.0
            for _, b in hold.iterrows():
                if b["low"] <= sl_px:
                    ret = -5.0; break
                elif b["high"] >= tp_px:
                    ret = 10.0; break

            records.append({
                "base_date": st_date,
                "strategy": f"P1_日柄待ち(Day {wait_days}寄り参入/保有5日/利10/損5)",
                "ret": ret,
            })

        # --- パターンB: ストップ高終値ライン（支持線）タッチで翌日参入 ---
        # ストップ高の翌日(Day 1)以降、終値が ST高終値の -3%〜+3% まで押した最初の日の翌日寄りで買い
        for d in range(1, 10):
            idx_d = st_idx + d
            if idx_d >= len(cqi):
                break
            b_d = cqi.iloc[idx_d]
            # ST高終値付近（0.97〜1.03）にタッチしたか
            if b_d["low"] <= st_close * 1.03 and b_d["high"] >= st_close * 0.97:
                # 翌営業日寄り参入
                if idx_d + 1 < len(cqi):
                    e_bar = cqi.iloc[idx_d + 1]
                    e_px = e_bar["open"]
                    if e_px > 0:
                        hold = cqi.iloc[idx_d + 1: min(idx_d + 7, len(cqi))]
                        tp_px = e_px * 1.10
                        sl_px = e_px * 0.95
                        ret = (hold["close"].values[-1] / e_px - 1.0) * 100.0
                        for _, b in hold.iterrows():
                            if b["low"] <= sl_px:
                                ret = -5.0; break
                            elif b["high"] >= tp_px:
                                ret = 10.0; break
                        records.append({
                            "base_date": st_date,
                            "strategy": "P2_ST高終値支持線タッチ後翌日寄り買い(保有5日/利10/損5)",
                            "ret": ret,
                        })
                break  # 最初のタッチのみ

        # --- パターンC: 初反発（連続陰線のあと、初めて陽線が出た翌日寄り買い） ---
        for d in range(2, 10):
            idx_d = st_idx + d
            if idx_d >= len(cqi):
                break
            prev_b = cqi.iloc[idx_d - 1]
            cur_b = cqi.iloc[idx_d]
            # 前日陰線かつ当日陽線（反転シグナル）
            if prev_b["close"] < prev_b["open"] and cur_b["close"] > cur_b["open"]:
                if idx_d + 1 < len(cqi):
                    e_bar = cqi.iloc[idx_d + 1]
                    e_px = e_bar["open"]
                    if e_px > 0:
                        hold = cqi.iloc[idx_d + 1: min(idx_d + 7, len(cqi))]
                        tp_px = e_px * 1.10
                        sl_px = e_px * 0.95
                        ret = (hold["close"].values[-1] / e_px - 1.0) * 100.0
                        for _, b in hold.iterrows():
                            if b["low"] <= sl_px:
                                ret = -5.0; break
                            elif b["high"] >= tp_px:
                                ret = 10.0; break
                        records.append({
                            "base_date": st_date,
                            "strategy": "P3_初陽線(陰線から初反転)翌日寄り買い(保有5日/利10/損5)",
                            "ret": ret,
                        })
                break

        # --- パターンD: 出来高枯れ（出来高が20日平均の50%以下まで冷却） ---
        for d in range(3, 11):
            idx_d = st_idx + d
            if idx_d >= len(cqi):
                break
            b_d = cqi.iloc[idx_d]
            v = b_d.get("volume")
            v_ma = b_d.get("vol_ma20")
            if pd.notna(v) and pd.notna(v_ma) and v_ma > 0 and v < v_ma * 0.5:
                if idx_d + 1 < len(cqi):
                    e_bar = cqi.iloc[idx_d + 1]
                    e_px = e_bar["open"]
                    if e_px > 0:
                        hold = cqi.iloc[idx_d + 1: min(idx_d + 7, len(cqi))]
                        tp_px = e_px * 1.10
                        sl_px = e_px * 0.95
                        ret = (hold["close"].values[-1] / e_px - 1.0) * 100.0
                        for _, b in hold.iterrows():
                            if b["low"] <= sl_px:
                                ret = -5.0; break
                            elif b["high"] >= tp_px:
                                ret = 10.0; break
                        records.append({
                            "base_date": st_date,
                            "strategy": "P4_出来高急減(冷却)翌日寄り買い(保有5日/利10/損5)",
                            "ret": ret,
                        })
                break

        # --- パターンE: 25日移動平均線(MA25)タッチ反発 ---
        for d in range(3, 14):
            idx_d = st_idx + d
            if idx_d >= len(cqi):
                break
            b_d = cqi.iloc[idx_d]
            ma25 = b_d.get("ma25")
            if pd.notna(ma25) and ma25 > 0 and b_d["low"] <= ma25 <= b_d["high"]:
                if idx_d + 1 < len(cqi):
                    e_bar = cqi.iloc[idx_d + 1]
                    e_px = e_bar["open"]
                    if e_px > 0:
                        hold = cqi.iloc[idx_d + 1: min(idx_d + 7, len(cqi))]
                        tp_px = e_px * 1.10
                        sl_px = e_px * 0.95
                        ret = (hold["close"].values[-1] / e_px - 1.0) * 100.0
                        for _, b in hold.iterrows():
                            if b["low"] <= sl_px:
                                ret = -5.0; break
                            elif b["high"] >= tp_px:
                                ret = 10.0; break
                        records.append({
                            "base_date": st_date,
                            "strategy": "P5_MA25タッチ翌日寄り買い(保有5日/利10/損5)",
                            "ret": ret,
                        })
                break

    return pd.DataFrame(records)


def summarize(df: pd.DataFrame, mid_date) -> pd.DataFrame:
    rows = []
    for strat, g in df.groupby("strategy"):
        arr = g["ret"].values
        g_h1 = g[g["base_date"] <= mid_date]
        g_h2 = g[g["base_date"] > mid_date]

        r1 = g_h1["ret"].mean() if len(g_h1) else np.nan
        r2 = g_h2["ret"].mean() if len(g_h2) else np.nan
        w1 = (g_h1["ret"] > 0).mean() * 100 if len(g_h1) else np.nan
        w2 = (g_h2["ret"] > 0).mean() * 100 if len(g_h2) else np.nan

        rows.append({
            "戦略パターン": strat,
            "サンプル数": len(g),
            "全勝率": f"{(arr > 0).mean()*100:.1f}%",
            "全平均R": round(arr.mean(), 2),
            "前半R": round(r1, 2),
            "後半R": round(r2, 2),
            "判定": "◎両プラス" if (r1 > 0 and r2 > 0) else ("△片方" if (r1 > 0 or r2 > 0) else "×全滅"),
        })
    res = pd.DataFrame(rows)
    return res.sort_values("全平均R", ascending=False)


def run(quotes: pd.DataFrame, fin=None, listed=None):
    print("=" * 76)
    print("ストップ高後 14日以内の『勝ち筋パターン』網羅的検証")
    print("※ 手数料・スリッページ未考慮 / 投資助言ではありません")
    print("=" * 76)

    df = search_14d_patterns(quotes, fin=fin, listed=listed)
    if df.empty:
        print("データがありませんでした。")
        return

    unique_dates = sorted(df["base_date"].unique())
    mid_date = unique_dates[len(unique_dates) // 2]
    print(f"総トレードシミュレーション数: {len(df):,} 件 / 分割境界日: {mid_date.strftime('%Y-%m-%d')}\n")

    res = summarize(df, mid_date)
    print(res.to_string(index=False))
