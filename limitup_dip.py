# -*- coding: utf-8 -*-
"""
limitup_dip.py - ストップ高後の下ヒゲ指値・当日リバウンド日計り戦略の検証

検証の作法:
- 始値より下(-X%)で指値買い、当日中に+Y%で利確、未達は大引け決済
- 前半・後半の2分割検証で再現性を確認
- タイプ別（仕手的特徴／決算／説明不能）に分類
- 手数料・スリッページは未考慮（明記）
- 投資助言ではない（明記）
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import limitup
import limitup_analyze

DIP_TARGETS = [3, 4, 5, 6, 8]       # 始値からの下落指値(%)
TAKE_PROFITS = [3, 5, 8, 10]       # 買値からの利確幅(%)
STOP_LOSSES = [None, -3, -5]       # 損切り幅(%)


def simulate_dip_trades(quotes: pd.DataFrame, fin=None, listed=None, days=(1, 2, 3)) -> pd.DataFrame:
    st = limitup.detect_limit_up(quotes)
    if st.empty:
        return pd.DataFrame()

    try:
        reason_df = limitup_analyze.classify_reason(quotes, st, fin=fin, listed=listed)
        reason_df["event_id"] = reason_df["code"].astype(str) + "_" + pd.to_datetime(reason_df["date"]).dt.strftime('%Y%m%d')
        reason_map = dict(zip(reason_df["event_id"], reason_df["reason_type"]))
    except Exception:
        reason_map = {}

    quotes_sorted = quotes.sort_values(["code", "date"]).copy()
    grouped = {code: group.reset_index(drop=True) for code, group in quotes_sorted.groupby("code")}

    trades = []

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

        for d in days:
            t_idx = st_idx + d
            if t_idx >= len(cqi):
                break
            bar = cqi.iloc[t_idx]
            o = bar["open"]
            h = bar["high"]
            l = bar["low"]
            c = bar["close"]

            if o <= 0 or np.isnan(o):
                continue

            for dip in DIP_TARGETS:
                buy_px = o * (1.0 - dip / 100.0)
                # 当日の安値が指値以下なら約定
                if l <= buy_px:
                    # 買えた場合の値動きシミュレーション
                    for tp in TAKE_PROFITS:
                        tp_px = buy_px * (1.0 + tp / 100.0)
                        for sl in STOP_LOSSES:
                            # 判定
                            # 1. 損切りがある場合
                            if sl is not None:
                                sl_px = buy_px * (1.0 + sl / 100.0)
                                hit_sl = l <= sl_px
                                hit_tp = h >= tp_px
                                if hit_sl and hit_tp:
                                    # 保守的に損切り優先
                                    ret = sl
                                elif hit_sl:
                                    ret = sl
                                elif hit_tp:
                                    ret = tp
                                else:
                                    # 引け成行
                                    ret = (c / buy_px - 1.0) * 100.0
                            else:
                                # 損切りなし（届かなければ大引け決済）
                                if h >= tp_px:
                                    ret = tp
                                else:
                                    ret = (c / buy_px - 1.0) * 100.0

                            trades.append({
                                "event_id": ev_id,
                                "base_date": pd.to_datetime(st_date),
                                "day": d,
                                "reason": reason,
                                "dip": dip,
                                "tp": tp,
                                "sl": "なし" if sl is None else f"{sl}%",
                                "ret": ret,
                                "is_win": 1 if ret > 0 else 0,
                            })

    return pd.DataFrame(trades)


def summarize_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (day, dip, tp, sl), g in df.groupby(["day", "dip", "tp", "sl"]):
        arr = g["ret"].values
        rows.append({
            "Day": int(day),
            "指値(始値比)": f"-{dip}%",
            "利確": f"+{tp}%",
            "損切": sl,
            "約定件数": len(g),
            "勝率(%)": round((arr > 0).mean() * 100.0, 1),
            "平均R(%)": round(arr.mean(), 2),
            "中央値(%)": round(float(np.median(arr)), 2),
        })
    res = pd.DataFrame(rows)
    return res.sort_values(["Day", "平均R(%)"], ascending=[True, False])


def run(quotes: pd.DataFrame, fin=None, listed=None):
    print("=" * 72)
    print("ストップ高後 下ヒゲ指値・当日リバウンド日計り戦略検証 (Day 1〜3)")
    print("※ 手数料・スリッページ未考慮 / 投資助言ではありません")
    print("=" * 72)

    df = simulate_dip_trades(quotes, fin=fin, listed=listed, days=(1, 2, 3))
    if df.empty:
        print("有効なトレードデータがありませんでした。")
        return

    print(f"約定シミュレーション総数: {len(df):,} レコード")

    # Dayごとに平均R上位の組み合わせを表示
    sum_all = summarize_trades(df)

    for d in [1, 2, 3]:
        print(f"\n--- 【全期間】Day {d} 平均リターン上位10組み合わせ ---")
        sub = sum_all[sum_all["Day"] == d].head(10)
        print(sub.to_string(index=False))

    # 前半・後半の分割検証（Day 1 と Day 2 の最良候補について）
    unique_dates = sorted(df["base_date"].unique())
    mid_date = unique_dates[len(unique_dates) // 2]
    df_h1 = df[df["base_date"] <= mid_date]
    df_h2 = df[df["base_date"] > mid_date]

    print(f"\n--- 【分割検証】Day 1〜2 の主要組み合わせ（前半 vs 後半） ---")
    sum_h1 = summarize_trades(df_h1)
    sum_h2 = summarize_trades(df_h2)

    # 注目する組み合わせをピックアップ
    keys = [
        (1, "-5%", "+5%", "なし"),
        (1, "-5%", "+3%", "なし"),
        (1, "-6%", "+5%", "なし"),
        (1, "-6%", "+3%", "なし"),
        (2, "-4%", "+3%", "なし"),
        (2, "-5%", "+5%", "なし"),
    ]

    split_rows = []
    for day, dip, tp, sl in keys:
        row_all = sum_all[(sum_all["Day"] == day) & (sum_all["指値(始値比)"] == dip) & (sum_all["利確"] == tp) & (sum_all["損切"] == sl)]
        row_h1 = sum_h1[(sum_h1["Day"] == day) & (sum_h1["指値(始値比)"] == dip) & (sum_h1["利確"] == tp) & (sum_h1["損切"] == sl)]
        row_h2 = sum_h2[(sum_h2["Day"] == day) & (sum_h2["指値(始値比)"] == dip) & (sum_h2["利確"] == tp) & (sum_h2["損切"] == sl)]

        if not row_all.empty and not row_h1.empty and not row_h2.empty:
            split_rows.append({
                "Day": day, "条件": f"買{dip}/利{tp}/損{sl}",
                "全体R(%)": row_all.iloc[0]["平均R(%)"], "全体勝率": f"{row_all.iloc[0]['勝率(%)']}%",
                "前半R(%)": row_h1.iloc[0]["平均R(%)"], "前半勝率": f"{row_h1.iloc[0]['勝率(%)']}%",
                "後半R(%)": row_h2.iloc[0]["平均R(%)"], "後半勝率": f"{row_h2.iloc[0]['勝率(%)']}%",
            })
    print(pd.DataFrame(split_rows).to_string(index=False))

    # タイプ別（仕手的特徴 vs 決算 vs 説明不能）
    print(f"\n--- 【タイプ別】Day 1 (買-5%/利+5%/損なし) の成績比較 ---")
    type_rows = []
    for reason, g in df[(df["day"] == 1) & (df["dip"] == 5) & (df["tp"] == 5) & (df["sl"] == "なし")].groupby("reason"):
        arr = g["ret"].values
        type_rows.append({
            "タイプ": reason,
            "約定件数": len(g),
            "勝率(%)": round((arr > 0).mean() * 100.0, 1),
            "平均R(%)": round(arr.mean(), 2),
            "中央値(%)": round(float(np.median(arr)), 2),
        })
    print(pd.DataFrame(type_rows).to_string(index=False))
