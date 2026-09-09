# -*- coding: utf-8 -*-
"""
limitup_all_hypotheses.py - 3つの勝利の仮説を一括検証（修正版）

検証の作法:
- 全て事実にもとづいて判定（推測なし）
- 日足OHLCの時系列順序の先読みバイアス（高値が安値より前につく現象）を排除
  → 日中リバウンドは「終値(Close)ベース」で厳密に判定
- 前半・後半の2分割検証で再現性を確認
- 対相場比較・手数料未考慮を明記
- 投資助言ではない（明記）
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import limitup
import limitup_analyze
import refine
from badnews import _latest_fin_before, _bad_flags, _prep_fin_extended
from preearnings import _next_earnings_date
from preentry import _biz_days_until


def run_all(quotes: pd.DataFrame, fin=None, listed=None):
    print("=" * 76)
    print("ストップ高後の3大仮説 一括総合検証（厳格・時系列バイアス排除版）")
    print("※ 手数料・スリッページ未考慮 / 投資助言ではありません")
    print("=" * 76)

    st = limitup.detect_limit_up(quotes)
    if st.empty:
        print("ストップ高が検出されませんでした。")
        return

    print(f"ストップ高総数: {len(st):,} 件")

    try:
        reason_df = limitup_analyze.classify_reason(quotes, st, fin=fin, listed=listed)
        reason_df["event_id"] = reason_df["code"].astype(str) + "_" + pd.to_datetime(reason_df["date"]).dt.strftime('%Y%m%d')
        reason_map = dict(zip(reason_df["event_id"], reason_df["reason_type"]))
    except Exception:
        reason_map = {}

    quotes_sorted = quotes.sort_values(["code", "date"]).copy()
    grouped = {code: group.reset_index(drop=True) for code, group in quotes_sorted.groupby("code")}

    unique_dates = sorted(pd.to_datetime(st["date"]).unique())
    mid_date = unique_dates[len(unique_dates) // 2]
    print(f"分割検証の境界日: {mid_date.strftime('%Y-%m-%d')} (前半 / 後半)")

    # =========================================================================
    # ■ 仮説1: 仕手・急騰株の「Day 1 下ヒゲ指値 → 大引け終値決済（純日計り）」
    # =========================================================================
    print("\n" + "=" * 76)
    print("【仮説1 検証】仕手・説明不能株「Day 1 下ヒゲ指値買い → 大引け終値決済」")
    print("（※日足の高値が安値より前につく先読みバイアスを排除し、終値で厳密決済）")
    print("=" * 76)

    h1_records = []
    for _, row in st.iterrows():
        code = row["code"]
        st_date = pd.to_datetime(row["date"])
        ev_id = f"{code}_{st_date.strftime('%Y%m%d')}"
        reason = reason_map.get(ev_id, "説明不能")

        if reason not in ["仕手的特徴", "説明不能"]:
            continue

        if code not in grouped:
            continue
        cqi = grouped[code]
        match_idx = cqi.index[cqi["date"] == row["date"]].tolist()
        if not match_idx or match_idx[0] + 1 >= len(cqi):
            continue

        bar1 = cqi.iloc[match_idx[0] + 1]
        o, l, c = bar1["open"], bar1["low"], bar1["close"]
        if o <= 0:
            continue

        # 翌日(Day 2)始値のデータがあるか
        next_o = cqi.iloc[match_idx[0] + 2]["open"] if match_idx[0] + 2 < len(cqi) else np.nan

        for dip in [3, 4, 5, 6, 7, 8]:
            buy_px = o * (1.0 - dip / 100.0)
            if l <= buy_px:  # 指値にタッチ（約定）
                # 1. 大引け決済リターン
                ret_close = (c / buy_px - 1.0) * 100.0
                # 2. 翌日寄り付き決済リターン
                ret_next_open = (next_o / buy_px - 1.0) * 100.0 if pd.notna(next_o) and next_o > 0 else np.nan

                h1_records.append({
                    "base_date": st_date,
                    "dip": f"-{dip}%",
                    "ret_close": ret_close,
                    "ret_next_open": ret_next_open,
                })

    h1_df = pd.DataFrame(h1_records)
    if not h1_df.empty:
        h1_rows = []
        for dip, g in h1_df.groupby("dip"):
            arr_c = g["ret_close"].values
            g_h1 = g[g["base_date"] <= mid_date]
            g_h2 = g[g["base_date"] > mid_date]

            r1_c = g_h1["ret_close"].mean() if len(g_h1) else np.nan
            r2_c = g_h2["ret_close"].mean() if len(g_h2) else np.nan
            w_all = (arr_c > 0).mean() * 100.0

            # 翌日寄り決済の平均
            arr_no = g["ret_next_open"].dropna().values
            r_no = arr_no.mean() if len(arr_no) else np.nan

            h1_rows.append({
                "指値位置": dip,
                "約定件数": len(g),
                "当日終値勝率": f"{w_all:.1f}%",
                "当日終値平均R": round(arr_c.mean(), 2),
                "前半終値R": round(r1_c, 2),
                "後半終値R": round(r2_c, 2),
                "翌日寄り平均R": round(r_no, 2) if pd.notna(r_no) else "—",
            })
        res_h1 = pd.DataFrame(h1_rows)
        # dip値でソート
        res_h1["_sort"] = res_h1["指値位置"].str.extract(r'(-?\d+)').astype(int).abs()
        res_h1 = res_h1.sort_values("_sort").drop(columns="_sort")
        print(res_h1.to_string(index=False))

    # =========================================================================
    # ■ 仮説2: 「3日横ばい（エネルギー充填）」からの4日目順張り
    # =========================================================================
    print("\n" + "=" * 76)
    print("【仮説2 検証】ストップ高後「3日横ばい(±5%)」からの4日目順張り（7日目決済）")
    print("=" * 76)

    h2_records = []
    for _, row in st.iterrows():
        code = row["code"]
        st_date = pd.to_datetime(row["date"])
        st_close = row["close"]

        if code not in grouped:
            continue
        cqi = grouped[code]
        match_idx = cqi.index[cqi["date"] == row["date"]].tolist()
        if not match_idx:
            continue
        st_idx = match_idx[0]

        obs = cqi.iloc[st_idx + 1: st_idx + 4]
        if len(obs) < 3:
            continue
        ch3 = (obs["close"].values[-1] / st_close - 1.0) * 100.0
        if not (-5.0 <= ch3 <= 5.0):
            continue

        entry_idx = st_idx + 4
        if entry_idx >= len(cqi):
            continue
        entry_price = cqi["open"].values[entry_idx]
        if entry_price <= 0:
            continue

        hold = cqi.iloc[entry_idx: entry_idx + 4]
        if len(hold) == 0:
            continue

        for tp in [5, 8, 10, 15]:
            for sl in [-3, -5, -8]:
                tp_px = entry_price * (1.0 + tp / 100.0)
                sl_px = entry_price * (1.0 + sl / 100.0)
                hit_tp, hit_sl = False, False
                ret = 0.0
                for _, b in hold.iterrows():
                    if b["low"] <= sl_px and b["high"] >= tp_px:
                        ret = sl; hit_sl = True; break
                    elif b["low"] <= sl_px:
                        ret = sl; hit_sl = True; break
                    elif b["high"] >= tp_px:
                        ret = tp; hit_tp = True; break
                if not hit_tp and not hit_sl:
                    ret = (hold["close"].values[-1] / entry_price - 1.0) * 100.0

                h2_records.append({
                    "base_date": st_date,
                    "tp": f"+{tp}%",
                    "sl": f"{sl}%",
                    "ret": ret,
                })

    h2_df = pd.DataFrame(h2_records)
    if not h2_df.empty:
        h2_rows = []
        for (tp, sl), g in h2_df.groupby(["tp", "sl"]):
            arr_all = g["ret"].values
            g_h1 = g[g["base_date"] <= mid_date]
            g_h2 = g[g["base_date"] > mid_date]
            r1 = g_h1["ret"].mean() if len(g_h1) else np.nan
            r2 = g_h2["ret"].mean() if len(g_h2) else np.nan

            h2_rows.append({
                "利確": tp, "損切": sl,
                "取引数": len(g),
                "全勝率": f"{(arr_all > 0).mean()*100:.1f}%",
                "全平均R": round(arr_all.mean(), 2),
                "前半R": round(r1, 2),
                "後半R": round(r2, 2),
            })
        res_h2 = pd.DataFrame(h2_rows).sort_values("全平均R", ascending=False)
        print(res_h2.to_string(index=False))

    # =========================================================================
    # ■ 仮説3: 本命戦略への還流（高勝率業種 × 決算通過直後の押し目）
    # =========================================================================
    print("\n" + "=" * 76)
    print("【仮説3 検証】ストップ高を出した銘柄が、その後『押し目本命条件』に合流した成績")
    print("=" * 76)

    sec_map = {}
    if listed is not None and "sector33" in listed.columns:
        sec_map = dict(zip(listed["code"].astype(str), listed["sector33"]))

    fin_ext = _prep_fin_extended(fin) if fin is not None else pd.DataFrame()
    fin_by_code = {}
    if fin is not None and "code" in fin.columns and "disclosed_date" in fin.columns:
        for cg, gg in fin.groupby("code"):
            fin_by_code[str(cg)] = sorted(pd.to_datetime(gg["disclosed_date"]).tolist())

    qi = quotes.sort_values(["code", "date"]).copy()
    g = qi.groupby("code")
    h = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / h - 1.0
    qi_by_code = {str(c): gg.reset_index(drop=True) for c, gg in qi.groupby("code")}

    h3_records = []
    for _, row in st.iterrows():
        code = str(row["code"])
        st_date = pd.to_datetime(row["date"])
        sec = sec_map.get(code, "")
        if sec not in refine.HIGH_WIN_SECTORS:
            continue

        cqi = qi_by_code.get(code)
        if cqi is None:
            continue
        match_idx = cqi.index[cqi["date"] == row["date"]].tolist()
        if not match_idx:
            continue
        st_idx = match_idx[0]

        for offset in range(10, 61):
            chk_idx = st_idx + offset
            if chk_idx + 10 >= len(cqi):
                break
            r = cqi.iloc[chk_idx].copy()
            r["sector33"] = sec
            chk_date = pd.Timestamp(r["date"])

            if not refine._f_pullback(r):
                continue

            earnings = _next_earnings_date(fin_by_code, code, chk_date)
            du = _biz_days_until(chk_date, earnings)
            if du is not None and du <= 20:
                continue

            fin_row = _latest_fin_before(fin_ext, code, chk_date)
            flags = _bad_flags(fin_row)
            if flags["予想未達"] or flags["減益"] or flags["減収"]:
                continue

            entry = r["close"]
            fut = cqi.iloc[chk_idx + 1: chk_idx + 11]["close"].values
            if len(fut) < 10 or entry <= 0:
                continue

            path10 = [(p / entry - 1.0) * 100.0 for p in fut]
            ret5 = path10[4] if len(path10) >= 5 else path10[-1]
            ret5_stop = -5.0 if any(p <= -5.0 for p in path10[:5]) else ret5
            hit_up10 = any(p >= 10.0 for p in path10)
            hit_dn5 = any(p <= -5.0 for p in path10)

            h3_records.append({
                "base_date": chk_date,
                "code": code,
                "ret5": ret5,
                "ret5_stop": ret5_stop,
                "hit_up10": 1 if hit_up10 else 0,
                "hit_dn5": 1 if hit_dn5 else 0,
            })
            break

    h3_df = pd.DataFrame(h3_records)
    if not h3_df.empty:
        arr_all = h3_df["ret5_stop"].values
        g_h1 = h3_df[h3_df["base_date"] <= mid_date]
        g_h2 = h3_df[h3_df["base_date"] > mid_date]

        print(f"合流該当件数: {len(h3_df):,} 件")
        print(f"  ・全期間: 平均(損切5%){arr_all.mean():+.2f}% / 勝率{(arr_all>0).mean()*100:.1f}%")
        print(f"           2週で+10%到達率 {h3_df['hit_up10'].mean()*100:.1f}% / -5%到達率 {h3_df['hit_dn5'].mean()*100:.1f}%")
        if len(g_h1) and len(g_h2):
            print(f"  ・前半（{len(g_h1)}件）: 平均(損切5%){g_h1['ret5_stop'].mean():+.2f}% / 勝率{(g_h1['ret5_stop']>0).mean()*100:.1f}%")
            print(f"  ・後半（{len(g_h2)}件）: 平均(損切5%){g_h2['ret5_stop'].mean():+.2f}% / 勝率{(g_h2['ret5_stop']>0).mean()*100:.1f}%")
    else:
        print("ストップ高後に対象条件へ合流した銘柄がありませんでした。")
