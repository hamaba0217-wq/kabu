# -*- coding: utf-8 -*-
"""
streak_industry_filter.py - 5連騰シグナル × 業種・決算・悪材料回避の絞り込み検証
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import refine
from badnews import _bad_flags

def run_filter_opt(quotes: pd.DataFrame, fin=None, listed=None):
    print("=" * 76)
    print("5連騰シグナル × 業種・悪材料回避の『高勝率絞り込み』検証")
    print("※ 手数料・スリッページ未考慮 / 投資助言ではありません")
    print("=" * 76)

    sec_map = {}
    if listed is not None and "sector33" in listed.columns:
        sec_map = dict(zip(listed["code"].astype(str), listed["sector33"]))

    qi = quotes.sort_values(["code", "date"]).reset_index(drop=True)
    g = qi.groupby("code")

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

    bad_set = _bad_flags(fin) if fin is not None else set()
    df["has_bad"] = [1 if (str(row["code"]), pd.to_datetime(row["date"]).strftime("%Y-%m-%d")) in bad_set else 0 for _, row in df.iterrows()]

    codes_str = df["code"].astype(str)
    df["sec"] = codes_str.map(sec_map).fillna("その他")
    df["is_high_win_sec"] = df["sec"].isin(refine.HIGH_WIN_SECTORS).astype(int)

    all_dates = sorted(df["date"].unique())
    mid_date = pd.to_datetime(all_dates[len(all_dates) // 2])

    sub5 = df[df["streak"] >= 5].copy()
    print("5連騰シグナル全サンプル集計完了")

    filters = [
        ("基準: 5連騰のみ (全銘柄)", sub5["streak"] >= 5),
        ("F1: 5連騰 × 悪材料なし", (sub5["streak"] >= 5) & (sub5["has_bad"] == 0)),
        ("F2: 5連騰 × 高勝率業種(金融/倉庫等)", (sub5["streak"] >= 5) & (sub5["is_high_win_sec"] == 1)),
        ("F3: 5連騰 × 悪材料なし × 高勝率業種", (sub5["streak"] >= 5) & (sub5["has_bad"] == 0) & (sub5["is_high_win_sec"] == 1)),
        ("F4: 5連騰 × 悪材料なし × 一般業種", (sub5["streak"] >= 5) & (sub5["has_bad"] == 0) & (sub5["is_high_win_sec"] == 0)),
    ]

    rows = []
    for label, mask in filters:
        sub = sub5[mask]
        n = len(sub)
        if n < 30: continue
        ret = (sub["next_close"] / sub["close"] - 1.0) * 100.0

        h1 = sub[sub["date"] <= mid_date]
        h2 = sub[sub["date"] > mid_date]
        r1 = (h1["next_close"] / h1["close"] - 1.0) * 100.0 if len(h1) else np.array([])
        r2 = (h2["next_close"] / h2["close"] - 1.0) * 100.0 if len(h2) else np.array([])

        rows.append({
            "フィルター条件": label,
            "件数": n,
            "勝率": f"{(ret > 0).mean()*100:.1f}%",
            "平均R(%)": round(ret.mean(), 2),
            "中央値(%)": round(float(np.median(ret)), 2),
            "前半R(%)": round(r1.mean(), 2) if len(r1) else np.nan,
            "後半R(%)": round(r2.mean(), 2) if len(r2) else np.nan,
            "判定": "◎両プラス" if (len(r1) and len(r2) and r1.mean() > 0 and r2.mean() > 0) else ("△片方" if (len(r1) and len(r2) and (r1.mean() > 0 or r2.mean() > 0)) else "×全滅"),
        })

    res_df = pd.DataFrame(rows).sort_values("平均R(%)", ascending=False)
    print("--- 【5連騰シグナル × 業種・ニュース（悪材料回避）の検証結果】 ---")
    print(res_df.to_string(index=False))
