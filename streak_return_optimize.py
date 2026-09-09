# -*- coding: utf-8 -*-
"""
streak_return_optimize.py - 5連騰シグナル点灯後の「買値×利確×損切り」網羅的最適化
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def run_return_opt(quotes: pd.DataFrame, fin=None, listed=None):
    print("=" * 76)
    print("5連騰シグナル銘柄の『平均リターン最大化（買値・利確・損切り）』検証")
    print("※ 手数料・スリッページ未考慮 / 投資助言ではありません")
    print("=" * 76)

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

    sub5 = qi[qi["streak"] >= 5].copy()
    all_dates = sorted(sub5["date"].unique())
    mid_date = pd.to_datetime(all_dates[len(all_dates) // 2])

    print("5連騰シグナル対象サンプル集計完了")

    if len(sub5) == 0:
        print("該当サンプルがありません。")
        return

    by_code = {c: (grp["open"].to_numpy(), grp["high"].to_numpy(), grp["low"].to_numpy(), grp["close"].to_numpy(), grp["date"].to_numpy())
               for c, grp in qi.groupby("code")}

    results = []

    for hold_days in [1, 3, 5]:
        for tp in [3.0, 5.0, 7.0, 10.0]:
            for sl in [None, -2.0, -3.0, -5.0]:
                rets = []
                dates_list = []

                for _, row in sub5.iterrows():
                    code = row["code"]
                    d = row["date"]
                    arrs = by_code.get(code)
                    if arrs is None: continue
                    o_arr, h_arr, l_arr, c_arr, d_arr = arrs
                    pos = np.searchsorted(d_arr, np.datetime64(pd.Timestamp(d)), side="right")
                    if pos + hold_days >= len(d_arr):
                        continue

                    entry_px = o_arr[pos]
                    if not entry_px or entry_px <= 0: continue

                    fut_h = h_arr[pos : pos + hold_days]
                    fut_l = l_arr[pos : pos + hold_days]
                    fut_c = c_arr[pos : pos + hold_days]

                    tp_px = entry_px * (1.0 + tp / 100.0)
                    sl_px = entry_px * (1.0 + sl / 100.0) if sl is not None else -999.0

                    hit = False
                    trade_ret = 0.0
                    for h_val, l_val, c_val in zip(fut_h, fut_l, fut_c):
                        hit_sl = (sl is not None and l_val <= sl_px)
                        hit_tp = (h_val >= tp_px)
                        if hit_sl and hit_tp:
                            trade_ret = sl; hit = True; break
                        elif hit_sl:
                            trade_ret = sl; hit = True; break
                        elif hit_tp:
                            trade_ret = tp; hit = True; break

                    if not hit:
                        trade_ret = (fut_c[-1] / entry_px - 1.0) * 100.0

                    rets.append(trade_ret)
                    dates_list.append(d)

                if not rets: continue
                rets_arr = np.array(rets)
                dates_arr = np.array(dates_list)

                h1_mask = dates_arr <= np.datetime64(mid_date)
                r1 = rets_arr[h1_mask]
                r2 = rets_arr[~h1_mask]

                results.append({
                    "保有日数": f"{hold_days}営業日",
                    "利確": f"+{tp}%",
                    "損切": "なし" if sl is None else f"{sl}%",
                    "件数": len(rets_arr),
                    "勝率": f"{(rets_arr > 0).mean()*100:.1f}%",
                    "平均R(%)": round(rets_arr.mean(), 2),
                    "中央値(%)": round(float(np.median(rets_arr)), 2),
                    "前半R(%)": round(r1.mean(), 2) if len(r1) else np.nan,
                    "後半R(%)": round(r2.mean(), 2) if len(r2) else np.nan,
                    "判定": "◎両プラス" if (len(r1) and len(r2) and r1.mean() > 0 and r2.mean() > 0) else ("△片方" if (len(r1) and len(r2) and (r1.mean() > 0 or r2.mean() > 0)) else "×全滅"),
                })

    res_df = pd.DataFrame(results).sort_values("平均R(%)", ascending=False)
    print("--- 【5連騰シグナル後の利確・損切り・保有日数 最適化結果】 ---")
    print(res_df.head(15).to_string(index=False))
