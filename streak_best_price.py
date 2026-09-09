# -*- coding: utf-8 -*-
"""
streak_best_price.py - 5連騰シグナル点灯後の「ベストな買値（日中OHLCベースの最適化）」検証
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def run_best_price(quotes: pd.DataFrame, fin=None, listed=None):
    print("=" * 76)
    print("5連騰シグナル銘柄の『ベストな買値（日中OHLCベース）』最適化検証")
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

    print(f"5連騰シグナル対象サンプル: {len(sub5):,} 件 / 分割境界: {mid_date.strftime('%Y-%m-%d')}\n")

    if len(sub5) == 0:
        print("該当サンプルがありません。")
        return

    by_code = {c: (grp["open"].to_numpy(), grp["high"].to_numpy(), grp["low"].to_numpy(), grp["close"].to_numpy(), grp["date"].to_numpy())
               for c, grp in qi.groupby("code")}

    # 翌営業日の日中OHLCを使った買値バリエーションの検証
    # 保有期間: 翌日大引けまで（1日） および 3営業日保有
    # 買値ルール:
    # 1. 翌日始値 (Open)
    # 2. 翌日安値 (Low) ※理論上の最安値で買えた場合（指値が奇跡的に刺さった仮定）
    # 3. 翌日終値 (Close) ※引け買い
    # 4. 始値から -2% の指値（安値がそれ以下なら約定、不可なら見送り）
    # 5. 始値から -3% の指値

    rules = [
        ("翌日始値(Open)で成行", lambda o,h,l,c: o),
        ("翌日安値(Low) ※理論最安値", lambda o,h,l,c: l),
        ("翌日終値(Close)で成行", lambda o,h,l,c: c),
        ("始値 -2% 指値 (刺されば)", lambda o,h,l,c: o * 0.98 if l <= o * 0.98 else np.nan),
        ("始値 -3% 指値 (刺されば)", lambda o,h,l,c: o * 0.97 if l <= o * 0.97 else np.nan),
    ]

    results = []

    for label, px_func in rules:
        for hold_days in [1, 3]:
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

                o, h, l, c = o_arr[pos], h_arr[pos], l_arr[pos], c_arr[pos]
                entry_px = px_func(o, h, l, c)
                if pd.isna(entry_px) or entry_px <= 0:
                    continue

                # 出口：hold_days後の終値
                exit_px = c_arr[pos + hold_days - 1]
                if not exit_px or exit_px <= 0: continue

                trade_ret = (exit_px / entry_px - 1.0) * 100.0
                rets.append(trade_ret)
                dates_list.append(d)

            if not rets: continue
            rets_arr = np.array(rets)
            dates_arr = np.array(dates_list)

            h1_mask = dates_arr <= np.datetime64(mid_date)
            r1 = rets_arr[h1_mask]
            r2 = rets_arr[~h1_mask]

            results.append({
                "買値エントリーモデル": label,
                "保有日数": f"{hold_days}営業日",
                "約定件数": len(rets_arr),
                "勝率": f"{(rets_arr > 0).mean()*100:.1f}%",
                "平均R(%)": round(rets_arr.mean(), 2),
                "中央値(%)": round(float(np.median(rets_arr)), 2),
                "前半R(%)": round(r1.mean(), 2) if len(r1) else np.nan,
                "後半R(%)": round(r2.mean(), 2) if len(r2) else np.nan,
                "判定": "◎両プラス" if (len(r1) and len(r2) and r1.mean() > 0 and r2.mean() > 0) else ("△片方" if (len(r1) and len(r2) and (r1.mean() > 0 or r2.mean() > 0)) else "×全滅"),
            })

    res_df = pd.DataFrame(results).sort_values("平均R(%)", ascending=False)
    print("--- 【5連騰シグナル後のベストな買値モデル 検証結果】 ---")
    print(res_df.to_string(index=False))
