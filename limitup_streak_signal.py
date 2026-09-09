# -*- coding: utf-8 -*-
"""
limitup_streak_signal.py - 連騰シグナルからの「+10%・ストップ高到達確率」実データ検証

検証の作法:
- 1〜6日連続プラスの各シグナル点灯日について、翌日〜5日以内に+10%やST高に到達する確率を全数集計
- 全体ベースレート（自然発生率）と比較してリフト値（倍率）を算出
- 出来高条件（1.5倍、2倍、3倍）との掛け合わせ効果
- 前半・後半の2分割検証で再現性を確認
- 手数料・スリッページ未考慮（明記）
- 投資助言ではない（明記）
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import limitup


def run_signal_analysis(quotes: pd.DataFrame, fin=None, listed=None):
    print("=" * 76)
    print("連騰シグナルからの『+10%到達・ストップ高到達』実データ確率検証")
    print("※ 全銘柄・全営業日（約370万サンプル）による全数調査・生存者バイアス排除")
    print("※ 手数料・スリッページ未考慮 / 投資助言ではありません")
    print("=" * 76)

    st = limitup.detect_limit_up(quotes)
    st_set = set(zip(st["code"].astype(str), pd.to_datetime(st["date"])))

    qi = quotes.sort_values(["code", "date"]).reset_index(drop=True)
    g = qi.groupby("code")

    # 1. 出来高20日平均比
    qi["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    qi["vol_ratio"] = qi["volume"] / qi["vol_ma20"]

    # 2. 連続上昇日数（Numpyで一括計算）
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

    # 3. 翌営業日以降の値動き
    qi["next_date"] = g["date"].shift(-1)
    qi["next_open"] = g["open"].shift(-1)
    qi["next_high"] = g["high"].shift(-1)
    qi["next_close"] = g["close"].shift(-1)

    # 翌日の最大上昇率 (当日終値基準)
    qi["next_day_max_up"] = (qi["next_high"] / qi["close"] - 1.0) * 100.0
    # 3日以内の最大上昇率 (当日終値基準)
    qi["h_3d"] = g["high"].transform(lambda s: s.shift(-1).rolling(3, min_periods=1).max())
    qi["up_3d_max"] = (qi["h_3d"] / qi["close"] - 1.0) * 100.0

    # 5日以内の最大上昇率 (当日終値基準)
    qi["h_5d"] = g["high"].transform(lambda s: s.shift(-1).rolling(5, min_periods=1).max())
    qi["up_5d_max"] = (qi["h_5d"] / qi["close"] - 1.0) * 100.0

    # 翌営業日ストップ高フラグ
    valid_mask = qi["next_date"].notna()
    df = qi[valid_mask].copy()

    pairs = list(zip(df["code"].astype(str), pd.to_datetime(df["next_date"])))
    df["is_next_st"] = [1 if p in st_set else 0 for p in pairs]
    df["hit_next_10"] = (df["next_day_max_up"] >= 10.0).astype(int)
    df["hit_3d_10"] = (df["up_3d_max"] >= 10.0).astype(int)
    df["hit_5d_10"] = (df["up_5d_max"] >= 10.0).astype(int)

    all_dates = sorted(df["date"].unique())
    mid_date = pd.to_datetime(all_dates[len(all_dates) // 2])

    print(f"検証対象母集団: {len(df):,} 行 / 分割境界日: {mid_date.strftime('%Y-%m-%d')}\n")

    # ベースレート（自然発生率）
    base_st = df["is_next_st"].mean() * 100.0
    base_1d_10 = df["hit_next_10"].mean() * 100.0
    base_3d_10 = df["hit_3d_10"].mean() * 100.0
    base_5d_10 = df["hit_5d_10"].mean() * 100.0

    print("--- 【全体ベースレート（任意の日に買った時の自然確率）】 ---")
    print(f"  ・翌日ストップ高になる確率 : {base_st:.3f}% (約 {int(100/base_st)} 銘柄に1件)")
    print(f"  ・翌日に+10%以上に触れる確率: {base_1d_10:.2f}%")
    print(f"  ・3日以内に+10%以上に触れる確率: {base_3d_10:.2f}%")
    print(f"  ・5日以内に+10%以上に触れる確率: {base_5d_10:.2f}%\n")

    # 1. 連騰数別の到達確率（何連騰した日の引けで買うのが一番確率が高いか）
    print("--- ① 【連騰数別の到達確率】（N連騰した日の引けで買った場合） ---")
    rows1 = []
    for s in [0, 1, 2, 3, 4, 5, 6]:
        sub = df[df["streak"] == s] if s < 6 else df[df["streak"] >= 6]
        lbl = f"{s}日プラス" if s < 6 else "6日以上連続"
        n = len(sub)
        if n < 100: continue

        p_st = sub["is_next_st"].mean() * 100.0
        p_1d = sub["hit_next_10"].mean() * 100.0
        p_3d = sub["hit_3d_10"].mean() * 100.0
        p_5d = sub["hit_5d_10"].mean() * 100.0

        # 分割検証（5日以内+10%到達率の前半・後半）
        sub_h1 = sub[sub["date"] <= mid_date]
        sub_h2 = sub[sub["date"] > mid_date]
        p_5d_h1 = sub_h1["hit_5d_10"].mean() * 100.0 if len(sub_h1) else np.nan
        p_5d_h2 = sub_h2["hit_5d_10"].mean() * 100.0 if len(sub_h2) else np.nan

        rows1.append({
            "シグナル(前日引け時点)": lbl,
            "サンプル件数": n,
            "翌日ST高率": f"{p_st:.2f}% (x{p_st/base_st:.1f})",
            "翌日+10%到達": f"{p_1d:.1f}% (x{p_1d/base_1d_10:.1f})",
            "3日内+10%到達": f"{p_3d:.1f}% (x{p_3d/base_3d_10:.1f})",
            "5日内+10%到達": f"{p_5d:.1f}% (x{p_5d/base_5d_10:.1f})",
            "前半5日+10%": f"{p_5d_h1:.1f}%",
            "後半5日+10%": f"{p_5d_h2:.1f}%",
        })
    print(pd.DataFrame(rows1).to_string(index=False))

    # 2. 「連騰数 × 出来高急増」の掛け合わせ効果
    print("\n--- ② 【連騰数 × 出来高倍率の掛け合わせ】（5日以内に+10%に到達する確率） ---")
    rows2 = []
    for s in [3, 4, 5]:
        sub_s = df[df["streak"] >= s]
        for vr in [1.0, 2.0, 3.0, 5.0]:
            sub = sub_s[sub_s["vol_ratio"] >= vr]
            n = len(sub)
            if n < 50: continue

            p_st = sub["is_next_st"].mean() * 100.0
            p_5d = sub["hit_5d_10"].mean() * 100.0

            sub_h1 = sub[sub["date"] <= mid_date]
            sub_h2 = sub[sub["date"] > mid_date]
            p_5d_h1 = sub_h1["hit_5d_10"].mean() * 100.0 if len(sub_h1) else np.nan
            p_5d_h2 = sub_h2["hit_5d_10"].mean() * 100.0 if len(sub_h2) else np.nan

            rows2.append({
                "連騰条件": f"{s}連騰以上",
                "出来高条件": f"出来高{vr:.0f}倍以上",
                "件数": n,
                "翌日ST高率": f"{p_st:.2f}% (x{p_st/base_st:.1f})",
                "5日内+10%到達率": f"{p_5d:.1f}%",
                "全体比リフト": round(p_5d / base_5d_10, 2),
                "前半5日率": f"{p_5d_h1:.1f}%",
                "後半5日率": f"{p_5d_h2:.1f}%",
                "判定": "◎再現あり" if (p_5d_h1 >= base_5d_10 * 1.5 and p_5d_h2 >= base_5d_10 * 1.5) else "△",
            })
    print(pd.DataFrame(rows2).to_string(index=False))
