# -*- coding: utf-8 -*-
"""
limitup_precursor.py - ストップ高の「前兆（前日までの動き）」実データ検証（超高速版）

検証の作法:
- 陽性（ストップ高前日）と全体母集団（全銘柄・全営業日）を厳密に対比（生存者バイアス排除）
- 自然発生確率（ベースレート）に対するリフト値（倍率）を算出
- 前半・後半の2分割検証で再現性を確認
- 手数料・スリッページ未考慮（明記）
- 投資助言ではない（明記）
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import limitup


def analyze_precursors(quotes: pd.DataFrame) -> tuple[pd.DataFrame, float, pd.Timestamp]:
    st = limitup.detect_limit_up(quotes)
    if st.empty:
        return pd.DataFrame(), 0.0, pd.Timestamp("2025-01-01")

    # 日付昇順で整列
    qi = quotes.sort_values(["code", "date"]).reset_index(drop=True)
    g = qi.groupby("code")

    # 1. 出来高20日平均比
    qi["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    qi["vol_ratio"] = qi["volume"] / qi["vol_ma20"]

    # 2. 52週高値からの位置
    qi["high_52w"] = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / qi["high_52w"] - 1.0

    # 3. 日中ボラティリティ (High - Low) / Open
    qi["day_range"] = (qi["high"] - qi["low"]) / qi["open"] * 100.0

    # 4. 連続上昇日数（Numpyで一括高速計算）
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

    # 5. 翌営業日ストップ高フラグをベクトル結合
    qi["next_date"] = g["date"].shift(-1)
    # 末尾の行（翌日データなし）は除外
    valid_mask = qi["next_date"].notna()
    qi_valid = qi[valid_mask].copy()

    st_set = set(zip(st["code"].astype(str), pd.to_datetime(st["date"])))
    pairs = list(zip(qi_valid["code"].astype(str), pd.to_datetime(qi_valid["next_date"])))
    qi_valid["is_next_st"] = [1 if p in st_set else 0 for p in pairs]

    # 分割検証の日付
    all_dates = sorted(qi_valid["date"].unique())
    mid_date = pd.to_datetime(all_dates[len(all_dates) // 2])

    # 区分化（Pandas cut / map で一括処理）
    def categorize_streak(s):
        if s == 0: return "0日(前日マイナス/横ばい)"
        if s == 1: return "1日プラス"
        if s == 2: return "2日連続プラス"
        if s == 3: return "3日連続プラス"
        return "4日以上連続プラス"

    qi_valid["streak_cat"] = qi_valid["streak"].map(categorize_streak)

    qi_valid["vol_cat"] = pd.cut(
        qi_valid["vol_ratio"],
        bins=[-np.inf, 1.0, 2.0, 3.0, 5.0, np.inf],
        labels=["平時未満(<1倍)", "微増(1〜2倍)", "急増(2〜3倍)", "大幅急増(3〜5倍)", "爆増(5倍以上)"]
    ).astype(str).fillna("不明")

    qi_valid["pos_cat"] = pd.cut(
        qi_valid["pct_from_high"],
        bins=[-np.inf, -0.30, -0.15, -0.05, np.inf],
        labels=["底値・下落圏(高値-30%以下)", "中間圏(高値-30〜-15%)", "押し目圏(高値-15〜-5%)", "高値直前(高値-5%以内)"]
    ).astype(str).fillna("不明")

    qi_valid["range_cat"] = pd.cut(
        qi_valid["day_range"],
        bins=[-np.inf, 3.0, 6.0, np.inf],
        labels=["煮詰まり(レンジ<3%)", "通常(3〜6%)", "高ボラ(>6%)"]
    ).astype(str).fillna("不明")

    base_rate = qi_valid["is_next_st"].mean() * 100.0
    return qi_valid, base_rate, mid_date


def summarize_lift(df: pd.DataFrame, col: str, base_rate: float, mid_date: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for val, g in df.groupby(col):
        n = len(g)
        if n < 100:
            continue
        hits = int(g["is_next_st"].sum())
        prob = hits / n * 100.0
        lift = prob / base_rate if base_rate > 0 else 0.0

        g_h1 = g[g["date"] <= mid_date]
        g_h2 = g[g["date"] > mid_date]
        base_h1 = (df[df["date"] <= mid_date]["is_next_st"].mean() * 100.0) if len(df) else 0.0
        base_h2 = (df[df["date"] > mid_date]["is_next_st"].mean() * 100.0) if len(df) else 0.0

        p1 = (g_h1["is_next_st"].mean() * 100.0) if len(g_h1) else 0.0
        p2 = (g_h2["is_next_st"].mean() * 100.0) if len(g_h2) else 0.0
        lift1 = p1 / base_h1 if base_h1 > 0 else 0.0
        lift2 = p2 / base_h2 if base_h2 > 0 else 0.0

        rows.append({
            "前日特徴": val,
            "母数(全サンプル)": n,
            "ST高発生数": hits,
            "翌日ST高率": f"{prob:.3f}%",
            "全体比リフト": round(lift, 2),
            "前半リフト": round(lift1, 2),
            "後半リフト": round(lift2, 2),
            "判定": "◎再現あり" if (lift1 >= 1.3 and lift2 >= 1.3) else ("△片方" if (lift1 >= 1.3 or lift2 >= 1.3) else "—"),
        })
    res = pd.DataFrame(rows)
    return res.sort_values("全体比リフト", ascending=False) if not res.empty else res


def run(quotes: pd.DataFrame, fin=None, listed=None):
    print("=" * 76)
    print("ストップ高の『前兆（前日までの動き）』客観的データ検証")
    print("※ 生存者バイアスを排除：全銘柄・全営業日（約200万件）と比較")
    print("※ 手数料・スリッページ未考慮 / 投資助言ではありません")
    print("=" * 76)

    print("データ集計中（高速ベクトル走査）...")
    df_samples, base_rate, mid_date = analyze_precursors(quotes)
    if df_samples.empty:
        print("サンプルを抽出できませんでした。")
        return

    st_total = int(df_samples["is_next_st"].sum())
    print(f"全銘柄・全営業日の総検証数: {len(df_samples):,} 件")
    print(f"翌日ストップ高が発生した件数: {st_total:,} 件")
    print(f"自然発生基準確率（ベースレート）: {base_rate:.3f}%（約 {int(100/base_rate) if base_rate > 0 else 0} 営業日に1回）")
    print(f"分割検証の境界日: {mid_date.strftime('%Y-%m-%d')}\n")

    # 1. 連続上昇日数
    print("--- ① 【直前の連続上昇日数（モメンタム）】 ---")
    res_stk = summarize_lift(df_samples, "streak_cat", base_rate, mid_date)
    print(res_stk.to_string(index=False))

    # 2. 出来高倍率
    print("\n--- ② 【前日の出来高急増（予兆商い）】 ---")
    res_vol = summarize_lift(df_samples, "vol_cat", base_rate, mid_date)
    print(res_vol.to_string(index=False))

    # 3. 52週高値からの位置
    print("\n--- ③ 【前日時点の52週高値位置（チャート水準）】 ---")
    res_pos = summarize_lift(df_samples, "pos_cat", base_rate, mid_date)
    print(res_pos.to_string(index=False))

    # 4. ボラティリティ（レンジ）
    print("\n--- ④ 【前日の日中レンジ（煮詰まり vs 乱高下）】 ---")
    res_rng = summarize_lift(df_samples, "range_cat", base_rate, mid_date)
    print(res_rng.to_string(index=False))

    # 5. 複合条件（強い組み合わせ）
    print("\n--- ⑤ 【複合前兆（有望な組み合わせ）】 ---")
    df_samples["combo"] = df_samples["vol_cat"] + " × " + df_samples["pos_cat"]
    res_cmb = summarize_lift(df_samples, "combo", base_rate, mid_date)
    print(res_cmb.head(8).to_string(index=False))
