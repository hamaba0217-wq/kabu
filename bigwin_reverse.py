# -*- coding: utf-8 -*-
"""
bigwin_reverse.py - 短期大勝ち銘柄（10営業日以内に+15%以上急騰）の急騰前共通点 逆算分析

検証の作法:
- 10営業日以内に+15%以上の高値に到達したイベント（大勝ち群）を全数抽出
- 全体母集団（ベースレート）と対比し、急騰前の特徴のリフト値（倍率）を算出（生存者バイアス排除）
- 7大観点（位置、出来高変化、ローソク足、業績、時価総額、業種、信用需給）を網羅検証
- 前半・後半の2分割検証で再現性を確認
- 手数料・スリッページ未考慮（明記）
- 投資助言ではない（明記）
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import refine
from badnews import _latest_fin_before, _prep_fin_extended

WIN_HORIZON = 10      # 保有上限（営業日）
WIN_THRESHOLD = 15.0  # +15%以上の高値到達を「大勝ち」と定義
STEP_DAYS = 3         # サンプリング間隔（高速化と十分な母数の両立）


def run_analysis(quotes: pd.DataFrame, fin=None, listed=None, margin=None):
    print("=" * 76)
    print(f"短期急騰株の『急騰前共通点』逆算分析（{WIN_HORIZON}営業日以内に+{WIN_THRESHOLD:.0f}%以上到達）")
    print("※ 生存者バイアス完全排除：全銘柄・全営業日のベースレートと比較")
    print("※ 手数料・スリッページ未考慮 / 投資助言ではありません")
    print("=" * 76)

    # 銘柄属性
    sec_map, name_map, mkt_map = {}, {}, {}
    if listed is not None:
        if "sector33" in listed.columns:
            sec_map = dict(zip(listed["code"].astype(str), listed["sector33"]))
        if "company_name" in listed.columns:
            name_map = dict(zip(listed["code"].astype(str), listed["company_name"]))
        if "market" in listed.columns:
            mkt_map = dict(zip(listed["code"].astype(str), listed["market"]))

    # 決算前年比
    fin_ext = _prep_fin_extended(fin) if fin is not None else pd.DataFrame()
    if not fin_ext.empty and "operating_profit" in fin_ext.columns:
        fin_ext = fin_ext.sort_values(["code", "disclosed_date"])
        fin_ext["op_prev"] = fin_ext.groupby(["code", "period_type"])["operating_profit"].shift(1)

    # 株式数（時価総額用）
    shares_map = {}
    if fin is not None and "shares_outstanding" in fin.columns:
        f_sorted = fin.sort_values("disclosed_date").dropna(subset=["shares_outstanding"])
        shares_map = dict(f_sorted.groupby("code")["shares_outstanding"].last())

    # 信用残
    margin_by_code = {}
    if margin is not None and len(margin):
        mm = margin.copy()
        mm["date"] = pd.to_datetime(mm["date"])
        for c in ("long_margin", "short_margin"):
            if c in mm.columns:
                mm[c] = pd.to_numeric(mm[c], errors="coerce")
        margin_by_code = {str(c): g.sort_values("date") for c, g in mm.groupby("code")}

    # 指標算出
    qi = quotes.sort_values(["code", "date"]).reset_index(drop=True)
    g = qi.groupby("code")

    qi["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    qi["vol_ratio"] = qi["volume"] / qi["vol_ma20"]

    qi["high_52w"] = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / qi["high_52w"] - 1.0

    qi["day_range"] = (qi["high"] - qi["low"]) / qi["open"] * 100.0
    qi["is_yang"] = (qi["close"] > qi["open"]).astype(int)
    # 直近3日の陽線数
    qi["yang_3d"] = g["is_yang"].transform(lambda s: s.rolling(3, min_periods=3).sum())

    # 未来10営業日の最高値
    qi["fut_high_10d"] = g["high"].transform(lambda s: s.shift(-1).rolling(WIN_HORIZON, min_periods=WIN_HORIZON).max())
    qi["max_up_10d"] = (qi["fut_high_10d"] / qi["close"] - 1.0) * 100.0

    # 有効データ（未来10日分が存在する行）
    df_valid = qi[qi["max_up_10d"].notna()].copy()
    all_dates = sorted(df_valid["date"].unique())
    mid_date = pd.to_datetime(all_dates[len(all_dates) // 2])

    # サンプリング走査（STEP_DAYS刻みで全銘柄評価）
    sampled_dates = set(all_dates[::STEP_DAYS])
    df_s = df_valid[df_valid["date"].isin(sampled_dates)].copy()

    # 大勝ちフラグ (+15%以上到達)
    df_s["is_bigwin"] = (df_s["max_up_10d"] >= WIN_THRESHOLD).astype(int)

    base_rate = df_s["is_bigwin"].mean() * 100.0
    n_total = len(df_s)
    n_wins = int(df_s["is_bigwin"].sum())

    print(f"検証母集団: {n_total:,} サンプル（全上場銘柄 × 各起点）")
    print(f"大勝ち（10日以内+{WIN_THRESHOLD:.0f}%到達）発生数: {n_wins:,} 件")
    print(f"自然発生基準確率（ベースレート）: {base_rate:.2f}%")
    print(f"分割境界日: {mid_date.strftime('%Y-%m-%d')}\n")

    # 区分化
    # 1. 位置
    df_s["cat_pos"] = pd.cut(
        df_s["pct_from_high"],
        bins=[-np.inf, -0.30, -0.15, -0.05, np.inf],
        labels=["底値圏(高値-30%↓)", "中間圏(高値-30〜-15%)", "押し目圏(高値-15〜-5%)", "高値圏(高値-5%↑)"]
    ).astype(str).fillna("不明")

    # 2. 出来高
    df_s["cat_vol"] = pd.cut(
        df_s["vol_ratio"],
        bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
        labels=["出来高枯れ(<0.5倍)", "平時(0.5〜1倍)", "微増(1〜2倍)", "急増(2倍以上)"]
    ).astype(str).fillna("不明")

    # 3. 日中レンジ（前日ボラティリティ）
    df_s["cat_range"] = pd.cut(
        df_s["day_range"],
        bins=[-np.inf, 2.5, 5.0, np.inf],
        labels=["煮詰まり(レンジ<2.5%)", "通常(2.5〜5%)", "拡大(>5%)"]
    ).astype(str).fillna("不明")

    # 4. 直近3日の陽線数
    df_s["cat_yang"] = df_s["yang_3d"].map({
        0: "0回(直近3日連続陰線等)",
        1: "1回陽線",
        2: "2回陽線",
        3: "3回連続陽線"
    }).fillna("不明")

    # 5. 時価総額
    codes_str = df_s["code"].astype(str)
    mcaps = [df_s.iloc[i]["close"] * shares_map.get(codes_str.iloc[i], 0) / 1e8 for i in range(len(df_s))]
    df_s["mcap_oku"] = mcaps
    df_s["cat_mcap"] = pd.cut(
        df_s["mcap_oku"],
        bins=[-np.inf, 50, 150, 500, np.inf],
        labels=["超小型(<50億)", "小型(50〜150億)", "中型(150〜500億)", "大型(>500億)"]
    ).astype(str).fillna("不明")

    # 6. 業種
    secs = codes_str.map(sec_map).fillna("その他")
    df_s["sec"] = secs
    def categorize_sec(s):
        if s in refine.HIGH_WIN_SECTORS:
            return "★高勝率業種(金融/倉庫/水産等)"
        if s in ["電気機器", "精密機器", "機械", "非鉄金属", "化学"]:
            return "製造・素材・ハイテク"
        if s in ["情報・通信業", "サービス業"]:
            return "IT・サービス・グロース"
        return "その他業種"
    df_s["cat_sec"] = df_s["sec"].map(categorize_sec)

    # 7. 信用倍率
    LAG = 7
    def get_mr_cat(code, as_of):
        if code not in margin_by_code:
            return "不明"
        cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=LAG)
        g_m = margin_by_code[code]
        past = g_m[g_m["date"] <= cutoff]
        if past.empty: return "不明"
        r = past.iloc[-1]
        lm, sm = r.get("long_margin"), r.get("short_margin")
        if lm and sm and sm > 0:
            ratio = lm / sm
            if ratio < 1.0: return "売り長(倍率<1)"
            if ratio <= 3.0: return "中立(1〜3倍)"
            return "買い長(>3倍)"
        return "不明"

    # サンプル行のみ信用残紐付け
    dates_list = df_s["date"].values
    codes_list = df_s["code"].astype(str).values
    df_s["cat_mr"] = [get_mr_cat(c, d) for c, d in zip(codes_list, dates_list)]

    def _eval_category(col_name: str, title: str):
        print(f"--- 【{title}】 ---")
        rows = []
        for val, g_sub in df_s.groupby(col_name):
            n = len(g_sub)
            if n < 100: continue
            hits = int(g_sub["is_bigwin"].sum())
            prob = hits / n * 100.0
            lift = prob / base_rate if base_rate > 0 else 0.0

            g1 = g_sub[g_sub["date"] <= mid_date]
            g2 = g_sub[g_sub["date"] > mid_date]
            b1 = (df_s[df_s["date"] <= mid_date]["is_bigwin"].mean() * 100.0)
            b2 = (df_s[df_s["date"] > mid_date]["is_bigwin"].mean() * 100.0)

            p1 = (g1["is_bigwin"].mean() * 100.0) if len(g1) else 0.0
            p2 = (g2["is_bigwin"].mean() * 100.0) if len(g2) else 0.0
            l1 = p1 / b1 if b1 > 0 else 0.0
            l2 = p2 / b2 if b2 > 0 else 0.0

            rows.append({
                "急騰前特徴": val,
                "母数": n,
                "急騰到達数": hits,
                "大勝ち確率": f"{prob:.2f}%",
                "全体比リフト": round(lift, 2),
                "前半リフト": round(l1, 2),
                "後半リフト": round(l2, 2),
                "判定": "◎再現あり" if (l1 >= 1.25 and l2 >= 1.25) else ("△片方" if (l1 >= 1.25 or l2 >= 1.25) else "—"),
            })
        res = pd.DataFrame(rows).sort_values("全体比リフト", ascending=False)
        print(res.to_string(index=False) + "\n")
        return res

    _eval_category("cat_pos", "1. 52週高値からの位置（チャート水準）")
    _eval_category("cat_vol", "2. 直前の出来高（出来高枯れ vs 急増）")
    _eval_category("cat_range", "3. 直前の日中レンジ（煮詰まり vs 拡大）")
    _eval_category("cat_yang", "4. 直近3日の陽線数（モメンタム）")
    _eval_category("cat_mcap", "5. 時価総額帯（企業の規模）")
    _eval_category("cat_sec", "6. 業種グループ")
    _eval_category("cat_mr", "7. 信用需給（信用倍率）")

    # 複合ベストパターンの探索
    print("--- 【8. 複合ベスト条件（上位特徴の掛け合わせ）】 ---")
    df_s["combo_best"] = df_s["cat_mcap"] + " × " + df_s["cat_pos"] + " × " + df_s["cat_vol"]
    rows_c = []
    for val, g_sub in df_s.groupby("combo_best"):
        n = len(g_sub)
        if n < 100: continue
        hits = int(g_sub["is_bigwin"].sum())
        prob = hits / n * 100.0
        lift = prob / base_rate if base_rate > 0 else 0.0

        g1 = g_sub[g_sub["date"] <= mid_date]
        g2 = g_sub[g_sub["date"] > mid_date]
        b1 = (df_s[df_s["date"] <= mid_date]["is_bigwin"].mean() * 100.0)
        b2 = (df_s[df_s["date"] > mid_date]["is_bigwin"].mean() * 100.0)

        p1 = (g1["is_bigwin"].mean() * 100.0) if len(g1) else 0.0
        p2 = (g2["is_bigwin"].mean() * 100.0) if len(g2) else 0.0
        l1 = p1 / b1 if b1 > 0 else 0.0
        l2 = p2 / b2 if b2 > 0 else 0.0

        rows_c.append({
            "複合条件": val,
            "母数": n,
            "急騰到達数": hits,
            "大勝ち確率": f"{prob:.2f}%",
            "全体比リフト": round(lift, 2),
            "前半リフト": round(l1, 2),
            "後半リフト": round(l2, 2),
            "判定": "◎再現あり" if (l1 >= 1.5 and l2 >= 1.5) else ("△片方" if (l1 >= 1.5 or l2 >= 1.5) else "—"),
        })
    res_c = pd.DataFrame(rows_c).sort_values("全体比リフト", ascending=False)
    print(res_c.head(10).to_string(index=False))
