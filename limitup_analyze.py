# -*- coding: utf-8 -*-
"""
ストップ高後の値動きを分析する。
  py main.py limitup-analyze   （main.pyに登録）

分析内容
--------
1. 全銘柄・全期間でストップ高を検出（limitup.detect_limit_up）
2. 各ストップ高の「その後」を9パターンに分類（起点＝ストップ高の終値）
3. パターンの頻度を表に
4. タイプ別（決算・業種・仕手的特徴・説明不能）に分布が違うか
5. 参入戦略の検証：
   ・3日観察→4日目参入 と 5日観察→6日目参入 の2通り
   ・観察期間の動きで「勝てるパターン」を見分け、その後の勝率を測る

9パターン（起点＝ストップ高当日の終値、その後7営業日で判定）
  爆騰       : +50%以上（1.5倍）
  続伸       : +20〜+50%
  小幅上昇   : +5〜+20%
  連続ST高   : 翌日もストップ高
  急騰頻発   : 日中の振れ(安値→高値)が+10%以上の日が2日以上（勢い継続）
  もみ合い   : ±5%以内
  すぐ戻る   : ストップ高の上昇分の大半を失う
  緩やか下落 : -5〜-20%
  急落       : -20%以上
（判定は上から優先。複数該当は上位を採用）

事実にもとづく注意
  ・手数料・スリッページ未考慮。終値/日中値ベースの理論値。
  ・「勝てるパターン」は過去の傾向。将来を保証しない。分割検証で確認する。
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import limitup

FORWARD = 7          # ストップ高後、何営業日追うか
SURGE_INTRADAY = 10.0   # 「急騰」とみなす日中振れ幅(%)
SURGE_DAYS_MIN = 2      # 急騰頻発とみなす日数


def _future_window(cqi, st_idx, n):
    """ストップ高日(st_idx)の翌日からn営業日のOHLCを返す。"""
    return cqi.iloc[st_idx + 1: st_idx + 1 + n]


def classify(cqi, st_idx, st_close):
    """ストップ高後7営業日を9パターンに分類する。"""
    fut = _future_window(cqi, st_idx, FORWARD)
    if len(fut) == 0:
        return "データ不足"
    highs = fut["high"].values
    lows = fut["low"].values
    closes = fut["close"].values
    # 起点(ストップ高終値)からの、期間中の最大上昇・最終騰落
    max_up = (highs.max() / st_close - 1) * 100
    min_dn = (lows.min() / st_close - 1) * 100
    last = (closes[-1] / st_close - 1) * 100
    # 日中振れ幅が+10%以上の日数（安値→高値）
    with np.errstate(divide="ignore", invalid="ignore"):
        intraday = (highs / lows - 1) * 100
    surge_days = int((intraday >= SURGE_INTRADAY).sum())
    # 翌日もストップ高か（翌日終値が前日+制限値幅）
    renzoku = False
    if len(fut) >= 1:
        nxt_close = fut["close"].values[0]
        w = limitup.price_limit(st_close)
        if w == w and nxt_close - st_close >= w * 0.995:
            renzoku = True

    # 上から優先で判定
    if max_up >= 50:
        return "爆騰(+50%↑)"
    if renzoku:
        return "連続ST高"
    if surge_days >= SURGE_DAYS_MIN:
        return "急騰頻発"
    if max_up >= 20:
        return "続伸(+20〜50%)"
    if min_dn <= -20:
        return "急落(-20%↓)"
    if max_up >= 5 and last >= 5:
        return "小幅上昇(+5〜20%)"
    if min_dn <= -20 or last <= -20:
        return "急落(-20%↓)"
    if last <= -5:
        return "緩やか下落(-5〜-20%)"
    if -5 < last < 5:
        return "もみ合い(±5%)"
    # ストップ高の上昇分をすぐ失った（起点近辺〜下）
    return "すぐ戻る"


PATTERN_ORDER = ["爆騰(+50%↑)", "連続ST高", "急騰頻発", "続伸(+20〜50%)",
                 "小幅上昇(+5〜20%)", "もみ合い(±5%)", "すぐ戻る",
                 "緩やか下落(-5〜-20%)", "急落(-20%↓)", "データ不足"]


def analyze(quotes, fin=None, listed=None):
    """ストップ高を検出して分類し、頻度表を返す。"""
    print("ストップ高を検出中…", flush=True)
    st = limitup.detect_limit_up(quotes)
    print(f"  ストップ高（終値ベース）: {len(st):,}件 検出", flush=True)
    if st.empty:
        return None

    q = quotes.sort_values(["code", "date"])
    qi_by_code = {c: g.reset_index(drop=True) for c, g in q.groupby("code")}
    # 各銘柄の日付→行番号
    idx_by_code = {c: {pd.Timestamp(d): i for i, d in enumerate(g["date"])}
                   for c, g in qi_by_code.items()}

    results = []
    for _, r in st.iterrows():
        code = r["code"]
        cqi = qi_by_code.get(code)
        if cqi is None:
            continue
        st_idx = idx_by_code[code].get(pd.Timestamp(r["date"]))
        if st_idx is None or st_idx + 1 >= len(cqi):
            continue
        pat = classify(cqi, st_idx, r["close"])
        results.append({"code": code, "date": r["date"],
                        "gain_pct": r["gain_pct"], "pattern": pat})
    rdf = pd.DataFrame(results)
    return rdf, st


def print_summary(rdf):
    """パターンの頻度を表で出力。"""
    n = len(rdf)
    print("\n" + "=" * 60)
    print(f"ストップ高 後の値動きパターン（全{n:,}件、起点=ST高終値、7営業日）")
    print("=" * 60)
    counts = rdf["pattern"].value_counts()
    print(f"{'パターン':<20}{'件数':>8}{'割合':>8}")
    print("-" * 40)
    for pat in PATTERN_ORDER:
        c = int(counts.get(pat, 0))
        if c == 0:
            continue
        print(f"{pat:<20}{c:>8,}{c/n*100:>7.1f}%")
    print("-" * 40)
    print("※手数料・スリッページ未考慮。過去の傾向で将来を保証しません。")


def classify_reason(quotes, st, fin=None, listed=None):
    """各ストップ高を、原因タイプで分類する（データで測れる範囲）。
      決算タイプ    : ST高の直近5営業日以内に決算発表があった
      業種タイプ    : 同業種の他銘柄も同時期に上昇していた（未実装時はスキップ）
      仕手的特徴    : 出来高が普段の何倍か・売買代金が小さいか・突発性
      説明不能      : 上記に当てはまらない（見えないニュースの可能性）
    ※「仕手」と断定はしない。特徴の程度で評価する。"""
    q = quotes.sort_values(["code", "date"])
    qi_by_code = {c: g.reset_index(drop=True) for c, g in q.groupby("code")}
    idx_by_code = {c: {pd.Timestamp(d): i for i, d in enumerate(g["date"])}
                   for c, g in qi_by_code.items()}

    # 決算発表日（code -> 日付のset）
    fin_dates = {}
    if fin is not None and len(fin) and "disclosed_date" in fin.columns:
        for c, g in fin.groupby("code"):
            fin_dates[str(c)] = set(pd.to_datetime(g["disclosed_date"]).dt.normalize())

    rows = []
    for _, r in st.iterrows():
        code = str(r["code"])
        cqi = qi_by_code.get(code)
        if cqi is None:
            continue
        st_idx = idx_by_code[code].get(pd.Timestamp(r["date"]))
        if st_idx is None:
            continue

        # 決算タイプ：直近5営業日以内に決算発表
        is_earnings = False
        if code in fin_dates:
            recent_dates = set(pd.to_datetime(
                cqi["date"].iloc[max(0, st_idx - 5): st_idx + 1]).dt.normalize())
            if fin_dates[code] & recent_dates:
                is_earnings = True

        # 仕手的特徴：出来高が普段（過去20日平均）の何倍か
        vol_ratio = None
        if "volume" in cqi.columns and st_idx >= 20:
            base_vol = cqi["volume"].iloc[st_idx - 20: st_idx].mean()
            cur_vol = cqi["volume"].iloc[st_idx]
            if base_vol and base_vol > 0:
                vol_ratio = cur_vol / base_vol
        # 売買代金（小型か）
        turnover = None
        if "turnover_value" in cqi.columns:
            turnover = cqi["turnover_value"].iloc[st_idx]
        elif "close" in cqi.columns and "volume" in cqi.columns:
            turnover = cqi["close"].iloc[st_idx] * cqi["volume"].iloc[st_idx]

        # 仕手的特徴スコア（0〜3）：出来高急増・小型・低位
        shite_score = 0
        if vol_ratio is not None and vol_ratio >= 10:
            shite_score += 1   # 出来高10倍以上
        if turnover is not None and turnover < 500_000_000:
            shite_score += 1   # 売買代金5億未満（小型・薄い）
        if r["close"] < 1000:
            shite_score += 1   # 低位株（仕手が好む）

        # タイプ判定（優先順）
        if is_earnings:
            rtype = "決算"
        elif shite_score >= 2:
            rtype = "仕手的特徴"
        else:
            rtype = "説明不能"   # 決算でなく仕手特徴も弱い＝見えないニュースの可能性

        rows.append({
            "code": code, "date": r["date"], "reason_type": rtype,
            "出来高倍率": round(vol_ratio, 1) if vol_ratio else None,
            "仕手スコア": shite_score,
        })
    return pd.DataFrame(rows)


def print_reason_summary(rdf, reason_df):
    """タイプ別の頻度と、タイプ×パターンのクロス集計。"""
    merged = rdf.merge(reason_df[["code", "date", "reason_type"]],
                       on=["code", "date"], how="left")
    print("\n" + "=" * 60)
    print("ストップ高の原因タイプ別 分布")
    print("=" * 60)
    n = len(merged)
    tc = merged["reason_type"].value_counts()
    for t in ["決算", "仕手的特徴", "説明不能"]:
        c = int(tc.get(t, 0))
        print(f"  {t:<10}: {c:>6,}件 ({c/n*100:>4.1f}%)")
    print("\n※「仕手的特徴」＝出来高急増・小型・低位のうち2つ以上。仕手と断定はしない。")
    print("※「説明不能」＝決算でなく仕手特徴も弱い。見えないニュースの可能性（特定不可）。")

    # タイプ×パターンのクロス（各タイプで、その後どのパターンが多いか）
    print("\n" + "=" * 60)
    print("原因タイプ別・その後の値動きパターン（縦=タイプ、割合）")
    print("=" * 60)
    for t in ["決算", "仕手的特徴", "説明不能"]:
        sub = merged[merged["reason_type"] == t]
        if len(sub) == 0:
            continue
        print(f"\n■ {t}（{len(sub):,}件）")
        pc = sub["pattern"].value_counts()
        for pat in PATTERN_ORDER:
            c = int(pc.get(pat, 0))
            if c == 0:
                continue
            print(f"    {pat:<20}{c/len(sub)*100:>5.1f}%")


def entry_strategy(quotes, st, observe_days, hold_days=None):
    """観察→参入戦略の検証。
    observe_days: ストップ高後、何営業日観察するか（3 or 5）
    参入: 観察最終日の翌営業日の始値で買う。
    決済: hold_days営業日後の終値、または期間中の高値/安値も記録。

    観察期間の特徴で分類:
      観察上昇  : 観察期間で株価が上がった（終値ベース、ST高終値比 +5%以上）
      観察横ばい: ±5%以内
      観察下落  : -5%以下
    それぞれ、参入後のリターン（対相場は別途）を集計。"""
    if hold_days is None:
        hold_days = 7 - observe_days  # 残りの営業日で決済
    q = quotes.sort_values(["code", "date"])
    qi_by_code = {c: g.reset_index(drop=True) for c, g in q.groupby("code")}
    idx_by_code = {c: {pd.Timestamp(d): i for i, d in enumerate(g["date"])}
                   for c, g in qi_by_code.items()}

    rows = []
    for _, r in st.iterrows():
        code = r["code"]
        cqi = qi_by_code.get(code)
        if cqi is None:
            continue
        st_idx = idx_by_code[code].get(pd.Timestamp(r["date"]))
        if st_idx is None:
            continue
        st_close = r["close"]
        # 観察期間の終値（ST高翌日〜observe_days日）
        obs = cqi.iloc[st_idx + 1: st_idx + 1 + observe_days]
        if len(obs) < observe_days:
            continue
        obs_last_close = obs["close"].values[-1]
        obs_change = (obs_last_close / st_close - 1) * 100
        # 参入：観察最終日の翌営業日の始値
        entry_idx = st_idx + 1 + observe_days
        if entry_idx >= len(cqi):
            continue
        entry_price = cqi["open"].values[entry_idx]
        if entry_price <= 0 or entry_price != entry_price:
            continue
        # 決済期間（参入日〜hold_days）の高値・安値・最終終値
        hold = cqi.iloc[entry_idx: entry_idx + hold_days]
        if len(hold) == 0:
            continue
        hi = hold["high"].values.max()
        lo = hold["low"].values.min()
        last_c = hold["close"].values[-1]
        max_up = (hi / entry_price - 1) * 100      # 参入後の最大上昇（高値で売れれば）
        max_dn = (lo / entry_price - 1) * 100      # 参入後の最大下落（安値まで）
        final = (last_c / entry_price - 1) * 100   # 保有終了時のリターン
        # 観察期間の特徴で分類
        if obs_change >= 5:
            obs_type = "観察上昇"
        elif obs_change <= -5:
            obs_type = "観察下落"
        else:
            obs_type = "観察横ばい"
        rows.append({
            "code": code, "date": r["date"], "obs_type": obs_type,
            "obs_change": round(obs_change, 1),
            "参入後_最大上昇%": round(max_up, 1),
            "参入後_最大下落%": round(max_dn, 1),
            "参入後_最終%": round(final, 1),
        })
    return pd.DataFrame(rows)


def print_entry_summary(edf, observe_days):
    """参入戦略の集計を表で出力。"""
    if edf is None or edf.empty:
        print(f"\n【{observe_days}日観察→{observe_days+1}日目参入】該当なし")
        return
    n = len(edf)
    print("\n" + "=" * 64)
    print(f"【{observe_days}日観察 → {observe_days+1}日目の始値で参入】（全{n:,}件）")
    print("=" * 64)
    print(f"{'観察期の動き':<12}{'件数':>7}{'平均最大上昇':>11}{'平均最大下落':>11}"
          f"{'平均最終':>9}{'最終勝率':>9}")
    print("-" * 64)
    for obs_type in ["観察上昇", "観察横ばい", "観察下落"]:
        g = edf[edf["obs_type"] == obs_type]
        if len(g) == 0:
            continue
        c = len(g)
        avg_up = g["参入後_最大上昇%"].mean()
        avg_dn = g["参入後_最大下落%"].mean()
        avg_fin = g["参入後_最終%"].mean()
        win = (g["参入後_最終%"] > 0).mean() * 100
        print(f"{obs_type:<12}{c:>7,}{avg_up:>10.1f}%{avg_dn:>10.1f}%"
              f"{avg_fin:>8.1f}%{win:>8.0f}%")
    print("-" * 64)
    # 全体
    print(f"{'全体':<12}{n:>7,}{edf['参入後_最大上昇%'].mean():>10.1f}%"
          f"{edf['参入後_最大下落%'].mean():>10.1f}%"
          f"{edf['参入後_最終%'].mean():>8.1f}%"
          f"{(edf['参入後_最終%']>0).mean()*100:>8.0f}%")
    print("※平均最大上昇＝参入後、高値まで取れた場合の平均。平均最終＝保有終了時の平均。")
    print("※手数料・スリッページ未考慮。対相場超過は別途要検証。")


def _split_half(rdf, st):
    """検出したストップ高を、日付で前半・後半に分ける（分割検証用）。"""
    if rdf.empty:
        return None, None
    dates = pd.to_datetime(rdf["date"])
    mid = dates.min() + (dates.max() - dates.min()) / 2
    first = rdf[dates <= mid]
    second = rdf[dates > mid]
    return first, second


def supply_analysis(rdf, st, quotes, margin=None, short_ratio=None, listed=None):
    """ストップ高時点の信用売り残・空売り比率で、その後のパターンが違うか検証。
    仮説：売り残が多い（踏み上げ余地がある）ほど、その後も上がりやすいのでは？
    ※これまでの検証では信用・空売りは単独では効かなかった。ストップ高局面で
      効くかは未知。効かなければ「効かない」と分かるのが検証の価値。"""
    if margin is None or not len(margin):
        print("\n（信用残データがないため、信用の分析はスキップ）")
        return
    # ストップ高日ごとに、直近の信用売り残・信用倍率を引く
    mm = margin.sort_values(["code", "date"]).copy()
    marg_by_code = {c: g for c, g in mm.groupby("code")}

    def latest_before(g, d, col):
        sub = g[g["date"] <= d]
        if len(sub) == 0 or col not in sub.columns:
            return None
        v = sub[col].values[-1]
        return float(v) if v == v else None

    merged = rdf.merge(st[["code", "date", "close"]], on=["code", "date"], how="left")
    rows = []
    for _, r in merged.iterrows():
        g = marg_by_code.get(r["code"])
        if g is None:
            continue
        d = pd.Timestamp(r["date"])
        long_m = latest_before(g, d, "long_margin")
        short_m = latest_before(g, d, "short_margin")
        ratio = (long_m / short_m) if (long_m and short_m and short_m > 0) else None
        rows.append({"pattern": r["pattern"], "信用倍率": ratio,
                     "信用売り残": short_m})
    sdf = pd.DataFrame(rows).dropna(subset=["信用倍率"])
    if sdf.empty:
        print("\n（信用残を紐付けられたストップ高がなく、分析スキップ）")
        return

    print("\n" + "=" * 60)
    print("信用倍率別・ストップ高後のパターン（踏み上げ仮説の検証）")
    print("=" * 60)
    # 信用倍率で3分位（低い=売り長=踏み上げ余地大）
    sdf["倍率帯"] = pd.qcut(sdf["信用倍率"], 3,
                          labels=["低(売り長・踏み上げ余地)", "中", "高(買い長)"],
                          duplicates="drop")
    good = ["爆騰(+50%↑)", "連続ST高", "急騰頻発", "続伸(+20〜50%)"]
    for band in sdf["倍率帯"].cat.categories:
        sub = sdf[sdf["倍率帯"] == band]
        if len(sub) == 0:
            continue
        good_pct = sub["pattern"].isin(good).mean() * 100
        print(f"  {band:<22} n={len(sub):>5,}  "
              f"その後さらに上昇(爆騰/連続/頻発/続伸)= {good_pct:>4.1f}%")
    print("※踏み上げ仮説が正しければ、売り長(低倍率)ほど上昇率が高いはず。")
    print("※これまで信用は単独では効かなかった。ここで差が出るかは事実で判断。")


def run(quotes, fin=None, listed=None, margin=None, short_ratio=None):
    """ストップ高分析の全体を実行する。"""
    print("=" * 64)
    print("ストップ高 後の値動き分析（直近2年・全銘柄）")
    print("=" * 64)

    out = analyze(quotes, fin, listed)
    if out is None:
        print("ストップ高が検出されませんでした。")
        return
    rdf, st = out

    # 1. パターンの頻度
    print_summary(rdf)

    # 2. 分割検証（前半・後半で分布が再現するか）
    first, second = _split_half(rdf, st)
    if first is not None and len(first) and len(second):
        print("\n" + "=" * 60)
        print("【分割検証】前半・後半で、パターン分布は再現するか")
        print("=" * 60)
        for label, part in [("前半", first), ("後半", second)]:
            print(f"\n■ {label}（{len(part):,}件）")
            pc = part["pattern"].value_counts()
            for pat in PATTERN_ORDER[:5]:  # 主要5パターン
                c = int(pc.get(pat, 0))
                print(f"    {pat:<20}{c/len(part)*100:>5.1f}%")

    # 3. 原因タイプ別
    reason_df = classify_reason(quotes, st, fin, listed)
    if not reason_df.empty:
        print_reason_summary(rdf, reason_df)

    # 4. 信用・空売りの分析（踏み上げ仮説）
    supply_analysis(rdf, st, quotes, margin, short_ratio, listed)

    # 5. 参入戦略（3日観察・5日観察）
    for obs in [3, 5]:
        edf = entry_strategy(quotes, st, observe_days=obs)
        print_entry_summary(edf, obs)

    print("\n" + "=" * 64)
    print("分析完了。手数料・スリッページ未考慮。過去の傾向で将来を保証しません。")
    print("=" * 64)
