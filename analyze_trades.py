# -*- coding: utf-8 -*-
"""
テクニカル戦略の勝敗要因分析

technical.py の3戦略（A決算モメンタム / Bボックス反発 / Cブレイクアウト）で
発生した全トレードを、エントリー時点の特徴とともに記録し、
「どんな特徴のトレードが勝ちやすい/負けやすいか」を集計します。

逆算分析と同じ原則
------------------
・勝ちトレードだけを見ない。全トレードを同じ土俵で集計する（生存者バイアス回避）
・「勝ちトレードの平均時価総額」ではなく「その時価総額帯の勝率」を見る
・エントリー時点で分かる特徴だけを使う（結果は特徴に混ぜない）

分析する特徴（as_of時点で判明しているもの）
--------------------------------------------
・時価総額
・売買代金（絶対額）
・株価
・出来高急増度（20日平均比）
・52週高値からの位置
・エントリー前1か月の値動き（既に上げすぎか）
・業種
・相場環境（その日の市場全体の方向）

空売り・信用残は J-Quants Standard プランが必要なため、現状は対象外。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import backtest
import config
import technical


def _collect_trades(quotes, fin, listed, step_days=10):
    """3戦略の全トレードを、エントリー特徴つきで1つのDataFrameに集める。"""
    qi = technical.add_indicators(quotes)

    # 追加の特徴量を計算
    g = qi.groupby("code")
    qi["ret_1m"] = g["close"].transform(lambda s: s / s.shift(20) - 1.0)  # 直近1か月騰落
    qi["high_52w"] = g["close"].transform(
        lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / qi["high_52w"] - 1.0

    by_code = technical._build_index(qi)
    all_dates = np.sort(qi["date"].unique())

    # 発行済株式数（時価総額用）を最新の決算から引く
    fin_sorted = fin.sort_values("disclosed_date")
    shares = (fin_sorted.dropna(subset=["shares_outstanding"])
              .groupby("code")["shares_outstanding"].last())

    # 市場・業種
    meta = listed.set_index("code")[["market", "sector33"]] if "sector33" in listed.columns \
        else listed.set_index("code")[["market"]]

    fin2 = technical._prep_fin(fin)

    # 相場環境（各日の全銘柄前日比中央値）を事前計算
    daily_med = _market_direction(qi)

    rows = []
    for name, spec in technical.STRATEGIES.items():
        horizon = spec["horizon"]
        start_idx, end_idx = 80, len(all_dates) - horizon
        if end_idx <= start_idx:
            continue
        for as_of in all_dates[start_idx:end_idx:step_days]:
            snap = technical._snapshot(qi, as_of)
            if snap.empty:
                continue

            if spec["entry"] == "breakout":
                codes = technical.entries_breakout(snap)
            elif spec["entry"] == "box":
                codes = technical.entries_box_bottom(snap)
            else:
                codes = technical.entries_earnings_momentum(snap, fin2, as_of)
            if not codes:
                continue

            snap_i = snap.set_index("code")
            mkt_dir = daily_med.get(pd.Timestamp(as_of), np.nan)

            for code in codes:
                path, ma_path = technical._forward(by_code, code, as_of, horizon)
                if path is None:
                    continue
                if spec["exit"] == "box":
                    r, reason = technical._exit_box(path)
                else:
                    r, reason = technical._exit_trend(path, ma_path)

                if code not in snap_i.index:
                    continue
                s = snap_i.loc[code]
                price = float(s["close"])
                sh = shares.get(code, np.nan)
                mcap = price * sh if pd.notna(sh) else np.nan

                rows.append({
                    "戦略": name,
                    "as_of": pd.Timestamp(as_of).date(),
                    "code": code,
                    "return": r,
                    "is_win": r > 0,
                    "reason": reason,
                    # --- エントリー時点の特徴 ---
                    "時価総額_億": mcap / 1e8 if pd.notna(mcap) else np.nan,
                    "売買代金_億": float(s["turnover_value"]) / 1e8,
                    "株価": price,
                    "出来高急増": float(s.get("turnover_spike", np.nan)),
                    "高値からの位置": float(s.get("pct_from_high", np.nan)) * 100
                        if pd.notna(s.get("pct_from_high", np.nan)) else np.nan,
                    "直近1M騰落": float(s.get("ret_1m", np.nan)) * 100
                        if pd.notna(s.get("ret_1m", np.nan)) else np.nan,
                    "業種": meta.loc[code, "sector33"] if code in meta.index
                        and "sector33" in meta.columns else None,
                    "相場環境": mkt_dir * 100 if pd.notna(mkt_dir) else np.nan,
                })
    return pd.DataFrame(rows)


def _market_direction(qi):
    """各日の全銘柄前日比の中央値（相場環境の代理）。"""
    q = qi.sort_values("date").copy()
    q["chg"] = q.groupby("code")["close"].transform(lambda s: s / s.shift(1) - 1.0)
    return q.groupby("date")["chg"].median()


# ---------------------------------------------------------------------------
# 特徴別の勝率分析
# ---------------------------------------------------------------------------

def _lift_table(df, col, bins, labels):
    sub = df[df[col].notna()].copy()
    if len(sub) < 10:
        return None
    sub["_b"] = pd.cut(sub[col], bins=bins, labels=labels)
    base = sub["is_win"].mean()
    g = sub.groupby("_b", observed=True).agg(
        件数=("is_win", "size"),
        勝率=("is_win", "mean"),
        平均リターン=("return", "mean"),
    ).reset_index()
    g["勝率リフト"] = (g["勝率"] / base).round(2) if base > 0 else np.nan
    g["勝率"] = (g["勝率"] * 100).round(1)
    g["平均リターン"] = (g["平均リターン"] * 100).round(2)
    return g, base


def _classify(r):
    """1トレードのリターンを6段階に分類する。"""
    if r >= 0.10:
        return "大勝ち(+10%〜)"
    if r >= 0.03:
        return "普通に勝ち(+3〜10%)"
    if r > 0:
        return "小勝ち(0〜+3%)"
    if r > -0.03:
        return "小負け(0〜-3%)"
    if r > -0.10:
        return "普通に負け(-3〜-10%)"
    return "大負け(-10%〜)"


CLASS_ORDER = ["大勝ち(+10%〜)", "普通に勝ち(+3〜10%)", "小勝ち(0〜+3%)",
               "小負け(0〜-3%)", "普通に負け(-3〜-10%)", "大負け(-10%〜)"]


def _distribution(df):
    """6段階分類の分布を返す。"""
    cls = df["return"].apply(_classify)
    counts = cls.value_counts()
    total = len(df)
    lines = []
    for c in CLASS_ORDER:
        n = int(counts.get(c, 0))
        pct = n / total * 100 if total else 0
        # そのクラスの平均リターンと、全体への寄与（平均リターンへの貢献）
        sub = df[cls == c]
        contrib = sub["return"].sum() / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        lines.append((c, n, pct, contrib, bar))
    return lines


def _dist_by_group(df, col, bins, labels):
    """特徴帯ごとに『大勝ち率』と『大負け率』を出す。"""
    sub = df[df[col].notna()].copy()
    if len(sub) < 10:
        return None
    sub["_b"] = pd.cut(sub[col], bins=bins, labels=labels)
    sub["_cls"] = sub["return"].apply(_classify)
    rows = []
    for b, g in sub.groupby("_b", observed=True):
        n = len(g)
        big_win = (g["_cls"] == "大勝ち(+10%〜)").mean() * 100
        big_lose = (g["_cls"] == "大負け(-10%〜)").mean() * 100
        avg = g["return"].mean() * 100
        rows.append((str(b), n, big_win, big_lose, avg))
    return rows


def analyze(df):
    L = []
    A = L.append
    A("=" * 70)
    A("テクニカル3戦略の勝敗要因分析（全トレード・生存者バイアスなし）")
    A("=" * 70)
    A(f"総トレード数: {len(df):,} / 全体勝率: {df['is_win'].mean()*100:.1f}%")
    A(f"平均リターン: {df['return'].mean()*100:+.2f}%")
    A("")

    # --- リターンの6段階分布 ---
    A("■ 勝ち負けの内訳（6段階）")
    A("  分類                  件数    割合   平均への寄与")
    for c, n, pct, contrib, bar in _distribution(df):
        A(f"  {c:<18} {n:>6}  {pct:5.1f}%   {contrib:+6.2f}%  {bar}")
    A("")
    A("  ※「平均への寄与」= その層が全体平均リターンをどれだけ押し上げ/下げたか。")
    A("    大負けの寄与が大きければ『たまの大負けが全体を潰している』と分かる。")
    A("")
    A("※各特徴帯の『勝率』= その特徴を持つトレードが勝った割合。")
    A("  リフト1.0が全体平均。1.2なら平均の1.2倍勝ちやすい特徴帯。")
    A("")

    specs = [
        ("時価総額_億", "時価総額（億円）",
         [0, 50, 100, 300, 1000, 1e9], ["〜50", "50-100", "100-300", "300-1000", "1000〜"]),
        ("売買代金_億", "売買代金（億円）",
         [0, 1, 3, 10, 50, 1e9], ["〜1", "1-3", "3-10", "10-50", "50〜"]),
        ("株価", "株価（円）",
         [0, 200, 500, 1000, 3000, 1e9], ["〜200", "200-500", "500-1000", "1000-3000", "3000〜"]),
        ("出来高急増", "出来高急増（20日平均比）",
         [0, 1, 2, 3, 5, 1e9], ["〜1", "1-2", "2-3", "3-5", "5〜"]),
        ("高値からの位置", "52週高値からの位置(%)",
         [-1e9, -50, -30, -15, -5, 1e9], ["〜-50", "-50〜-30", "-30〜-15", "-15〜-5", "-5〜高値圏"]),
        ("直近1M騰落", "直近1か月の騰落(%)",
         [-1e9, -10, 0, 10, 30, 1e9], ["〜-10", "-10〜0", "0-10", "10-30", "30〜"]),
        ("相場環境", "エントリー日の相場（全銘柄中央値%）",
         [-1e9, -1, 0, 1, 1e9], ["〜-1(下落)", "-1〜0", "0〜1", "1〜(上昇)"]),
    ]

    for col, title, bins, labels in specs:
        res = _lift_table(df, col, bins, labels)
        if res is None:
            continue
        g, base = res
        A(f"■ {title}")
        for _, r in g.iterrows():
            bar = "█" * int(max(0, r["勝率リフト"]) * 8) if pd.notna(r["勝率リフト"]) else ""
            A(f"  {str(r['_b']):>14} : 勝率{r['勝率']:5.1f}%  "
              f"リフト{r['勝率リフト'] if pd.notna(r['勝率リフト']) else 0:4.2f}  "
              f"平均{r['平均リターン']:+6.2f}%  (n={int(r['件数']):>5}) {bar}")
        A("")

    # --- 主要特徴ごとの「大勝ち率 vs 大負け率」 ---
    A("■ 特徴帯ごとの大勝ち率・大負け率（損小利大が成立しているか）")
    A("  大勝ち(+10%〜)が多く大負け(-10%〜)が少ない帯 = 有利")
    A("")
    for col, title, bins, labels in [
        ("時価総額_億", "時価総額(億円)",
         [0, 50, 100, 300, 1000, 1e9], ["〜50", "50-100", "100-300", "300-1000", "1000〜"]),
        ("株価", "株価(円)",
         [0, 200, 500, 1000, 3000, 1e9], ["〜200", "200-500", "500-1000", "1000-3000", "3000〜"]),
        ("高値からの位置", "52週高値から(%)",
         [-1e9, -50, -30, -15, -5, 1e9], ["〜-50", "-50〜-30", "-30〜-15", "-15〜-5", "-5〜"]),
        ("相場環境", "エントリー日の相場(%)",
         [-1e9, -1, 0, 1, 1e9], ["〜-1(下落)", "-1〜0", "0〜1", "1〜(上昇)"]),
    ]:
        rows = _dist_by_group(df, col, bins, labels)
        if not rows:
            continue
        A(f"  【{title}】")
        for b, n, bw, bl, avg in rows:
            edge = "◎" if bw > bl * 1.3 else ("×" if bl > bw * 1.3 else " ")
            A(f"    {b:>12} : 大勝ち{bw:5.1f}%  大負け{bl:5.1f}%  平均{avg:+6.2f}%  (n={n:>5}) {edge}")
        A("")

    # 業種別
    if "業種" in df.columns and df["業種"].notna().any():
        A("■ 業種別（n≥50・勝率上位/下位5）")
        sec = df[df["業種"].notna()].groupby("業種").agg(
            件数=("is_win", "size"), 勝率=("is_win", "mean"),
            平均リターン=("return", "mean")).reset_index()
        sec = sec[sec["件数"] >= 50]
        base = df["is_win"].mean()
        sec["リフト"] = (sec["勝率"] / base).round(2)
        sec = sec.sort_values("勝率", ascending=False)
        for _, r in pd.concat([sec.head(5), sec.tail(5)]).drop_duplicates().iterrows():
            A(f"  {r['業種']:<14} : 勝率{r['勝率']*100:5.1f}%  リフト{r['リフト']:4.2f}  "
              f"平均{r['平均リターン']*100:+6.2f}%  (n={int(r['件数'])})")
        A("")

    # 戦略別の勝敗も併記
    A("■ 戦略別")
    for name, gg in df.groupby("戦略"):
        A(f"  {name:<16} : 勝率{gg['is_win'].mean()*100:5.1f}%  "
          f"平均{gg['return'].mean()*100:+6.2f}%  (n={len(gg):,})")
    A("")

    A("=" * 70)
    A("読み方と注意")
    A("=" * 70)
    A("・リフトが高く n が大きい特徴帯 = 再現性のある勝ちパターンの候補。")
    A("・リフトが高くても n が小さい帯は偶然かもしれない。")
    A("・平均リターンも必ず併記。勝率が高くても平均がマイナスなら")
    A("  『小さく勝って大きく負ける』パターン。")
    A("・ここで見えた特徴は『仮説』。エントリー条件に加えて再検証し、")
    A("  別期間でも再現するか確認すること（過剰最適化を避ける）。")
    A("・手数料・スリッページ未考慮。高頻度な戦略ほど実際は不利。")
    A("・空売り・信用残は Standard プランが必要なため未分析。")
    return "\n".join(L)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    from sources import JQuants, JST
    import fundamentals
    jq = JQuants()
    print("キャッシュからデータを読み込み中...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    fin = jq.financials(config.BACKTEST_LOOKBACK_DAYS)
    listed = jq.listed()
    # 会社予想は financials() のキャッシュに含まれるため再取得は不要
    print(f"  株価 {len(quotes):,}行 / 決算 {len(fin):,}行 / 銘柄 {len(listed):,}")

    print("\n全トレードを収集中（3戦略）...")
    trades = _collect_trades(quotes, fin, listed)
    if trades.empty:
        print("トレードがありませんでした。")
        return
    print(f"  {len(trades):,} トレードを収集\n")

    report = analyze(trades)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    csv = os.path.join(config.OUTPUT_DIR, f"{today}_trade_analysis.csv")
    trades.to_csv(csv, index=False, encoding="utf-8-sig")
    rep = os.path.join(config.OUTPUT_DIR, f"{today}_trade_analysis.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n全トレード明細: {csv}")
    print(f"分析レポート  : {rep}")
    print("\n勝ち/負けで絞るには、CSVを is_win 列でフィルタしてください。")


if __name__ == "__main__":
    main()
