# -*- coding: utf-8 -*-
"""
Role 0: 相場環境判定（自動化できる範囲）

J-Quants のデータだけで計算できるものに限定しています。

計算するも
- 市場全体の売買代金トレンド（資金が入っているか出ているか）
- グロース市場の売買代金トレンド（小型株にお金が向いているか）
- 上昇銘柄比率（市場の広がり）

自動化できないもの（メール本文に注意書きとして入れます）
- VIX指数、米10年債利回り、ドル円 … J-Quantsの対象外
- 地政学イベント、日銀・FOMCの日程 … 人が見る必要あり

つまりこの判定は「国内株式市場に資金が向かっているか」だけを見ています。
海外発のリスクオフは検知できません。そこは割り切りです。
"""

from __future__ import annotations

import pandas as pd


def judge(quotes: pd.DataFrame, listed: pd.DataFrame | None = None) -> dict:
    """直近の相場環境を判定する。

    戻り値: {"level": "GREEN"/"YELLOW"/"RED", "reasons": [...], "metrics": {...}}
    """
    q = quotes.sort_values("date")
    dates = q["date"].drop_duplicates().sort_values()
    if len(dates) < 25:
        return {"level": "UNKNOWN", "reasons": ["データ不足で判定できません"], "metrics": {}}

    # 日次の市場全体売買代金
    daily = q.groupby("date")["turnover_value"].sum()
    latest = float(daily.iloc[-1])
    ma20 = float(daily.tail(20).mean())
    ma60 = float(daily.tail(60).mean()) if len(daily) >= 60 else ma20
    turnover_ratio = latest / ma20 if ma20 else float("nan")
    trend_ratio = ma20 / ma60 if ma60 else float("nan")

    # グロース市場の売買代金比率
    growth_share = float("nan")
    if listed is not None and "market" in listed.columns:
        m = q.merge(listed[["code", "market"]], on="code", how="left")
        last_day = m[m["date"] == m["date"].max()]
        total = last_day["turnover_value"].sum()
        g = last_day[last_day["market"].fillna("").str.contains("グロース")]["turnover_value"].sum()
        if total:
            growth_share = float(g / total)

    # 上昇銘柄比率（直近日）
    last2 = q[q["date"].isin(dates.tail(2))]
    piv = last2.pivot_table(index="code", columns="date", values="close")
    adv_ratio = float("nan")
    if piv.shape[1] == 2:
        chg = piv.iloc[:, 1] / piv.iloc[:, 0] - 1
        adv_ratio = float((chg > 0).mean())

    reasons, score = [], 0

    if pd.notna(trend_ratio):
        if trend_ratio >= 1.05:
            score += 1
            reasons.append(f"市場全体の売買代金が拡大傾向（20日平均が60日平均の{trend_ratio:.2f}倍）")
        elif trend_ratio <= 0.90:
            score -= 1
            reasons.append(f"市場全体の売買代金が縮小傾向（20日平均が60日平均の{trend_ratio:.2f}倍）")
        else:
            reasons.append(f"市場全体の売買代金は横ばい（{trend_ratio:.2f}倍）")

    if pd.notna(adv_ratio):
        if adv_ratio >= 0.60:
            score += 1
            reasons.append(f"上昇銘柄が{adv_ratio:.0%}と広い")
        elif adv_ratio <= 0.35:
            score -= 1
            reasons.append(f"上昇銘柄が{adv_ratio:.0%}と狭い")
        else:
            reasons.append(f"上昇銘柄比率 {adv_ratio:.0%}")

    if pd.notna(growth_share):
        reasons.append(f"グロース市場が売買代金全体の{growth_share:.1%}")

    level = "GREEN" if score >= 1 else ("RED" if score <= -1 else "YELLOW")

    return {
        "level": level,
        "reasons": reasons,
        "metrics": {
            "turnover_latest": latest,
            "turnover_ma20": ma20,
            "turnover_ratio": turnover_ratio,
            "trend_ratio": trend_ratio,
            "growth_share": growth_share,
            "advance_ratio": adv_ratio,
        },
    }


LEVEL_LABEL = {
    "GREEN":   ("🟢 GREEN", "通常運用"),
    "YELLOW":  ("🟡 YELLOW", "新規は半分のサイズ"),
    "RED":     ("🔴 RED", "新規エントリー停止"),
    "UNKNOWN": ("⚪ 判定不能", "データ不足"),
}
