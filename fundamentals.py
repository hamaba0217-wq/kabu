# -*- coding: utf-8 -*-
"""
Role 4（経営者評価）と Role 7（執行設計）

この2つは **AIに判断させません。** 計算で出るからです。

- Role 4: 会社予想と実績の乖離履歴 → 達成率で機械的にランク付け
- Role 7: 損切り・利確価格 → 単なる算術

LLMに計算をさせると、もっともらしい間違いが混ざります。
確定的に出せるものは確定的に出す、が原則です。
"""

from __future__ import annotations

import pandas as pd

import config

# 会社予想の列名候補（V2は短縮名、V1は長い名前）
# 会社予想（通期）の実測列名。連結 FOP/FSales、単体 FNCOP/FNCSales。
FORECAST_COLUMNS = {
    "fc_operating_profit": ["FOP", "FNCOP", "FcOP", "ForecastOperatingProfit"],
    "fc_net_sales":        ["FSales", "FNCSales", "FcSales", "ForecastNetSales"],
}


def attach_forecasts(fin_raw: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    """正規化済みの決算データに、会社予想の列を足す。

    連結予想（FOP/FSales）が空の会社は、単体予想（FNCOP/FNCSales）で補う。
    候補リストは連結→単体の順に並べ、前の列が欠損した行だけ次で埋める。
    """
    out = fin.copy()
    for canon, cands in FORECAST_COLUMNS.items():
        series = None
        for c in cands:
            if c not in fin_raw.columns:
                continue
            col = pd.to_numeric(fin_raw[c], errors="coerce").values
            series = col if series is None else pd.Series(series).fillna(
                pd.Series(col)).values
        if series is not None:
            out[canon] = series
    return out


def management_score(fin_code: pd.DataFrame) -> dict:
    """1銘柄の「会社予想 vs 実績」履歴からRole 4のランクを出す。

    fin_code: その銘柄の決算データ（disclosed_date昇順）
    """
    g = fin_code.sort_values("disclosed_date")
    fy = g[g["period_type"].astype(str).str.upper().isin(["FY", "4Q", "通期"])]

    if "fc_operating_profit" not in g.columns or len(fy) < 2:
        return {"rank": "判定不能", "reason": "会社予想データが取得できませんでした",
                "history": [], "achieved": None}

    rows, hits = [], []
    prev_fc = None
    for _, r in fy.iterrows():
        actual = r.get("operating_profit")
        if prev_fc is not None and pd.notna(actual) and prev_fc and prev_fc != 0:
            rate = actual / prev_fc
            rows.append({"期": str(r.get("period_end"))[:10],
                         "会社予想": prev_fc, "実績": actual,
                         "達成率": round(rate * 100, 1)})
            hits.append(rate)
        prev_fc = r.get("fc_operating_profit")

    if not hits:
        return {"rank": "判定不能", "reason": "予想と実績の対応が取れませんでした",
                "history": [], "achieved": None}

    over = sum(1 for h in hits if h >= 1.0)
    under_big = sum(1 for h in hits if h < 0.9)

    if under_big >= 2:
        rank, reason = "C", f"{under_big}期で会社予想を1割以上下回っています（除外推奨）"
    elif over >= 3:
        rank, reason = "S", f"{over}期で会社予想を上回っています"
    elif over >= len(hits) * 0.6:
        rank, reason = "A", "概ね会社予想を達成しています"
    else:
        rank, reason = "B", "達成率が安定しません"

    return {"rank": rank, "reason": reason, "history": rows[-5:],
            "achieved": round(sum(hits) / len(hits) * 100, 1)}


def execution_plan(close: float) -> dict:
    """Role 7: 執行ルールを価格に落とす。"""
    return {
        "想定エントリー": round(close),
        "ハード損切り(-25%)": round(close * 0.75),
        "利確1 (+50%)": round(close * 1.50),
        "利確2 (+100%)": round(close * 2.00),
        "利確3 (+200%)": round(close * 3.00),
        "建玉上限": "総資金の7%まで（全損想定）",
        "保有期限": "65営業日",
        "シナリオ撤退": "減益転落 / 下方修正 / 想定外の大型増資 / 経営陣交代",
    }
