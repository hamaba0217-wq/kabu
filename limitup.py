# -*- coding: utf-8 -*-
"""
ストップ高を検出し、その後の値動きを分析する基盤。

ストップ高の定義（JPX公式の制限値幅テーブルにもとづく）
------------------------------------------------------
  制限値幅は「前日終値（基準値段）」の価格帯で決まる。
  当日終値が「前日終値 + 制限値幅」以上なら、ストップ高（で引けた）とみなす。
  ※ザラ場で一時的にストップ高でも終値が下なら除外（終値ベースで厳密に判定）。
  ※分割・併合のあった日は前日比がずれるが、稀なので無視できる誤差。

事実にもとづく注意
  ・終値ベースの判定なので「ストップ高で引けた」もの。ザラ場タッチは含まない。
  ・値幅制限の臨時拡大（4倍ルール等）は考慮しない（近似）。
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# JPX公式：基準値段（前日終値）→ 制限値幅（円）。2026年時点。
# (上限未満, 制限値幅) のリスト。価格が上限未満ならその値幅。
_LIMIT_TABLE = [
    (100, 30), (200, 50), (500, 80), (700, 100), (1000, 150),
    (1500, 300), (2000, 400), (3000, 500), (5000, 700), (7000, 1000),
    (10000, 1500), (15000, 3000), (20000, 4000), (30000, 5000),
    (50000, 7000), (70000, 10000), (100000, 15000), (150000, 30000),
    (200000, 40000), (300000, 50000), (500000, 70000), (700000, 100000),
    (1000000, 150000), (1500000, 300000), (2000000, 400000),
    (3000000, 500000), (5000000, 700000), (7000000, 1000000),
    (10000000, 1500000), (15000000, 3000000), (20000000, 4000000),
    (30000000, 5000000), (50000000, 7000000),
]
_LIMIT_ABOVE_50M = 10000000  # 5000万円以上


def price_limit(base_price: float) -> float:
    """基準値段（前日終値）に対する制限値幅（円）を返す。"""
    if base_price is None or base_price != base_price:  # NaN
        return np.nan
    for upper, width in _LIMIT_TABLE:
        if base_price < upper:
            return width
    return _LIMIT_ABOVE_50M


def detect_limit_up(quotes: pd.DataFrame, tol: float = 0.995) -> pd.DataFrame:
    """各銘柄について、終値ベースでストップ高だった日を検出する。
    quotes: code, date, close（分割調整済みでよい）。
    返り値: ストップ高だった (code, date, close, prev_close, gain_pct) のDataFrame。
    tol: 制限値幅の何割以上でストップ高とみなすか（呼値の丸め等の誤差を吸収。0.995）。
    """
    q = quotes[["code", "date", "close"]].dropna().sort_values(["code", "date"])
    rows = []
    for code, g in q.groupby("code"):
        closes = g["close"].values
        dates = g["date"].values
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            cur = closes[i]
            if prev <= 0:
                continue
            width = price_limit(prev)
            if width != width:  # NaN
                continue
            up = cur - prev
            # 当日終値が「前日終値+制限値幅」の tol 倍以上ならストップ高で引けたとみなす
            if up >= width * tol:
                rows.append({
                    "code": str(code),
                    "date": pd.Timestamp(dates[i]),
                    "close": float(cur),
                    "prev_close": float(prev),
                    "gain_pct": round((cur / prev - 1) * 100, 2),
                    "limit_width": width,
                })
    return pd.DataFrame(rows)
