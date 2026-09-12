# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd


def _prev_business_day(d):
    prev = d - pd.Timedelta(days=1)
    while prev.weekday() >= 5:  # 5=土, 6=日
        prev -= pd.Timedelta(days=1)
    return prev


def fetch_dividends(codes, max_codes=60):
    result = {}
    try:
        import yfinance as yf
    except ImportError:
        print("  [配当] yfinance未インストール。配当情報はスキップします", flush=True)
        return result

    codes = list(codes)[:max_codes]
    ok, ng = 0, 0
    for code in codes:
        # コードの正規化: 4桁または5桁の場合の処理 (例: "71730" -> "7173")
        c_str = str(code).strip()
        if len(c_str) > 4 and c_str.endswith("0"):
            c_str = c_str[:4]
        
        ticker = f"{c_str}.T"
        try:
            t = yf.Ticker(ticker)
            info = t.info
            ex = info.get("exDividendDate")
            last_div = info.get("lastDividendValue")
            if ex is None:
                ng += 1
                continue
            ex_date = pd.Timestamp(ex, unit="s")
            kenri = _prev_business_day(ex_date)
            entry = {"権利付最終日": str(kenri.date())}
            if last_div is not None and last_div == last_div:
                entry["100株配当"] = round(float(last_div) * 100)
            result[str(code)] = entry
            ok += 1
        except Exception:
            ng += 1
            continue
    print(f"  [配当] yfinanceから取得: 成功{ok}件 / 失敗{ng}件", flush=True)
    return result
