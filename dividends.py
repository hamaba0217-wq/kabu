# -*- coding: utf-8 -*-
"""
yfinance（Yahoo Finance）から、候補銘柄の配当情報を取得する。
  ・権利付最終日（= 配当落ち日 exDividendDate の前営業日）
  ・100株あたり配当（直近1回分 × 100）

重要な注意（事実）
------------------
  ・yfinance は非公式ライブラリ。Yahoo側の都合で突然止まることがある。
    → 取得に失敗しても全体を止めない（空欄で続行）。
  ・返るのは exDividendDate（配当落ち日）。配当をもらうために持つべきは
    その前営業日（権利付最終日）なので、1営業日戻して計算する。
  ・金額は「直近1回分」。次回も同額とは限らない。
  ・すべて参考値。正確な日付・金額は証券会社や企業IRで要確認。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def _prev_business_day(d):
    """前営業日を返す（土日のみ考慮。祝日は考慮しないので注記で補う）。"""
    prev = d - pd.Timedelta(days=1)
    while prev.weekday() >= 5:  # 5=土, 6=日
        prev -= pd.Timedelta(days=1)
    return prev


def fetch_dividends(codes, max_codes=60):
    """銘柄コードのリストから配当情報を取得。
    返り値: {code: {"権利付最終日": str, "100株配当": float}} の辞書。
    取得できない銘柄は辞書に入らない（呼び出し側で空欄扱い）。"""
    result = {}
    try:
        import yfinance as yf
    except ImportError:
        print("  [配当] yfinance未インストール。配当情報はスキップします", flush=True)
        return result

    codes = list(codes)[:max_codes]  # 負荷制限：多すぎる時は上限まで
    ok, ng = 0, 0
    for code in codes:
        ticker = f"{code}.T"
        try:
            t = yf.Ticker(ticker)
            info = t.info
            ex = info.get("exDividendDate")
            last_div = info.get("lastDividendValue")
            if ex is None:
                ng += 1
                continue
            ex_date = pd.Timestamp(ex, unit="s")
            # 未来の配当落ち日でなければ（過去のものしか無い場合）、履歴から次を推定しない。
            # 素直に「直近に判明している配当落ち日」を使い、権利付最終日に変換。
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
