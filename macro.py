# -*- coding: utf-8 -*-
"""
macro.py — 株価以外の外部要因（公式・無料データのみ）

  FRED（米セントルイス連銀）: VIX（恐怖指数）, 米10年金利, ドル円, WTI原油, S&P500, NASDAQ
  財務省: 日本国債10年・2年金利（国債金利情報CSV）

先読み防止：米国市場は東京の引け後に動き、財務省の金利は当日夕方の公表のため、
すべて「1営業日後から使える」ように1日ずらして使う（rule_lab 側で処理）。
data/macro.pkl に保存（取得失敗時は前回分を使う）。
"""
from __future__ import annotations

import io
import os
import re

import pandas as pd
import requests

FRED = {
    "vix": "VIXCLS", "us10y": "DGS10", "usdjpy": "DEXJPUS",
    "wti": "DCOILWTICO", "sp500": "SP500", "nasdaq": "NASDAQCOM",
}
MOF_URLS = ["https://www.mof.go.jp/jgbs/reference/interest_rate/data/jgbcm_all.csv",
            "https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm.csv"]
PATH = os.path.join("data", "macro.pkl")
UA = {"User-Agent": "Mozilla/5.0 (research script)"}


def _fred(sid):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2019-01-01"
    df = pd.read_csv(io.StringIO(requests.get(url, headers=UA, timeout=60).text))
    df.columns = ["date", "v"]
    df["date"] = pd.to_datetime(df["date"])
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    return df.dropna().set_index("date")["v"]


_ERA = {"S": 1925, "H": 1988, "R": 2018}


def _wareki(s):
    m = re.match(r"([SHR])(\d+)\.(\d+)\.(\d+)", str(s))
    if not m:
        return pd.NaT
    return pd.Timestamp(_ERA[m[1]] + int(m[2]), int(m[3]), int(m[4]))


def _mof():
    out = []
    for u in MOF_URLS:
        try:
            t = requests.get(u, headers=UA, timeout=60).content.decode("cp932", errors="ignore")
            df = pd.read_csv(io.StringIO(t), skiprows=1)
            df["date"] = df["基準日"].map(_wareki)
            df = df.dropna(subset=["date"])
            for col, name in (("10年", "jgb10y"), ("2年", "jgb2y")):
                df[name] = pd.to_numeric(df[col], errors="coerce")
            out.append(df[["date", "jgb10y", "jgb2y"]])
        except Exception as e:
            print(f"  [警告] 財務省CSV取得失敗 {u}: {e}")
    if not out:
        return None
    return pd.concat(out).drop_duplicates("date", keep="last").set_index("date").sort_index()


def fetch():
    cols = {}
    for name, sid in FRED.items():
        try:
            cols[name] = _fred(sid)
        except Exception as e:
            print(f"  [警告] FRED {sid} 取得失敗: {e}")
    df = pd.DataFrame(cols)
    j = _mof()
    if j is not None:
        df = df.join(j, how="outer")
    df = df.sort_index()
    df = df[df.index >= "2019-01-01"]
    if len(df):
        os.makedirs("data", exist_ok=True)
        old = load()
        if old is not None:
            df = df.combine_first(old)
        df.to_pickle(PATH)
    print(f"  外部指標 {list(df.columns)} {df.index.min()}〜{df.index.max()}")
    return df


def load():
    try:
        return pd.read_pickle(PATH)
    except Exception:
        return None


if __name__ == "__main__":
    print(fetch().tail())
