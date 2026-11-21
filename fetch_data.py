# -*- coding: utf-8 -*-
"""
fetch_data.py — 勝率分析用データを data/ に少しずつ蓄積（古い日→新しい日へ）

・1回の実行は時間予算（環境変数 FETCH_MINUTES、既定 50分）内で取得し、40日ごとに保存。
  GitHub Actions の時間切れでも途中までのデータが残り、次回はその続きから取る。
・株価は「未調整の生データ + 調整係数」で保存し、分割調整は rule_lab が全期間通して行う
  （取得の区切りごとに調整すると分割前後で価格がずれるため）。
・状態は data/fetch_status.txt（ready=最新まで取得済み / partial=途中）
"""
import datetime as dt
import json
import os
import time

import pandas as pd

import cache
from sources import JQuants, QUOTE_COLUMNS, normalize, JST

START = pd.Timestamp(os.environ.get("FETCH_START", "2021-06-01"))
BUDGET = float(os.environ.get("FETCH_MINUTES", "50")) * 60
T0 = time.time()
PROG = os.path.join("data", "fetch_progress.json")


def _prog():
    try:
        return json.load(open(PROG))
    except Exception:
        return {}


def _set_prog(name, day):
    p = _prog()
    p[name] = str(day.date())
    os.makedirs("data", exist_ok=True)
    json.dump(p, open(PROG, "w"))


def _norm_q(raw):
    df = normalize(raw, QUOTE_COLUMNS, required=["code", "date", "close", "volume"], label="株価日足")
    df["code"] = df["code"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume", "turnover_value", "adj_factor"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def grow(name, fetch_raw, norm, keys):
    cur = cache.load(name)
    done = _prog().get(name)
    start = START if done is None else pd.Timestamp(done) + pd.Timedelta(days=1)
    end = pd.Timestamp(dt.datetime.now(JST).date())
    days = pd.bdate_range(start, end)
    if len(days) == 0:
        print(f"  {name}: 最新まで取得済み（{done}まで）")
        return True
    print(f"  {name}: {days[0].date()}〜 {len(days)}営業日分が未取得")
    chunk = 40
    for i in range(0, len(days), chunk):
        if time.time() - T0 > BUDGET:
            print(f"  {name}: 時間予算に達したので中断（次回続きから）")
            return False
        a, b = days[i], days[min(i + chunk, len(days)) - 1]
        s = dt.datetime.combine(a.date(), dt.time(), tzinfo=JST)
        e = dt.datetime.combine(b.date(), dt.time(), tzinfo=JST)
        raw = fetch_raw(s, e)
        if raw is not None and len(raw):
            cur = cache.merge(cur, norm(raw), keys=keys)
            cache.save(name, cur)
        # 当日分はまだ出ていない可能性があるので、進捗は「前営業日まで」に留める
        last_ok = b if b.date() < end.date() else b - pd.offsets.BDay(1)
        _set_prog(name, last_ok)
        print(f"    {name}: 〜{b.date()} 保存（{0 if cur is None else len(cur)}行）")
    return True


def main():
    jq = JQuants()
    ok_q = grow("raw_quotes", jq.raw_quotes, _norm_q, ["code", "date"])
    if ok_q:
        grow("fin", jq.raw_fin, jq._normalize_fin, ["code", "disclosed_date"])
    try:
        if cache.load("listed") is None or ok_q:
            jq.listed(use_cache=True)
    except Exception as e:
        print("listed error", e)
    q = cache.load("raw_quotes")
    status = "ready" if ok_q else "partial"
    if q is not None:
        print(f"raw_quotes {len(q)}行 {q['date'].min().date()}〜{q['date'].max().date()} 銘柄 {q['code'].nunique()}")
    os.makedirs("data", exist_ok=True)
    open(os.path.join("data", "fetch_status.txt"), "w").write(status)
    print("FETCH_STATUS", status)


if __name__ == "__main__":
    main()
