# -*- coding: utf-8 -*-
"""
売られすぎ戦略の分割検証（アウトオブサンプル）

対象戦略
--------
「25日移動平均から -30〜-20% 乖離した銘柄を、利確なしで保有15日」
selltiming で平均リターン最高（+9%）だった組み合わせ。

検証内容
--------
過去データを日付で前半・後半に2分割し、それぞれ独立に：
  ・平均リターン / 勝率 / 中央値
  ・対相場超過（同期間の市場平均に対する超過）
を測る。両期間でプラスなら再現性のある本物。
前半だけプラスなら「その時期に合っただけ」の偶然。

さらに、比較のため乖離区分を複数見て、-30〜-20 が
本当に両期間で高いかを確認する。

先読み防止：指標は as_of 時点、値動きは as_of 以降。
対相場超過：各エントリー日・保有期間の市場平均リターンを引く。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import backtest
import config
import technical
from oversold import add_oversold_indicators


HOLD = 15               # 保有営業日（selltimingの最良）
STOP_LOSS = -0.10       # 損切り（-10%固定）
TAKE_PROFIT = 0.99      # 実質利確なし
STEP_DAYS = 5

DEV_ZONES = [
    ("移動平均-10〜-5乖離", -10.0, -5.0),
    ("移動平均-20〜-10乖離", -20.0, -10.0),
    ("移動平均-30〜-20乖離", -30.0, -20.0),
    ("移動平均-30以下乖離", -1e9, -30.0),
]


def _simulate(path, tp=TAKE_PROFIT, sl=STOP_LOSS):
    for px in path:
        ret = px - 1.0
        if ret <= sl:
            return sl
        if ret >= tp:
            return tp
    return path[-1] - 1.0 if len(path) else 0.0


def _collect(quotes, step_days=STEP_DAYS):
    qi = add_oversold_indicators(quotes)
    by_code = technical._build_index(qi)
    all_dates = np.sort(qi["date"].unique())
    base_index = {c: (v[0], v[1]) for c, v in by_code.items()}

    print("  売られすぎ銘柄のエントリーを収集中...")
    entries = []
    start_idx, end_idx = 260, len(all_dates) - HOLD
    for as_of in all_dates[start_idx:end_idx:step_days]:
        snap = qi[qi["date"] == as_of]
        if snap.empty:
            continue
        for _, s in snap.iterrows():
            dev = s["ma25_dev"]
            if pd.isna(dev):
                continue
            code = s["code"]
            arrs = by_code.get(code)
            if arrs is None:
                continue
            close_arr, dates_arr = arrs[0], arrs[1]
            pos = np.searchsorted(dates_arr, np.datetime64(pd.Timestamp(as_of)), side="right")
            if pos == 0:
                continue
            entry = close_arr[pos - 1]
            if not entry or entry <= 0:
                continue
            fut = close_arr[pos:pos + HOLD]
            if len(fut) < HOLD * 0.5:
                continue
            ret = _simulate(fut / entry)
            entries.append({"as_of": pd.Timestamp(as_of), "dev": dev, "ret": ret})
    print(f"  サンプル: {len(entries):,} 件")
    return entries, base_index


def _evaluate(entries, base_index, lo, hi, date_lo=None, date_hi=None):
    base_cache = {}
    def _base(as_of):
        if as_of not in base_cache:
            base_cache[as_of] = backtest.market_baseline(base_index, as_of, HOLD)
        return base_cache[as_of]

    rets, excess = [], []
    for e in entries:
        if not (lo <= e["dev"] < hi if hi < 0 else lo <= e["dev"] <= hi):
            continue
        if date_lo is not None and e["as_of"] < date_lo:
            continue
        if date_hi is not None and e["as_of"] >= date_hi:
            continue
        rets.append(e["ret"])
        b = _base(e["as_of"])
        if pd.notna(b):
            excess.append(e["ret"] - b)
    if len(rets) < 20:
        return None
    arr = np.array(rets)
    return {
        "件数": len(arr),
        "平均%": round(arr.mean() * 100, 2),
        "勝率%": round((arr > 0).mean() * 100, 1),
        "中央%": round(np.median(arr) * 100, 2),
        "対相場超過%": round(np.median(excess) * 100, 2) if excess else None,
    }


def run(quotes):
    entries, base_index = _collect(quotes)
    dates = sorted(e["as_of"] for e in entries)
    mid = dates[len(dates) // 2]
    print(f"  分割点: {mid.date()}（前半 <  / 後半 >= ）")

    rows = []
    for zname, lo, hi in DEV_ZONES:
        full = _evaluate(entries, base_index, lo, hi)
        first = _evaluate(entries, base_index, lo, hi, date_hi=mid)
        second = _evaluate(entries, base_index, lo, hi, date_lo=mid)
        rows.append((zname, full, first, second))
    return rows, mid


def summarize(rows, mid):
    L = ["=" * 80,
         "売られすぎ戦略の分割検証（利確なし・保有15日・損切-10%）",
         "=" * 80, ""]
    L.append(f"分割点: {mid.date()}  前半=これより前 / 後半=これ以降")
    L.append("各区分で、全期間・前半・後半の 平均リターン / 勝率 / 対相場超過 を見る。")
    L.append("両期間とも対相場超過プラスなら、再現性のある本物。")
    L.append("")

    def _fmt(d):
        if d is None:
            return "（件数不足）"
        ex = f"{d['対相場超過%']:+.2f}" if d['対相場超過%'] is not None else "--"
        return (f"件数{d['件数']:>5}  平均{d['平均%']:+6.2f}%  "
                f"勝率{d['勝率%']:4.1f}%  対相場超過{ex}%")

    for zname, full, first, second in rows:
        L.append("=" * 80)
        L.append(f"【{zname}】")
        L.append(f"  全期間 : {_fmt(full)}")
        L.append(f"  前半   : {_fmt(first)}")
        L.append(f"  後半   : {_fmt(second)}")
        # 判定
        if first and second and first["対相場超過%"] is not None and second["対相場超過%"] is not None:
            if first["対相場超過%"] > 0 and second["対相場超過%"] > 0:
                verdict = "◎ 両期間とも対相場プラス＝再現性あり（本物の可能性）"
            elif first["平均%"] > 0 and second["平均%"] > 0:
                verdict = "○ 両期間とも平均プラスだが、対相場では片方以上マイナス"
            else:
                verdict = "× どちらかの期間でマイナス＝偶然の可能性"
            L.append(f"  判定   : {verdict}")
        L.append("")

    L += ["=" * 80, "読み方", "=" * 80,
          "・『対相場超過』が前半・後半どちらもプラスなら、時期を問わず市場に勝てる。",
          "・平均リターンがプラスでも対相場超過がマイナスなら、",
          "  それは相場全体が上げていただけで、戦略の力ではない。",
          "・前半だけプラスで後半マイナスは、その時期にたまたま合っただけ。",
          "・深い乖離（-30〜-20）が両期間で高ければ、売られすぎ反発は本物。",
          "・手数料・スリッページ未考慮。利確なしは保有中の下振れリスクも大きい。"]
    return "\n".join(L)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    from sources import JQuants, JST
    jq = JQuants()
    print("キャッシュからデータを読み込み中...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    print(f"  株価 {len(quotes):,}行")

    rows, mid = run(quotes)
    report = summarize(rows, mid)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    path = os.path.join(config.OUTPUT_DIR, f"{today}_validate_oversold.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nレポート: {path}")


if __name__ == "__main__":
    main()
