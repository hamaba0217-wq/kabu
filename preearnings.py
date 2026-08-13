# -*- coding: utf-8 -*-
"""
決算前手仕舞いによる安定性向上の検証（ボックス反発）

仮説
----
決算発表は株価が最も予測不能に動く瞬間（ギャンブル）。
保有中に決算をまたぐと、戦略の想定外の大きな損益が発生する。
決算の1営業日前に手仕舞えば、決算ギャンブルを避けて安定性が上がるはず。

決算予定日の推定（バックテストでの再現）
----------------------------------------
実運用では「次の決算日」は会社が事前公表するので分かる。
バックテストでこれを再現するため、四半期決算が約90日ごとに規則的に
出ることを利用し、「直近の開示日 + 約90日」を次回決算予定日と推定する。
（厳密な予定日ではないが、決算が近いゾーンは捉えられる）

検証内容
--------
ボックス反発について、以下を比較：
  ・通常（決算をまたいでも保有継続）
  ・決算1営業日前に手仕舞い

保有中に決算予定日が来る場合、その1営業日前の終値で決済する。
決算前に+15%や-7%に達していれば、そちらが優先（通常通り）。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import backtest
import config
import technical


EARNINGS_INTERVAL = 90       # 四半期決算の間隔（暦日）
INTERVAL_TOL = 25            # 推定のゆらぎ許容


def _next_earnings_date(fin_by_code, code, as_of):
    """as_of 時点で見込まれる「次の決算予定日」を推定する。

    直近の開示日 + 90日 を次回予定とする。
    as_of より後で、かつ最も近い推定日を返す。無ければ None。
    """
    dates = fin_by_code.get(code)
    if dates is None or len(dates) == 0:
        return None
    past = [d for d in dates if d <= pd.Timestamp(as_of)]
    if not past:
        return None
    last_disc = max(past)
    # 直近開示から約90日ごとに、as_ofより後の最初の予定日を探す
    est = last_disc + pd.Timedelta(days=EARNINGS_INTERVAL)
    while est <= pd.Timestamp(as_of):
        est += pd.Timedelta(days=EARNINGS_INTERVAL)
    return est


def _exit_box_with_earnings(path_dates, path, target, stop,
                            earnings_date=None):
    """ボックス反発の決済 + 決算1営業日前の強制手仕舞い。

    path_dates: 各営業日の日付配列（pathと同じ長さ）
    earnings_date: 推定決算日。これの1営業日前で手仕舞う。
    """
    for i, px in enumerate(path):
        ret = px - 1.0
        # 通常の利確・損切りを優先
        if ret <= stop:
            return stop, "損切り"
        if ret >= target:
            return target, "反発利確"
        # 決算前手仕舞い判定：次の営業日が決算日なら、今日の終値で手仕舞う
        if earnings_date is not None and i + 1 < len(path_dates):
            if pd.Timestamp(path_dates[i + 1]) >= earnings_date > pd.Timestamp(path_dates[i]):
                return ret, "決算前手仕舞い"
    return (path[-1] - 1.0 if len(path) else 0.0), "期限切れ"


def _forward_with_dates(by_code, code, as_of, horizon):
    """pathと、対応する日付配列を返す。"""
    arrs = by_code.get(code)
    if arrs is None:
        return None, None
    close_arr, dates_arr = arrs[0], arrs[1]
    pos = np.searchsorted(dates_arr, np.datetime64(pd.Timestamp(as_of)), side="right")
    if pos == 0:
        return None, None
    entry = close_arr[pos - 1]
    if not entry or entry <= 0:
        return None, None
    fut = close_arr[pos:pos + horizon]
    fut_dates = dates_arr[pos:pos + horizon]
    if len(fut) < horizon * 0.5:
        return None, None
    return fut / entry, fut_dates


def run(quotes, fin, listed, step_days=7):
    qi = technical.add_indicators(quotes)
    by_code = technical._build_index(qi)
    all_dates = np.sort(qi["date"].unique())
    base_index = {c: (v[0], v[1]) for c, v in by_code.items()}

    # 銘柄ごとの開示日リスト（決算予定日の推定に使う）
    fin_by_code = {}
    for code, g in fin.groupby("code"):
        fin_by_code[code] = sorted(pd.to_datetime(g["disclosed_date"]).tolist())

    print("  ボックス反発のエントリー候補を収集中...")
    spec = technical.STRATEGIES["B_ボックス反発"]
    horizon = spec["horizon"]
    entries = []
    s_i, e_i = 80, len(all_dates) - horizon
    for as_of in all_dates[s_i:e_i:step_days]:
        snap = technical._snapshot(qi, as_of)
        if snap.empty:
            continue
        codes = technical.entries_box_bottom(snap)
        if not codes:
            continue
        for code in codes:
            path, path_dates = _forward_with_dates(by_code, code, as_of, horizon)
            if path is None:
                continue
            earnings = _next_earnings_date(fin_by_code, code, as_of)
            entries.append({"as_of": pd.Timestamp(as_of), "horizon": horizon,
                            "path": path, "path_dates": path_dates,
                            "earnings": earnings})
    print(f"  候補: {len(entries):,} 件")

    base_cache = {}
    def _base(as_of, h):
        k = (as_of, h)
        if k not in base_cache:
            base_cache[k] = backtest.market_baseline(base_index, as_of, h)
        return base_cache[k]

    def _evaluate(use_earnings_exit):
        rets, period_map, reasons = [], {}, []
        for e in entries:
            ed = e["earnings"] if use_earnings_exit else None
            r, reason = _exit_box_with_earnings(
                e["path_dates"], e["path"], 0.15, -0.07, earnings_date=ed)
            rets.append(r); reasons.append(reason)
            period_map.setdefault((e["as_of"], e["horizon"]), []).append(r)
        arr = np.array(rets)
        excess = []
        for (as_of, h), rs in period_map.items():
            b = _base(as_of, h)
            if pd.notna(b):
                excess.append(np.median(rs) - b)
        rc = pd.Series(reasons).value_counts()
        return {
            "取引数": len(arr),
            "勝率%": round((arr > 0).mean() * 100, 1),
            "平均%": round(arr.mean() * 100, 2),
            "中央%": round(np.median(arr) * 100, 2),
            "対相場超過%": round(np.median(excess) * 100, 2) if excess else None,
            "決算前手仕舞い率": round((pd.Series(reasons) == "決算前手仕舞い").mean() * 100, 1),
            "reasons": rc,
        }

    normal = _evaluate(False)
    with_exit = _evaluate(True)
    return normal, with_exit


def summarize(normal, with_exit):
    L = ["=" * 70,
         "決算前手仕舞いの効果（ボックス反発・決算1営業日前に手仕舞い）",
         "=" * 70, ""]
    df = pd.DataFrame([
        {"ルール": "通常（決算をまたぐ）", **{k: normal[k] for k in
         ["取引数", "勝率%", "平均%", "中央%", "対相場超過%"]}},
        {"ルール": "決算前手仕舞い", **{k: with_exit[k] for k in
         ["取引数", "勝率%", "平均%", "中央%", "対相場超過%"]}},
    ])
    L.append(df.to_string(index=False))
    L.append("")
    L.append(f"決算前手仕舞いが発生した割合: {with_exit['決算前手仕舞い率']}%")
    L.append("")
    L.append("【決算前手仕舞いルールでの決済理由内訳】")
    total = with_exit["reasons"].sum()
    for reason, cnt in with_exit["reasons"].items():
        L.append(f"  {reason:<14} : {cnt/total*100:5.1f}%  (n={cnt:,})")
    L += ["", "=" * 70, "読み方", "=" * 70,
          "・『対相場超過%』と『勝率%』が改善していれば、決算回避が有効。",
          "・特に『中央値%』の安定（マイナス幅の縮小）に注目。",
          "  決算ギャンブルを避けると、大きな外れが減るはず。",
          "・決算前手仕舞い率が低すぎる場合、そもそも保有中に決算をまたぐ",
          "  ケースが少ない（保有30日 vs 決算90日間隔のため）。",
          "・決算日は『直近開示+90日』の推定。実際の予定日とはズレる。",
          "・有効なら分割検証へ。手数料・スリッページ未考慮。"]
    return "\n".join(L)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    from sources import JQuants, JST
    jq = JQuants()
    print("キャッシュからデータを読み込み中...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    fin = jq.financials(config.BACKTEST_LOOKBACK_DAYS)
    listed = jq.listed()
    print(f"  株価 {len(quotes):,}行 / 決算 {len(fin):,}行 / 銘柄 {len(listed):,}")

    normal, with_exit = run(quotes, fin, listed)
    report = summarize(normal, with_exit)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    path = os.path.join(config.OUTPUT_DIR, f"{today}_preearnings.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nレポート: {path}")


if __name__ == "__main__":
    main()
