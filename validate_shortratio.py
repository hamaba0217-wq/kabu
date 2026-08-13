# -*- coding: utf-8 -*-
"""
④ 空売り比率が高い業種 × 売られすぎ の分割検証

supply-demand2 で、唯一 対相場超過がプラス（全期間+1.45%）になった
「空売り比率が高い業種の売られすぎ銘柄」を、前半・後半に分けて検証する。

全期間203件と少ないため、分割すると各期間100件前後になる。
両期間とも対相場超過プラスなら再現性あり。片方でもマイナスなら偶然の可能性。

事実にもとづく検証：
  ・supply_demand2 の収集・空売り比率紐付けロジックを再利用（推測なし）
  ・先読み防止（空売り比率は7日前までの値）
  ・対相場超過で市場と比較
  ・手数料・スリッページ未考慮
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import backtest
import config
import supply_demand2 as sd2


def _evaluate(entries, base_index, pred, date_lo=None, date_hi=None):
    base_cache = {}
    def _base(as_of):
        if as_of not in base_cache:
            base_cache[as_of] = backtest.market_baseline(base_index, as_of, sd2.HOLD)
        return base_cache[as_of]
    rets, excess = [], []
    for e in entries:
        if not pred(e):
            continue
        if date_lo is not None and e["as_of"] < date_lo:
            continue
        if date_hi is not None and e["as_of"] >= date_hi:
            continue
        rets.append(e["ret"])
        b = _base(e["as_of"])
        if pd.notna(b):
            excess.append(e["ret"] - b)
    if len(rets) < 10:
        return None
    arr = np.array(rets)
    return {
        "件数": len(arr),
        "平均%": round(arr.mean() * 100, 2),
        "勝率%": round((arr > 0).mean() * 100, 1),
        "対相場超過%": round(np.median(excess) * 100, 2) if excess else None,
    }


def run(quotes, listed, short_ratio):
    entries, base_index = sd2._collect(quotes, listed)
    # 空売り比率を紐付け（supply_demand2 と同じロジック）
    sd2.run_short_ratio(entries, base_index, short_ratio)  # entriesに short_ratio を付与

    vals = np.array([e["short_ratio"] for e in entries if pd.notna(e.get("short_ratio"))])
    if len(vals) < 40:
        print("  [!] 空売り比率の紐付けが不足。分割検証できません。")
        return None, None, None

    # 複数の閾値（上位何%を含めるか）で見る。緩めるほど母数が増える。
    # top_pct: 空売り比率の上位何%を「高い」とみなすか
    thresholds = []
    for top_pct in (30, 50, 70, 100):   # 100%=全紐付け銘柄（閾値なし）
        cut = float(np.percentile(vals, 100 - top_pct)) if top_pct < 100 else float(vals.min())
        thresholds.append((top_pct, cut))

    # 分割点は全紐付け銘柄で決める（閾値によらず同じ日付で切る）
    linked = [e for e in entries if pd.notna(e.get("short_ratio"))]
    dates = sorted(e["as_of"] for e in linked)
    mid = dates[len(dates) // 2]
    print(f"  紐付け {len(linked)}件。分割点: {mid.date()}")

    results = []
    for top_pct, cut in thresholds:
        pred = (lambda c: (lambda e: pd.notna(e.get("short_ratio")) and e["short_ratio"] >= c))(cut)
        row = {
            "閾値": ("全紐付け(閾値なし)" if top_pct == 100 else f"上位{top_pct}% (比率≥{cut:.2f})"),
            "全期間": _evaluate(entries, base_index, pred),
            "前半": _evaluate(entries, base_index, pred, date_hi=mid),
            "後半": _evaluate(entries, base_index, pred, date_lo=mid),
        }
        results.append(row)

    # 参考：基準（売られすぎのみ）
    pred_all = lambda e: True
    base = {
        "全期間": _evaluate(entries, base_index, pred_all),
        "前半": _evaluate(entries, base_index, pred_all, date_hi=mid),
        "後半": _evaluate(entries, base_index, pred_all, date_lo=mid),
    }
    return results, base, mid


def summarize(results, base, mid):
    L = ["=" * 78,
         "④ 空売り比率が高い業種 × 売られすぎ の分割検証（複数閾値）",
         "=" * 78, ""]
    L.append(f"分割点：{mid.date()}（前半＝これより前 / 後半＝これ以降）")
    L.append("土台：移動平均-20〜-10乖離・利確なし・保有15日・損切-10%")
    L.append("閾値を緩める（上位%を広げる）ほど母数が増える。")
    L.append("母数を増やしても両期間プラスが保たれるか＝条件の頑健さを見る。")
    L.append("")

    def _fmt(d):
        if d is None:
            return "（件数不足で評価不能）"
        ex = f"{d['対相場超過%']:+.2f}" if d['対相場超過%'] is not None else "--"
        return (f"件数{d['件数']:>5}  平均{d['平均%']:+6.2f}%  "
                f"勝率{d['勝率%']:4.1f}%  対相場超過{ex}%")

    for row in results:
        L.append("=" * 78)
        L.append(f"【{row['閾値']}】")
        for k in ("全期間", "前半", "後半"):
            L.append(f"  {k:<4}: {_fmt(row[k])}")
        f, s = row["前半"], row["後半"]
        if f and s and f["対相場超過%"] is not None and s["対相場超過%"] is not None:
            fe, se = f["対相場超過%"], s["対相場超過%"]
            if fe > 0 and se > 0:
                L.append(f"  判定: ◎ 両期間プラス（前{fe:+.2f}/後{se:+.2f}）")
            elif fe > 0 or se > 0:
                L.append(f"  判定: △ 片方のみプラス（前{fe:+.2f}/後{se:+.2f}）")
            else:
                L.append(f"  判定: × 両期間マイナス（前{fe:+.2f}/後{se:+.2f}）")
        L.append("")

    L.append("=" * 78)
    L.append("【参考：売られすぎのみ（基準・閾値なし）】")
    for k in ("全期間", "前半", "後半"):
        L.append(f"  {k:<4}: {_fmt(base[k])}")
    L += ["", "=" * 78, "読み方", "=" * 78,
          "・閾値を緩めて母数を増やしても両期間プラスなら、条件は頑健。",
          "・緩めると崩れる（片方マイナスになる）なら、厳しい閾値だけの偶然。",
          "・件数が各数十のうちは、偶然の振れが残る点に注意。",
          "・手数料・スリッページ未考慮。"]
    return "\n".join(L)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    from sources import JQuants, JST
    jq = JQuants()
    print("キャッシュからデータを読み込み中...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    listed = jq.listed()
    print(f"  株価 {len(quotes):,}行 / 銘柄 {len(listed):,}")
    print("空売り比率を取得します...")
    short_ratio = jq.short_ratio(config.BACKTEST_LOOKBACK_DAYS)
    print(f"  空売り比率 {len(short_ratio):,}行")

    results, base, mid = run(quotes, listed, short_ratio)
    if results is None:
        return
    report = summarize(results, base, mid)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    with open(os.path.join(config.OUTPUT_DIR, f"{today}_validate_shortratio.txt"),
              "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nレポート: output/{today}_validate_shortratio.txt")


if __name__ == "__main__":
    main()
