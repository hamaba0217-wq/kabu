# -*- coding: utf-8 -*-
"""
大勝ち(+10%)狙いの絞り込み探索

refine.py が「勝率・対相場超過」を最大化したのに対し、
これは視点を変えて「大勝ち率（+10%以上の割合）」を最大化する。

土台と掛け合わせる条件
----------------------
土台: 高勝率業種（保険・銀行・証券・倉庫運輸・水産農林）
      ※bigwinで大勝ち率も高かった業種群
掛け合わせ:
  ・直近1M下落     … 直近20営業日 -10%以下（bigwin ネット+19.7%）
  ・相場下落日      … エントリー日の全銘柄中央値が 0未満（bigwin ネット+18.4%）

評価軸（refineと違う点）
------------------------
・大勝ち率(+10%以上)      … 大きく取れた割合
・大負け率(-10%以下)      … 大きく損した割合
・ネット(大勝ち率-大負け率)… 損小利大が成立しているか
・平均リターン            … 総合成績
・対相場超過              … 市場に勝っているか

大勝ち狙いは勝率とトレードオフになりうる。勝率より「たまに大きく、
めったに大きく負けない」を重視する見方。
"""

from __future__ import annotations

import datetime as dt
import itertools
import os

import numpy as np
import pandas as pd

import backtest
import config
import technical
from refine import _f_high_win_sector, HIGH_WIN_SECTORS

BIG_WIN = 0.10
BIG_LOSE = -0.10


def _f_down_1m(row):
    r = row.get("ret_1m")
    if pd.isna(r):
        return False
    return r <= -0.10


def _f_down_market(row):
    m = row.get("_mkt")
    if pd.isna(m):
        return False
    return m < 0.0


EXTRA_AXES = [
    ("直近1M下落", _f_down_1m),
    ("相場下落日", _f_down_market),
]


def run(quotes, fin, listed, step_days=7, min_trades=100):
    qi = technical.add_indicators(quotes)
    g = qi.groupby("code")
    qi["ret_1m"] = g["close"].transform(lambda s: s / s.shift(20) - 1.0)
    h = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / h - 1.0
    sec = (listed.set_index("code")["sector33"]
           if "sector33" in listed.columns else pd.Series(dtype=object))

    by_code = technical._build_index(qi)
    all_dates = np.sort(qi["date"].unique())
    fin2 = technical._prep_fin(fin)
    base_index = {c: (v[0], v[1]) for c, v in by_code.items()}

    q = qi.sort_values("date").copy()
    q["chg"] = q.groupby("code")["close"].transform(lambda s: s / s.shift(1) - 1.0)
    daily_dir = q.groupby("date")["chg"].median()

    # 高勝率業種を土台に候補収集
    print("  高勝率業種を土台に、エントリー候補を収集中...")
    entries = []
    for name, spec in technical.STRATEGIES.items():
        horizon = spec["horizon"]
        s_i, e_i = 80, len(all_dates) - horizon
        if e_i <= s_i:
            continue
        for as_of in all_dates[s_i:e_i:step_days]:
            snap = technical._snapshot(qi, as_of)
            if snap.empty:
                continue
            if spec["entry"] == "breakout":
                codes = technical.entries_breakout(snap)
            elif spec["entry"] == "box":
                codes = technical.entries_box_bottom(snap)
            else:
                codes = technical.entries_earnings_momentum(snap, fin2, as_of)
            if not codes:
                continue
            mkt = daily_dir.get(pd.Timestamp(as_of), np.nan)
            snap_i = snap.set_index("code")
            for code in codes:
                if code not in snap_i.index:
                    continue
                row = snap_i.loc[code].copy()
                row["sector33"] = sec.get(code)
                row["_mkt"] = mkt
                if not _f_high_win_sector(row):    # 土台：高勝率業種
                    continue
                path, ma_path = technical._forward(by_code, code, as_of, horizon)
                if path is None:
                    continue
                if spec["exit"] == "box":
                    r, _ = technical._exit_box(path)
                else:
                    r, _ = technical._exit_trend(path, ma_path)
                entries.append({"as_of": pd.Timestamp(as_of), "horizon": horizon,
                                "ret": r, "row": row})
    print(f"  高勝率業種の候補: {len(entries):,} 件")

    base_cache = {}
    def _base(as_of, horizon):
        k = (as_of, horizon)
        if k not in base_cache:
            base_cache[k] = backtest.market_baseline(base_index, as_of, horizon)
        return base_cache[k]

    results = []
    m = len(EXTRA_AXES)
    for bits in itertools.product([0, 1], repeat=m):
        active = [EXTRA_AXES[i][1] for i in range(m) if bits[i]]
        parts = ["高勝率業種"] + [EXTRA_AXES[i][0] for i in range(m) if bits[i]]
        label = " + ".join(parts)

        rets, period_map = [], {}
        for e in entries:
            if all(f(e["row"]) for f in active):
                rets.append(e["ret"])
                period_map.setdefault((e["as_of"], e["horizon"]), []).append(e["ret"])

        if len(rets) < min_trades:
            results.append({"組み合わせ": label, "取引数": len(rets),
                            "大勝ち%": None, "大負け%": None, "ネット%": None,
                            "平均%": None, "対相場超過%": None})
            continue

        arr = np.array(rets)
        bw = (arr >= BIG_WIN).mean() * 100
        bl = (arr <= BIG_LOSE).mean() * 100
        excess = []
        for (as_of, horizon), rs in period_map.items():
            b = _base(as_of, horizon)
            if pd.notna(b):
                excess.append(np.median(rs) - b)
        results.append({
            "組み合わせ": label,
            "取引数": len(arr),
            "大勝ち%": round(bw, 1),
            "大負け%": round(bl, 1),
            "ネット%": round(bw - bl, 1),
            "平均%": round(arr.mean() * 100, 2),
            "対相場超過%": round(np.median(excess) * 100, 2) if excess else None,
        })

    df = pd.DataFrame(results)
    df["_s"] = df["ネット%"].fillna(-999)
    return df.sort_values("_s", ascending=False).drop(columns="_s").reset_index(drop=True)


def summarize(df):
    L = ["=" * 78,
         "高勝率業種 × 大勝ち条件の探索（大勝ち率=ネットを最大化）",
         "=" * 78, ""]
    L.append(df.to_string(index=False))
    L += ["", "=" * 78, "読み方", "=" * 78,
          "・全行が『高勝率業種』を土台に持つ。",
          "・『ネット%』(大勝ち率-大負け率)が大きい組み合わせが、",
          "  『たまに大きく取り、めったに大きく負けない』＝大勝ち狙いの本命。",
          "・ただし条件を足すほど取引数が減る。100未満は評価対象外。",
          "・『平均%』『対相場超過%』も併せて見る。大勝ち率が高くても",
          "  平均が低ければ、外れも多いということ。",
          "・勝率狙い(refine)とは別の路線。こちらは『大きく取る』方向。",
          "・ここで有望でも、別期間での再現確認（分割検証）が必要。",
          "・手数料・スリッページ未考慮。"]
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

    df = run(quotes, fin, listed)
    report = summarize(df)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    path = os.path.join(config.OUTPUT_DIR, f"{today}_bigwin_refine.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n比較表: {path}")


if __name__ == "__main__":
    main()
