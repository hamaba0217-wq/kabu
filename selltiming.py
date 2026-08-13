# -*- coding: utf-8 -*-
"""
売り方の最適化：どの利確ラインが平均リターン・勝率が高いか

profit_targets は「利確ラインへの到達率」だけを見た。
これは「指値がどれくらい約定するか」で、届かなかった時の損失が入っていない。

このモジュールは、届かなかった場合も含めて決済し、
利確ライン × 損切りライン × 保有日数 ごとの
・平均リターン（トータル・損切りや期限切れ込み）
・勝率
を出す。「どこで売るのが一番いいか」の答えを出すのが目的。

決済ルール（運用に直結）
------------------------
エントリー後、保有N営業日以内で：
  ・利確ライン +TP% に到達 → +TP% で利確
  ・損切りライン -SL% に到達 → -SL% で損切り
  ・どちらも未達で N日経過 → その日の終値で手仕舞い
（同日に両方なら損切り優先＝保守的）

対象銘柄
--------
売られすぎ（25日移動平均から下方乖離）した銘柄。
乖離の全区分と、有望だった -20〜-10% 区分の両方で見る。

先読み防止：指標は as_of 時点、値動きは as_of 以降。
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


TAKE_PROFITS = [0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.99]  # 0.99=実質利確なし
STOP_LOSSES = [-0.07, -0.10]
HOLD_DAYS = [5, 10, 15, 20]
STEP_DAYS = 5

# 売られすぎ区分（oversoldで高リターンだった区分を明示的に含める）
DEV_ZONES = [
    ("全売られすぎ(乖離マイナス全部)", -1e9, 0.0),
    ("移動平均-10〜-5乖離", -10.0, -5.0),
    ("移動平均-20〜-10乖離", -20.0, -10.0),
    ("移動平均-30〜-20乖離", -30.0, -20.0),
    ("移動平均-30以下乖離", -1e9, -30.0),
]


def _simulate(path, tp, sl):
    """保有期間内で利確/損切り/期限切れを判定。"""
    for px in path:
        ret = px - 1.0
        if ret <= sl:
            return sl
        if ret >= tp:
            return tp
    return path[-1] - 1.0 if len(path) else 0.0


def run(quotes, listed, step_days=STEP_DAYS, stop_loss=-0.10):
    qi = add_oversold_indicators(quotes)
    by_code = technical._build_index(qi)
    all_dates = np.sort(qi["date"].unique())

    print("  売られすぎ銘柄のエントリーを収集中...")
    max_hold = max(HOLD_DAYS)
    entries = []
    start_idx, end_idx = 260, len(all_dates) - max_hold
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
            paths = {}
            for hd in HOLD_DAYS:
                fut = close_arr[pos:pos + hd]
                if len(fut) >= hd * 0.5:
                    paths[hd] = fut / entry
            if paths:
                entries.append({"dev": dev, "paths": paths})
    print(f"  サンプル: {len(entries):,} 件")

    def _avg(subset, hd, tp, sl):
        rets = []
        for e in subset:
            p = e["paths"].get(hd)
            if p is not None:
                rets.append(_simulate(p, tp, sl))
        if len(rets) < 30:
            return None, None, 0
        arr = np.array(rets)
        return arr.mean() * 100, (arr > 0).mean() * 100, len(arr)

    # 売られすぎ区分ごとに、保有期間×利確ラインの平均リターン表を作る
    results = {}
    for zname, lo, hi in DEV_ZONES:
        subset = [e for e in entries if lo <= e["dev"] < hi] if hi < 0 else \
                 [e for e in entries if lo <= e["dev"] <= hi]
        # 平均リターン表（行=利確、列=保有期間）
        avg_rows, wr_rows = [], []
        for tp in TAKE_PROFITS:
            avg_row = {"利確%": ("なし" if tp >= 0.99 else f"+{int(tp*100)}%")}
            wr_row = {"利確%": ("なし" if tp >= 0.99 else f"+{int(tp*100)}%")}
            for hd in HOLD_DAYS:
                avg, wr, n = _avg(subset, hd, tp, stop_loss)
                avg_row[f"{hd}日"] = avg
                wr_row[f"{hd}日"] = wr
            avg_rows.append(avg_row)
            wr_rows.append(wr_row)
        n_total = len(subset)
        results[zname] = {
            "avg": pd.DataFrame(avg_rows),
            "wr": pd.DataFrame(wr_rows),
            "n": n_total,
        }
    return results, stop_loss


def _fmt(v):
    return f"{v:>6.2f}" if v is not None and not pd.isna(v) else "    --"


def summarize(results, stop_loss):
    results_d, _ = results if isinstance(results, tuple) else (results, stop_loss)
    L = ["=" * 78,
         f"売り方の最適化：保有期間 × 利確ライン（損切り{int(stop_loss*100)}%固定）",
         "=" * 78, ""]
    L.append("各区分で、行=利確ライン・列=保有期間 の平均リターン(%)を出す。")
    L.append("『利確なし』= 利確せず保有期間の最終値。利確・損切り・期限切れ込みの")
    L.append("トータル平均リターン。どちらが効くか（保有を延ばす vs 利確を上げる）が分かる。")
    L.append("")

    for zname, d in results_d.items():
        if d["n"] < 30:
            continue
        L.append("=" * 78)
        L.append(f"【{zname}】 サンプル {d['n']:,} 件")
        L.append("=" * 78)
        L.append("■ 平均リターン(%)  （行=利確 / 列=保有期間）")
        avg = d["avg"]
        hd_cols = [c for c in avg.columns if c != "利確%"]
        header = "    利確\\保有 " + "".join(f"{c:>8}" for c in hd_cols)
        L.append(header)
        for _, row in avg.iterrows():
            line = f"    {row['利確%']:>8} " + "".join(_fmt(row[c]) + "  " for c in hd_cols)
            L.append(line)
        # 最高セルを特定
        best_val, best_pos = -1e9, None
        for _, row in avg.iterrows():
            for c in hd_cols:
                if row[c] is not None and not pd.isna(row[c]) and row[c] > best_val:
                    best_val = row[c]; best_pos = (row["利確%"], c)
        if best_pos:
            L.append(f"    → 平均最高: 利確{best_pos[0]} × 保有{best_pos[1]} = {best_val:.2f}%")
        L.append("")

    L += ["=" * 78, "この表の読み方（保有期間 vs 利確ライン）", "=" * 78,
          "・横に見る（同じ利確で保有を延ばす）と、保有期間の効果が分かる。",
          "・縦に見る（同じ保有で利確を上げる）と、利確ラインの効果が分かる。",
          "・数字が大きく動くほうが、平均リターンへの影響が大きい＝効く要因。",
          "・『利確なし』の行が高ければ、利確せず持ち切るほうが良いということ。",
          "  逆に利確ありが高ければ、頭打ちさせたほうが良い（ダマシ上げを避ける）。",
          "・売られすぎ区分を上下に比べ、深い乖離ほどリターンが高いか確認。",
          "  oversoldで見た-20〜-10や-30〜-20が、実際の売買でも高いか。",
          "・これはトータル平均で、対相場超過ではない。手数料・滑り未考慮。"]
    return "\n".join(L)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    from sources import JQuants, JST
    jq = JQuants()
    print("キャッシュからデータを読み込み中...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    listed = jq.listed()
    print(f"  株価 {len(quotes):,}行 / 銘柄 {len(listed):,}")

    results, sl = run(quotes, listed)
    report = summarize(results, sl)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    frames = []
    for zname, d in results.items():
        avg = d["avg"].copy(); avg["区分"] = zname; avg["指標"] = "平均リターン"
        wr = d["wr"].copy(); wr["区分"] = zname; wr["指標"] = "勝率"
        frames.extend([avg, wr])
    if frames:
        pd.concat(frames).to_csv(
            os.path.join(config.OUTPUT_DIR, f"{today}_selltiming.csv"),
            index=False, encoding="utf-8-sig")
        print(f"\n比較表: output/{today}_selltiming.csv")


if __name__ == "__main__":
    main()
