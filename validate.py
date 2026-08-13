# -*- coding: utf-8 -*-
"""
分割検証（アウトオブサンプル検証）

「押し目ゾーン + 高勝率業種」が本物か、この2年にたまたま合っただけかを
見極める。過去データを前半・後半に分け、両方で成績を出す。

考え方
------
過去2年を、日付で前半・後半に2分割する。
・前半（古い1年）: この期間だけで各組み合わせの成績を測る
・後半（新しい1年）: 同じ組み合わせを、前半とは独立に測る

もし「押し目+高勝率業種」が前半・後半どちらでも超過プラスなら、
特定の期間に依存しない＝本物の可能性が高い。
前半だけプラスで後半マイナスなら、過去に合っただけの偶然。

これは機械学習でいう「訓練/テスト分割」に相当する、
過剰最適化を見破る標準的な方法。

比較する組み合わせ
------------------
最良候補（押し目+高勝率業種）と、その構成要素を並べ、
全期間・前半・後半で横比較する。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import backtest
import config
import technical
from refine import (_f_pullback, _f_high_win_sector, _f_mild_down_1m,
                    HIGH_WIN_SECTORS)


# 検証する組み合わせ: (ラベル, 条件関数のリスト)
COMBOS = [
    ("押し目のみ",            [_f_pullback]),
    ("押し目+高勝率業種",     [_f_pullback, _f_high_win_sector]),
    ("押し目+高勝率+直近下げ", [_f_pullback, _f_high_win_sector, _f_mild_down_1m]),
]


def _collect(quotes, fin, listed, step_days=7):
    """全エントリー候補を、日付・特徴つきで集める（分割前）。"""
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
            snap_i = snap.set_index("code")
            for code in codes:
                if code not in snap_i.index:
                    continue
                row = snap_i.loc[code].copy()
                row["sector33"] = sec.get(code)
                path, ma_path = technical._forward(by_code, code, as_of, horizon)
                if path is None:
                    continue
                if spec["exit"] == "box":
                    r, _ = technical._exit_box(path)
                else:
                    r, _ = technical._exit_trend(path, ma_path)
                entries.append({"as_of": pd.Timestamp(as_of), "horizon": horizon,
                                "ret": r, "row": row})
    return entries, base_index


def _evaluate(entries, base_index, combo_funcs, date_lo=None, date_hi=None,
              min_trades=50):
    """指定期間のエントリーに組み合わせを適用し、成績を返す。"""
    base_cache = {}
    def _base(as_of, horizon):
        k = (as_of, horizon)
        if k not in base_cache:
            base_cache[k] = backtest.market_baseline(base_index, as_of, horizon)
        return base_cache[k]

    rets, period_map = [], {}
    for e in entries:
        d = e["as_of"]
        if date_lo is not None and d < date_lo:
            continue
        if date_hi is not None and d >= date_hi:
            continue
        if all(f(e["row"]) for f in combo_funcs):
            rets.append(e["ret"])
            period_map.setdefault((d, e["horizon"]), []).append(e["ret"])

    if len(rets) < min_trades:
        return {"取引数": len(rets), "平均%": None, "勝率%": None,
                "対相場超過%": None, "勝った回": "件数不足"}

    excess = []
    for (as_of, horizon), rs in period_map.items():
        b = _base(as_of, horizon)
        if pd.notna(b):
            excess.append(np.median(rs) - b)
    arr = np.array(rets); ex = np.array(excess)
    return {
        "取引数": len(arr),
        "平均%": round(arr.mean() * 100, 2),
        "勝率%": round((arr > 0).mean() * 100, 1),
        "対相場超過%": round(np.median(ex) * 100, 2) if len(ex) else None,
        "勝った回": f"{int((ex > 0).sum())}/{len(ex)}" if len(ex) else "—",
    }


def run(quotes, fin, listed):
    print("  全期間のエントリー候補を収集中...")
    entries, base_index = _collect(quotes, fin, listed)
    if not entries:
        return None, None
    dates = sorted(e["as_of"] for e in entries)
    lo, hi = dates[0], dates[-1]
    mid = lo + (hi - lo) / 2
    print(f"  期間: {lo.date()} 〜 {hi.date()}")
    print(f"  分割点: {mid.date()}（前半 / 後半）")

    rows = []
    for label, funcs in COMBOS:
        full = _evaluate(entries, base_index, funcs)
        first = _evaluate(entries, base_index, funcs, date_hi=mid)
        second = _evaluate(entries, base_index, funcs, date_lo=mid)
        rows.append({
            "組み合わせ": label,
            "全体_取引": full["取引数"], "全体_超過%": full["対相場超過%"],
            "前半_取引": first["取引数"], "前半_超過%": first["対相場超過%"],
            "後半_取引": second["取引数"], "後半_超過%": second["対相場超過%"],
        })
    return pd.DataFrame(rows), (lo, mid, hi)


def summarize(df, period):
    lo, mid, hi = period
    L = ["=" * 78,
         "分割検証（アウトオブサンプル）：本物か、過去に合っただけかを見極める",
         "=" * 78, "",
         f"前半: {lo.date()} 〜 {mid.date()}   後半: {mid.date()} 〜 {hi.date()}", ""]
    L.append(df.to_string(index=False))
    L += ["", "=" * 78, "判定の仕方", "=" * 78,
          "・前半・後半の『超過%』が両方プラス → 期間に依存しない＝本物の可能性大。",
          "・前半だけプラスで後半マイナス → この2年前半にたまたま合っただけ。",
          "・両方マイナス → そもそも効いていない。",
          "・取引数が『件数不足』の期は、分割で母数が減ったため参考外。",
          "",
          "・最重要: 『押し目+高勝率業種』が前半・後半どちらでも超過プラスなら、",
          "  今日の探索で見つけた唯一の、再現性のある手がかりになる。",
          "・ただし超過はわずか。手数料を引くと消える水準であることは変わらない。",
          "・これは2分割の簡易検証。より厳密には、複数の期間・別の年でも要確認。"]
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

    df, period = run(quotes, fin, listed)
    if df is None:
        print("データ不足で検証できませんでした。")
        return
    report = summarize(df, period)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    path = os.path.join(config.OUTPUT_DIR, f"{today}_validate.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n比較表: {path}")


if __name__ == "__main__":
    main()
