# -*- coding: utf-8 -*-
"""
エントリーフィルターの検証（1つずつ足して確かめる）

勝敗要因分析（analyze_trades.py）で見つかった「勝ちやすい条件」を、
実際にエントリーフィルターとして足したとき、成績が改善するかを検証します。

最重要の原則：1つずつ足す
--------------------------
複数のフィルターを同時に足すと、過去データに合わせ込むだけ（過剰最適化）で、
どれが効いたのかも分からなくなります。前回の教訓です。
このモジュールは「フィルターなし（基準）」と「フィルター1つだけ」を並べ、
差分だけを見ます。

検証するフィルター（analyze_trades の発見にもとづく）
----------------------------------------------------
  F0: なし（基準）
  F1: 上昇相場の日を除外   … 相場中央値が +1% 以上の日は見送る
  F2: 高値から-15〜-5%限定 … 適度な押し目のゾーンだけ
  F3: 急騰後を除外         … 直近1か月で +10% 以上なら見送る

各フィルターは基準(F0)からの差分で評価する。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import backtest
import technical
import config


# フィルター定義: 名前 → 判定関数
# 判定関数は (snap_row, market_dir) を受け取り、True なら「エントリー可」
def _f_none(row, mkt):
    return True


def _f_no_up_market(row, mkt):
    """F1: 相場が大きく上昇している日(+1%以上)は見送る。"""
    if pd.isna(mkt):
        return True
    return mkt < 0.01


def _f_pullback_zone(row, mkt):
    """F2: 52週高値から -15〜-5% のゾーンだけ。"""
    p = row.get("pct_from_high")
    if pd.isna(p):
        return False          # 位置が分からないものは通さない
    return -0.15 <= p <= -0.05


def _f_no_overheated(row, mkt):
    """F3: 直近1か月で +10% 以上上げた銘柄は見送る。"""
    r = row.get("ret_1m")
    if pd.isna(r):
        return True
    return r < 0.10


FILTERS = [
    ("F0_なし(基準)",        _f_none),
    ("F1_上昇相場を除外",    _f_no_up_market),
    ("F2_押し目ゾーン限定",  _f_pullback_zone),
    ("F3_急騰後を除外",      _f_no_overheated),
]


def _add_extra_indicators(qi):
    """フィルター判定に必要な指標を足す。"""
    g = qi.groupby("code")
    if "ret_1m" not in qi.columns:
        qi["ret_1m"] = g["close"].transform(lambda s: s / s.shift(20) - 1.0)
    if "pct_from_high" not in qi.columns:
        h = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
        qi["pct_from_high"] = qi["close"] / h - 1.0
    return qi


def run(quotes, fin, listed, step_days=10):
    qi = technical.add_indicators(quotes)
    qi = _add_extra_indicators(qi)
    by_code = technical._build_index(qi)
    all_dates = np.sort(qi["date"].unique())
    fin2 = technical._prep_fin(fin)

    # 相場環境（各日の全銘柄前日比中央値）
    q = qi.sort_values("date").copy()
    q["chg"] = q.groupby("code")["close"].transform(lambda s: s / s.shift(1) - 1.0)
    daily_dir = q.groupby("date")["chg"].median()

    results = []
    for fname, ffunc in FILTERS:
        print(f"\n{'='*56}\n▶ {fname}\n{'='*56}")
        rets, excess = [], []

        for name, spec in technical.STRATEGIES.items():
            horizon = spec["horizon"]
            start_idx, end_idx = 80, len(all_dates) - horizon
            if end_idx <= start_idx:
                continue
            for as_of in all_dates[start_idx:end_idx:step_days]:
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

                # ★ここでフィルターを適用（1つだけ）
                kept = []
                for code in codes:
                    if code not in snap_i.index:
                        continue
                    if ffunc(snap_i.loc[code], mkt):
                        kept.append(code)
                if not kept:
                    continue

                period = []
                for code in kept:
                    path, ma_path = technical._forward(by_code, code, as_of, horizon)
                    if path is None:
                        continue
                    if spec["exit"] == "box":
                        r, _ = technical._exit_box(path)
                    else:
                        r, _ = technical._exit_trend(path, ma_path)
                    rets.append(r); period.append(r)

                base = backtest.market_baseline(
                    {c: (v[0], v[1]) for c, v in by_code.items()}, as_of, horizon)
                if period and pd.notna(base):
                    excess.append(np.median(period) - base)

        if not rets:
            results.append({"フィルター": fname, "取引数": 0})
            continue
        arr = np.array(rets); ex = np.array(excess)
        results.append({
            "フィルター": fname,
            "取引数": len(arr),
            "平均リターン%": round(arr.mean() * 100, 2),
            "中央値%": round(np.median(arr) * 100, 2),
            "勝率%": round((arr > 0).mean() * 100, 1),
            "対相場 超過%": round(np.median(ex) * 100, 2) if len(ex) else None,
            "相場に勝った回": f"{int((ex > 0).sum())}/{len(ex)}" if len(ex) else "—",
        })

    return pd.DataFrame(results)


def summarize(df):
    L = ["=" * 74,
         "エントリーフィルター検証（基準 F0 から1つずつ足して比較）",
         "=" * 74, ""]
    L.append(df.to_string(index=False))
    L += ["", "=" * 74, "読み方", "=" * 74,
          "・F0（フィルターなし）が基準。各フィルターは1つだけ足している。",
          "・「対相場 超過%」が F0 より改善したフィルターが有効な候補。",
          "・取引数が大きく減っていないかも確認。減りすぎ＝機会損失。",
          "・改善しても、この2年での結果。別期間で再現するか要確認。",
          "・複数を同時に足すのは、1つずつの効果を確認してから。",
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

    print("\n【全組み合わせ探索】フィルター3種のon/off・8通りを一括検証します。")
    df = run_grid(quotes, fin, listed)
    report = summarize_grid(df)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    path = os.path.join(config.OUTPUT_DIR, f"{today}_filters_grid.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n比較表: {path}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 全組み合わせ探索（グリッドサーチ）
# ---------------------------------------------------------------------------

import itertools


# 組み合わせ対象のフィルター（F0は常時オフ=なしなので除く）
GRID_FILTERS = [
    ("上昇相場除外", _f_no_up_market),
    ("押し目ゾーン", _f_pullback_zone),
    ("急騰後除外",   _f_no_overheated),
]


def _passes(active_funcs, row, mkt):
    """有効な全フィルターをANDで適用。1つでもFalseなら通さない。"""
    for f in active_funcs:
        if not f(row, mkt):
            return False
    return True


def run_grid(quotes, fin, listed, step_days=10, min_trades=100):
    """全フィルターのon/off全組み合わせ(2^3=8通り)を試し、成績を一覧化する。"""
    qi = technical.add_indicators(quotes)
    qi = _add_extra_indicators(qi)
    by_code = technical._build_index(qi)
    all_dates = np.sort(qi["date"].unique())
    fin2 = technical._prep_fin(fin)

    q = qi.sort_values("date").copy()
    q["chg"] = q.groupby("code")["close"].transform(lambda s: s / s.shift(1) - 1.0)
    daily_dir = q.groupby("date")["chg"].median()
    base_index = {c: (v[0], v[1]) for c, v in by_code.items()}

    # 事前に「全エントリー候補」を1度だけ集める（各組み合わせで使い回す）
    # (as_of, code, horizon, exit種別, row, mkt) を貯める
    print("  全エントリー候補を収集中...")
    entries = []
    for name, spec in technical.STRATEGIES.items():
        horizon = spec["horizon"]
        start_idx, end_idx = 80, len(all_dates) - horizon
        if end_idx <= start_idx:
            continue
        for as_of in all_dates[start_idx:end_idx:step_days]:
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
                path, ma_path = technical._forward(by_code, code, as_of, horizon)
                if path is None:
                    continue
                if spec["exit"] == "box":
                    r, _ = technical._exit_box(path)
                else:
                    r, _ = technical._exit_trend(path, ma_path)
                entries.append({
                    "as_of": pd.Timestamp(as_of), "horizon": horizon,
                    "ret": r, "row": snap_i.loc[code], "mkt": mkt,
                })
    print(f"  候補 {len(entries):,} 件。全組み合わせを評価します。")

    # 相場ベースラインを (as_of,horizon) ごとに1度だけ計算（使い回し）
    base_cache = {}
    def _base(as_of, horizon):
        key = (as_of, horizon)
        if key not in base_cache:
            base_cache[key] = backtest.market_baseline(base_index, as_of, horizon)
        return base_cache[key]

    results = []
    n = len(GRID_FILTERS)
    for bits in itertools.product([0, 1], repeat=n):
        active = [GRID_FILTERS[i][1] for i in range(n) if bits[i]]
        label_parts = [GRID_FILTERS[i][0] for i in range(n) if bits[i]]
        label = " + ".join(label_parts) if label_parts else "なし(基準)"

        # 各エントリーにフィルター適用
        rets = []
        period_map = {}   # (as_of,horizon) -> [ret,...]  超過計算用
        for e in entries:
            if _passes(active, e["row"], e["mkt"]):
                rets.append(e["ret"])
                period_map.setdefault((e["as_of"], e["horizon"]), []).append(e["ret"])

        if len(rets) < min_trades:
            results.append({"組み合わせ": label, "取引数": len(rets),
                            "平均%": None, "勝率%": None,
                            "対相場超過%": None, "勝った回": "件数不足"})
            continue

        excess = []
        for (as_of, horizon), rs in period_map.items():
            b = _base(as_of, horizon)
            if pd.notna(b):
                excess.append(np.median(rs) - b)
        arr = np.array(rets); ex = np.array(excess)
        results.append({
            "組み合わせ": label,
            "取引数": len(arr),
            "平均%": round(arr.mean() * 100, 2),
            "勝率%": round((arr > 0).mean() * 100, 1),
            "対相場超過%": round(np.median(ex) * 100, 2) if len(ex) else None,
            "勝った回": f"{int((ex > 0).sum())}/{len(ex)}" if len(ex) else "—",
        })

    df = pd.DataFrame(results)
    # 対相場超過で降順ソート（Noneは末尾）
    df["_sort"] = df["対相場超過%"].fillna(-999)
    df = df.sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)
    return df


def summarize_grid(df):
    L = ["=" * 78,
         "全組み合わせ探索（フィルター3種のon/off・8通り）／対相場超過で降順",
         "=" * 78, ""]
    L.append(df.to_string(index=False))
    L += ["", "=" * 78, "【重要】この一覧の読み方と警告", "=" * 78,
          "・最上段が『過去2年で最も成績が良かった組み合わせ』です。",
          "  ただし、これを鵜呑みにしてはいけません。",
          "・数百通りから最良を選ぶと、たまたま過去に合っただけの偶然を",
          "  拾う危険が非常に高い（過剰最適化・多重比較問題）。",
          "・見るべきは『上位が似た傾向で固まっているか』。",
          "  例: 上位すべてに『上昇相場除外』が入っていれば、それは本物の可能性。",
          "  上位がバラバラなら、偶然の並びで信頼できない。",
          "・取引数が大きく減った組み合わせは、勝率が高くても機会損失が大きい。",
          "・次のステップ: 有望な組み合わせを、前半・後半に分けて再検証し、",
          "  『過去で良かっただけ』でないかを必ず確認すること。",
          "・手数料・スリッページ未考慮。"]
    return "\n".join(L)
