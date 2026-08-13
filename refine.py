# -*- coding: utf-8 -*-
"""
押し目ゾーンを軸にした絞り込み探索

グリッド探索で「押し目ゾーン（52週高値から-15〜-5%）」が唯一の
本物の手がかりと分かった。これを土台に固定し、勝敗分析で有望だった
2軸を重ねて、勝率と対相場超過が最も高まる組み合わせを探す。

重ねる軸
--------
・高勝率業種       … analyze_trades で勝率上位だった業種に限定
                     （保険・銀行・証券・倉庫運輸・水産農林など）
・直近1か月やや下げ … 直近20営業日の騰落が -10〜0%（適度に押した銘柄）

安全装置
--------
取引数が min_trades 未満の組み合わせは「件数不足」として除外。
数百トレード残る範囲でのみ、勝率・超過リターンを評価する。

※ここで出た「最良」も、別期間での再現確認（分割検証）を通すまでは
  暫定候補として扱うこと。
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


# analyze_trades（実データ）で勝率が全体平均を明確に上回った業種
HIGH_WIN_SECTORS = {
    "保険業", "銀行業", "証券･商品先物取引業", "証券・商品先物取引業",
    "倉庫･運輸関連業", "倉庫・運輸関連業", "水産・農林業", "水産･農林業",
}


def _f_pullback(row):
    p = row.get("pct_from_high")
    if pd.isna(p):
        return False
    return -0.15 <= p <= -0.05


def _f_high_win_sector(row):
    return row.get("sector33") in HIGH_WIN_SECTORS


def _f_mild_down_1m(row):
    r = row.get("ret_1m")
    if pd.isna(r):
        return False
    return -0.10 <= r < 0.0


# 重ねる軸（押し目ゾーンは常に土台として固定）
EXTRA_AXES = [
    ("高勝率業種", _f_high_win_sector),
    ("直近1M下げ", _f_mild_down_1m),
]


def run(quotes, fin, listed, step_days=10, min_trades=200):
    qi = technical.add_indicators(quotes)
    g = qi.groupby("code")
    qi["ret_1m"] = g["close"].transform(lambda s: s / s.shift(20) - 1.0)
    h = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / h - 1.0

    # 業種を紐づけ
    sec = (listed.set_index("code")["sector33"]
           if "sector33" in listed.columns else pd.Series(dtype=object))

    by_code = technical._build_index(qi)
    all_dates = np.sort(qi["date"].unique())
    fin2 = technical._prep_fin(fin)
    base_index = {c: (v[0], v[1]) for c, v in by_code.items()}

    # 全エントリー候補を収集（押し目ゾーンで既に絞る＝土台）
    print("  押し目ゾーンを土台に、エントリー候補を収集中...")
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
                # 土台：押し目ゾーンを満たさないものは最初から除外
                if not _f_pullback(row):
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
    print(f"  押し目ゾーン内の候補: {len(entries):,} 件")

    base_cache = {}
    def _base(as_of, horizon):
        k = (as_of, horizon)
        if k not in base_cache:
            base_cache[k] = backtest.market_baseline(base_index, as_of, horizon)
        return base_cache[k]

    # 追加軸の全on/off組み合わせ（2軸=4通り）
    results = []
    m = len(EXTRA_AXES)
    for bits in itertools.product([0, 1], repeat=m):
        active = [EXTRA_AXES[i][1] for i in range(m) if bits[i]]
        parts = ["押し目ゾーン"] + [EXTRA_AXES[i][0] for i in range(m) if bits[i]]
        label = " + ".join(parts)

        rets, period_map = [], {}
        for e in entries:
            if all(f(e["row"]) for f in active):
                rets.append(e["ret"])
                period_map.setdefault((e["as_of"], e["horizon"]), []).append(e["ret"])

        if len(rets) < min_trades:
            results.append({"組み合わせ": label, "取引数": len(rets),
                            "平均%": None, "勝率%": None, "対相場超過%": None,
                            "勝った回": "件数不足"})
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
    df["_s"] = df["対相場超過%"].fillna(-999)
    return df.sort_values("_s", ascending=False).drop(columns="_s").reset_index(drop=True)


def summarize(df):
    L = ["=" * 74,
         "押し目ゾーン × 絞り込み軸の探索（勝率・超過を最大化）",
         "=" * 74, ""]
    L.append(df.to_string(index=False))
    L += ["", "=" * 74, "読み方", "=" * 74,
          "・全行が『押し目ゾーン』を土台に持つ（前回グリッドで唯一有効だった軸）。",
          "・そこに高勝率業種／直近1M下げ を重ねた効果を見る。",
          "・対相場超過がプラスに転じた組み合わせがあれば、初の『市場超え』候補。",
          "・取引数が200未満は『件数不足』として評価対象外にしている。",
          "・ここで有望でも、別期間での再現確認（分割検証）を通すまでは暫定。",
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
    path = os.path.join(config.OUTPUT_DIR, f"{today}_refine.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n比較表: {path}")


if __name__ == "__main__":
    main()
