# -*- coding: utf-8 -*-
"""
③④ 需給（信用残・空売り比率・投資部門別）× 売られすぎ の検証

機関投資家フレームワークの「③資金フロー・④需給」を、
Standardプランで取れる3種のデータで検証する。

検証する需給指標
----------------
  ③信用残（銘柄別・週次）:
    信用買い残 … 多いと「戻り売り圧力」。売られすぎても戻りにくい可能性
    信用倍率 = 買い残 / 売り残 … 高いと買い方に傾き、上値が重い
  ④業種別空売り比率:
    空売り比率が高い業種 … 踏み上げ期待 or 弱気。反発の燃料になるか
  ④投資部門別:
    海外勢の売買動向 … 相場を動かす主体（今回は参考表示）

土台
----
売られすぎ（移動平均-20〜-10乖離・利確なし・保有15日・損切-10%）に、
需給条件を足して勝率・対相場超過が上がるかを見る。

先読み防止：需給データは as_of 以前の直近公表値を使う。値動きは as_of 以降。
週次データは公表遅延（週第2営業日）を考慮し、as_of の1週間前までの値を使う。
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


HOLD = 15
STOP_LOSS = -0.10
STEP_DAYS = 5
DEV_MIN, DEV_MAX = -20.0, -10.0
MARGIN_LAG_DAYS = 7      # 信用残は週次・公表遅延あり。7日前までの値を使う（先読み防止）


def _latest_margin_before(margin_by_code, code, as_of):
    """as_of の MARGIN_LAG_DAYS 日前までに公表された、直近の信用残。"""
    g = margin_by_code.get(code)
    if g is None:
        return None
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=MARGIN_LAG_DAYS)
    past = g[g["date"] <= cutoff]
    if past.empty:
        return None
    return past.iloc[-1]


def run(quotes, margin, listed, step_days=STEP_DAYS):
    qi = add_oversold_indicators(quotes)
    by_code = technical._build_index(qi)
    all_dates = np.sort(qi["date"].unique())
    base_index = {c: (v[0], v[1]) for c, v in by_code.items()}

    # 信用残を銘柄別に整理（信用倍率も計算）
    m = margin.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["code", "date"])
    # 列の存在を確認（実データの列名ゆらぎに対応）
    if "long_margin" not in m.columns or "short_margin" not in m.columns:
        print(f"  [!] 信用残に必要な列がありません。実際の列: {list(m.columns)}")
        print("      long_margin（信用買い残）/ short_margin（信用売り残）が必要です。")
        raise SystemExit("信用残の列名が想定と異なります。sources.py の MARGIN_COLUMNS を確認してください。")
    for c in ("long_margin", "short_margin"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["margin_ratio"] = np.where((m["short_margin"].notna()) & (m["short_margin"] > 0),
                                 m["long_margin"] / m["short_margin"], np.nan)
    margin_by_code = {c: g for c, g in m.groupby("code")}

    print("  売られすぎ銘柄を収集し、信用残を紐付け中...")
    entries = []
    start_idx, end_idx = 260, len(all_dates) - HOLD
    for as_of in all_dates[start_idx:end_idx:step_days]:
        snap = qi[qi["date"] == as_of]
        if snap.empty:
            continue
        sub = snap[(snap["ma25_dev"] >= DEV_MIN) & (snap["ma25_dev"] <= DEV_MAX)]
        for _, s in sub.iterrows():
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
            path = fut / entry
            ret = STOP_LOSS
            hit = False
            for px in path:
                if px - 1.0 <= STOP_LOSS:
                    ret = STOP_LOSS; hit = True; break
            if not hit:
                ret = path[-1] - 1.0

            mrow = _latest_margin_before(margin_by_code, code, as_of)
            margin_ratio = mrow["margin_ratio"] if mrow is not None else np.nan
            # 信用買い残の対出来高（株数）。出来高で正規化して銘柄横断で比較可能に
            long_margin = mrow["long_margin"] if mrow is not None else np.nan

            entries.append({"as_of": pd.Timestamp(as_of), "ret": ret,
                            "margin_ratio": margin_ratio,
                            "long_margin": long_margin})
    print(f"  サンプル: {len(entries):,} 件")
    have = sum(1 for e in entries if pd.notna(e["margin_ratio"]))
    print(f"  信用残を紐付けできた: {have:,} 件")

    base_cache = {}
    def _base(as_of):
        if as_of not in base_cache:
            base_cache[as_of] = backtest.market_baseline(base_index, as_of, HOLD)
        return base_cache[as_of]

    def _eval(pred, label):
        rets, excess = [], []
        for e in entries:
            if not pred(e):
                continue
            rets.append(e["ret"])
            b = _base(e["as_of"])
            if pd.notna(b):
                excess.append(e["ret"] - b)
        if len(rets) < 20:
            return {"条件": label, "件数": len(rets), "勝率%": None,
                    "平均%": None, "対相場超過%": None}
        arr = np.array(rets)
        return {"条件": label, "件数": len(arr),
                "勝率%": round((arr > 0).mean() * 100, 1),
                "平均%": round(arr.mean() * 100, 2),
                "対相場超過%": round(np.median(excess) * 100, 2) if excess else None}

    # 信用倍率の中央値で高低を分ける
    ratios = [e["margin_ratio"] for e in entries if pd.notna(e["margin_ratio"])]
    med = float(np.median(ratios)) if ratios else 1.0

    rows = [
        _eval(lambda e: True, "売られすぎのみ(基準)"),
        _eval(lambda e: pd.notna(e["margin_ratio"]) and e["margin_ratio"] < med,
              f"＋信用倍率<中央値{med:.1f}(売り方多め=戻り軽い)"),
        _eval(lambda e: pd.notna(e["margin_ratio"]) and e["margin_ratio"] >= med,
              f"＋信用倍率≥中央値{med:.1f}(買い方多め=戻り重い)"),
    ]
    return pd.DataFrame(rows), med


def summarize(df, med):
    L = ["=" * 74,
         "③ 信用残（信用倍率）× 売られすぎ の検証",
         "=" * 74, ""]
    L.append("土台＝売られすぎ（移動平均-20〜-10乖離・利確なし・保有15日・損切-10%）。")
    L.append("信用倍率 = 信用買い残 ÷ 信用売り残。")
    L.append("高い＝買い方に偏り＝戻り売り圧力が重い、という仮説を検証する。")
    L.append("")
    L.append(df.to_string(index=False))
    L += ["", "=" * 74, "読み方", "=" * 74,
          "・仮説：信用買い残が多い（信用倍率が高い）銘柄は、売られすぎても",
          "  戻り売りに押されて反発しにくい。逆に売り方が多いと踏み上げで戻りやすい。",
          "・『信用倍率<中央値』が基準より勝率・超過が高ければ、仮説が正しい。",
          "  ＝需給の軽い（売り方が多い）売られすぎ銘柄を選ぶべき。",
          "・信用残は週次・公表遅延ありのため、7日前までの値を使用（先読み防止）。",
          "・有効なら分割検証へ。手数料・スリッページ未考慮。"]
    return "\n".join(L)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    from sources import JQuants, JST
    jq = JQuants()
    print("キャッシュからデータを読み込み中...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    listed = jq.listed()
    print(f"  株価 {len(quotes):,}行 / 銘柄 {len(listed):,}")
    print("\n信用取引週末残高を取得します（Standardプラン）...")
    margin = jq.margin(config.BACKTEST_LOOKBACK_DAYS)
    print(f"  信用残 {len(margin):,}行  実際の列: {list(margin.columns)}")

    if margin.empty:
        print("\n  [!] 信用残データが取得できませんでした。")
        print("      Standardプランの契約状況と、APIキーをご確認ください。")
        return

    df, med = run(quotes, margin, listed)
    report = summarize(df, med)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    path = os.path.join(config.OUTPUT_DIR, f"{today}_supply_demand.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n比較表: {path}")


if __name__ == "__main__":
    main()
