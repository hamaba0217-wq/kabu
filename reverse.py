# -*- coding: utf-8 -*-
"""
逆算分析ツール（reverse analysis）

「3か月で+100%以上になった銘柄は、急騰する前にどんな特徴を持っていたか」を、
全上場銘柄を対象に集計します。

生存者バイアスへの対策（最重要）
--------------------------------
勝者だけを見ると、必ず間違えます。このツールは以下を守ります。

1. **勝者と敗者を同じ土俵で比較する**
   「+100%になった銘柄の平均時価総額は○億」だけでは無意味。
   「同じ時価総額帯で、+100%になった "割合"」を見る。分母に敗者を必ず入れる。

2. **急騰前の情報だけを使う**
   起点日（as_of）までに分かっていた特徴のみ。
   起点より後の株価・決算は、特徴量として一切使わない。

3. **複数の起点で集計する**
   1時点の勝者の特徴は、その時期の偶然かもしれない。
   毎月起点をずらし、どの起点でも共通して効く特徴を探す。

出力
----
- 勝者(+100%)と全体で、各特徴の分布がどう違うか
- 特徴帯ごとの「勝者になった割合」（＝リフト）
- どの特徴が勝率を押し上げているかのランキング

このツールは条件を "決める" ものではありません。
"仮説を作る" ものです。ここで見えた特徴は、必ず別の期間の
バックテストで検証してから採用してください（さもないと過剰最適化）。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import cache
import config
import screen
from sources import JQuants, JST, code4

WIN_THRESHOLD = 1.00      # +100%以上を「勝者」とする
HORIZON = 60              # 3か月 ≒ 60営業日
STEP_DAYS = 21            # 起点を1か月ずつずらす


def _forward_return_fast(close_arr, dates_arr, as_of_ts, horizon=HORIZON):
    """事前に取り出した1銘柄の終値配列から、先行リターンを計算する。

    close_arr, dates_arr は「その銘柄だけ」の昇順配列。
    全体データを毎回検索しないので桁違いに速い。
    """
    import numpy as _np
    # as_of までの最後の終値（entry）と、それ以降 horizon 本
    idx = _np.searchsorted(dates_arr, as_of_ts, side="right")
    if idx == 0:
        return None, None
    entry = close_arr[idx - 1]
    if not entry or entry <= 0:
        return None, None
    fut = close_arr[idx:idx + horizon]
    if len(fut) < horizon * 0.6:
        return None, None
    return fut.max() / entry - 1, fut[-1] / entry - 1


def build_dataset(quotes, fin, listed):
    """各起点×各銘柄について、急騰前の特徴と3か月後の結果を1行にする。"""
    q = screen.add_rolling(quotes, config.MA_WINDOW)
    dates = np.sort(q["date"].unique())
    start_i = max(config.MA_WINDOW, 30)
    end_i = len(dates) - HORIZON
    if end_i <= start_i:
        raise SystemExit("データ期間が不足しています。")
    as_of_list = dates[start_i:end_i:STEP_DAYS]

    # 【高速化の肝】銘柄ごとに終値・日付の配列を1度だけ作る。
    # これで、起点ごとに全データを検索し直す無駄をなくす。
    print(f"    銘柄ごとの株価を索引化中...（{q['code'].nunique():,}銘柄）")
    by_code = {}
    for code, g in q.sort_values("date").groupby("code"):
        by_code[code] = (g["close"].to_numpy(), g["date"].to_numpy())

    n_steps = len(as_of_list)
    print(f"    {n_steps}個の起点 × 各銘柄を評価します")

    rows = []
    for step_i, as_of in enumerate(as_of_list, 1):
        as_of_ts = np.datetime64(pd.Timestamp(as_of))
        snap = screen.snapshot_at(q, as_of)
        yoy = screen.yoy_table(fin[fin["disclosed_date"] <= pd.Timestamp(as_of)])
        if yoy.empty:
            continue
        snap = snap.merge(yoy, on="code", how="left").merge(
            listed[["code", "company_name", "market", "sector33"]],
            on="code", how="left")

        if "shares_outstanding" in snap.columns:
            snap["market_cap"] = snap["close"] * snap["shares_outstanding"]
        else:
            snap["market_cap"] = np.nan

        for _, r in snap.iterrows():
            arrs = by_code.get(r["code"])
            if arrs is None:
                continue
            mx, fin_ret = _forward_return_fast(arrs[0], arrs[1], as_of_ts)
            if mx is None:
                continue
            rows.append({
                "as_of": pd.Timestamp(as_of).date(),
                "code": r["code"],
                "company_name": r.get("company_name"),
                "sector33": r.get("sector33"),
                "market": r.get("market"),
                # --- 急騰前に分かっていた特徴 ---
                "market_cap_oku": (r["market_cap"] / 1e8
                                   if pd.notna(r["market_cap"]) else np.nan),
                "turnover_oku": r["turnover_value"] / 1e8,
                "turnover_spike": r.get("turnover_spike"),
                "op_yoy": r.get("op_yoy"),
                "sales_yoy": r.get("sales_yoy"),
                "pct_from_high": r.get("pct_from_high"),
                "price": r["close"],
                # --- 結果（分析対象。特徴には使わない）---
                "max_return": mx,
                "final_return": fin_ret,
                "is_winner": mx >= WIN_THRESHOLD,
            })
        print(f"    起点 {step_i}/{n_steps}（{pd.Timestamp(as_of).date()}） 累計 {len(rows):,}件")
    return pd.DataFrame(rows)


def _bucket_lift(df, col, bins, labels):
    """特徴を区間に分け、区間ごとの勝者割合（リフト）を出す。"""
    sub = df[df[col].notna()].copy()
    if len(sub) == 0:
        return None
    sub["_bucket"] = pd.cut(sub[col], bins=bins, labels=labels)
    base = sub["is_winner"].mean()
    g = sub.groupby("_bucket", observed=True).agg(
        件数=("is_winner", "size"),
        勝者数=("is_winner", "sum"),
        勝者割合=("is_winner", "mean"),
    ).reset_index()
    g["全体比リフト"] = g["勝者割合"] / base if base > 0 else np.nan
    g["勝者割合"] = (g["勝者割合"] * 100).round(1)
    g["全体比リフト"] = g["全体比リフト"].round(2)
    return g, base


def analyze(df):
    lines = []
    A = lines.append
    n = len(df)
    nw = int(df["is_winner"].sum())
    A("=" * 64)
    A("逆算分析：+100%以上になった銘柄の『急騰前』の特徴")
    A("=" * 64)
    A(f"対象サンプル : {n:,}件（起点×銘柄）")
    A(f"うち勝者     : {nw}件（{nw/n*100:.1f}%）= 3か月で最大+100%以上に到達")
    A("")
    A("※ 割合は『その特徴帯の銘柄が勝者になった確率』です。")
    A("  勝者の中でその特徴が多い、ではありません（生存者バイアスを避けるため）。")
    A("")

    specs = [
        ("market_cap_oku", "時価総額(億円)",
         [0, 50, 100, 300, 500, 1000, 1e9],
         ["〜50", "50-100", "100-300", "300-500", "500-1000", "1000〜"]),
        ("op_yoy", "営業利益 前年同期比(%)",
         [-1e9, 0, 30, 100, 300, 1e9],
         ["赤字/減益", "0-30", "30-100", "100-300", "300〜"]),
        ("sales_yoy", "売上高 前年同期比(%)",
         [-1e9, 0, 15, 30, 100, 1e9],
         ["減収", "0-15", "15-30", "30-100", "100〜"]),
        ("turnover_spike", "売買代金(20日平均比)",
         [0, 1, 2, 3, 5, 1e9],
         ["〜1倍", "1-2倍", "2-3倍", "3-5倍", "5倍〜"]),
        ("turnover_oku", "売買代金(億円)",
         [0, 1, 3, 5, 10, 1e9],
         ["〜1", "1-3", "3-5", "5-10", "10〜"]),
        ("pct_from_high", "52週高値からの位置(%)",
         [-1e9, -50, -30, -15, -5, 1e9],
         ["〜-50", "-50〜-30", "-30〜-15", "-15〜-5", "-5〜高値圏"]),
        ("price", "株価(円)",
         [0, 200, 500, 1000, 3000, 1e9],
         ["〜200", "200-500", "500-1000", "1000-3000", "3000〜"]),
    ]

    for col, name, bins, labels in specs:
        res = _bucket_lift(df, col, bins, labels)
        if res is None:
            continue
        g, base = res
        A(f"■ {name}")
        for _, row in g.iterrows():
            bar = "█" * int(row["全体比リフト"] * 10) if pd.notna(row["全体比リフト"]) else ""
            A(f"  {str(row['_bucket']):>12} : "
              f"勝者{row['勝者割合']:5.1f}%  "
              f"リフト{row['全体比リフト'] if pd.notna(row['全体比リフト']) else 0:4.2f}  "
              f"(n={int(row['件数']):>4}) {bar}")
        A("")

    # 業種の偏り
    A("■ 業種別の勝者割合（上位10・n≥20のみ）")
    sec = df[df["sector33"].notna()].groupby("sector33").agg(
        件数=("is_winner", "size"), 勝者割合=("is_winner", "mean")).reset_index()
    sec = sec[sec["件数"] >= 20].sort_values("勝者割合", ascending=False).head(10)
    base = df["is_winner"].mean()
    for _, row in sec.iterrows():
        lift = row["勝者割合"] / base if base > 0 else 0
        A(f"  {row['sector33']:<16} : 勝者{row['勝者割合']*100:5.1f}%  "
          f"リフト{lift:4.2f}  (n={int(row['件数'])})")
    A("")

    A("=" * 64)
    A("読み方と注意")
    A("=" * 64)
    A("・リフト1.0 = 全体平均と同じ。1.5なら平均の1.5倍勝ちやすい特徴帯。")
    A("・リフトが高い帯でも n が小さいものは偶然かもしれません。")
    A("・ここで見えた特徴は『仮説』です。条件に採用する前に、")
    A("  別の期間のバックテストで再現するか必ず確認してください。")
    A("・全部を同時に条件化すると、過去に合わせ込むだけ（過剰最適化）。")
    A("  効きそうな特徴を1つずつ足して検証するのが正道です。")
    return "\n".join(lines)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    jq = JQuants()
    print("キャッシュからデータを読み込み中...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    fin = jq.financials(config.BACKTEST_LOOKBACK_DAYS)
    listed = jq.listed()
    print(f"  株価 {len(quotes):,}行 / 決算 {len(fin):,}行 / 銘柄 {len(listed):,}\n")

    print("逆算データセットを構築中...")
    df = build_dataset(quotes, fin, listed)
    if df.empty:
        print("データが作れませんでした。")
        return
    print(f"  {len(df):,}件のサンプルを作成\n")

    report = analyze(df)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    csv_path = os.path.join(config.OUTPUT_DIR, f"{today}_reverse_data.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    rep_path = os.path.join(config.OUTPUT_DIR, f"{today}_reverse_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n詳細データ: {csv_path}")
    print(f"レポート  : {rep_path}")
    print("\n勝者だけを抽出したい場合は、CSVを is_winner=True で絞ってください。")


if __name__ == "__main__":
    main()
