# -*- coding: utf-8 -*-
"""
1年（252営業日）以内に5倍(+400%)になった銘柄を抽出する。

定義（事実にもとづく明確な基準）
--------------------------------
  ある起点日の終値を基準に、その後252営業日以内の最高終値が
  基準の5倍以上になった銘柄を「5倍株」とする。
  各銘柄について、最も達成が早かった1回を記録する。

出力
----
  銘柄・業種・起点日・5倍到達日・かかった日数・起点株価・ピーク株価・倍率

注意
----
  ・終値ベース。ザラ場の瞬間高値は見ない。
  ・キャッシュにある期間（約2年強）内での検出。それ以前は対象外。
  ・株式分割の調整はJ-Quantsの調整後終値に依存（AdjustmentClose）。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import config

WINDOW = 252          # 1年（営業日）
MULTIPLE = 5.0        # 5倍
MIN_PRICE = 50        # 起点株価の下限（低位すぎる株の乱高下を除外）


def find(quotes, listed):
    sec_name = {}
    name_map = {}
    if "sector33" in listed.columns:
        for _, r in listed.iterrows():
            sec_name[str(r["code"])] = r["sector33"]
    if "company_name" in listed.columns:
        for _, r in listed.iterrows():
            name_map[str(r["code"])] = r["company_name"]

    rows = []
    for code, g in quotes.groupby("code"):
        g = g.sort_values("date")
        closes = g["close"].values
        dates = g["date"].values
        n = len(closes)
        if n < 30:
            continue
        best = None
        for i in range(n):
            entry = closes[i]
            if not entry or entry < MIN_PRICE:
                continue
            # i以降 WINDOW日以内の最高値
            hi_end = min(i + WINDOW + 1, n)
            window = closes[i + 1:hi_end]
            if len(window) == 0:
                continue
            peak = window.max()
            if peak >= entry * MULTIPLE:
                # 5倍到達した最初の日
                peak_rel = np.argmax(window >= entry * MULTIPLE)
                peak_idx = i + 1 + peak_rel
                days = peak_idx - i
                mult = closes[peak_idx] / entry
                if best is None or days < best["かかった日数"]:
                    best = {
                        "code": str(code),
                        "銘柄名": name_map.get(str(code), ""),
                        "業種": sec_name.get(str(code), ""),
                        "起点日": pd.Timestamp(dates[i]).date(),
                        "5倍到達日": pd.Timestamp(dates[peak_idx]).date(),
                        "かかった日数": int(days),
                        "起点株価": round(float(entry), 1),
                        "到達株価": round(float(closes[peak_idx]), 1),
                        "倍率": round(float(mult), 2),
                    }
        if best is not None:
            rows.append(best)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("倍率", ascending=False).reset_index(drop=True)
    return df


def _attach_precursors(df, quotes, fin, listed, margin, short_ratio):
    """各5倍株に、起点日直前の指標を付ける。precursorの測定ロジックを流用。"""
    import precursor as pc
    from badnews import _prep_fin_extended

    fin_ext = _prep_fin_extended(fin)
    if "operating_profit" in fin_ext.columns:
        fin_ext = fin_ext.sort_values(["code", "disclosed_date"])
        fin_ext["op_prev"] = fin_ext.groupby(["code", "period_type"])["operating_profit"].shift(1)

    sec_code = {}
    if "sector33_code" in listed.columns:
        for _, r in listed.iterrows():
            sec_code[str(r["code"])] = str(r["sector33_code"])

    qv = quotes.sort_values(["code", "date"])
    tv_by_code = {c: (g["turnover_value"].values, g["date"].values)
                  for c, g in qv.groupby("code")}
    margin_by_code = {}
    if margin is not None and len(margin):
        m = margin.copy(); m["date"] = pd.to_datetime(m["date"])
        for c in ("long_margin", "short_margin"):
            if c in m.columns:
                m[c] = pd.to_numeric(m[c], errors="coerce")
        margin_by_code = {str(c): g.sort_values("date") for c, g in m.groupby("code")}
    sr_by_sec = {}
    if short_ratio is not None and len(short_ratio):
        sr = short_ratio.copy(); sr["date"] = pd.to_datetime(sr["date"])
        sr_by_sec = {str(s): g.sort_values("date") for s, g in sr.groupby("sector33")}
    LAG = 7

    def _latest(g, as_of, col):
        cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=LAG)
        past = g[g["date"] <= cutoff]
        if past.empty:
            return None
        v = past.iloc[-1].get(col)
        return float(v) if pd.notna(v) else None

    sales_g, op_v, mr_v, sr_v, tv_v = [], [], [], [], []
    for _, r in df.iterrows():
        code = str(r["code"]); as_of = r["起点日"]
        sg, op = pc._fin_growth(fin_ext, code, pd.Timestamp(as_of))
        sales_g.append(round(sg, 1) if sg is not None else None)
        op_v.append(op)
        mr = None
        g = margin_by_code.get(code)
        if g is not None:
            lm = _latest(g, as_of, "long_margin"); sm = _latest(g, as_of, "short_margin")
            if lm and sm and sm > 0:
                mr = round(lm / sm, 2)
        mr_v.append(mr)
        srt = None
        scode = sec_code.get(code)
        if scode and scode in sr_by_sec:
            v = _latest(sr_by_sec[scode], as_of, "short_ratio")
            srt = round(v, 3) if v is not None else None
        sr_v.append(srt)
        arrs = tv_by_code.get(code); tvr = None
        if arrs is not None:
            tvvals, dates = arrs
            idx = np.searchsorted(dates, np.datetime64(pd.Timestamp(as_of)))
            r2 = pc._turnover_ratio(tvvals, idx) if 0 < idx < len(tvvals) else None
            tvr = round(r2, 2) if r2 is not None else None
        tv_v.append(tvr)

    df = df.copy()
    df["起点前_売上前年比%"] = sales_g
    df["起点前_営業利益"] = op_v
    df["起点前_信用倍率"] = mr_v
    df["起点前_業種空売り比率"] = sr_v
    df["起点前_売買代金比"] = tv_v
    return df


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    from sources import JQuants, JST
    jq = JQuants()
    print("キャッシュからデータを読み込み中...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    listed = jq.listed()
    print(f"  株価 {len(quotes):,}行 / 銘柄 {len(listed):,}")
    # 分割調整が効いているかの確認（adj_factor列が株価データにあるか）
    if "adj_factor" not in quotes.columns:
        print("\n  [!] 注意：この株価キャッシュには分割調整情報(adj_factor)がありません。")
        print("      分割・併合が調整されず、見かけ上の異常な倍率が出る可能性があります。")
        print("      正しい結果を得るには、株価キャッシュを削除して再取得してください：")
        print("        del data\\quotes.pkl  （PowerShell）")
        print("      その後もう一度このコマンドを実行してください。")
        print("")
    print(f"\n1年({WINDOW}営業日)以内に {MULTIPLE:.0f}倍 になった銘柄を探しています...")

    df = find(quotes, listed)
    print(f"\n{'='*88}")
    print(f"1年以内に{MULTIPLE:.0f}倍になった銘柄  （倍率が高い順）")
    print(f"{'='*88}")
    if df.empty:
        print("該当する銘柄はありませんでした。")
        return
    print(f"該当 {len(df)} 件\n")
    print(df.to_string(index=False))

    # 業種別の集計
    print(f"\n{'-'*88}")
    print("業種別の件数（5倍株がどの業種に多いか）:")
    sec_counts = df["業種"].value_counts()
    for sec, cnt in sec_counts.items():
        print(f"  {sec}: {cnt}件")

    # 各5倍株に、起点日直前の前兆データ（売上前年比・信用倍率・空売り比率・
    # 売買代金比）を付けた詳細版を作る。傾向分析用。
    print(f"\n{'-'*88}")
    print("各銘柄に起点日直前の指標を付与しています（傾向分析用）...")
    try:
        fin = jq.financials(config.BACKTEST_LOOKBACK_DAYS)
        try:
            margin = jq.margin(config.BACKTEST_LOOKBACK_DAYS)
        except Exception:
            margin = None
        try:
            short_ratio = jq.short_ratio(config.BACKTEST_LOOKBACK_DAYS)
        except Exception:
            short_ratio = None
        df = _attach_precursors(df, quotes, fin, listed, margin, short_ratio)
        print("  付与完了（売上前年比・営業利益・信用倍率・業種空売り比率・売買代金比）")
    except Exception as e:
        print(f"  [!] 前兆データの付与に失敗（基本情報のみ出力）: {e}")

    today = dt.datetime.now(JST).date().isoformat()
    path = os.path.join(config.OUTPUT_DIR, f"{today}_fivebagger.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nCSV: {path}")
    print("\n※終値ベース・キャッシュ期間内での検出。株式分割は調整後終値に依存。")


if __name__ == "__main__":
    main()
