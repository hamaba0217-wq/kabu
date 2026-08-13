# -*- coding: utf-8 -*-
"""
5倍株が「上がる前」に共通するサインを探す（前兆分析）。

各5倍株の起点日の直前の状態を測り、5倍にならなかった銘柄と比較する。
生存者バイアスを避けるため、必ず「5倍株 vs 非5倍株」で対比する。

見る前兆（J-Quantsで取れるもの）
--------------------------------
  ① 業種      … 5倍株はどの業種に多いか（全銘柄の業種構成と比較）
  ② 業績の伸び … 起点日直前の直近決算で、売上・営業利益が前年比プラスか
  ③ 売買代金   … 起点日前20日の平均売買代金が、その前(60日)平均の何倍か
                 （初動の出来高増加を捉える）

方法
----
  5倍株の各起点日を「陽性サンプル」とする。
  比較対照として、5倍にならなかった銘柄のランダムな時点を「陰性サンプル」とし、
  同じ3指標を測って、陽性と陰性で差があるかを見る。
  差が出れば、それが「上がる前のサイン」の候補。

注意（事実）
-----------
  ・プレスリリース数・記事数はJ-Quantsで取れないため未対応。
  ・分割調整済みの株価を使用（fivebaggerと同じ）。
  ・過去の傾向であり、将来を保証しない。手数料等は無関係（分析のみ）。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import config
import technical
from badnews import _latest_fin_before, _prep_fin_extended
from fivebagger import find as find_fivebaggers, WINDOW


VOL_RECENT = 20      # 直近の売買代金を測る日数
VOL_BASE = 60        # 平常時の売買代金を測る日数（その前）
N_NEGATIVE = 2000    # 陰性サンプル数（比較対照）


def _turnover_ratio(closes_dates_vals, idx):
    """起点idx直前20日平均売買代金 ÷ その前60日平均。初動の出来高増を測る。"""
    tv = closes_dates_vals
    if idx < VOL_RECENT + VOL_BASE:
        return None
    recent = tv[idx - VOL_RECENT:idx].mean()
    base = tv[idx - VOL_RECENT - VOL_BASE:idx - VOL_RECENT].mean()
    if base and base > 0:
        return recent / base
    return None


def _fin_growth(fin_ext, code, as_of):
    """起点日直前の直近決算で、売上・営業利益が前年比プラスか。"""
    row = _latest_fin_before(fin_ext, code, as_of)
    if row is None:
        return None, None
    sales = row.get("net_sales")
    sales_prev = row.get("sales_prev")
    op = row.get("operating_profit")
    op_prev = row.get("op_prev") if "op_prev" in row else None
    sales_growth = None
    if pd.notna(sales) and pd.notna(sales_prev) and sales_prev not in (0, None):
        sales_growth = (sales / sales_prev - 1.0) * 100
    return sales_growth, (op if pd.notna(op) else None)


def _had_upward_revision(fin_ext, code, as_of, within_days=90):
    """起点日前 within_days 日以内に、その銘柄が上方修正を開示したか。"""
    if "is_upward_revision" not in fin_ext.columns:
        return None
    g = fin_ext[(fin_ext["code"] == str(code)) &
                (fin_ext["disclosed_date"] <= pd.Timestamp(as_of)) &
                (fin_ext["disclosed_date"] >= pd.Timestamp(as_of) - pd.Timedelta(days=within_days))]
    if g.empty:
        return None   # 期間内に決算開示がない＝判定不能
    return bool(g["is_upward_revision"].any())


def analyze(quotes, fin, listed, margin=None, short_ratio=None):
    fivebaggers = find_fivebaggers(quotes, listed)
    if fivebaggers.empty:
        return None
    fb_codes = set(fivebaggers["code"])

    fin_ext = _prep_fin_extended(fin)
    # 営業利益の前年比も足す
    if "operating_profit" in fin_ext.columns:
        fin_ext = fin_ext.sort_values(["code", "disclosed_date"])
        fin_ext["op_prev"] = fin_ext.groupby(["code", "period_type"])["operating_profit"].shift(1)
    # 上方修正フラグ：会社予想営業利益が、同じ銘柄の前回開示より引き上げられたか。
    # （材料＝ニュースの一種。TDnetが取れない代わりに、業績予想の上方修正で測る）
    if "fc_operating_profit" in fin_ext.columns:
        fin_ext = fin_ext.sort_values(["code", "disclosed_date"])
        fin_ext["fc_op_prev"] = fin_ext.groupby("code")["fc_operating_profit"].shift(1)
        def _upward(row):
            cur, prev = row.get("fc_operating_profit"), row.get("fc_op_prev")
            if pd.notna(cur) and pd.notna(prev) and prev not in (0, None):
                if cur > prev * 1.03:   # 予想を3%超引き上げ＝上方修正とみなす
                    return True
            return False
        fin_ext["is_upward_revision"] = fin_ext.apply(_upward, axis=1)

    sec_name = {}
    sec_code = {}
    if "sector33" in listed.columns:
        for _, r in listed.iterrows():
            sec_name[str(r["code"])] = r["sector33"]
    if "sector33_code" in listed.columns:
        for _, r in listed.iterrows():
            sec_code[str(r["code"])] = str(r["sector33_code"])

    qv = quotes.sort_values(["code", "date"])
    tv_by_code = {c: (g["turnover_value"].values, g["date"].values)
                  for c, g in qv.groupby("code")}

    # 業種トレンド分析用：業種ごとの平均株価指数を作る。
    # 各銘柄の終値を起点で正規化し、業種内で日ごとに平均＝業種の値動き指数。
    sector_index = {}   # sector33_code -> (dates(np), index_values(np))
    qv2 = qv.copy()
    qv2["sec"] = qv2["code"].map(sec_code)
    for sec, g in qv2.groupby("sec"):
        if not sec or sec == "None":
            continue
        # 各銘柄を最初の終値で正規化してから、日付ごとに平均
        piv = g.pivot_table(index="date", columns="code", values="close", aggfunc="last")
        piv = piv / piv.ffill().bfill().iloc[0]   # 各銘柄を初日=1に正規化
        idx = piv.mean(axis=1)   # 業種平均指数
        sector_index[str(sec)] = (idx.index.values, idx.values)

    SEC_TREND_DAYS = 60   # 業種トレンドを測る期間（起点前60営業日）

    def _sector_trend(scode, as_of):
        """起点前60営業日の、その業種指数の変化率(%)。業種全体の上昇トレンドを測る。"""
        arr = sector_index.get(str(scode))
        if arr is None:
            return None
        dates, vals = arr
        idx = np.searchsorted(dates, np.datetime64(pd.Timestamp(as_of)))
        if idx < SEC_TREND_DAYS or idx >= len(vals):
            return None
        past = vals[idx - SEC_TREND_DAYS]
        now = vals[idx]
        if past and past > 0:
            return (now / past - 1.0) * 100
        return None

    # 信用残（銘柄別・週次）を銘柄ごとに整理
    margin_by_code = {}
    if margin is not None and len(margin):
        m = margin.copy()
        m["date"] = pd.to_datetime(m["date"])
        for c in ("long_margin", "short_margin"):
            if c in m.columns:
                m[c] = pd.to_numeric(m[c], errors="coerce")
        m = m.sort_values(["code", "date"])
        margin_by_code = {str(c): g for c, g in m.groupby("code")}

    # 空売り比率（業種別・週次）を業種コードごとに整理
    sr_by_sec = {}
    if short_ratio is not None and len(short_ratio):
        sr = short_ratio.copy()
        sr["date"] = pd.to_datetime(sr["date"])
        sr = sr.sort_values(["sector33", "date"])
        sr_by_sec = {str(s): g for s, g in sr.groupby("sector33")}

    LAG = 7  # 週次データの公表遅延（先読み防止）

    def _latest_val(g, date_col, as_of, col):
        cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=LAG)
        past = g[g[date_col] <= cutoff]
        if past.empty:
            return None
        v = past.iloc[-1].get(col)
        return float(v) if pd.notna(v) else None

    # 陽性サンプル：5倍株の起点日
    def _measure(code, as_of_date):
        arrs = tv_by_code.get(code)
        if arrs is None:
            return None
        tvvals, dates = arrs
        idx = np.searchsorted(dates, np.datetime64(pd.Timestamp(as_of_date)))
        if idx <= 0 or idx >= len(tvvals):
            return None
        tv_ratio = _turnover_ratio(tvvals, idx)
        sg, op = _fin_growth(fin_ext, code, pd.Timestamp(as_of_date))
        # 信用倍率（買い残÷売り残、先読み防止で7日前まで）
        margin_ratio = None
        g = margin_by_code.get(str(code))
        if g is not None:
            lm = _latest_val(g, "date", as_of_date, "long_margin")
            sm = _latest_val(g, "date", as_of_date, "short_margin")
            if lm and sm and sm > 0:
                margin_ratio = lm / sm
        # 業種空売り比率
        short_ratio_sec = None
        scode = sec_code.get(str(code))
        if scode and scode in sr_by_sec:
            short_ratio_sec = _latest_val(sr_by_sec[scode], "date", as_of_date, "short_ratio")
        # 業種トレンド（起点前60日、その業種全体の上昇率）
        sec_trend = _sector_trend(scode, as_of_date) if scode else None
        # 上方修正（起点前90日以内に業績予想の上方修正を出したか）＝材料の代替指標
        upward = _had_upward_revision(fin_ext, code, pd.Timestamp(as_of_date))
        return {"sector": sec_name.get(code), "turnover_ratio": tv_ratio,
                "sales_growth": sg, "op": op,
                "margin_ratio": margin_ratio, "short_ratio": short_ratio_sec,
                "sector_trend": sec_trend, "upward": upward}

    pos = []
    for _, r in fivebaggers.iterrows():
        m = _measure(str(r["code"]), r["起点日"])
        if m:
            pos.append(m)

    # 陰性サンプル：非5倍株のランダム時点
    rng = np.random.default_rng(42)
    non_fb = [c for c in tv_by_code if c not in fb_codes]
    neg = []
    tries = 0
    while len(neg) < N_NEGATIVE and tries < N_NEGATIVE * 5:
        tries += 1
        code = str(rng.choice(non_fb))
        tvvals, dates = tv_by_code[code]
        if len(dates) < VOL_RECENT + VOL_BASE + WINDOW + 10:
            continue
        idx = int(rng.integers(VOL_RECENT + VOL_BASE, len(dates) - WINDOW))
        m = _measure(code, pd.Timestamp(dates[idx]).date())
        if m:
            neg.append(m)

    return fivebaggers, pos, neg, sec_name


def _summarize_group(samples, label):
    n = len(samples)
    tv = [s["turnover_ratio"] for s in samples if s["turnover_ratio"] is not None]
    sg = [s["sales_growth"] for s in samples if s["sales_growth"] is not None]
    op_pos = [s for s in samples if s["op"] is not None and s["op"] > 0]
    mr = [s["margin_ratio"] for s in samples if s.get("margin_ratio") is not None]
    srt = [s["short_ratio"] for s in samples if s.get("short_ratio") is not None]
    st = [s["sector_trend"] for s in samples if s.get("sector_trend") is not None]
    uw = [s["upward"] for s in samples if s.get("upward") is not None]
    L = [f"【{label}】 サンプル {n}件"]
    if uw:
        L.append(f"  起点前90日の上方修正: 出していた割合 {sum(uw)/len(uw)*100:.1f}% "
                 f"（判定できた {len(uw)}件中）")
    if st:
        L.append(f"  業種トレンド（起点前60日の業種指数の変化）: 中央値 {np.median(st):+.1f}% "
                 f"/ プラスの割合 {sum(1 for x in st if x>0)/len(st)*100:.1f}%")
    if tv:
        L.append(f"  売買代金比（直近20日÷前60日）: 中央値 {np.median(tv):.2f}倍 "
                 f"/ 平均 {np.mean(tv):.2f}倍")
        big = sum(1 for x in tv if x >= 2.0)
        L.append(f"    うち2倍以上に急増していた割合: {big/len(tv)*100:.1f}%")
    if sg:
        L.append(f"  直近決算の売上前年比: 中央値 {np.median(sg):+.1f}% "
                 f"/ プラスの割合 {sum(1 for x in sg if x>0)/len(sg)*100:.1f}%")
    if samples:
        L.append(f"  直近決算が営業黒字だった割合: {len(op_pos)/n*100:.1f}%")
    if mr:
        L.append(f"  信用倍率（買残÷売残）: 中央値 {np.median(mr):.2f} "
                 f"（データ有 {len(mr)}件）")
    if srt:
        L.append(f"  業種空売り比率: 中央値 {np.median(srt):.3f} "
                 f"（データ有 {len(srt)}件）")
    return "\n".join(L), {"tv": tv, "sg": sg, "mr": mr, "srt": srt, "st": st,
                          "uw": uw}


def summarize(result):
    fivebaggers, pos, neg, sec_name = result
    L = ["=" * 80,
         "5倍株の『上がる前のサイン』分析（5倍株 vs 非5倍株）",
         "=" * 80, ""]
    L.append(f"5倍株: {len(fivebaggers)}件 / 起点日で測定できた: {len(pos)}件")
    L.append(f"比較対照（非5倍株のランダム時点）: {len(neg)}件")
    L.append("")

    # ① 業種の偏り
    L.append("─" * 80)
    L.append("① 業種：5倍株はどの業種から出やすいか")
    L.append("─" * 80)
    fb_sec = fivebaggers["業種"].value_counts()
    all_sec = pd.Series(sec_name).value_counts()
    L.append("  5倍株の業種内訳（上位）と、全銘柄に占めるその業種の割合:")
    for sec, cnt in fb_sec.head(10).items():
        total = int(all_sec.get(sec, 0))
        rate = (cnt / total * 100) if total else 0
        L.append(f"    {sec}: 5倍株{cnt}件 / 全{total}社 = その業種の{rate:.1f}%が5倍化")
    L.append("")

    # ②③ 売買代金・業績
    L.append("─" * 80)
    L.append("②③ 起点日直前の 売買代金・業績（陽性=5倍株 vs 陰性=非5倍株）")
    L.append("─" * 80)
    pos_txt, pos_d = _summarize_group(pos, "5倍株の起点日直前")
    neg_txt, neg_d = _summarize_group(neg, "非5倍株のランダム時点")
    L.append(pos_txt)
    L.append("")
    L.append(neg_txt)
    L.append("")

    # 差の判定
    L.append("─" * 80)
    L.append("判定：5倍株だけに見られる差（前兆の候補）")
    L.append("─" * 80)
    if pos_d["uw"] and neg_d["uw"]:
        pm = sum(pos_d["uw"]) / len(pos_d["uw"]) * 100
        nm = sum(neg_d["uw"]) / len(neg_d["uw"]) * 100
        L.append(f"・上方修正を出していた割合: 5倍株 {pm:.1f}% vs 非5倍株 {nm:.1f}%")
        if pm > nm * 1.5:
            L.append("  → 5倍株は起点前に上方修正（材料）を出していた（前兆候補◎）")
        elif pm > nm:
            L.append("  → 5倍株のほうがやや高いが差は小さい（弱い前兆）")
        else:
            L.append("  → 差がない。上方修正は前兆とは言えない")
    if pos_d["st"] and neg_d["st"]:
        pm, nm = np.median(pos_d["st"]), np.median(neg_d["st"])
        L.append(f"・業種トレンド（起点前60日）: 5倍株 {pm:+.1f}% vs 非5倍株 {nm:+.1f}%")
        if pm > nm + 3:
            L.append("  → 5倍株は起点前、その業種全体が上昇していた（前兆候補◎）")
        elif pm > nm:
            L.append("  → 5倍株のほうがやや高いが差は小さい（弱い前兆）")
        else:
            L.append("  → 差がない。業種トレンドは前兆とは言えない")
    if pos_d["tv"] and neg_d["tv"]:
        pm, nm = np.median(pos_d["tv"]), np.median(neg_d["tv"])
        L.append(f"・売買代金比の中央値: 5倍株 {pm:.2f}倍 vs 非5倍株 {nm:.2f}倍")
        if pm > nm * 1.2:
            L.append("  → 5倍株は起点前に売買代金が明確に増えていた（前兆候補◎）")
        elif pm > nm:
            L.append("  → 5倍株のほうがやや高いが差は小さい（弱い前兆）")
        else:
            L.append("  → 差がない。売買代金の増加は前兆とは言えない")
    if pos_d["sg"] and neg_d["sg"]:
        pm, nm = np.median(pos_d["sg"]), np.median(neg_d["sg"])
        L.append(f"・売上前年比の中央値: 5倍株 {pm:+.1f}% vs 非5倍株 {nm:+.1f}%")
        if pm > nm + 5:
            L.append("  → 5倍株は起点前に業績の伸びが強かった（前兆候補◎）")
        elif pm > nm:
            L.append("  → 5倍株のほうがやや高いが差は小さい（弱い前兆）")
        else:
            L.append("  → 差がない。業績の伸びは前兆とは言えない")
    if pos_d["mr"] and neg_d["mr"]:
        pm, nm = np.median(pos_d["mr"]), np.median(neg_d["mr"])
        L.append(f"・信用倍率の中央値: 5倍株 {pm:.2f} vs 非5倍株 {nm:.2f}")
        if abs(pm - nm) < 0.2:
            L.append("  → ほぼ差がない。信用倍率は前兆とは言えない")
        elif pm > nm:
            L.append("  → 5倍株は起点前、買い残が相対的に多かった（買い方に人気）")
        else:
            L.append("  → 5倍株は起点前、売り残が相対的に多かった（踏み上げ余地）")
    if pos_d["srt"] and neg_d["srt"]:
        pm, nm = np.median(pos_d["srt"]), np.median(neg_d["srt"])
        L.append(f"・業種空売り比率の中央値: 5倍株 {pm:.3f} vs 非5倍株 {nm:.3f}")
        if abs(pm - nm) < 0.02:
            L.append("  → ほぼ差がない。空売り比率は前兆とは言えない")
        elif pm > nm:
            L.append("  → 5倍株の業種は起点前、空売り比率が高かった（踏み上げ素地）")
        else:
            L.append("  → 5倍株の業種は起点前、空売り比率が低かった")
    L += ["", "=" * 80,
          "・生存者バイアスを避けるため5倍株と非5倍株を対比した。差が小さい指標は",
          "  『5倍株だけの特徴』とは言えない（多くの銘柄が同じ状態のため）。",
          "・プレスリリース数・記事数はJ-Quantsで取得不可のため未対応。",
          "・過去の傾向であり将来を保証しない。"]
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
    if "adj_factor" not in quotes.columns:
        print("\n  [!] 株価に分割調整情報がありません。del data\\quotes.pkl で再取得してください。")
    print("信用残・空売り比率を取得します...")
    try:
        margin = jq.margin(config.BACKTEST_LOOKBACK_DAYS)
    except Exception:
        margin = None
    try:
        short_ratio = jq.short_ratio(config.BACKTEST_LOOKBACK_DAYS)
    except Exception:
        short_ratio = None
    print("\n5倍株を抽出し、その起点日直前の前兆を分析します...")

    result = analyze(quotes, fin, listed, margin=margin, short_ratio=short_ratio)
    if result is None:
        print("5倍株が見つからないため分析できません。")
        return
    print("\n" + summarize(result))

    today = dt.datetime.now(JST).date().isoformat()
    path = os.path.join(config.OUTPUT_DIR, f"{today}_precursor.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summarize(result))
    print(f"\nレポート: {path}")


if __name__ == "__main__":
    main()
