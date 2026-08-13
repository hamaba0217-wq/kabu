# -*- coding: utf-8 -*-
"""
決定論的チェックリスト（Role 6 の置き換え）

このモジュールは **文章を生成しません。**
すべての判定は、APIから取得した数値とその計算結果のみに基づきます。

各判定は必ず以下を返します。
  判定 : 適合 / 不適合 / 不明
  式   : 実際の数値を入れた計算式（利用者が検算できる形）
  閾値 : どの基準と比べたか
  出典 : どのAPIのどの項目から取ったか

「ランクB」のような結論だけを出さないための構造です。
式が出ていれば、利用者が自分で検算し、閾値そのものが妥当かを判断できます。

「不明」の扱いについて
----------------------
「調べた結果、問題がなかった」＝ 適合（良い）
「調べられなかった」        ＝ 不明 → 不適合として扱う

この2つは別物です。後者を通すと、たとえば発行済株式数が欠損しているだけの
銘柄が「増資なし」として通過します。埋めない、通さない。

このチェックが見ていないもの
----------------------------
競合の参入、訴訟、経営陣の交代、規制変更、マクロの急変など。
「全項目通過」は「調べた6項目に問題がなかった」であり、
「問題がない」ではありません。
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

PASS, FAIL, UNKNOWN = "適合", "不適合", "不明"


def _r(name, status, formula, value, threshold, source, note=""):
    return {"項目": name, "判定": status, "式": formula, "実測値": value,
            "閾値": threshold, "出典": source, "備考": note}


def _unknown(name, source, reason):
    return _r(name, UNKNOWN, "計算不可（データ欠損）", None, "—", source, reason)


# ---------------------------------------------------------------------------
# 個別チェック
# ---------------------------------------------------------------------------

def check_dilution(fin_code: pd.DataFrame) -> dict:
    """希薄化: 発行済株式数が1年で10%以上増えていないか。

    増資・新株予約権の行使は発行済株式数に必ず反映されるため、
    株式数の推移だけで機械的に検知できます。
    """
    src = "J-Quants 決算サマリー（発行済株式数）"
    if "shares_outstanding" not in fin_code.columns:
        return _unknown("希薄化", src, "発行済株式数の列が取得できませんでした")
    g = fin_code.sort_values("disclosed_date").dropna(subset=["shares_outstanding"])
    if len(g) < 2:
        return _unknown("希薄化", src, "比較できる開示が不足しています")

    now = g.iloc[-1]
    past = g[g["disclosed_date"] <= now["disclosed_date"] - pd.Timedelta(days=300)]
    if past.empty:
        return _unknown("希薄化", src, "1年前の開示がありません")
    base, cur = float(past.iloc[-1]["shares_outstanding"]), float(now["shares_outstanding"])
    if base <= 0:
        return _unknown("希薄化", src, "株式数が0または欠損")

    inc = cur / base - 1
    formula = (f"{cur:,.0f}株 ÷ {base:,.0f}株 − 1 = {inc:+.2%}"
               f"（{str(past.iloc[-1]['disclosed_date'])[:10]} → "
               f"{str(now['disclosed_date'])[:10]}）")
    status = FAIL if inc >= 0.10 else PASS
    return _r("希薄化", status, formula, f"{inc:+.1%}", "+10%未満", src,
              "1年で株式数が10%以上増加。増資または新株予約権の行使" if status == FAIL else "")


def check_earnings_in_window(code, earnings_cal, as_of, hold_days=90) -> dict:
    src = "J-Quants 決算発表日"
    if earnings_cal is None or earnings_cal.empty:
        return _unknown("保有期間中の決算", src, "決算発表日を取得できませんでした")
    df = earnings_cal[earnings_cal["code"].astype(str).str[:4] == str(code)[:4]]
    end = as_of + dt.timedelta(days=hold_days)
    up = df[(df["date"] >= pd.Timestamp(as_of)) & (df["date"] <= pd.Timestamp(end))]
    window = f"{as_of} 〜 {end}"
    if up.empty:
        return _r("保有期間中の決算", PASS, f"{window} に発表予定なし", "なし",
                  "保有期間内に発表が無いこと", src)
    d = up["date"].min().date()
    return _r("保有期間中の決算", FAIL,
              f"{window} 内の {d}（{(d - as_of).days}日後）に発表予定",
              str(d), "保有期間内に発表が無いこと", src,
              "この日に想定外の下落が起きうる。建玉サイズでの調整が必要")


def check_already_run_up(quotes_code: pd.DataFrame, days: int = 60) -> dict:
    src = f"J-Quants 株価日足（{days}営業日）"
    g = quotes_code.sort_values("date")
    if len(g) < days:
        return _unknown("直近の急騰", src, f"{days}営業日分の株価が不足（{len(g)}日分）")
    now, past = float(g["close"].iloc[-1]), float(g["close"].iloc[-days])
    ret = now / past - 1
    formula = (f"{now:,.0f}円 ÷ {past:,.0f}円 − 1 = {ret:+.1%}"
               f"（{str(g['date'].iloc[-days])[:10]} → {str(g['date'].iloc[-1])[:10]}）")
    status = FAIL if ret >= 1.00 else PASS
    return _r("直近の急騰", status, formula, f"{ret:+.1%}", "+100%未満", src,
              "既に2倍以上。3倍余地が乏しい可能性" if status == FAIL else "")


def check_turnover_vs_fundamentals(turnover_spike, op_yoy) -> dict:
    src = "J-Quants 株価日足＋決算サマリー"
    if turnover_spike is None or pd.isna(turnover_spike) or op_yoy is None or pd.isna(op_yoy):
        return _unknown("需給先行", src, "売買代金または営業利益YoYが欠損")
    formula = (f"当日売買代金 ÷ 20日平均 = {turnover_spike:.2f}倍　"
               f"かつ　営業利益YoY = {op_yoy:+.1f}%")
    status = FAIL if (turnover_spike >= 3.0 and op_yoy < 30) else PASS
    return _r("需給先行", status, formula,
              f"{turnover_spike:.1f}倍 / {op_yoy:+.1f}%",
              "売買代金3倍以上 かつ 営業益YoY+30%未満 なら不適合", src,
              "出来高だけが先行し、業績の裏付けが薄い" if status == FAIL else "")


def check_psr(market_cap, net_sales, period_type) -> dict:
    src = "J-Quants 決算サマリー（時価総額 ÷ 年換算売上）"
    if not market_cap or not net_sales or net_sales <= 0:
        return _unknown("PSR", src, "時価総額または売上高が欠損")
    mult = {"1Q": 4, "2Q": 2, "3Q": 4/3}.get(str(period_type).upper(), 1)
    psr = market_cap / (net_sales * mult)
    formula = (f"時価総額 {market_cap/1e8:,.1f}億円 ÷ "
               f"（{period_type}累計売上 {net_sales/1e8:,.1f}億円 × {mult:g}）"
               f" = {psr:.1f}倍")
    status = FAIL if psr >= 10 else PASS
    return _r("PSR", status, formula, f"{psr:.1f}倍", "10倍未満", src,
              "成長が既に株価に織り込まれている可能性" if status == FAIL else "")


def check_management(mgmt: dict) -> dict:
    src = "J-Quants 決算サマリー（会社予想 vs 実績）"
    rank = mgmt.get("rank", "判定不能")
    if rank == "判定不能":
        return _unknown("経営者の予想達成率", src, mgmt.get("reason", ""))
    hist = mgmt.get("history") or []
    formula = "　/　".join(
        f"{h['期']}: 実績{h['実績']:,.0f} ÷ 予想{h['会社予想']:,.0f} = {h['達成率']}%"
        for h in hist) or "履歴なし"
    status = FAIL if rank in ("B", "C") else PASS
    return _r("経営者の予想達成率", status, formula,
              f"ランク{rank}（平均 {mgmt.get('achieved')}%）",
              "3期以上100%超=S / 概ね達成=A（S・Aのみ適合）", src,
              mgmt.get("reason", ""))


def check_large_holding(flag) -> dict:
    src = "EDINET 書類一覧API（docTypeCode 350/360）"
    if flag is None:
        return _unknown("大量保有報告書", src, "EDINET APIキーが未設定")
    return _r("大量保有報告書", PASS, "直近30日の提出書類を証券コードで照合",
              "提出あり" if flag else "なし", "参考情報（不合格にはしない）", src,
              "5%超保有者の異動。機関投資家の参入シグナル" if flag else "")


# ---------------------------------------------------------------------------
# まとめ
# ---------------------------------------------------------------------------

STRICT_ITEMS = {"希薄化", "保有期間中の決算", "直近の急騰", "需給先行",
                "PSR", "経営者の予想達成率"}

# このチェックが見ていない失敗要因（レポートに明記する）
NOT_COVERED = [
    "競合の参入・価格競争", "訴訟・行政処分", "経営陣の交代",
    "規制変更", "取引先の集中リスク", "海外発のマクロ急変（VIX・米金利）",
]


def run_all(row, quotes_code, fin_code, mgmt, earnings_cal, as_of) -> dict:
    results = [
        check_management(mgmt),
        check_dilution(fin_code),
        check_earnings_in_window(str(row.get("code", "")), earnings_cal, as_of),
        check_already_run_up(quotes_code),
        check_turnover_vs_fundamentals(row.get("turnover_spike"), row.get("op_yoy")),
        check_psr(row.get("market_cap"), row.get("net_sales"), row.get("period_type")),
        check_large_holding(row.get("large_holding_filed")),
    ]
    strict = [r for r in results if r["項目"] in STRICT_ITEMS]
    n_fail = sum(1 for r in strict if r["判定"] == FAIL)
    n_unknown = sum(1 for r in strict if r["判定"] == UNKNOWN)
    return {
        "results": results,
        "適合": sum(1 for r in strict if r["判定"] == PASS),
        "不適合": n_fail,
        "不明": n_unknown,
        "全項目通過": (n_fail == 0 and n_unknown == 0),
    }


# ---------------------------------------------------------------------------
# スクリーニング条件（Role 2）の内訳
# ---------------------------------------------------------------------------

def screening_basis(row, cfg) -> list[dict]:
    """なぜこの銘柄が抽出されたのかを、条件ごとに式で示す。"""
    out = []

    def add(name, formula, value, threshold, source):
        out.append({"項目": name, "式": formula, "実測値": value,
                    "閾値": threshold, "出典": source})

    pt = row.get("period_type", "")
    cap, tv = row.get("market_cap_oku"), row.get("turnover_oku")
    op, sa, hi = row.get("op_yoy"), row.get("sales_yoy"), row.get("pct_from_high")

    if pd.notna(cap):
        add("時価総額", f"終値 × 発行済株式数 = {cap:,.1f}億円", f"{cap:,.1f}億円",
            f"{cfg.MIN_MARKET_CAP/1e8:.0f}〜{cfg.MAX_MARKET_CAP/1e8:.0f}億円",
            "J-Quants 株価日足＋決算サマリー")
    if pd.notna(tv):
        add("売買代金", f"当日の売買代金 = {tv:,.2f}億円", f"{tv:,.2f}億円",
            f"{cfg.MIN_TURNOVER_VALUE/1e8:.1f}億円以上", "J-Quants 株価日足")
    if pd.notna(op):
        add("営業利益 前年同期比",
            f"今期{pt}累計 ÷ 前年同{pt}累計 − 1 = {op:+.1f}%", f"{op:+.1f}%",
            f"+{cfg.MIN_OP_PROFIT_YOY:.0%}以上 または黒字転換", "J-Quants 決算サマリー")
    if pd.notna(sa):
        add("売上高 前年同期比",
            f"今期{pt}累計 ÷ 前年同{pt}累計 − 1 = {sa:+.1f}%", f"{sa:+.1f}%",
            f"+{cfg.MIN_SALES_YOY:.0%}以上", "J-Quants 決算サマリー")
    if pd.notna(hi):
        add("52週高値からの位置", f"終値 ÷ 52週高値 − 1 = {hi:+.1f}%", f"{hi:+.1f}%",
            "参考値（条件なし）", "J-Quants 株価日足")
    return out
