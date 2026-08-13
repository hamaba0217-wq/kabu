# -*- coding: utf-8 -*-
"""
日次 / 週次のメールレポート

設計方針
--------
「手を動かす時間がない」前提なので、**件名だけで判断できる**ようにしています。

  日次: [株スクリーン] 7/23 候補3件 🟢       ← 件名で件数と地合いがわかる
        [株スクリーン] 7/23 候補なし 🟡       ← 開かずに捨ててよい
        [株スクリーン] ⚠️ エラー               ← ここだけは開く

失敗しても通知が来るようにしてあります。
自動化で一番怖いのは「動いていないことに気づかないまま数週間経つ」ことなので。
"""

from __future__ import annotations

import datetime as dt
import os
import traceback

import pandas as pd

import backtest
import checks
import config
import fundamentals
import mailer
import regime
import screen
from sources import JQuants, JST, code4, fetch_disclosures

HISTORY_DIR = "history"


def _load(jq: JQuants, lookback: int):
    quotes = jq.quotes(lookback)
    fin = jq.financials(config.FIN_LOOKBACK_DAYS)
    listed = jq.listed()
    return quotes, fin, listed


def _delay_warning(data_date) -> str:
    """データの鮮度をチェックし、遅延が大きければ警告文を返す。"""
    lag = (dt.datetime.now(JST).date() - data_date).days
    if lag > 30:
        return (f'<p class="warn">⚠️ データが{lag}日前のものです（Freeプランは12週間遅延）。'
                f'この内容は売買判断には使えません。検証用としてご覧ください。</p>')
    if lag > 7:
        return f'<p class="warn">⚠️ データが{lag}日前のものです。</p>'
    return ""


def _basis_table(rows) -> str:
    """スクリーニング通過理由を式つきで表にする。"""
    if not rows:
        return ""
    body = "".join(
        f'<tr><td class="l">{r["項目"]}</td><td class="l">{r["式"]}</td>'
        f'<td class="l">{r["閾値"]}</td><td class="l">{r["出典"]}</td></tr>'
        for r in rows)
    return ('<table><tr><th class="l">条件</th><th class="l">計算式</th>'
            f'<th class="l">閾値</th><th class="l">出典</th></tr>{body}</table>')


def _disclosure_table(df) -> str:
    """開示書類の一覧。書類名はEDINETの原文のまま、要約はしない。"""
    if df is None or len(df) == 0:
        return "<p>直近30日の提出書類はありません（またはEDINET未設定）。</p>"
    body = "".join(
        f'<tr><td class="l">{r["submit_date"]}</td>'
        f'<td class="l">{r["doc_type"]}</td>'
        f'<td class="l">{r["doc_description"]}</td>'
        f'<td class="l">{r["filer_name"]}</td></tr>'
        for _, r in df.head(15).iterrows())
    return ('<table><tr><th class="l">提出日</th><th class="l">書類種別</th>'
            f'<th class="l">書類名（EDINET原文）</th><th class="l">提出者</th></tr>'
            f'{body}</table>')


def _check_html(code, name, basis, mgmt, plan, res, docs) -> str:
    """1銘柄分のレポート。すべての判定に計算式と出典を併記する。"""
    color = "green" if res["全項目通過"] else "red"
    head = ("<b>全項目通過</b>" if res["全項目通過"]
            else f'<b>不通過</b>（不適合{res["不適合"]}件 / 不明{res["不明"]}件）')

    rows = "".join(
        f'<tr><td class="l">{r["項目"]}</td>'
        f'<td class="l">{r["判定"]}</td>'
        f'<td class="l">{r["式"]}</td>'
        f'<td class="l">{r["閾値"]}</td>'
        f'<td class="l">{r["出典"]}</td>'
        f'<td class="l">{r["備考"]}</td></tr>'
        for r in res["results"])

    plan_rows = "".join(f'<tr><td class="l">{k}</td><td class="l">{v}</td></tr>'
                        for k, v in plan.items())

    not_covered = "、".join(checks.NOT_COVERED)

    return f"""
<div class="box {color}">
<h3>{code} {name} — {head}</h3>

<b>1. なぜ抽出されたか（Role 2 の通過理由）</b>
{_basis_table(basis)}

<b>2. 検証チェック（Role 4・6）</b>
<table>
<tr><th class="l">項目</th><th class="l">判定</th><th class="l">計算式</th>
    <th class="l">閾値</th><th class="l">出典</th><th class="l">備考</th></tr>
{rows}
</table>

<b>3. 直近30日の提出書類（EDINET・原文のまま）</b>
<p style="font-size:12px;color:#666;">要約・解釈は行っていません。原文をご確認ください。</p>
{_disclosure_table(docs)}

<b>4. 執行設計（Role 7）</b>
<table>{plan_rows}</table>

<p style="font-size:12px;color:#666;">
このチェックが見ていない失敗要因: {not_covered}<br>
「全項目通過」は「調べた6項目に問題がなかった」であり、「問題がない」ではありません。
</p>
</div>"""


# ---------------------------------------------------------------------------
# 日次
# ---------------------------------------------------------------------------

def daily() -> None:
    today = dt.datetime.now(JST).date()
    try:
        jq = JQuants()
        quotes, fin, listed = _load(jq, config.PRICE_LOOKBACK_DAYS)
        data_date = quotes["date"].max().date()

        rg = regime.judge(quotes, listed)
        label, action = regime.LEVEL_LABEL[rg["level"]]

        prices = screen.latest_prices(quotes, config.MA_WINDOW)
        yoy = screen.yoy_table(fin)
        picks, log = screen.run_screen(prices, yoy, listed, pd.DataFrame())
        out = screen.to_output(picks) if len(picks) else pd.DataFrame()

        # 履歴を残す（週次で追跡するため）
        os.makedirs(HISTORY_DIR, exist_ok=True)
        if len(out):
            out.to_csv(f"{HISTORY_DIR}/{today}.csv", index=False, encoding="utf-8-sig")

        n = len(out)

        # 「出たときだけ知らせる」設定なら、候補ゼロの日は送らずに終える
        if n == 0 and getattr(config, "DAILY_MAIL_ONLY_WHEN_HITS", False):
            print(f"候補0件のためメール送信をスキップしました（{today}）")
            return

        subject = f"[株スクリーン] {today:%-m/%-d} " + \
                  (f"候補{n}件 " if n else "候補なし ") + label.split()[0]

        body = [
            f'<div class="box {rg["level"].lower()}">'
            f'<b>相場環境: {label}</b> — {action}<br>'
            + "<br>".join(f"・{r}" for r in rg["reasons"]) +
            '</div>',
            _delay_warning(data_date),
            f"<p>データ最終日: {data_date} / 抽出: <b>{n}件</b></p>",
        ]

        if n:
            cols = [c for c in ["code4", "company_name", "market", "close",
                                "market_cap_oku", "turnover_oku", "sales_yoy",
                                "op_yoy", "large_holding_filed"] if c in out.columns]
            disp = out[cols].rename(columns={
                "code4": "コード", "company_name": "銘柄名", "market": "市場",
                "close": "終値", "market_cap_oku": "時価総額(億)",
                "turnover_oku": "売買代金(億)", "sales_yoy": "売上YoY%",
                "op_yoy": "営業益YoY%", "large_holding_filed": "大量保有",
            })
            body.append(mailer.df_to_html(disp, left_cols=("コード", "銘柄名", "市場")))
            body.append(
                "<p><b>この時点ではまだ買い候補ではありません。</b><br>"
                "事業性・経営者・需給・反対意見（Role 3〜7）の確認が済むまでは"
                "監視リストです。時間が取れるときにチャットへ貼ってください。</p>"
            )
        else:
            body.append("<p>条件を満たす銘柄はありませんでした。"
                        "これは正常な結果です。</p>")

        body.append("<details><summary>絞り込みの経過</summary><pre>"
                    + "\n".join(log) + "</pre></details>")

        footer = ("VIX・米金利・ドル円・イベント日程は自動取得の対象外です。"
                  "相場環境の判定は国内の資金動向のみに基づきます。<br>")
        mailer.send(subject, mailer.wrap(f"日次スクリーニング {today}", "".join(body), footer))
        print(f"送信しました: {subject}")

    except Exception:
        err = traceback.format_exc()
        print(err)
        try:
            mailer.send(
                f"[株スクリーン] ⚠️ エラー {today:%-m/%-d}",
                mailer.wrap("日次処理でエラーが発生しました",
                            f"<pre>{err}</pre>"
                            "<p>このメッセージをそのままチャットに貼ってください。</p>"),
            )
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# 週次
# ---------------------------------------------------------------------------

def weekly() -> None:
    today = dt.datetime.now(JST).date()
    try:
        jq = JQuants()
        quotes, fin, listed = _load(jq, 400)
        data_date = quotes["date"].max().date()

        rg = regime.judge(quotes, listed)
        label, action = regime.LEVEL_LABEL[rg["level"]]

        body = [
            f'<div class="box {rg["level"].lower()}">'
            f'<b>相場環境: {label}</b> — {action}<br>'
            + "<br>".join(f"・{r}" for r in rg["reasons"]) + '</div>',
            _delay_warning(data_date),
        ]

        # --- 今週の抽出をまとめる ---
        files = sorted(os.listdir(HISTORY_DIR)) if os.path.isdir(HISTORY_DIR) else []
        recent, all_hist = [], []
        for f in files:
            if not f.endswith(".csv"):
                continue
            d = dt.date.fromisoformat(f[:-4])
            df = pd.read_csv(f"{HISTORY_DIR}/{f}", dtype={"code": str, "code4": str})
            df["抽出日"] = d
            all_hist.append(df)
            if (today - d).days <= 7:
                recent.append(df)

        if recent:
            r = pd.concat(recent).drop_duplicates(subset=["code"], keep="last")
            body.append(f"<h3>今週の新規候補（{len(r)}銘柄）</h3>")
            cols = [c for c in ["code4", "company_name", "market", "close",
                                "market_cap_oku", "sales_yoy", "op_yoy", "抽出日"]
                    if c in r.columns]
            disp = r[cols].rename(columns={
                "code4": "コード", "company_name": "銘柄名", "market": "市場",
                "close": "抽出時株価", "market_cap_oku": "時価総額(億)",
                "sales_yoy": "売上YoY%", "op_yoy": "営業益YoY%"})
            body.append(mailer.df_to_html(disp, left_cols=("コード", "銘柄名", "市場")))
        else:
            body.append("<h3>今週の新規候補</h3><p>ありませんでした。</p>")

        # --- 過去の抽出銘柄がその後どうなったか ---
        if all_hist:
            hist = pd.concat(all_hist)
            latest_px = (quotes.sort_values("date").groupby("code").tail(1)
                         [["code", "close"]].rename(columns={"close": "現在値"}))
            hist["code"] = hist["code"].astype(str)
            trk = hist.drop_duplicates(subset=["code"], keep="first").merge(
                latest_px, on="code", how="left")
            trk = trk[trk["現在値"].notna() & (trk["close"] > 0)]
            if len(trk):
                trk["騰落率%"] = ((trk["現在値"] / trk["close"] - 1) * 100).round(1)
                trk["経過日数"] = (data_date - pd.to_datetime(trk["抽出日"]).dt.date).apply(lambda x: x.days)
                trk = trk.sort_values("騰落率%", ascending=False)

                body.append("<h3>これまでの抽出銘柄の追跡</h3>")
                body.append(
                    f"<p>対象 {len(trk)}銘柄 / 中央値 <b>{trk['騰落率%'].median():+.1f}%</b> / "
                    f"プラスの銘柄 {(trk['騰落率%'] > 0).sum()}件 / "
                    f"3倍達成 {(trk['騰落率%'] >= 200).sum()}件</p>")
                cols = [c for c in ["code4", "company_name", "抽出日", "経過日数",
                                    "close", "現在値", "騰落率%"] if c in trk.columns]
                disp = trk[cols].rename(columns={"code4": "コード",
                                                 "company_name": "銘柄名",
                                                 "close": "抽出時"})
                body.append(mailer.df_to_html(disp.head(30),
                                              left_cols=("コード", "銘柄名")))
                body.append(
                    "<p><b>ここが最も重要な数字です。</b> 中央値がマイナスで推移するなら、"
                    "スクリーニング条件が機能していません。"
                    "3か月分の追跡がたまった時点で条件を見直してください。</p>")

        # --- 詳細チェック（Role 4・6・7）---
        # LLMは使いません。すべてAPIの数値と計算結果のみです。
        if recent:
            body.append("<h3>詳細チェック（Role 4・6・7）</h3>")
            body.append("<p>すべての判定は、APIから取得した数値とその計算結果のみに基づきます。"
                        "文章生成は行っていません。"
                        "<b>「不明」の項目がある銘柄は不合格として扱います。</b></p>")

            # 会社予想は financials() のキャッシュに含まれるため再取得は不要
            fin_fc = fin
            cal = jq.earnings_calendar()
            all_docs = fetch_disclosures(30)

            cleared = []
            for _, row in r.head(config.MAX_DETAIL_CHECK).iterrows():
                code = str(row["code"])
                name = row.get("company_name", "")
                fin_code = fin_fc[fin_fc["code"] == code]
                q_code = quotes[quotes["code"] == code]
                mgmt = fundamentals.management_score(fin_code)
                plan = fundamentals.execution_plan(float(row["close"]))

                # チェックに必要な生値を補う
                info = dict(row)
                if len(fin_code):
                    last = fin_code.sort_values("disclosed_date").iloc[-1]
                    info["net_sales"] = last.get("net_sales")
                    info["period_type"] = last.get("period_type")
                if "market_cap_oku" in info and pd.notna(info.get("market_cap_oku")):
                    info["market_cap"] = float(info["market_cap_oku"]) * 1e8

                res = checks.run_all(info, q_code, fin_code, mgmt, cal, data_date)
                basis = checks.screening_basis(info, config)
                docs = (all_docs[all_docs["code"] == code[:4]]
                        .sort_values("submit_date", ascending=False)
                        if len(all_docs) else all_docs)
                if res["全項目通過"]:
                    cleared.append((row, mgmt, plan, res))
                body.append(_check_html(row.get("code4", code), name,
                                        basis, mgmt, plan, res, docs))

            body.insert(
                len(body) - min(len(r), config.MAX_DETAIL_CHECK),
                f"<p><b>全項目を通過した銘柄: {len(cleared)}件</b>"
                f"（チェック対象 {min(len(r), config.MAX_DETAIL_CHECK)}件）</p>")

                # 日次ジョブが動いているかの確認（履歴ファイルの日付で判断）
        if files:
            last = dt.date.fromisoformat(sorted(files)[-1][:-4])
            gap = (today - last).days
            if gap > 10:
                body.insert(1,
                    f'<p class="warn">⚠️ 日次の抽出履歴が{gap}日間更新されていません。'
                    f'GitHubのActionsタブを確認してください。</p>')

        footer = ("すべての判定はAPIの数値と計算結果のみに基づき、推測は含みません。"
                  "「不明」の項目がある銘柄は不合格としています。<br>"
                  "全項目を通過した銘柄も、買い推奨ではありません。"
                  "最終的な購入判断はご自身で行ってください。<br>"
                  "候補ゼロの日は日次メールを送らない設定です。"
                  "この週次メールが届いていれば、システムは正常に動いています。<br>")
        subject = f"[株スクリーン 週次] {today:%-m/%-d} " + label.split()[0]
        mailer.send(subject, mailer.wrap(f"週次レポート {today}", "".join(body), footer))
        print(f"送信しました: {subject}")

    except Exception:
        err = traceback.format_exc()
        print(err)
        try:
            mailer.send(f"[株スクリーン 週次] ⚠️ エラー {today:%-m/%-d}",
                        mailer.wrap("週次処理でエラーが発生しました", f"<pre>{err}</pre>"))
        except Exception:
            pass
        raise
