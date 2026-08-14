# -*- coding: utf-8 -*-
"""
推奨候補の履歴を記録し、その後の結果を追跡して実績ページを作る。

流れ
----
1. 毎朝 web-report 実行時に、その日の「特に有力」候補を history/picks.csv に追記
   （買い指値＝推奨日の終値。同じ銘柄を同じ日に二重記録しない）
2. その後、各推奨銘柄を最新株価まで追跡し、
   利確+10%到達 / 損切り-5%到達 / 追跡中 を判定
3. 実績ページ（docs/results.html）を生成

事実にもとづく注意
------------------
  ・買い指値＝推奨日の終値。翌日その値で約定できたと仮定（実際は前後する）。
  ・利確+10%/損切り-5%は終値ベース判定。ザラ場の瞬間は見ない。
  ・手数料・スリッページ未考慮。過去の記録であり将来を保証しない。
  ・記録開始日より前の推奨は復元できない（記録していないため）。
"""

from __future__ import annotations

import datetime as dt
import html
import os

import numpy as np
import pandas as pd

TARGET_PCT = 10.0     # 利確
STOP_PCT = -5.0       # 損切り
HIST_DIR = "history"
HIST_FILE = os.path.join(HIST_DIR, "picks.csv")
OUT_DIR = "docs"


def record_picks(strong_df, as_of):
    """その日の有力候補を履歴に追記する。同じ銘柄×同じ日付は重複させない。"""
    if strong_df is None or strong_df.empty:
        return
    os.makedirs(HIST_DIR, exist_ok=True)
    as_of_str = str(pd.Timestamp(as_of).date())
    rows = []
    for _, r in strong_df.iterrows():
        close = r.get("終値")
        if not (close and close == close):
            continue
        entry = float(close)
        rows.append({
            "推奨日": as_of_str,
            "code": str(r.get("code", "")),
            "銘柄名": r.get("銘柄名", ""),
            "業種": r.get("業種", ""),
            "買い指値": round(entry, 1),
            "利確": round(entry * (1 + TARGET_PCT / 100), 1),
            "損切り": round(entry * (1 + STOP_PCT / 100), 1),
        })
    if not rows:
        return
    new = pd.DataFrame(rows)
    if os.path.exists(HIST_FILE):
        old = pd.read_csv(HIST_FILE, dtype={"code": str})
        # 同じ推奨日×codeの重複を除く（同じ日に複数回実行しても増えない）
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["推奨日", "code"], keep="first")
    else:
        merged = new
    merged.to_csv(HIST_FILE, index=False, encoding="utf-8-sig")


def track(quotes):
    """履歴の各推奨を、その後の株価で追跡し結果を付ける。
    quotes: 分割調整済みの株価（code, date, close）。"""
    if not os.path.exists(HIST_FILE):
        return pd.DataFrame()
    hist = pd.read_csv(HIST_FILE, dtype={"code": str})
    if hist.empty:
        return hist

    q = quotes.sort_values(["code", "date"])
    closes_by_code = {c: (g["date"].values, g["close"].values)
                      for c, g in q.groupby("code")}

    results = []
    for _, r in hist.iterrows():
        code = str(r["code"])
        entry = float(r["買い指値"])
        rec_date = pd.Timestamp(r["推奨日"])
        arrs = closes_by_code.get(code)
        status = "データなし"
        peak_pct = None
        low_pct = None
        last_pct = None
        result_date = None
        if arrs is not None:
            dates, closes = arrs
            # 推奨日の翌営業日以降を追跡
            mask = dates > np.datetime64(rec_date)
            fut_dates = dates[mask]
            fut_closes = closes[mask]
            if len(fut_closes) > 0:
                rel = (fut_closes / entry - 1.0) * 100
                peak_pct = float(rel.max())
                low_pct = float(rel.min())
                last_pct = float(rel[-1])
                # 利確・損切りの到達を、時系列で先に来たほうで判定
                status = "追跡中"
                for i, pr in enumerate(rel):
                    if pr >= TARGET_PCT:
                        status = "利確+10%到達"
                        result_date = str(pd.Timestamp(fut_dates[i]).date())
                        break
                    if pr <= STOP_PCT:
                        status = "損切り-5%到達"
                        result_date = str(pd.Timestamp(fut_dates[i]).date())
                        break
            else:
                status = "追跡中（翌日以降のデータ待ち）"
        results.append({
            **r.to_dict(),
            "現在まで最大%": round(peak_pct, 1) if peak_pct is not None else None,
            "現在まで最小%": round(low_pct, 1) if low_pct is not None else None,
            "最新時点%": round(last_pct, 1) if last_pct is not None else None,
            "結果": status,
            "結果確定日": result_date or "",
        })
    return pd.DataFrame(results)


def _esc(v):
    return html.escape(str(v))


def _summary_stats(df):
    """確定した結果（利確/損切り到達）の勝率などを集計。"""
    done = df[df["結果"].isin(["利確+10%到達", "損切り-5%到達"])]
    n_done = len(done)
    n_win = (done["結果"] == "利確+10%到達").sum()
    tracking = (df["結果"].astype(str).str.startswith("追跡中")).sum()
    return n_done, int(n_win), int(tracking)


def build_results_html(df):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [f"""<!DOCTYPE html>
<html lang="ja"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>推奨の実績</title>
<style>
  body {{ font-family: -apple-system, "Hiragino Kaku Gothic ProN", sans-serif;
         margin: 0; padding: 16px; background: #f5f6f8; color: #1a1a2e; }}
  .wrap {{ max-width: 760px; margin: 0 auto; }}
  h1 {{ font-size: 20px; margin: 8px 0; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 16px; }}
  .nav a {{ display:inline-block; margin-right:12px; color:#1565c0; text-decoration:none;
           font-size:14px; }}
  .stats {{ background:#fff; border-radius:12px; padding:16px; margin:12px 0;
           box-shadow:0 1px 4px rgba(0,0,0,.08); font-size:14px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:8px;
          background:#fff; border-radius:8px; overflow:hidden; }}
  th, td {{ padding:7px 5px; text-align:right; border-bottom:1px solid #eee; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
  th {{ background:#eef1f5; }}
  .win {{ color:#2e7d32; font-weight:600; }}
  .lose {{ color:#c62828; font-weight:600; }}
  .track {{ color:#888; }}
  .note {{ color:#888; font-size:12px; margin-top:20px; line-height:1.6; }}
</style></head><body><div class="wrap">
<div class="nav"><a href="index.html">← 今日の候補</a><a href="results.html">推奨の実績</a></div>
<h1>推奨の実績（すべての記録）</h1>
<div class="meta">更新: {now}</div>
"""]

    if df is None or df.empty:
        parts.append('<div class="stats">まだ記録がありません。'
                     '毎朝の実行で推奨が記録され、その後の結果がここに貯まります。</div>')
    else:
        n_done, n_win, n_track = _summary_stats(df)
        wr = f"{n_win/n_done*100:.0f}%" if n_done else "—"
        parts.append(f"""<div class="stats">
記録した推奨: <b>{len(df)}件</b>／結果確定: <b>{n_done}件</b>
（利確到達 {n_win}件・損切り到達 {n_done-n_win}件、勝率 <b>{wr}</b>）／
追跡中: {n_track}件
</div>""")
        # 新しい推奨が上に来るよう推奨日で降順
        df = df.sort_values("推奨日", ascending=False)
        head = ["推奨日", "銘柄名", "買い指値", "利確", "損切り",
                "最新時点%", "最大%", "最小%", "結果"]
        thead = "".join(f"<th>{_esc(h)}</th>" for h in head)
        rows = []
        for _, r in df.iterrows():
            res = str(r["結果"])
            cls = "win" if "利確" in res else ("lose" if "損切り" in res else "track")
            tds = [
                _esc(r["推奨日"]),
                f'{_esc(r["銘柄名"])}<br><span style="color:#999">{_esc(r["code"])}</span>',
                f'{r["買い指値"]:,.1f}',
                f'{r["利確"]:,.1f}',
                f'{r["損切り"]:,.1f}',
                _esc(r.get("最新時点%", "")),
                _esc(r.get("現在まで最大%", "")),
                _esc(r.get("現在まで最小%", "")),
                f'<span class="{cls}">{_esc(res)}</span>',
            ]
            rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in tds) + "</tr>")
        parts.append(f"<table><thead><tr>{thead}</tr></thead>"
                     f"<tbody>{''.join(rows)}</tbody></table>")

    parts.append("""<div class="note">
※ 買い指値＝推奨日の終値。翌日その値で約定できたと仮定した追跡です（実際は前後します）。<br>
※ 利確+10%・損切り-5%は終値ベース判定。到達順で先に来たほうを結果とします。<br>
※ 手数料・スリッページは未考慮。過去の記録であり、将来を保証しません。<br>
※ 投資助言ではありません。売買の判断と責任は利用者ご自身にあります。
</div></div></body></html>""")
    return "".join(parts)


def build_and_save_results(quotes):
    """追跡して実績ページを保存する。"""
    df = track(quotes)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.html"), "w", encoding="utf-8") as f:
        f.write(build_results_html(df))
    return df
