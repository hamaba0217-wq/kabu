# -*- coding: utf-8 -*-
"""
今日の運用候補を、スマホで見やすいHTMLページにして出力する。
GitHub Actions で毎朝実行し、GitHub Pages で公開する用途。

出力先: docs/index.html （GitHub Pages が docs/ を公開する設定にする）
"""

from __future__ import annotations

import datetime as dt
import html
import os

import numpy as np
import pandas as pd

import config
from candidates_best import find as find_best

MIN_N = 30
STOP_PCT = -5.0
TARGET_PCT = 10.0
OUT_DIR = "docs"


def _pick_strong(df):
    """運用サマリーに出す『特に有力』な候補を絞る（candidates_best と同基準）。"""
    d = df.copy()
    if "過去該当n" in d.columns:
        d = d[d["過去該当n"] >= MIN_N]
    if "平均_損切5%" in d.columns:
        d = d[d["平均_損切5%"] > 0]
    if "2週で+10%到達%" in d.columns and "2週で-5%到達%" in d.columns:
        d = d[d["2週で+10%到達%"].fillna(0) >= d["2週で-5%到達%"].fillna(999)]
    if "平均_損切5%" in d.columns:
        d = d.sort_values("平均_損切5%", ascending=False)
    return d


def _esc(v):
    return html.escape(str(v))


def build_html(df, as_of):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    strong = _pick_strong(df) if not df.empty else pd.DataFrame()

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="ja"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>今日の株式運用候補</title>
<style>
  body {{ font-family: -apple-system, "Hiragino Kaku Gothic ProN", sans-serif;
         margin: 0; padding: 16px; background: #f5f6f8; color: #1a1a2e; }}
  .wrap {{ max-width: 720px; margin: 0 auto; }}
  h1 {{ font-size: 20px; margin: 8px 0; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 16px; }}
  .card {{ background: #fff; border-radius: 12px; padding: 16px; margin: 12px 0;
          box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .strong {{ border-left: 4px solid #2e7d32; }}
  .name {{ font-size: 17px; font-weight: 700; }}
  .sec {{ color: #666; font-size: 13px; }}
  .price {{ margin: 8px 0; font-size: 15px; }}
  .stop {{ color: #c62828; font-weight: 600; }}
  .target {{ color: #2e7d32; font-weight: 600; }}
  .stat {{ color: #444; font-size: 13px; margin-top: 6px; }}
  .none {{ background: #fff8e1; padding: 16px; border-radius: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
  th, td {{ padding: 6px 4px; text-align: right; border-bottom: 1px solid #eee; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  .note {{ color: #888; font-size: 12px; margin-top: 20px; line-height: 1.6; }}
</style></head><body><div class="wrap">
<h1>今日の株式運用候補</h1>
<div class="nav" style="margin-bottom:12px;">
  <a href="index.html" style="color:#1565c0;text-decoration:none;margin-right:12px;">今日の候補</a>
  <a href="results.html" style="color:#1565c0;text-decoration:none;">推奨の実績 →</a>
</div>
<div class="meta">データ時点: {_esc(as_of)} ／ 生成: {now}</div>
""")

    # 運用サマリー
    parts.append('<h1>★ 特に有力な候補</h1>')
    if strong.empty:
        parts.append('<div class="none">本日は、実績・リスクの条件を満たす'
                     '「特に有力」な候補はありません。無理に買う必要はありません。</div>')
    else:
        for _, r in strong.head(5).iterrows():
            close = r.get("終値")
            name = _esc(r.get("銘柄名", ""))
            code = _esc(r.get("code", ""))
            sec = _esc(r.get("業種", ""))
            if close and close == close:
                entry = float(close)
                stop = entry * (1 + STOP_PCT / 100)
                target = entry * (1 + TARGET_PCT / 100)
                parts.append(f"""<div class="card strong">
<div class="name">{name} <span class="sec">（{code}）{sec}</span></div>
<div class="price">買い指値 <b>{entry:,.1f}円</b>（現値）<br>
  <span class="target">売り/利確 {target:,.1f}円(+10%)</span> ／
  <span class="stop">損切り {stop:,.1f}円(-5%)</span></div>
<div class="stat">過去: 平均(損切5%){r.get('平均_損切5%'):+.2f}% ／
  2週で+10%到達 {r.get('2週で+10%到達%','?')}% ／
  -5%到達 {r.get('2週で-5%到達%','?')}% ／ n{r.get('過去該当n')}</div>
</div>""")

    # 候補一覧テーブル
    if not df.empty:
        parts.append(f'<h1>候補一覧（{len(df)}件）</h1>')
        cols = [("銘柄名", "銘柄"), ("業種", "業種"), ("高値からの下落%", "下落%"),
                ("平均_損切5%", "平均(損切5%)"), ("2週で+10%到達%", "+10%到達"),
                ("2週で-5%到達%", "-5%到達"), ("過去該当n", "n")]
        cols = [(c, lbl) for c, lbl in cols if c in df.columns]
        thead = "".join(f"<th>{_esc(lbl)}</th>" for _, lbl in cols)
        rows = []
        for _, r in df.iterrows():
            tds = "".join(f"<td>{_esc(r.get(c,''))}</td>" for c, _ in cols)
            rows.append(f"<tr>{tds}</tr>")
        parts.append(f"<table><thead><tr>{thead}</tr></thead>"
                     f"<tbody>{''.join(rows)}</tbody></table>")

    parts.append("""<div class="note">
※ このページは過去データにもとづく分析補助であり、投資助言ではありません。<br>
※ 売買の判断と結果の責任は、すべて利用者ご自身にあります。<br>
※ 損切り-5%・利確+10%は検証で用いた目安。手数料・スリッページは未考慮です。<br>
※ 「特に有力」= 過去該当n≧30・平均(損切5%)プラス・+10%到達≧-5%到達 を満たすもの。
</div></div></body></html>""")
    return "".join(parts)


def main():
    from sources import JQuants, JST
    jq = JQuants()
    print("データを読み込み中...")
    # 運用候補（押し目×高勝率業種）に必要な期間だけ取得する。
    # 52週高値の計算に1年強あれば足りるので、クラウドでも短時間で終わる。
    # BACKTEST_LOOKBACK_DAYS(2年)は重すぎる（クラウドで毎回ゼロから取得するため）。
    QUOTE_DAYS = 450   # 株価：52週高値+αに十分
    FIN_DAYS = 500     # 決算：前年比の計算に1年強
    quotes = jq.quotes(QUOTE_DAYS)
    fin = jq.financials(FIN_DAYS)
    listed = jq.listed()
    # 信用残・空売り比率は候補の補助情報。取得失敗しても候補は出せるので、
    # クラウドの負荷軽減のため短期間だけ試し、失敗しても続行する。
    try:
        margin = jq.margin(60)
    except Exception:
        margin = None
    try:
        short_ratio = jq.short_ratio(60)
    except Exception:
        short_ratio = None

    df, as_of = find_best(quotes, fin, listed, margin=margin, short_ratio=short_ratio)
    html_str = build_html(df, as_of)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"HTMLレポートを生成しました: {path}")
    print(f"候補 {len(df)}件 / データ時点 {as_of}")

    # 推奨を履歴に記録し、その後の結果を追跡して実績ページを生成
    import track_picks
    strong = _pick_strong(df) if not df.empty else df
    track_picks.record_picks(strong, as_of)
    res = track_picks.build_and_save_results(quotes)
    n_rec = len(res) if res is not None else 0
    print(f"推奨の実績ページを生成しました: {OUT_DIR}/results.html（記録 {n_rec}件）")


if __name__ == "__main__":
    main()
