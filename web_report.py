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
    from web_common import page_head, page_foot
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    strong = _pick_strong(df) if not df.empty else pd.DataFrame()

    parts = []
    parts.append(page_head("今日の運用候補", "index.html"))
    parts.append(f'<h1>今日の株式運用候補</h1>')
    parts.append(f'<div class="meta">データ時点: {_esc(as_of)} ／ 生成: {now}</div>')

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
                ("売買代金(3日平均)億", "代金億"),
                ("平均_損切5%", "平均(損切5%)"), ("2週で+10%到達%", "+10%到達"),
                ("2週で-5%到達%", "-5%到達"), ("過去該当n", "n")]
        cols = [(c, lbl) for c, lbl in cols if c in df.columns]
        thead = "".join(f"<th>{_esc(lbl)}</th>" for _, lbl in cols)
        rows = []
        for _, r in df.iterrows():
            tds = "".join(f"<td>{_esc(r.get(c,''))}</td>" for c, _ in cols)
            rows.append(f"<tr>{tds}</tr>")
        parts.append(f'<div class="tablewrap"><table><thead><tr>{thead}</tr></thead>'
                     f"<tbody>{''.join(rows)}</tbody></table></div>")

    note = ('<div class="note">'
            '※ 「特に有力」= 過去該当n≧30・平均(損切5%)プラス・+10%到達≧-5%到達 を満たすもの。'
            '損切り-5%・利確+10%は検証で用いた目安。</div>')
    parts.append(page_foot(note))
    return "".join(parts)


def main():
    from sources import JQuants, JST
    jq = JQuants()
    print("データを読み込み中...")
    # 通常はデータを限界まで使う（分析の質・勝率を優先）。data/ をキャッシュして差分取得。
    # ただし初回はキャッシュが無くフル取得が重い。環境変数 LIGHT_MODE=1 のときは
    # 期間を絞って軽く取得し、まずキャッシュを作る（初回のタイムアウト回避用）。
    # 初回のキャッシュ作成を確実にするため、当面は軽量モードを既定にする。
    # 環境変数 LIGHT_MODE=0 を明示したときだけフル取得する。
    # （キャッシュができたら、web.yml で LIGHT_MODE=0 に変えてフルに戻せる）
    _lm = os.environ.get("LIGHT_MODE", "").strip().lower()
    light = _lm not in ("0", "false", "no")   # 空・未設定・1・true はすべて軽量
    if light:
        print("  [LIGHT_MODE] 軽量取得（期間短縮）でキャッシュを作成/更新します", flush=True)
        quote_days, fin_days, sd_days = 400, 400, 60
    else:
        print("  [FULLモード] データを限界まで取得します", flush=True)
        quote_days = fin_days = sd_days = config.BACKTEST_LOOKBACK_DAYS
    quotes = jq.quotes(quote_days)
    print(f"  株価取得完了: {len(quotes):,}行", flush=True)
    fin = jq.financials(fin_days)
    print(f"  決算取得完了: {len(fin):,}行", flush=True)
    listed = jq.listed()
    print(f"  銘柄取得完了: {len(listed):,}件", flush=True)
    try:
        margin = jq.margin(sd_days)
        print(f"  信用残取得完了: {len(margin):,}行", flush=True)
    except Exception as e:
        print(f"  信用残スキップ: {e}", flush=True)
        margin = None
    try:
        short_ratio = jq.short_ratio(sd_days)
        print(f"  空売り比率取得完了: {len(short_ratio):,}行", flush=True)
    except Exception as e:
        print(f"  空売り比率スキップ: {e}", flush=True)
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

    # 追加の研究ページ（業種別成績・市場の状況・検証の記録）
    import web_extras
    made = web_extras.build_and_save_extras(quotes, listed, res)
    print(f"追加ページを生成しました: {', '.join(made)}")


if __name__ == "__main__":
    main()
