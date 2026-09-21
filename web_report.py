# -*- coding: utf-8 -*-
"""
web_report.py
1. daily_morning_routine を実行（0件でも必ず最新日付でCSVを生成）し、株価と推奨を受け取る
2. 今日の推奨を history/picks.csv に追記（蓄積）
3. docs/index.html（今日の候補）と docs/results.html（推奨のその後＝実績）を生成
   ※ J-Quantsは daily_morning_routine の1回だけ取得し、追跡でも再利用（二重取得しない）
"""
from __future__ import annotations
import os
import glob
import html
import datetime as dt
import pandas as pd

import config
from daily_morning_routine import run_morning_routine
import track_picks

DOCS = "docs"


def _esc(v):
    return html.escape(str(v))


# ---- 両ページ共通のスタイルとメニュー ----------------------------------
BASE_STYLE = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background: #f8f9fa; color: #333; line-height: 1.6; }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 16px; }
    h1 { color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; font-size: 1.5em; }
    h2 { color: #2c3e50; margin-top: 26px; font-size: 1.2em; border-left: 4px solid #1a73e8; padding-left: 10px; }
    .nav { display: flex; flex-wrap: wrap; gap: 6px; background: #1a73e8; padding: 8px 12px; position: sticky; top: 0; z-index: 10; }
    .nav a { color: #fff; text-decoration: none; font-weight: 600; font-size: 0.9em; padding: 6px 10px; border-radius: 6px; }
    .nav a.active { background: rgba(255,255,255,0.25); }
    .nav a:hover { background: rgba(255,255,255,0.15); }
    .date-badge { background: #e8f0fe; color: #1a73e8; padding: 5px 12px; border-radius: 4px; font-weight: bold; display: inline-block; margin-bottom: 20px; }
    .strategy-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
    .strategy-card { background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); padding: 20px; border-top: 4px solid #1a73e8; }
    .strategy-card.alt { border-top-color: #28a745; }
    .strategy-card h3 { margin-top: 0; color: #1a73e8; font-size: 1.1em; }
    .strategy-card.alt h3 { color: #28a745; }
    .strategy-card p { font-size: 0.9em; margin-bottom: 10px; }
    .strategy-card ul { font-size: 0.85em; padding-left: 20px; color: #555; }
    .tablewrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.08); margin-top: 15px; font-size: 0.9em; }
    th, td { padding: 11px 13px; text-align: center; border-bottom: 1px solid #eee; white-space: nowrap; }
    th { background: #1a73e8; color: #fff; font-weight: 600; cursor: pointer; user-select: none; }
    th[data-sortable]:after { content: " \\2195"; opacity: 0.5; font-size: 0.85em; }
    tr:hover { background: #f1f3f4; }
    .badge-main { background: #e8f0fe; color: #1a73e8; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }
    .badge-sub { background: #e6f4ea; color: #137333; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }
    .price-buy { font-weight: bold; color: #2c3e50; }
    .price-tp { font-weight: bold; color: #137333; }
    .price-sl { font-weight: bold; color: #d93025; }
    .win { color: #137333; font-weight: bold; }
    .lose { color: #d93025; font-weight: bold; }
    .track { color: #8a6d00; font-weight: bold; }
    .card { background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); padding: 16px 20px; margin-top: 12px; }
    .no-data { text-align: center; padding: 40px; color: #666; background: #fff; border-radius: 8px; margin-top: 15px; }
    .note { color: #888; font-size: 0.82em; margin-top: 16px; }
    footer { margin-top: 40px; text-align: center; font-size: 0.8em; color: #888; padding-bottom: 30px; }
    .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-top: 15px; }
    .chart-card { background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); padding: 12px 14px; }
    .chart-card .ttl { font-weight: 700; color: #2c3e50; font-size: 0.95em; }
    .chart-card .sub { color: #888; font-size: 0.8em; margin-bottom: 6px; }
    .chart-card .px { font-size: 0.8em; margin-top: 4px; }
    .chart-card .px b { color: #2c3e50; } .chart-card .px .tp { color: #137333; } .chart-card .px .sl { color: #d93025; }
    svg.candles { width: 100%; height: auto; display: block; border: 1px solid #f0f0f0; border-radius: 4px; }
    .chart-none { color: #999; font-size: 0.85em; padding: 20px; text-align: center; }
    .legend { font-size: 0.78em; color: #777; margin: 6px 0 0; }
    .legend .up { color: #e53935; font-weight: 700; } .legend .dn { color: #1e88e5; font-weight: 700; }
"""

# クリックで昇順↔降順に並び替え（数値/日付は数値として、それ以外は文字列として比較）
SORT_SCRIPT = """
<script>
document.querySelectorAll('table').forEach(function(tbl){
  var ths = tbl.querySelectorAll('thead th[data-sortable]');
  ths.forEach(function(th, idx){
    var col = th.cellIndex;
    var asc = true;
    th.addEventListener('click', function(){
      var tb = tbl.querySelector('tbody');
      var rows = Array.prototype.slice.call(tb.querySelectorAll('tr'));
      rows.sort(function(a,b){
        var x = a.cells[col].getAttribute('data-v') || a.cells[col].innerText;
        var y = b.cells[col].getAttribute('data-v') || b.cells[col].innerText;
        var nx = parseFloat(String(x).replace(/[^0-9.\\-]/g,''));
        var ny = parseFloat(String(y).replace(/[^0-9.\\-]/g,''));
        var both = !isNaN(nx) && !isNaN(ny);
        if (both) return asc ? nx-ny : ny-nx;
        return asc ? String(x).localeCompare(String(y),'ja') : String(y).localeCompare(String(x),'ja');
      });
      rows.forEach(function(r){ tb.appendChild(r); });
      asc = !asc;
    });
  });
});
</script>
"""


def _candles_svg(dfc, buy=None, tp=None, sl=None, months=6):
    """1銘柄の日足ローソクを SVG で描画（直近~6か月）。
    dfc: code の株価（date, open, high, low, close を含む）。外部JS不使用。"""
    if dfc is None or len(dfc) == 0:
        return '<div class="chart-none">チャートデータなし</div>'
    d = dfc.sort_values("date")
    need = [c for c in ("open", "high", "low", "close") if c in d.columns]
    if len(need) < 4:
        return '<div class="chart-none">OHLCデータなし</div>'
    n = int(months * 21)  # 1か月≒21営業日
    d = d.tail(n)
    o = d["open"].to_numpy(dtype=float)
    h = d["high"].to_numpy(dtype=float)
    lo = d["low"].to_numpy(dtype=float)
    c = d["close"].to_numpy(dtype=float)
    m = len(c)
    if m == 0:
        return '<div class="chart-none">チャートデータなし</div>'

    W, H = 560.0, 200.0
    padL, padR, padT, padB = 6.0, 52.0, 8.0, 16.0
    plotW = W - padL - padR
    plotH = H - padT - padB

    lows = [x for x in lo if x == x]
    highs = [x for x in h if x == x]
    pmin = min(lows) if lows else 0.0
    pmax = max(highs) if highs else 1.0
    for v in (buy, tp, sl):
        if v is not None and v == v:
            pmin = min(pmin, float(v)); pmax = max(pmax, float(v))
    if pmax <= pmin:
        pmax = pmin + 1.0
    rng = pmax - pmin

    def y(p):
        return padT + (pmax - p) / rng * plotH

    step = plotW / m
    bw = max(1.2, min(6.0, step * 0.65))
    up, down = "#e53935", "#1e88e5"  # 陽線=赤 / 陰線=青（日本式）

    parts = [f'<svg viewBox="0 0 {W:.0f} {H:.0f}" class="candles" '
             f'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">']
    parts.append(f'<rect x="0" y="0" width="{W:.0f}" height="{H:.0f}" fill="#fff"/>')

    # 補助線・目盛（上・中・下）
    for frac in (0.0, 0.5, 1.0):
        pv = pmax - frac * rng
        yy = y(pv)
        parts.append(f'<line x1="{padL:.1f}" y1="{yy:.1f}" x2="{padL+plotW:.1f}" y2="{yy:.1f}" '
                     f'stroke="#eee" stroke-width="1"/>')
        parts.append(f'<text x="{padL+plotW+4:.1f}" y="{yy+3:.1f}" font-size="9" fill="#999">{pv:,.0f}</text>')

    # ローソク
    for i in range(m):
        cx = padL + i * step + step / 2.0
        oi, hi, li, ci = o[i], h[i], lo[i], c[i]
        if not (hi == hi and li == li):
            continue
        col = up if (ci >= oi) else down
        parts.append(f'<line x1="{cx:.1f}" y1="{y(hi):.1f}" x2="{cx:.1f}" y2="{y(li):.1f}" '
                     f'stroke="{col}" stroke-width="1"/>')
        y1, y2 = y(oi), y(ci)
        top = min(y1, y2); bh = max(0.8, abs(y2 - y1))
        parts.append(f'<rect x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{col}"/>')

    # 指値ライン（買い=灰・利確=緑・損切り=赤）
    def hline(v, color, label):
        if v is None or not (v == v):
            return
        yy = y(float(v))
        parts.append(f'<line x1="{padL:.1f}" y1="{yy:.1f}" x2="{padL+plotW:.1f}" y2="{yy:.1f}" '
                     f'stroke="{color}" stroke-width="1" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{padL+2:.1f}" y="{yy-2:.1f}" font-size="9" fill="{color}">{label}</text>')
    hline(tp, "#137333", "利確")
    hline(buy, "#5f6368", "買い")
    hline(sl, "#d93025", "損切り")

    parts.append("</svg>")
    return "".join(parts)


def _nav(active):
    def cls(p):
        return ' class="active"' if p == active else ""
    return (f'<div class="nav">'
            f'<a href="index.html"{cls("index")}>📋 今日の候補</a>'
            f'<a href="recent.html"{cls("recent")}>🗓 過去10日の推奨</a>'
            f'<a href="results.html"{cls("results")}>📈 推奨のその後（実績）</a>'
            f'</div>')


def _page(title, active, body):
    return f"""<!DOCTYPE html>
<html lang="ja"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_esc(title)}</title>
<style>{BASE_STYLE}</style>
</head><body>
{_nav(active)}
<div class="wrap">
{body}
<footer><p>© 2026 株式投資自動スクリーニングシステム ／ 投資助言ではありません・自己責任・手数料未考慮</p></footer>
</div>
{SORT_SCRIPT}
</body></html>"""


# ---- 今日の候補ページ（index.html）------------------------------------
def _build_index(df, target_date, quotes=None):
    body = [f'<h1>今日の株式運用候補 ＆ 指値プラン</h1>',
            f'<div class="date-badge">基準日: {_esc(target_date)}</div>',
            '<h2>💡 検証方法・アプローチ別の特徴とおすすめ</h2>',
            '''<div class="strategy-container">
        <div class="strategy-card">
            <h3>📈 【本命セクター重視】アプローチ</h3>
            <p><strong>考え方:</strong> 過去のバックテストデータから「勝率55%以上」が実証されている高勝率セクター（海運、銀行、パルプ・紙など）に資金を集中させる方法。</p>
            <p><strong>こんな方におすすめ:</strong></p>
            <ul><li>相場全体のトレンドに逆らわず、確率的に優位な業種から手堅くリターンを狙いたい方</li>
                <li>セクターローテーションの波に乗って安定感を重視したい方</li></ul>
        </div>
        <div class="strategy-card alt">
            <h3>⚡ 【高スコア・モメンタム急増】アプローチ</h3>
            <p><strong>考え方:</strong> 52週高値から大幅に下落した大底圏（-30%以下）にあり、かつ本日の出来高が急増している銘柄をスコアリングで多角的に抽出する方法。</p>
            <p><strong>こんな方におすすめ:</strong></p>
            <ul><li>底打ちからの急反発（リバウンド）による大きな値幅や短期的な利益を狙いたい方</li>
                <li>話題性や出来高の勢い（モメンタム）を重視したダイナミックなトレードが好きな方</li></ul>
        </div>
    </div>''',
            '<h2>📋 本日のスクリーニング＆指値結果一覧</h2>']

    if df is not None and not df.empty:
        body.append('<p class="legend">売買代金5億円以上からスコア上位8銘柄を表示'
                    '（<b>【本命】条件クリア</b>＝高値-30%超×出来高急増×値幅大×地合い良、を優先）。'
                    '指値・利確・損切りは目安、投資助言ではありません。</p>')
        head = ["銘柄コード","企業名","業種","検証区分","おすすめ度(%)","買い指値","利確目標 (TP+10%)","損切ライン (SL-5%)","出来高倍率"]
        thead = "".join(f'<th data-sortable>{_esc(h)}</th>' for h in head)
        rows = []
        for _, r in df.iterrows():
            badge = "badge-main" if "本命" in str(r["区分"]) else "badge-sub"
            rows.append(
                "<tr>"
                f'<td><strong>{_esc(r["銘柄コード"])}</strong></td>'
                f'<td><strong>{_esc(r["企業名"])}</strong></td>'
                f'<td>{_esc(r["業種"])}</td>'
                f'<td><span class="{badge}">{_esc(r["区分"])}</span></td>'
                f'<td data-v="{r["おすすめ度(%)"]}"><strong>{r["おすすめ度(%)"]}%</strong></td>'
                f'<td class="price-buy" data-v="{r["買い指値"]}">{r["買い指値"]:,} 円</td>'
                f'<td class="price-tp" data-v="{r["利確目標(TP+10%)"]}">+{r["利確目標(TP+10%)"]:,} 円</td>'
                f'<td class="price-sl" data-v="{r["損切ライン(SL-5%)"]}">-{r["損切ライン(SL-5%)"]:,} 円</td>'
                f'<td data-v="{r["出来高倍率"]}">{r["出来高倍率"]} 倍</td>'
                "</tr>")
        body.append(f'<div class="tablewrap"><table><thead><tr>{thead}</tr></thead>'
                    f'<tbody>{"".join(rows)}</tbody></table></div>')

        # --- 各銘柄のローソク（6か月・日足）---
        body.append('<h2>📈 おすすめ銘柄のチャート（6か月・日足）</h2>')
        body.append('<p class="legend"><span class="up">■</span> 陽線（終値≥始値）　'
                    '<span class="dn">■</span> 陰線（終値&lt;始値）　'
                    '／ 点線：<span style="color:#5f6368">買い</span>・'
                    '<span style="color:#137333">利確+10%</span>・'
                    '<span style="color:#d93025">損切り-5%</span></p>')
        if quotes is not None and not quotes.empty:
            qs = quotes.copy()
            qs["code"] = qs["code"].astype(str)
            by_code = {code: g for code, g in qs.groupby("code")}
            cards = []
            for _, r in df.iterrows():
                try:
                    code = str(r["銘柄コード"])
                    dfc = by_code.get(code)
                    if dfc is None:
                        dfc = by_code.get(code[:4])
                    if dfc is None:
                        dfc = by_code.get(code + "0")
                    svg = _candles_svg(dfc, buy=r.get("買い指値"),
                                       tp=r.get("利確目標(TP+10%)"), sl=r.get("損切ライン(SL-5%)"))
                    def _yen(v):
                        try:
                            return f"{float(v):,.1f}円"
                        except (TypeError, ValueError):
                            return f"{_esc(v)}円"
                    cards.append(
                        '<div class="chart-card">'
                        f'<div class="ttl">{_esc(r["企業名"])} <span style="color:#999">({_esc(code)})</span></div>'
                        f'<div class="sub">{_esc(r.get("業種",""))}｜{_esc(r.get("区分",""))}</div>'
                        f'{svg}'
                        f'<div class="px"><b>買い {_yen(r.get("買い指値"))}</b>　'
                        f'<span class="tp">利確 {_yen(r.get("利確目標(TP+10%)"))}</span>　'
                        f'<span class="sl">損切 {_yen(r.get("損切ライン(SL-5%)"))}</span></div>'
                        '</div>')
                except Exception as e:
                    cards.append(f'<div class="chart-card"><div class="chart-none">チャート生成エラー: {_esc(e)}</div></div>')
            body.append(f'<div class="chart-grid">{"".join(cards)}</div>')
        else:
            body.append('<div class="chart-none">チャート用の株価データがありません'
                        '（クラウドの毎朝実行で表示されます）。</div>')
    else:
        body.append('<div class="no-data"><p>本日は候補を表示できませんでした'
                    '（株価データ未取得、または売買代金5億円以上の銘柄が抽出できませんでした）。'
                    '翌営業日の自動実行で更新されます。</p></div>')

    return _page("今日の株式運用候補 ＆ 指値プラン", "index", "\n".join(body))


# ---- 推奨のその後ページ（results.html）--------------------------------
def _build_results(tracked):
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")
    body = ['<h1>📈 推奨のその後（実績）</h1>',
            f'<div class="date-badge">更新: {now}</div>']

    if tracked is None or tracked.empty:
        body.append('<div class="no-data"><p>まだ記録がありません。'
                    '毎朝の実行で推奨が記録され、その後の結果がここに貯まっていきます。</p></div>')
        body.append('<div class="note">※ 買い指値＝推奨日の終値。翌日その値で約定できたと仮定した追跡です。'
                    '利確+10%・損切り-5%は終値ベース、到達順で先に来たほうを結果とします。手数料・スリッページ未考慮。</div>')
        return _page("推奨のその後（実績）", "results", "\n".join(body))

    done = tracked[tracked["結果"].isin(["利確+10%到達", "損切り-5%到達"])]
    n_done = len(done)
    n_win = int((done["結果"] == "利確+10%到達").sum())
    n_track = int(tracked["結果"].astype(str).str.startswith("追跡中").sum())
    wr = f"{n_win / n_done * 100:.0f}%" if n_done else "—"
    body.append(f'<div class="card">記録した推奨: <b>{len(tracked)}件</b>／'
                f'結果確定: <b>{n_done}件</b>（利確 {n_win}件・損切り {n_done - n_win}件、'
                f'勝率 <b>{wr}</b>）／追跡中: <b>{n_track}件</b></div>')

    tracked = tracked.sort_values("推奨日", ascending=False)
    head = ["推奨日","銘柄名","区分","買い指値","利確","損切り","最新時点%","最大%","最小%","結果"]
    thead = "".join(f'<th data-sortable>{_esc(h)}</th>' for h in head)
    rows = []
    for _, r in tracked.iterrows():
        res = str(r["結果"])
        cls = "win" if "利確" in res else ("lose" if "損切り" in res else "track")
        def num(v):
            return "" if v is None or (isinstance(v, float) and v != v) else v
        rows.append(
            "<tr>"
            f'<td>{_esc(r["推奨日"])}</td>'
            f'<td>{_esc(r["銘柄名"])}<br><span style="color:#999;font-size:0.85em">{_esc(r["code"])}</span></td>'
            f'<td>{_esc(r.get("区分",""))}</td>'
            f'<td data-v="{r["買い指値"]}">{r["買い指値"]:,.1f}</td>'
            f'<td data-v="{r["利確"]}">{r["利確"]:,.1f}</td>'
            f'<td data-v="{r["損切り"]}">{r["損切り"]:,.1f}</td>'
            f'<td data-v="{num(r.get("最新時点%"))}">{_esc(num(r.get("最新時点%")))}</td>'
            f'<td data-v="{num(r.get("現在まで最大%"))}">{_esc(num(r.get("現在まで最大%")))}</td>'
            f'<td data-v="{num(r.get("現在まで最小%"))}">{_esc(num(r.get("現在まで最小%")))}</td>'
            f'<td><span class="{cls}">{_esc(res)}</span></td>'
            "</tr>")
    body.append(f'<div class="tablewrap"><table><thead><tr>{thead}</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table></div>')
    body.append('<div class="note">※ 買い指値＝推奨日の終値。翌日その値で約定できたと仮定した追跡です。'
                '利確+10%・損切り-5%は終値ベース、到達順で先に来たほうを結果とします。'
                '記録開始より前の推奨は復元できません。手数料・スリッページ未考慮・投資助言ではありません。</div>')
    return _page("推奨のその後（実績）", "results", "\n".join(body))


def _build_recent(tracked, days=10):
    """直近10営業日分（推奨日ベース）の推奨を、日付ごとにまとめて表示。実データのみ。"""
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")
    body = ['<h1>🗓 過去10日の推奨</h1>',
            f'<div class="date-badge">更新: {now}</div>']

    if tracked is None or tracked.empty:
        body.append('<div class="no-data"><p>まだ記録がありません。'
                    '毎朝の実行で推奨が記録され、直近10営業日分がここに表示されます。</p></div>')
        return _page("過去10日の推奨", "recent", "\n".join(body))

    t = tracked.copy()
    t["推奨日"] = t["推奨日"].astype(str)
    recent_dates = sorted(t["推奨日"].unique(), reverse=True)[:days]
    t = t[t["推奨日"].isin(recent_dates)]

    # 全体サマリ（この10日分）
    done = t[t["結果"].isin(["利確+10%到達", "損切り-5%到達"])]
    n_done = len(done); n_win = int((done["結果"] == "利確+10%到達").sum())
    wr = f"{n_win / n_done * 100:.0f}%" if n_done else "—"
    body.append(f'<div class="card">直近{len(recent_dates)}営業日の推奨: <b>{len(t)}件</b>／'
                f'結果確定: <b>{n_done}件</b>（利確 {n_win}・損切り {n_done - n_win}、勝率 <b>{wr}</b>）</div>')

    def num(v):
        return "" if v is None or (isinstance(v, float) and v != v) else v

    for d in recent_dates:
        g = t[t["推奨日"] == d]
        body.append(f'<h2>{_esc(d)}（{len(g)}件）</h2>')
        head = ["銘柄名", "区分", "買い指値", "利確", "損切り", "最新時点%", "最大%", "最小%", "結果"]
        thead = "".join(f'<th data-sortable>{_esc(h)}</th>' for h in head)
        rows = []
        for _, r in g.iterrows():
            res = str(r["結果"])
            cls = "win" if "利確" in res else ("lose" if "損切り" in res else "track")
            rows.append(
                "<tr>"
                f'<td>{_esc(r["銘柄名"])}<br><span style="color:#999;font-size:0.85em">{_esc(r["code"])}</span></td>'
                f'<td>{_esc(r.get("区分", ""))}</td>'
                f'<td data-v="{r["買い指値"]}">{r["買い指値"]:,.1f}</td>'
                f'<td data-v="{r["利確"]}">{r["利確"]:,.1f}</td>'
                f'<td data-v="{r["損切り"]}">{r["損切り"]:,.1f}</td>'
                f'<td data-v="{num(r.get("最新時点%"))}">{_esc(num(r.get("最新時点%")))}</td>'
                f'<td data-v="{num(r.get("現在まで最大%"))}">{_esc(num(r.get("現在まで最大%")))}</td>'
                f'<td data-v="{num(r.get("現在まで最小%"))}">{_esc(num(r.get("現在まで最小%")))}</td>'
                f'<td><span class="{cls}">{_esc(res)}</span></td>'
                "</tr>")
        body.append(f'<div class="tablewrap"><table><thead><tr>{thead}</tr></thead>'
                    f'<tbody>{"".join(rows)}</tbody></table></div>')

    body.append('<div class="note">※ 買い指値＝推奨日の終値。翌日約定と仮定した実データ追跡。'
                '利確+10%・損切り-5%は終値ベース、到達順で先に来たほうを結果とします。'
                '手数料・スリッページ未考慮・投資助言ではありません。</div>')
    return _page("過去10日の推奨", "recent", "\n".join(body))


def generate_web_report(df=None, quotes=None, target_date=None):
    print("=" * 76)
    print("Webレポート（docs/index.html・results.html）生成中...")
    print("=" * 76)
    os.makedirs(DOCS, exist_ok=True)

    # df/quotes が渡されなければ、CSVから復元（indexのみ・追跡はスキップ）
    if df is None:
        pattern = os.path.join(config.OUTPUT_DIR, "*_morning_orders.csv")
        files = glob.glob(pattern)
        df = pd.DataFrame()
        target_date = target_date or dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date().isoformat()
        if files:
            latest = max(files, key=os.path.getmtime)
            try:
                target_date = os.path.basename(latest).split("_")[0]
                df = pd.read_csv(latest)
                print(f"  最新の指値CSVを読み込み: {latest}（{len(df)}件）")
            except Exception as e:
                print(f"  CSV読み込みエラー: {e}")

    if target_date is None:
        target_date = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date().isoformat()
    else:
        target_date = str(pd.Timestamp(target_date).date()) if not isinstance(target_date, str) else target_date

    # index.html
    try:
        html_index = _build_index(df, target_date, quotes=quotes)
    except Exception as e:
        import traceback; traceback.print_exc()
        html_index = _page("今日の株式運用候補", "index",
                           f'<h1>今日の株式運用候補</h1><div class="no-data">生成中にエラー: {_esc(e)}</div>')
    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_index)
    print(f"  docs/index.html を生成しました")

    # results.html（追跡）— quotes があるときだけ最新まで追跡。無くても既存履歴で生成
    tracked = None
    try:
        if df is not None and not df.empty:
            track_picks.record_from_orders(df, target_date)
        tracked = track_picks.track(quotes) if quotes is not None else track_picks.track(_dummy_quotes())
    except Exception as e:
        print(f"  追跡でエラー（実績ページは前回内容/空で継続）: {e}")
    with open(os.path.join(DOCS, "results.html"), "w", encoding="utf-8") as f:
        f.write(_build_results(tracked))
    print(f"  docs/results.html を生成しました")

    # recent.html（過去10日の推奨）
    try:
        with open(os.path.join(DOCS, "recent.html"), "w", encoding="utf-8") as f:
            f.write(_build_recent(tracked, days=10))
        print(f"  docs/recent.html を生成しました")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  recent.html 生成エラー: {e}")


def _dummy_quotes():
    """quotes が無い環境向け。履歴は表示するが追跡結果は『データなし』となる。"""
    return pd.DataFrame(columns=["code", "date", "close"])


def main():
    print(">>> 1. スクリーニング実行中（daily_morning_routine）...")
    df = quotes = target_date = None
    try:
        ret = run_morning_routine()
        if isinstance(ret, tuple):
            df, quotes, target_date = (list(ret) + [None, None, None])[:3]
    except SystemExit as e:
        # JQuants() はAPIキー未設定/ネット中断で SystemExit を投げる。
        # ここで握りつぶし、Webページ生成は必ず継続する（ジョブを赤くしない）。
        print(f"スクリーニング中断（APIキー/通信など）: {e}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"スクリーニング実行エラー: {e}")

    print(">>> 2. WebレポートHTML生成中（index.html・results.html）...")
    try:
        generate_web_report(df=df, quotes=quotes, target_date=target_date)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Web生成エラー: {e}")
    print(">>> 完了（ジョブは正常終了）")


if __name__ == "__main__":
    main()
