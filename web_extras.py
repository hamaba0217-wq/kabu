# -*- coding: utf-8 -*-
"""
追加の研究用ページを生成する。
  ・業種別成績（sectors.html）：推奨実績を業種で集計。どの業種が勝てているか。
  ・市場の状況（market.html）：TOPIXなど相場全体のトレンド（参考情報）。
  ・検証の記録（notes.html）：これまでの検証で分かった事実の記録。
"""

from __future__ import annotations

import datetime as dt
import html
import os

import numpy as np
import pandas as pd

from web_common import page_head, page_foot

OUT_DIR = "docs"


def _esc(v):
    return html.escape(str(v))


# ────────────────────────────────────────────────────────────
# 業種別成績（推奨実績を業種で集計）
# ────────────────────────────────────────────────────────────
def build_sectors_html(tracked):
    """tracked: track_picks.track() の結果（推奨と結果）。業種ごとに集計。"""
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [page_head("業種別成績", "sectors.html")]
    parts.append(f'<h1>業種別の成績</h1><div class="meta">更新: {now}</div>')
    parts.append('<p style="font-size:13px;color:#556;">推奨した銘柄を業種ごとにまとめ、'
                 'どの業種が実際に勝てているかを集計します。実績が貯まるほど傾向が見えます。</p>')

    if tracked is None or tracked.empty or "業種" not in tracked.columns:
        parts.append('<div class="none">まだ集計できる推奨がありません。'
                     '毎朝の記録が貯まると、業種別の成績がここに出ます。</div>')
        parts.append(page_foot())
        return "".join(parts)

    rows = []
    for sec, g in tracked.groupby("業種"):
        done = g[g["結果"].isin(["利確+10%到達", "損切り-5%到達"])]
        n_all = len(g)
        n_done = len(done)
        n_win = int((done["結果"] == "利確+10%到達").sum())
        wr = (n_win / n_done * 100) if n_done else None
        # 最新時点%の平均（追跡中も含む、全体の平均的な動き）
        last = pd.to_numeric(g.get("最新時点%"), errors="coerce").dropna()
        avg_last = float(last.mean()) if len(last) else None
        rows.append({
            "業種": sec, "推奨数": n_all, "確定": n_done, "利確": n_win,
            "損切り": n_done - n_win, "勝率": wr, "平均騰落": avg_last,
        })
    sdf = pd.DataFrame(rows)
    # 勝率→推奨数の順に並べる（勝率不明は後ろ）
    sdf["_sort"] = sdf["勝率"].fillna(-1)
    sdf = sdf.sort_values(["_sort", "推奨数"], ascending=False)

    parts.append('<div class="tablewrap"><table><thead><tr>'
                 '<th>業種</th><th>推奨</th><th>確定</th><th>利確</th><th>損切</th>'
                 '<th>勝率</th><th>平均騰落</th></tr></thead><tbody>')
    for _, r in sdf.iterrows():
        wr = f'{r["勝率"]:.0f}%' if r["勝率"] is not None and r["勝率"] == r["勝率"] else "—"
        al = r["平均騰落"]
        al_s = (f'<span class="{"pos" if al>=0 else "neg"}">{al:+.1f}%</span>'
                if al is not None and al == al else "—")
        parts.append(
            f'<tr><td>{_esc(r["業種"])}</td><td>{r["推奨数"]}</td>'
            f'<td>{r["確定"]}</td><td class="win">{r["利確"]}</td>'
            f'<td class="lose">{r["損切り"]}</td><td><b>{wr}</b></td>'
            f'<td>{al_s}</td></tr>'
        )
    parts.append('</tbody></table></div>')
    parts.append('<p class="note">勝率＝利確到達÷（利確到達＋損切り到達）。'
                 '確定件数が少ない業種の勝率はブレます。件数と併せて見てください。</p>')
    parts.append(page_foot())
    return "".join(parts)


# ────────────────────────────────────────────────────────────
# 市場の状況（相場全体のトレンド・参考情報）
# ────────────────────────────────────────────────────────────
def build_market_html(quotes, listed):
    """全銘柄の平均的な値動きから、相場全体のトレンドを簡易表示（参考）。"""
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [page_head("市場の状況", "market.html")]
    parts.append(f'<h1>市場の状況（参考）</h1><div class="meta">更新: {now}</div>')
    parts.append('<div class="none" style="background:#eef;">'
                 'これは<b>検証済みの指標ではなく参考情報</b>です。相場全体の地合いを'
                 'ざっくり掴むためのもので、売買の根拠にはしないでください。</div>')

    try:
        q = quotes.sort_values("date")
        # 全銘柄の等加重指数：各銘柄を期間内で正規化して日ごとに平均
        piv = q.pivot_table(index="date", columns="code", values="close", aggfunc="last")
        base = piv.ffill().bfill().iloc[0]
        idx = (piv / base).mean(axis=1)
        idx = idx.dropna()
        if len(idx) >= 60:
            def chg(days):
                if len(idx) > days:
                    return (idx.iloc[-1] / idx.iloc[-1-days] - 1) * 100
                return None
            d5, d20, d60 = chg(5), chg(20), chg(60)
            # 20日前と比べたトレンド判定
            def pill(v):
                if v is None: return '<span class="pill flat">—</span>'
                if v > 1: return f'<span class="pill up">上昇 {v:+.1f}%</span>'
                if v < -1: return f'<span class="pill down">下落 {v:+.1f}%</span>'
                return f'<span class="pill flat">横ばい {v:+.1f}%</span>'
            parts.append('<div class="card"><div class="name">全銘柄・等加重の値動き</div>'
                         '<div class="stat">')
            parts.append(f'直近5日: {pill(d5)}<br>')
            parts.append(f'直近20日（約1か月）: {pill(d20)}<br>')
            parts.append(f'直近60日（約3か月）: {pill(d60)}')
            parts.append('</div></div>')
            # 地合いのざっくり判定
            if d20 is not None:
                if d20 > 1:
                    msg = "全体はやや上昇基調。押し目候補は追い風になりやすい局面です。"
                elif d20 < -1:
                    msg = "全体はやや下落基調。候補が出ても、無理せず様子を見る選択肢もあります。"
                else:
                    msg = "全体は横ばい。個別銘柄の条件で淡々と判断する局面です。"
                parts.append(f'<div class="card"><div class="stat">{msg}</div></div>')
        else:
            parts.append('<div class="none">データが不足しています。</div>')
    except Exception as e:
        parts.append(f'<div class="none">市場データを集計できませんでした。</div>')

    parts.append(page_foot())
    return "".join(parts)


# ────────────────────────────────────────────────────────────
# 検証の記録（これまで分かった事実）
# ────────────────────────────────────────────────────────────
def build_notes_html():
    parts = [page_head("検証の記録", "notes.html")]
    parts.append('<h1>検証の記録</h1>')
    parts.append('<p style="font-size:13px;color:#556;">この戦略が「なぜこの形なのか」を、'
                 'これまでの検証結果とともに残します。将来、判断を見返すための土台です。</p>')

    parts.append('<div class="card"><h2 style="margin-top:0">採用している戦略</h2>'
                 '<div class="stat">'
                 '<b>押し目 × 高勝率業種 × 決算1か月回避 × 悪材料なし × 売買代金5億以上</b><br>'
                 '・押し目：52週高値から −15〜−5%（下げすぎでなく浅めの押し）<br>'
                 '・高勝率業種：保険・銀行・証券・倉庫運輸・水産農林<br>'
                 '・決算1か月回避：次の決算まで20営業日以内は除外<br>'
                 '・悪材料なし：減収・減益・予想未達を除外<br>'
                 '・売買代金5億以上：流動性のある銘柄に限定<br>'
                 '検証では対相場超過 約+0.95%・勝率62%。前半・後半どちらの期間でもプラス（再現性あり）。'
                 '</div></div>')

    parts.append('<div class="card"><h2 style="margin-top:0">分割検証を通った銘柄</h2>'
                 '<div class="stat">'
                 '期間を3分割し、どの期間でもプラスだった銘柄：<br>'
                 'りそなHD・八十二長野銀行・大光銀行・京葉銀行・大垣共立銀行・みずほFG。<br>'
                 '中でもりそなHDが最も安定（3期間ともプラス、該当回数も最多）。'
                 '</div></div>')

    parts.append('<div class="card"><h2 style="margin-top:0">効かないと分かったもの（偽物）</h2>'
                 '<div class="stat">'
                 '検証の結果、以下は前兆・優位性が確認できませんでした：<br>'
                 '・売られすぎ（25日乖離）：分割検証で崩壊<br>'
                 '・信用倍率・空売り比率：閾値が不安定<br>'
                 '・投資部門別（海外勢）：再現性なし<br>'
                 '通説として語られるものでも、対相場・分割検証で確かめると多くは残りませんでした。'
                 '</div></div>')

    parts.append('<div class="card"><h2 style="margin-top:0">5倍株の研究で分かったこと</h2>'
                 '<div class="stat">'
                 '1年で5倍になった銘柄（分割調整済み177件）を、非5倍株と対比して調べました。<br>'
                 '・売買代金の増加、業種トレンド、空売り比率：<b>前兆にならず</b><br>'
                 '・売上の伸び、売り残の多さ：<b>ごく弱い差のみ</b><br>'
                 '・業種の偏りはある：非鉄金属31%・精密機器15%・電気機器13%・医薬品11%が5倍化（全体平均4%）<br>'
                 '結論：公開の数値データに5倍株の強い前兆はなく、原因はニュース・材料（適時開示）側にある。'
                 'それはTDnetアドオン（別料金）でしか取得できないことも確認済み。'
                 '</div></div>')

    parts.append('<div class="card"><h2 style="margin-top:0">検証の作法（この研究の原則）</h2>'
                 '<div class="stat">'
                 '・事実にもとづいて決める（推測で決め打ちしない）<br>'
                 '・一度出した結論を、根拠なく変えない<br>'
                 '・条件は1つずつ足し、分割検証で本物か確かめる<br>'
                 '・先読みを作り込まない。対相場超過で市場と比較する<br>'
                 '・手数料・スリッページ未考慮であることを明記する<br>'
                 '・不確実なことは、不確実なまま伝える'
                 '</div></div>')

    parts.append(page_foot())
    return "".join(parts)


def build_and_save_extras(quotes, listed, tracked):
    """3つの追加ページを生成して保存する。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    pages = {
        "sectors.html": build_sectors_html(tracked),
        "market.html": build_market_html(quotes, listed),
        "notes.html": build_notes_html(),
    }
    for fn, html_str in pages.items():
        with open(os.path.join(OUT_DIR, fn), "w", encoding="utf-8") as f:
            f.write(html_str)
    return list(pages.keys())
