# -*- coding: utf-8 -*-
"""
IPO注目シグナル（日次）。上場3か月以内の銘柄で「本日ルールが点灯した」ものを、
どのルールか（分類ラベル）付きで抽出・記録・追跡する。

検出ルール（本日点灯＝その日に条件を満たした最初の日）
  ・5日ブレイク       : 初期5日の高値を本日の終値で上抜け
  ・10日ブレイク      : 初期10日の高値を本日の終値で上抜け
  ・出来高2倍         : 5日ブレイク かつ 当日出来高が直近5日平均の2倍以上
  ・過熱+10%/+15%     : 5日ブレイク かつ 初期5日の上昇率が+10%/+15%以上
  ・上場来高値更新     : 本日の終値が上場来の終値高値を更新（勢い継続）

事実にもとづく注意
  ・買い指値＝本日終値。利確+20%/損切り-10%（最長20営業日で手仕舞い）を目安に表示。
  ・件数が少なくても情報として出す（買うかは利用者が判断）。
  ・手数料・スリッページ未考慮。過去検証で対相場超過プラス＋分割再現を確認したルール群。
"""
from __future__ import annotations
import datetime as dt
import html
import os

import numpy as np
import pandas as pd

WITHIN_DAYS = 60       # 上場3か月以内
TP = 20.0              # 利確(%)
SL = -10.0            # 損切り(%)
MAX_HOLD = 20          # 最長保有(営業日)
HIST_DIR = "history"
HIST_FILE = os.path.join(HIST_DIR, "ipo_picks.csv")
JST = dt.timezone(dt.timedelta(hours=9))


def _esc(v):
    return html.escape(str(v))


def _maps(listed):
    sec, name = {}, {}
    if listed is not None and len(listed):
        cols = listed.columns
        cc = "Code" if "Code" in cols else ("code" if "code" in cols else None)
        sc = next((c for c in ("sector33", "Sector33CodeName", "Sector17CodeName") if c in cols), None)
        nc = next((c for c in ("CompanyName", "company_name", "Name") if c in cols), None)
        if cc:
            k = listed[cc].astype(str).str.slice(0, 4)
            if sc:
                sec = dict(zip(k, listed[sc]))
            if nc:
                name = dict(zip(k, listed[nc]))
    return sec, name


def find_today_signals(quotes, listed=None):
    """本日（データ最新日）に点灯したIPOシグナルを抽出して返す。"""
    q = quotes.copy()
    q["code"] = q["code"].astype(str)
    q["date"] = pd.to_datetime(q["date"])
    all_dates = np.sort(q["date"].unique())
    if len(all_dates) == 0:
        return pd.DataFrame()
    latest = all_dates[-1]
    cutoff = all_dates[max(0, len(all_dates) - WITHIN_DAYS)]  # 直近60営業日以内に初出現＝新規上場
    sec, name = _maps(listed)

    rows = []
    for code, g in q.groupby("code"):
        g = g.sort_values("date")
        first = g["date"].values[0]
        if first < np.datetime64(cutoff):
            continue  # 新規上場でない
        if g["date"].values[-1] != latest:
            continue  # 本日取引なし（点灯判定できない）
        c = g["close"].values
        h = g["high"].values if "high" in g else c
        v = g["volume"].values if "volume" in g else np.ones(len(c))
        age = len(c)
        if age < 6 or not (c[0] > 0):
            continue
        tclose, pclose = c[-1], c[-2]
        first5h = np.nanmax(h[:5])
        runup5 = (np.nanmax(c[1:6]) / c[0] - 1) * 100
        vma5 = np.nanmean(v[-6:-1]) if age >= 6 else np.nan
        prevmax = np.nanmax(c[:-1])

        tags = []
        if tclose > first5h and pclose <= first5h:
            tags.append("5日ブレイク")
        if age >= 11:
            first10h = np.nanmax(h[:10])
            if tclose > first10h and pclose <= first10h:
                tags.append("10日ブレイク")
        if "5日ブレイク" in tags and vma5 and v[-1] >= 2 * vma5:
            tags.append("出来高2倍")
        if "5日ブレイク" in tags and runup5 >= 15:
            tags.append("過熱+15%")
        elif "5日ブレイク" in tags and runup5 >= 10:
            tags.append("過熱+10%")
        if tclose > prevmax:
            tags.append("上場来高値更新")

        if not tags:
            continue
        rows.append({
            "銘柄コード": code,
            "企業名": name.get(code[:4], "不明"),
            "業種": sec.get(code[:4], "その他"),
            "分類": " / ".join(tags),
            "上場来日数": age,
            "初期5日上昇率": round(float(runup5), 1),
            "出来高倍率": round(float(v[-1] / vma5), 2) if vma5 else None,
            "買い指値": round(float(tclose), 1),
            "利確+20%": round(float(tclose) * 1.20, 1),
            "利確+15%": round(float(tclose) * 1.15, 1),
            "損切り-10%": round(float(tclose) * 0.90, 1),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["上場来日数"]).reset_index(drop=True)
    return df


def record_signals(df, as_of):
    """本日のシグナルを履歴に追記（同一codeは初回のみ記録して追跡スパムを防ぐ）。"""
    if df is None or df.empty:
        return
    os.makedirs(HIST_DIR, exist_ok=True)
    as_of_str = str(pd.Timestamp(as_of).date())
    new = pd.DataFrame([{
        "推奨日": as_of_str,
        "code": str(r["銘柄コード"]),
        "銘柄名": r["企業名"],
        "業種": r["業種"],
        "分類": r["分類"],
        "買い指値": r["買い指値"],
        "利確": r["利確+20%"],
        "損切り": r["損切り-10%"],
    } for _, r in df.iterrows()])
    if os.path.exists(HIST_FILE):
        old = pd.read_csv(HIST_FILE, dtype={"code": str})
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["code"], keep="first")  # 1銘柄1回
    else:
        merged = new
    merged.to_csv(HIST_FILE, index=False, encoding="utf-8-sig")


def track_signals(quotes):
    """履歴の各シグナルを、その後の株価で +20%/-10%（最長20日）判定して結果を付ける。"""
    if not os.path.exists(HIST_FILE):
        return pd.DataFrame()
    hist = pd.read_csv(HIST_FILE, dtype={"code": str})
    if hist.empty:
        return hist
    q = quotes.copy()
    q["code"] = q["code"].astype(str)
    q["date"] = pd.to_datetime(q["date"])
    bycode = {c: g.sort_values("date") for c, g in q.groupby("code")}

    out = []
    for _, r in hist.iterrows():
        code = str(r["code"]); entry = float(r["買い指値"])
        rec = pd.Timestamp(r["推奨日"])
        g = bycode.get(code)
        status, last_pct, peak = "データなし", None, None
        if g is not None and entry > 0:
            m = g["date"].values > np.datetime64(rec)
            fc = g["close"].values[m]
            fh = (g["high"].values if "high" in g else g["close"].values)[m]
            fl = (g["low"].values if "low" in g else g["close"].values)[m]
            if len(fc):
                horizon = fc[:MAX_HOLD]; hh = fh[:MAX_HOLD]; ll = fl[:MAX_HOLD]
                peak = float((np.nanmax(hh) / entry - 1) * 100)
                last_pct = float((horizon[-1] / entry - 1) * 100)
                status = "追跡中"
                tp_p = entry * (1 + TP / 100); sl_p = entry * (1 + SL / 100)
                for i in range(len(horizon)):
                    if ll[i] <= sl_p:
                        status = "損切り-10%到達"; break
                    if hh[i] >= tp_p:
                        status = "利確+20%到達"; break
                else:
                    if len(horizon) >= MAX_HOLD:
                        status = "期間満了(20日)"
            else:
                status = "追跡中（翌日以降待ち）"
        out.append({**r.to_dict(),
                    "最大上昇%": round(peak, 1) if peak is not None else None,
                    "最新%": round(last_pct, 1) if last_pct is not None else None,
                    "結果": status})
    return pd.DataFrame(out)


# ---------- HTML（本体のみ。外枠/ナビは web_report が付与） ----------
def build_ipo_body(today_df, tracked):
    now = dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    b = ['<h1>🚀 IPO注目（上場3か月以内・ルール点灯）</h1>',
         f'<div class="date-badge">更新: {now}</div>',
         '<p class="legend">検証で「対相場超過プラス＋前半後半で再現」を確認したルール群。'
         '件数が少なくても点灯すれば表示します（買うかはご自身の判断で）。'
         '目安の出口：<b>利確+20%（or +15%）／損切り-10%／最長20営業日</b>。投資助言ではありません。</p>']

    # 本日のシグナル
    b.append('<h2>本日点灯したシグナル</h2>')
    if today_df is not None and not today_df.empty:
        head = ["銘柄コード", "企業名", "業種", "分類", "上場来日数", "初期5日上昇率", "出来高倍率",
                "買い指値", "利確+20%", "利確+15%", "損切り-10%"]
        thead = "".join(f'<th data-sortable>{_esc(h)}</th>' for h in head)
        rows = []
        for _, r in today_df.iterrows():
            rows.append(
                "<tr>"
                f'<td><strong>{_esc(r["銘柄コード"])}</strong></td>'
                f'<td><strong>{_esc(r["企業名"])}</strong></td>'
                f'<td>{_esc(r["業種"])}</td>'
                f'<td><span class="badge-sub">{_esc(r["分類"])}</span></td>'
                f'<td data-v="{r["上場来日数"]}">{r["上場来日数"]}日</td>'
                f'<td data-v="{r["初期5日上昇率"]}">{r["初期5日上昇率"]}%</td>'
                f'<td data-v="{r.get("出来高倍率") or 0}">{_esc(r.get("出来高倍率"))}倍</td>'
                f'<td class="price-buy" data-v="{r["買い指値"]}">{r["買い指値"]:,}円</td>'
                f'<td class="price-tp" data-v="{r["利確+20%"]}">{r["利確+20%"]:,}円</td>'
                f'<td class="price-tp" data-v="{r["利確+15%"]}">{r["利確+15%"]:,}円</td>'
                f'<td class="price-sl" data-v="{r["損切り-10%"]}">{r["損切り-10%"]:,}円</td>'
                "</tr>")
        b.append(f'<div class="tablewrap"><table><thead><tr>{thead}</tr></thead>'
                 f'<tbody>{"".join(rows)}</tbody></table></div>')
    else:
        b.append('<div class="no-data"><p>本日ルールが点灯したIPOはありません。</p></div>')

    # 追跡（実績）
    b.append('<h2>これまでのIPOシグナルの結果</h2>')
    if tracked is not None and not tracked.empty:
        done = tracked[tracked["結果"].isin(["利確+20%到達", "損切り-10%到達"])]
        nd = len(done); nw = int((done["結果"] == "利確+20%到達").sum())
        wr = f"{nw / nd * 100:.0f}%" if nd else "—"
        b.append(f'<div class="card">記録: <b>{len(tracked)}件</b>／結果確定: <b>{nd}件</b>'
                 f'（利確 {nw}・損切り {nd - nw}、勝率 <b>{wr}</b>）</div>')
        t = tracked.sort_values("推奨日", ascending=False)
        head = ["推奨日", "銘柄名", "分類", "買い指値", "利確", "損切り", "最大上昇%", "最新%", "結果"]
        thead = "".join(f'<th data-sortable>{_esc(h)}</th>' for h in head)
        rows = []
        for _, r in t.iterrows():
            res = str(r["結果"])
            cls = "win" if "利確" in res else ("lose" if "損切り" in res else "track")
            def _n(v):
                return "" if v is None or (isinstance(v, float) and v != v) else v
            rows.append(
                "<tr>"
                f'<td>{_esc(r["推奨日"])}</td>'
                f'<td>{_esc(r["銘柄名"])}<br><span style="color:#999;font-size:0.85em">{_esc(r["code"])}</span></td>'
                f'<td>{_esc(r.get("分類", ""))}</td>'
                f'<td data-v="{r["買い指値"]}">{r["買い指値"]:,.1f}</td>'
                f'<td data-v="{r["利確"]}">{r["利確"]:,.1f}</td>'
                f'<td data-v="{r["損切り"]}">{r["損切り"]:,.1f}</td>'
                f'<td data-v="{_n(r.get("最大上昇%"))}">{_esc(_n(r.get("最大上昇%")))}</td>'
                f'<td data-v="{_n(r.get("最新%"))}">{_esc(_n(r.get("最新%")))}</td>'
                f'<td><span class="{cls}">{_esc(res)}</span></td>'
                "</tr>")
        b.append(f'<div class="tablewrap"><table><thead><tr>{thead}</tr></thead>'
                 f'<tbody>{"".join(rows)}</tbody></table></div>')
    else:
        b.append('<div class="no-data"><p>まだ記録がありません。点灯が出た日から蓄積されます。</p></div>')

    b.append('<div class="note">※ 買い指値＝点灯日の終値。利確+20%・損切り-10%は高値安値ベース、'
             '最長20営業日で手仕舞い。手数料・スリッページ未考慮・投資助言ではありません。</div>')
    return "\n".join(b)
