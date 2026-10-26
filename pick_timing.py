# -*- coding: utf-8 -*-
"""
実際の推奨履歴（history/picks.csv）を使って、入り方（エントリー）× 出し方（イグジット）で
勝率・平均リターン・対相場超過がどう変わるかを総当たり検証する。
  py main.py pick-timing

検証する入り方（エントリー）
  推奨日終値        : 推奨日の終値で買う（現行の基準）
  翌日始値          : 翌営業日の寄りで買う
  2日目/3日目終値   : 推奨の2日後・3日後の終値で買う（様子見して入る）
  -2%押し目/-3%押し目: 推奨後5営業日以内に、推奨日終値から-2%/-3%下げたら買う
  高値ブレイク       : 推奨後3営業日以内に、推奨日の高値を上抜けたら買う（勢い確認）

検証する出し方（イグジット）
  利確/損切り総当たり: (+5/-3)(+8/-5)(+10/-5)(+15/-8)(+20/-10) ※高値安値ベース・同日は損切り優先
  時間決済          : 参入からN営業日後の終値で必ず売る（N=3,5,10,15,20）
  早期撤退          : M日以内に+3%に届かなければ即時終値で撤退（損失を浅く切る）

事実にもとづく注意
  ・手数料・スリッページ未考慮。推奨日＝記録した推奨の日。最長保有20営業日。
  ・分割検証（前半/後半）と対相場超過を併記（まぐれ排除）。
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HIST_FILE = os.path.join("history", "picks.csv")
MAX_HOLD = 20
TP_SL = [(5, -3), (8, -5), (10, -5), (15, -8), (20, -10)]
TIME_EXITS = [3, 5, 10, 15, 20]


# ---------- 相場（対相場超過の基準） ----------
def _market_daily_return(quotes):
    q = quotes.sort_values(["code", "date"]).copy()
    q["ret"] = q.groupby("code")["close"].pct_change()
    return q.groupby("date")["ret"].mean()


def _cum_market(mkt, dates):
    sub = mkt.reindex(pd.to_datetime(dates)).fillna(0.0)
    return (np.prod(1.0 + sub.values) - 1.0) * 100.0


# ---------- エントリー（推奨日indexから、参入index・参入価格を返す） ----------
def _entry(g, ri, kind):
    c = g["close"].values; o = g["open"].values if "open" in g else c
    h = g["high"].values if "high" in g else c
    lo = g["low"].values if "low" in g else c
    rec_close = c[ri]
    if kind == "推奨日終値":
        return ri, rec_close
    if kind == "翌日始値":
        return (ri + 1, o[ri + 1]) if ri + 1 < len(c) else (None, None)
    if kind == "2日目終値":
        return (ri + 2, c[ri + 2]) if ri + 2 < len(c) else (None, None)
    if kind == "3日目終値":
        return (ri + 3, c[ri + 3]) if ri + 3 < len(c) else (None, None)
    if kind in ("-2%押し目", "-3%押し目"):
        pct = 2 if "2" in kind else 3
        lvl = rec_close * (1 - pct / 100.0)
        for i in range(ri + 1, min(len(c), ri + 6)):
            if lo[i] <= lvl:
                return i, lvl
        return None, None
    if kind == "高値ブレイク":
        rec_high = h[ri]
        for i in range(ri + 1, min(len(c), ri + 4)):
            if h[i] > rec_high:
                return i, rec_high
        return None, None
    return None, None


# ---------- イグジット ----------
def _exit_tpsl(g, ei, entry, tp, sl, maxhold=MAX_HOLD):
    c = g["close"].values
    h = g["high"].values if "high" in g else c
    lo = g["low"].values if "low" in g else c
    if entry is None or not (entry > 0):
        return None
    tp_p = entry * (1 + tp / 100.0); sl_p = entry * (1 + sl / 100.0)
    end = min(len(c), ei + 1 + maxhold)
    for i in range(ei + 1, end):
        if lo[i] <= sl_p:
            return sl
        if h[i] >= tp_p:
            return tp
    if end - 1 <= ei:
        return None
    return (c[end - 1] / entry - 1) * 100.0


def _exit_time(g, ei, entry, n):
    c = g["close"].values
    if entry is None or not (entry > 0) or ei + n >= len(c):
        return None
    return (c[ei + n] / entry - 1) * 100.0


def _exit_earlycut(g, ei, entry, need=3.0, within=3, sl=-5, maxhold=MAX_HOLD):
    """within日以内に+need%に届かなければ、その時点の終値で撤退。届けば+need/-5のTP/SL継続。"""
    c = g["close"].values
    h = g["high"].values if "high" in g else c
    lo = g["low"].values if "low" in g else c
    if entry is None or not (entry > 0):
        return None
    need_p = entry * (1 + need / 100.0); sl_p = entry * (1 + sl / 100.0)
    reached = False
    end = min(len(c), ei + 1 + maxhold)
    for k, i in enumerate(range(ei + 1, end), start=1):
        if lo[i] <= sl_p:
            return sl
        if h[i] >= need_p:
            reached = True
        if (not reached) and k >= within:
            return (c[i] / entry - 1) * 100.0   # 伸びないので撤退
    if end - 1 <= ei:
        return None
    return (c[end - 1] / entry - 1) * 100.0


# ---------- 検証本体 ----------
def _rec_index(g, rec_date):
    for i, t in enumerate(g["date"]):
        if pd.Timestamp(t) >= pd.Timestamp(rec_date):
            return i
    return None


def _summ(rets):
    rets = np.array([r for r in rets if r is not None], dtype=float)
    if len(rets) == 0:
        return 0, 0.0, 0.0
    return len(rets), (rets > 0).mean() * 100, rets.mean()


def run(quotes, picks=None):
    if picks is None:
        if not os.path.exists(HIST_FILE):
            print("history/picks.csv がありません。"); return
        picks = pd.read_csv(HIST_FILE, dtype={"code": str})
    picks.columns = [c.strip().lstrip("﻿") for c in picks.columns]
    picks["code"] = picks["code"].astype(str)
    q = quotes.copy(); q["code"] = q["code"].astype(str); q["date"] = pd.to_datetime(q["date"])
    bycode = {c: g.sort_values("date").reset_index(drop=True) for c, g in q.groupby("code")}
    mkt = _market_daily_return(q)

    # 各推奨について、推奨日index を事前計算
    recs = []
    for _, r in picks.iterrows():
        g = bycode.get(str(r["code"]))
        if g is None:
            g = bycode.get(str(r["code"])[:4])
        if g is None:
            continue
        ri = _rec_index(g, r["推奨日"])
        if ri is None or ri >= len(g):
            continue
        recs.append((r["推奨日"], g, ri))
    print(f"検証対象の推奨: {len(recs)} 件（株価が取得できたもの）\n")
    if not recs:
        print("対象がありません。"); return

    entries = ["推奨日終値", "翌日始値", "2日目終値", "3日目終値", "-2%押し目", "-3%押し目", "高値ブレイク"]

    # (1) エントリー × 利確/損切り
    print("=" * 84)
    print("(1) 入り方 × 利確/損切り（勝率％ / 平均R％）　※最長20営業日保有")
    print("=" * 84)
    header = "入り方".ljust(12) + "".join(f"+{tp}/{sl}".rjust(14) for tp, sl in TP_SL)
    print(header)
    for ek in entries:
        cells = []
        for tp, sl in TP_SL:
            rets = []
            for _, g, ri in recs:
                ei, ep = _entry(g, ri, ek)
                rets.append(_exit_tpsl(g, ei, ep, tp, sl) if ei is not None else None)
            n, win, avg = _summ(rets)
            cells.append(f"{win:3.0f}%/{avg:+5.1f}".rjust(14) if n else "—".rjust(14))
        print(ek.ljust(12) + "".join(cells))

    # (2) エントリー × 時間決済
    print("\n" + "=" * 84)
    print("(2) 入り方 × 時間決済（N日後の終値で売る）　勝率％ / 平均R％")
    print("=" * 84)
    print("入り方".ljust(12) + "".join(f"{n}日".rjust(14) for n in TIME_EXITS))
    for ek in entries:
        cells = []
        for n in TIME_EXITS:
            rets = []
            for _, g, ri in recs:
                ei, ep = _entry(g, ri, ek)
                rets.append(_exit_time(g, ei, ep, n) if ei is not None else None)
            cnt, win, avg = _summ(rets)
            cells.append(f"{win:3.0f}%/{avg:+5.1f}".rjust(14) if cnt else "—".rjust(14))
        print(ek.ljust(12) + "".join(cells))

    # (3) 早期撤退ルール（伸びなければ即切る）vs 通常の-5%放置
    print("\n" + "=" * 84)
    print("(3) 早期撤退（M日以内に+3%未達なら撤退）の効果　※入り=推奨日終値・利確+10%")
    print("=" * 84)
    print(f"  {'ルール':<28}{'件数':>6}{'勝率':>7}{'平均R%':>9}")
    base = [_exit_tpsl(g, ri, g['close'].values[ri], 10, -5) for _, g, ri in recs]
    n, w, a = _summ(base); print(f"  {'通常（+10%/-5%・最長20日）':<28}{n:>6}{w:>6.0f}%{a:>9.1f}")
    for within in (2, 3, 5):
        rr = [_exit_earlycut(g, ri, g['close'].values[ri], need=3.0, within=within, sl=-5) for _, g, ri in recs]
        n, w, a = _summ(rr); print(f"  {f'{within}日以内+3%未達で撤退':<28}{n:>6}{w:>6.0f}%{a:>9.1f}")

    # (4) 有力な組合せを分割検証＋対相場超過
    print("\n" + "=" * 84)
    print("(4) 代表的な組合せの 分割検証＋対相場超過（まぐれ排除）")
    print("=" * 84)
    combos = [("推奨日終値", "TPSL", (10, -5)), ("推奨日終値", "TPSL", (15, -8)),
              ("3日目終値", "TPSL", (10, -5)), ("-3%押し目", "TPSL", (10, -5)),
              ("推奨日終値", "TIME", 10)]
    items = sorted(recs, key=lambda x: pd.Timestamp(x[0]))
    mid = len(items) // 2
    for ek, kind, param in combos:
        def ret_of(g, ri):
            ei, ep = _entry(g, ri, ek)
            if ei is None:
                return None
            return _exit_tpsl(g, ei, ep, param[0], param[1]) if kind == "TPSL" else _exit_time(g, ei, ep, param)
        label = f"{ek}×{('+%d/%d'%param) if kind=='TPSL' else ('%d日決済'%param)}"
        all_r = [ret_of(g, ri) for _, g, ri in recs]
        n, w, a = _summ(all_r)
        # 前半後半
        fh = [ret_of(g, ri) for _, g, ri in items[:mid]]
        sh = [ret_of(g, ri) for _, g, ri in items[mid:]]
        n1, w1, a1 = _summ(fh); n2, w2, a2 = _summ(sh)
        # 対相場超過（20日保有相当で近似：各参入から20日 or 決済まで）
        exc = []
        for d0, g, ri in recs:
            ei, ep = _entry(g, ri, ek)
            if ei is None:
                continue
            c = g["close"].values; dts = g["date"].values
            hold = param if kind == "TIME" else 20
            if ei + hold >= len(c) or not (ep > 0):
                continue
            raw = (c[ei + hold] / ep - 1) * 100
            mk = _cum_market(mkt, dts[ei + 1: ei + 1 + hold])
            exc.append(raw - mk)
        exm = np.mean(exc) if exc else float('nan')
        ok = (a1 > 0 and a2 > 0 and exm == exm and exm > 0)
        print(f"\n  ■ {label}")
        print(f"     全体 {n}件 勝率{w:.0f}% 平均{a:+.1f}%／前半{a1:+.1f}% 後半{a2:+.1f}%／対相場超過{exm:+.1f}% → {'◎候補' if ok else '×'}")

    # (5) 高勝率業種 × 現実的運用（翌日寄り/ブレイク × 短期決済）の再検証
    print("\n" + "=" * 84)
    print("(5) 業種を絞った『現実的運用』の再検証（翌日寄り/ブレイク × 短期決済）")
    print("   判定: 対相場超過プラス かつ 前半後半とも平均プラス → ◎")
    print("=" * 84)
    HIGH = ["銀行業", "卸売業"]

    def build_recs(sectors):
        out = []
        for _, r in picks.iterrows():
            if sectors is not None and str(r.get("業種", "")) not in sectors:
                continue
            g = bycode.get(str(r["code"]))
            if g is None:
                g = bycode.get(str(r["code"])[:4])
            if g is None:
                continue
            ri = _rec_index(g, r["推奨日"])
            if ri is None or ri >= len(g):
                continue
            out.append((pd.Timestamp(r["推奨日"]), g, ri))
        return sorted(out, key=lambda x: x[0])

    # 現実的な組合せ（＝翌日寄り or 高値ブレイクで入り、短期で出す）
    combos5 = [
        ("翌日始値×10日決済", "翌日始値", "TIME", 10),
        ("翌日始値×+10/-5(最長10日)", "翌日始値", "TPSL10", (10, -5)),
        ("高値ブレイク×10日決済", "高値ブレイク", "TIME", 10),
        ("高値ブレイク×+10/-5(最長10日)", "高値ブレイク", "TPSL10", (10, -5)),
        ("高値ブレイク×+15/-8(最長10日)", "高値ブレイク", "TPSL10", (15, -8)),
    ]
    for uni_label, sectors in [("全業種", None), ("銀行+卸売", HIGH), ("銀行のみ", ["銀行業"])]:
        rs = build_recs(sectors)
        print(f"\n【{uni_label}】 対象 {len(rs)}件")
        if len(rs) < 8:
            print("  件数が少なく判定不可"); continue
        mid = len(rs) // 2
        for label, ek, kind, param in combos5:
            def ret_of(g, ri):
                ei, ep = _entry(g, ri, ek)
                if ei is None:
                    return None
                if kind == "TIME":
                    return _exit_time(g, ei, ep, param)
                return _exit_tpsl(g, ei, ep, param[0], param[1], maxhold=10)
            allr = [ret_of(g, ri) for _, g, ri in rs]
            n, w, a = _summ(allr)
            fh = _summ([ret_of(g, ri) for _, g, ri in rs[:mid]])
            sh = _summ([ret_of(g, ri) for _, g, ri in rs[mid:]])
            # 対相場超過
            hold = param if kind == "TIME" else 10
            exc = []
            for _, g, ri in rs:
                ei, ep = _entry(g, ri, ek)
                if ei is None:
                    continue
                c = g["close"].values; dts = g["date"].values
                if ei + hold >= len(c) or not (ep > 0):
                    continue
                exc.append((c[ei + hold] / ep - 1) * 100 - _cum_market(mkt, dts[ei + 1: ei + 1 + hold]))
            exm = np.mean(exc) if exc else float('nan')
            ok = (fh[2] > 0 and sh[2] > 0 and exm == exm and exm > 0)
            if n:
                print(f"  {label:<26} {n:>4}件 勝率{w:3.0f}% 平均{a:+5.1f}%／前半{fh[2]:+5.1f} 後半{sh[2]:+5.1f}／超過{exm:+5.1f}% {'◎' if ok else '×'}")

    print("\n" + "=" * 84)
    print("※手数料・スリッページ未考慮。『対相場超過プラス＋前半後半プラス』を満たす組合せだけ採用候補。")
    print("=" * 84)
