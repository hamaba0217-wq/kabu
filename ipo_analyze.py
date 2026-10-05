# -*- coding: utf-8 -*-
"""
新規上場（IPO）直後の値動き検証。「上場3か月以内（＝初期60営業日）」に的を絞る。
  py main.py ipo-analyze          （2年分）
  py main.py ipo-analyze 4        （4年分：サンプルを増やして検証を安定させる）

検証する参入ルール（いろいろな入り方を比較）
  A. 日付参入   : 上場からD日目で買う（D=1,2,3,5,7,10）。※0=初日は初値未成立で買えない事が多い
  B. 押し目買い : 上場後、直近高値から -5/-10/-15% 下げた所で買う（買いやすい所）
  C. ブレイク   : 初期N日(=5)の高値を上抜けたら買う（勢いに乗る順張り）
  いずれも「終値で買う版」と「シグナル翌日の始値で買う版（現実的）」の両方を表示。

事実にもとづく注意
  ・利確/損切りは高値・安値ベース（日中に触れたら約定と仮定）。同日両touchは損切り優先。
  ・day0=初日終値は実際には約定しにくい（ストップ高比例配分・初値未成立）。参考値。
  ・手数料・スリッページ未考慮。サンプルが少ないと不安定（件数を必ず表示）。
  ・採用の条件は「対相場超過プラス＋前半後半で再現」。片方でも崩れたら不採用。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

WITHIN_DAYS = 60          # IPO判定の対象期間（上場3か月以内）
MIN_LIFE = 10
START_BUFFER = 15
ENTRY_DAYS = [1, 2, 3, 5, 7, 10]
DIPS = [5, 10, 15]
BREAKOUT_LOOKBACK = 5
TAKE_PROFITS = [5, 8, 10, 15, 20]
STOP_LOSSES = [-5, -8, -10]
HORIZON = 20              # 保有の上限＝約1か月（これを過ぎたら終値で決済。60日は持たない前提）


# ---------- データ準備 ----------
def _sector_name_maps(listed):
    sector_map, name_map = {}, {}
    if listed is not None and len(listed):
        cols = listed.columns
        code_col = "Code" if "Code" in cols else ("code" if "code" in cols else None)
        sec_col = next((c for c in ("sector33", "Sector33CodeName", "Sector17CodeName") if c in cols), None)
        name_col = next((c for c in ("CompanyName", "company_name", "Name") if c in cols), None)
        if code_col:
            key = listed[code_col].astype(str).str.slice(0, 4)
            if sec_col:
                sector_map = dict(zip(key, listed[sec_col]))
            if name_col:
                name_map = dict(zip(key, listed[name_col]))
    return sector_map, name_map


def detect_ipos(quotes):
    q = quotes.copy()
    q["code"] = q["code"].astype(str)
    q["date"] = pd.to_datetime(q["date"])
    all_dates = np.sort(q["date"].unique())
    if len(all_dates) == 0:
        return {}
    cutoff = all_dates[min(START_BUFFER, len(all_dates) - 1)]
    ipos = {}
    for code, g in q.groupby("code"):
        g = g.sort_values("date")
        if g["date"].values[0] <= np.datetime64(cutoff):
            continue
        if len(g) < MIN_LIFE:
            continue
        ipos[code] = g.reset_index(drop=True)
    return ipos


def _market_daily_return(quotes):
    q = quotes.sort_values(["code", "date"]).copy()
    q["ret"] = q.groupby("code")["close"].pct_change()
    return q.groupby("date")["ret"].mean()


def _cum_market_return(mkt, dates):
    sub = mkt.reindex(pd.to_datetime(dates)).fillna(0.0)
    return (np.prod(1.0 + sub.values) - 1.0) * 100.0


# ---------- 参入ルール（entry_index を返す関数で表現） ----------
def entry_on_day(day):
    def f(g):
        return day if len(g) > day else None
    return f


def entry_on_dip(dip_pct, start=1):
    """start日目以降、初値(day0終値)〜直近高値からdip_pct%下げた最初の日。"""
    def f(g):
        c = g["close"].values
        if len(c) <= start:
            return None
        run_high = c[0]
        for i in range(start, len(c)):
            run_high = max(run_high, c[i])
            if c[i] <= run_high * (1 - dip_pct / 100.0):
                return i
        return None
    return f


def entry_on_breakout(lookback=BREAKOUT_LOOKBACK):
    """最初のlookback日の高値を、以降で上抜けた最初の日。"""
    def f(g):
        h = g["high"].values if "high" in g else g["close"].values
        c = g["close"].values
        if len(c) <= lookback:
            return None
        base_high = np.nanmax(h[:lookback])
        for i in range(lookback, len(c)):
            if c[i] > base_high:
                return i
        return None
    return f


def entry_vol_breakout(lookback=5, vol_mult=2.0):
    """初期lookback日の高値を上抜け、かつ当日出来高が直近5日平均の vol_mult 倍以上。"""
    def f(g):
        h = g["high"].values if "high" in g else g["close"].values
        c = g["close"].values
        v = g["volume"].values if "volume" in g else np.ones(len(c))
        if len(c) <= lookback + 1:
            return None
        base_high = np.nanmax(h[:lookback])
        for i in range(lookback, len(c)):
            vma = np.nanmean(v[max(0, i - 5):i]) if i > 0 else np.nan
            if c[i] > base_high and vma and v[i] >= vol_mult * vma:
                return i
        return None
    return f


def entry_consec_up(n=3):
    """n日連続で終値が前日比プラス → その翌日の終値で買う。"""
    def f(g):
        c = g["close"].values
        if len(c) <= n + 1:
            return None
        up = np.diff(c) > 0  # up[k] = c[k+1]>c[k]
        for i in range(n, len(c)):
            if np.all(up[i - n:i]):  # 直近n日すべて上昇 → 当日iの終値で買う
                return i
        return None
    return f


def entry_gap_up(pct=5):
    """始値が前日終値から pct% 以上ギャップアップした日の終値で買う。"""
    def f(g):
        c = g["close"].values
        o = g["open"].values if "open" in g else c
        for i in range(1, len(c)):
            if o[i] >= c[i - 1] * (1 + pct / 100.0):
                return i
        return None
    return f


def entry_ma_cross(window=5):
    """終値が window日移動平均を下から上抜けた最初の日。"""
    def f(g):
        c = g["close"].values
        if len(c) <= window + 1:
            return None
        ma = pd.Series(c).rolling(window).mean().values
        for i in range(window, len(c)):
            if ma[i] == ma[i] and ma[i - 1] == ma[i - 1]:
                if c[i] > ma[i] and c[i - 1] <= ma[i - 1]:
                    return i
        return None
    return f


def entry_on_new_high(min_day=5):
    """min_day日目以降で、終値がそれまでの上場来終値高値を更新した最初の日に買う（勢い継続）。"""
    def f(g):
        c = g["close"].values
        if len(c) <= min_day:
            return None
        for i in range(min_day, len(c)):
            if c[i] > np.max(c[:i]):
                return i
        return None
    return f


def entry_hot_breakout(runup_min=15.0, lookback=5):
    """初期5日の最大上昇率が runup_min% 以上（＝勢いがある）IPOに限って、
    初期lookback日の高値を上抜けた最初の日に買う。※どちらも5日目までに判定可能＝先読みなし。"""
    base = entry_on_breakout(lookback)
    def f(g):
        ru = _m_early_runup(g)
        if ru is None or ru < runup_min:
            return None
        return base(g)
    return f


def filter_liquid(ipos, quotes, min_turnover=500_000_000, days=20):
    """上場後days日の平均売買代金が min_turnover 以上の銘柄だけに絞る。"""
    out = {}
    for code, g in ipos.items():
        if "turnover_value" in g.columns:
            to = g["turnover_value"].values[:days]
        else:
            c = g["close"].values[:days]
            v = g["volume"].values[:days] if "volume" in g else np.zeros(len(c))
            to = c * v
        if len(to) and np.nanmean(to) >= min_turnover:
            out[code] = g
    return out


# ---------- 約定シミュレーション ----------
def _simulate_from_index(g, entry_idx, tp, sl, horizon, next_open=False):
    """entry_idx の終値（next_open=Trueなら翌日始値）で買い、TP/SL判定。"""
    if entry_idx is None:
        return None
    c = g["close"].values
    h = g["high"].values if "high" in g else c
    lo = g["low"].values if "low" in g else c
    o = g["open"].values if "open" in g else c
    if next_open:
        ei = entry_idx + 1
        if ei >= len(c):
            return None
        entry = o[ei]
        start = ei
    else:
        ei = entry_idx
        entry = c[ei]
        start = ei
    if not (entry > 0):
        return None
    tp_p = entry * (1 + tp / 100.0)
    sl_p = entry * (1 + sl / 100.0)
    end = min(len(c), start + 1 + horizon)
    for i in range(start + 1, end):
        if lo[i] <= sl_p:
            return sl
        if h[i] >= tp_p:
            return tp
    if end - 1 <= start:
        return None
    return (c[end - 1] / entry - 1) * 100.0


def _eval_rule(ipos, entry_fn, tp, sl, next_open=False):
    """ルール全体の成績。(件数, 勝率, 平均R, 発生率) を返す。"""
    rs, triggered = [], 0
    for g in ipos.values():
        idx = entry_fn(g)
        if idx is None:
            continue
        triggered += 1
        r = _simulate_from_index(g, idx, tp, sl, HORIZON, next_open)
        if r is not None:
            rs.append(r)
    if not rs:
        return 0, 0.0, 0.0, triggered
    rs = np.array(rs, dtype=float)
    return len(rs), (rs > 0).mean() * 100, rs.mean(), triggered


def _split_check(ipos, entry_fn, tp, sl, next_open=False):
    items = sorted(ipos.items(), key=lambda kv: kv[1]["date"].values[0])
    mid = len(items) // 2
    out = []
    for part in (items[:mid], items[mid:]):
        rs = []
        for _, g in part:
            r = _simulate_from_index(g, entry_fn(g), tp, sl, HORIZON, next_open)
            if r is not None:
                rs.append(r)
        if rs:
            rs = np.array(rs)
            out.append((len(rs), (rs > 0).mean() * 100, rs.mean()))
        else:
            out.append((0, 0.0, 0.0))
    return out  # [(n,win,avg) 前半, 後半]


# ---------- 各種レポート ----------
def day_by_day(ipos):
    print("=" * 74)
    print(f"上場後の値動き（起点=初日終値 day0、全{len(ipos)}銘柄）")
    print("=" * 74)
    print(f"{'経過':>4}{'件数':>7}{'平均%':>8}{'中央%':>8}{'勝率':>7}{'最大上昇平均':>12}{'最大下落平均':>12}")
    for k in [1, 2, 3, 5, 10, 20, 30, 60]:
        rels, ups, downs = [], [], []
        for g in ipos.values():
            c = g["close"].values
            h = g["high"].values if "high" in g else c
            lo = g["low"].values if "low" in g else c
            if len(c) <= k or not (c[0] > 0):
                continue
            rels.append((c[k] / c[0] - 1) * 100)
            ups.append((np.nanmax(h[1:k + 1]) / c[0] - 1) * 100)
            downs.append((np.nanmin(lo[1:k + 1]) / c[0] - 1) * 100)
        if not rels:
            continue
        rels = np.array(rels)
        print(f"{k:>3}日{len(rels):>7}{rels.mean():>8.1f}{np.median(rels):>8.1f}"
              f"{(rels > 0).mean() * 100:>6.0f}%{np.mean(ups):>12.1f}{np.mean(downs):>12.1f}")


def _report_rules(title, ipos, rules, fixed_tp=15, fixed_sl=-8):
    """rules=[(ラベル, entry_fn)]。終値版と翌日始値版を並べて表示し、
       最良ラベルは分割検証も出す。"""
    print("\n" + "=" * 74)
    print(f"{title}（利確+{fixed_tp}% / 損切り{fixed_sl}%で比較）")
    print("=" * 74)
    print(f"{'ルール':<22}{'発生':>5}{'件数':>6}{'勝率':>7}{'平均R%':>9}{'翌日始値:勝率':>13}{'平均R%':>9}")
    best = None
    for label, fn in rules:
        n, win, avg, trig = _eval_rule(ipos, fn, fixed_tp, fixed_sl, next_open=False)
        n2, win2, avg2, _ = _eval_rule(ipos, fn, fixed_tp, fixed_sl, next_open=True)
        print(f"{label:<22}{trig:>5}{n:>6}{win:>6.0f}%{avg:>9.1f}{win2:>12.0f}%{avg2:>9.1f}")
        if n >= 10 and (best is None or avg > best[2]):
            best = (label, fn, avg)
    return best


def tp_sl_grid_for(ipos, entry_fn, label):
    print("\n" + "-" * 74)
    print(f"［{label}］利確/損切り 総当たり（終値参入）")
    print(f"    {'利確/損切り':<14}{'件数':>6}{'勝率':>7}{'平均R%':>9}")
    res = []
    for tp in TAKE_PROFITS:
        for sl in STOP_LOSSES:
            n, win, avg, _ = _eval_rule(ipos, entry_fn, tp, sl, next_open=False)
            if n:
                res.append((f"+{tp}%/{sl}%", n, win, avg))
    for name, n, win, avg in sorted(res, key=lambda x: x[3], reverse=True)[:5]:
        print(f"    {name:<14}{n:>6}{win:>6.0f}%{avg:>9.1f}")


def market_excess(ipos, quotes, entry_fn, label, hold=20):
    print("\n" + "-" * 74)
    print(f"【対相場超過】{label} → {hold}営業日後終値で売却")
    mkt = _market_daily_return(quotes)
    raws, excs = [], []
    for g in ipos.values():
        idx = entry_fn(g)
        if idx is None:
            continue
        c = g["close"].values; d = g["date"].values
        if len(c) <= idx + hold or not (c[idx] > 0):
            continue
        raw = (c[idx + hold] / c[idx] - 1) * 100
        mk = _cum_market_return(mkt, d[idx + 1: idx + 1 + hold])
        raws.append(raw); excs.append(raw - mk)
    if not raws:
        print("  データ不足"); return
    raws = np.array(raws); excs = np.array(excs)
    print(f"  件数{len(raws)}／生リターン平均{raws.mean():.1f}%（勝率{(raws>0).mean()*100:.0f}%）"
          f"／対相場超過平均{excs.mean():.1f}%（超過プラス率{(excs>0).mean()*100:.0f}%）")


def split_report(ipos, entry_fn, label, tp=15, sl=-8):
    print("\n" + "-" * 74)
    print(f"【分割検証】{label}・+{tp}%/{sl}%")
    (n1, w1, a1), (n2, w2, a2) = _split_check(ipos, entry_fn, tp, sl)
    print(f"  前半（{n1}件）勝率{w1:.0f}% 平均{a1:.1f}%  ／  後半（{n2}件）勝率{w2:.0f}% 平均{a2:.1f}%")
    ok = (a1 > 0 and a2 > 0)
    print(f"  → 前半後半とも平均プラス: {'YES（再現あり）' if ok else 'NO（まぐれの可能性）'}")


def sector_breakdown(ipos, listed, entry_fn, label, hold=20):
    sector_map, _ = _sector_name_maps(listed)
    print("\n" + "-" * 74)
    print(f"【業種別】{label} → {hold}営業日後終値")
    rows = {}
    for code, g in ipos.items():
        idx = entry_fn(g); c = g["close"].values
        if idx is None or len(c) <= idx + hold or not (c[idx] > 0):
            continue
        ret = (c[idx + hold] / c[idx] - 1) * 100
        rows.setdefault(sector_map.get(str(code)[:4], "その他"), []).append(ret)
    agg = [(s, len(v), float(np.mean(v)), (np.array(v) > 0).mean() * 100)
           for s, v in rows.items() if len(v) >= 5]  # 5件未満は不安定なので除外
    agg.sort(key=lambda x: x[2], reverse=True)
    print(f"    {'業種':<16}{'件数':>5}{'平均%':>8}{'勝率':>7}")
    for s, n, avg, win in agg[:8]:
        print(f"    {s:<16}{n:>5}{avg:>8.1f}{win:>6.0f}%")
    print("    ※5件未満の業種は除外（少数はまぐれ）。")


# ---------- 一括比較（採用判定つき） ----------
def _excess_mean(ipos, quotes, entry_fn, hold=20):
    mkt = _market_daily_return(quotes)
    excs = []
    for g in ipos.values():
        idx = entry_fn(g)
        if idx is None:
            continue
        c = g["close"].values; d = g["date"].values
        if len(c) <= idx + hold or not (c[idx] > 0):
            continue
        raw = (c[idx + hold] / c[idx] - 1) * 100
        mk = _cum_market_return(mkt, d[idx + 1: idx + 1 + hold])
        excs.append(raw - mk)
    return (float(np.mean(excs)), len(excs)) if excs else (float("nan"), 0)


def compare_all(rule_defs, quotes, tp=15, sl=-8):
    """rule_defs=[(ラベル, entry_fn, universe)]。各ルールを同じ基準で採点し一覧表示。
       判定: 対相場超過>0 かつ 前半後半とも平均>0 → ◎採用候補。"""
    print("\n" + "=" * 92)
    print(f"◆ 全ルール一括比較（利確+{tp}%/損切り{sl}%、保有は最大{HORIZON}日／対相場超過は20日保有）")
    print("=" * 92)
    print(f"{'ルール':<26}{'件数':>5}{'勝率':>6}{'平均R%':>8}{'対相場超過%':>11}{'前半':>7}{'後半':>7}{'判定':>8}")
    passed = []
    for label, fn, uni in rule_defs:
        n, win, avg, _ = _eval_rule(uni, fn, tp, sl)
        if n == 0:
            print(f"{label:<26}{'—':>5}{'':>6}{'':>8}{'':>11}{'':>7}{'':>7}{'データ不足':>8}")
            continue
        exc, _ = _excess_mean(uni, quotes, fn)
        (n1, w1, a1), (n2, w2, a2) = _split_check(uni, fn, tp, sl)
        split_ok = (a1 > 0 and a2 > 0)
        exc_ok = (exc == exc and exc > 0)
        verdict = "◎候補" if (split_ok and exc_ok) else "×"
        if split_ok and exc_ok:
            passed.append((label, fn, uni))
        print(f"{label:<26}{n:>5}{win:>5.0f}%{avg:>8.1f}{exc:>11.1f}{a1:>7.1f}{a2:>7.1f}{verdict:>8}")
    return passed


# ---------- 保有日数による成績の変化 ----------
HOLD_DAYS = [1, 2, 3, 4, 5, 7, 10, 12, 15, 20]   # 1か月(≒20営業日)以内を細かく


def holding_period_analysis(rules, quotes):
    """rules=[(ラベル, entry_fn, universe)]。参入後N日後の終値で売った時の
       勝率・平均・対相場超過を N 別に表示（＝何日持つのが良いか）。"""
    print("\n" + "=" * 86)
    print("◆ 保有日数による成績（ブレイクで買い → N営業日後の終値で売る）")
    print("=" * 86)
    mkt = _market_daily_return(quotes)
    for label, fn, uni in rules:
        print(f"\n■ {label}")
        print(f"    {'保有日数':>6}{'件数':>6}{'勝率':>7}{'平均R%':>9}{'中央R%':>9}{'対相場超過%':>12}")
        for N in HOLD_DAYS:
            rets, excs = [], []
            for g in uni.values():
                idx = fn(g)
                if idx is None:
                    continue
                c = g["close"].values; d = g["date"].values
                if len(c) <= idx + N or not (c[idx] > 0):
                    continue
                r = (c[idx + N] / c[idx] - 1) * 100
                mk = _cum_market_return(mkt, d[idx + 1: idx + 1 + N])
                rets.append(r); excs.append(r - mk)
            if not rets:
                print(f"    {N:>5}日{'—':>6}")
                continue
            rets = np.array(rets); excs = np.array(excs)
            print(f"    {N:>5}日{len(rets):>6}{(rets > 0).mean() * 100:>6.0f}%"
                  f"{rets.mean():>9.1f}{np.median(rets):>9.1f}{excs.mean():>12.1f}")
    print("\n  ※N日後の終値で必ず売る『時間決済』。利確/損切り(TP/SL)とは別の視点。")
    print("  ※勝率・平均・超過が最も高いNが『持つのに良い日数』の目安。")


# ---------- 注目度（初日人気・初期過熱）による分析 ----------
def _forward_from_day0(subset, ks=(10, 20, 30, 60)):
    """subset（code->g）の day0終値起点・k日後リターン分布。"""
    out = {}
    for k in ks:
        rels = []
        for g in subset.values():
            c = g["close"].values
            if len(c) > k and c[0] > 0:
                rels.append((c[k] / c[0] - 1) * 100)
        out[k] = np.array(rels) if rels else np.array([])
    return out


def _print_forward(title, buckets, ks=(10, 20, 30, 60)):
    print("\n" + "-" * 78)
    print(title)
    print(f"    {'区分':<10}{'件数':>5}" + "".join(f"{str(k)+'日平均':>9}{str(k)+'日勝率':>9}" for k in ks))
    for bname, subset in buckets:
        fw = _forward_from_day0(subset, ks)
        n = len(subset)
        cells = ""
        for k in ks:
            a = fw[k]
            if len(a):
                cells += f"{a.mean():>9.1f}{(a > 0).mean() * 100:>8.0f}%"
            else:
                cells += f"{'—':>9}{'—':>9}"
        print(f"    {bname:<10}{n:>5}{cells}")


def _tercile_split(ipos, metric_fn):
    """metric_fn(g)->値 で3分割。(高, 中, 低) の {code:g} を返す。"""
    vals = []
    for code, g in ipos.items():
        v = metric_fn(g)
        if v is not None and v == v:
            vals.append((code, v))
    if len(vals) < 6:
        return None
    arr = np.array([v for _, v in vals])
    q1, q2 = np.quantile(arr, [1 / 3, 2 / 3])
    hi, mid, lo = {}, {}, {}
    for code, v in vals:
        if v >= q2:
            hi[code] = ipos[code]
        elif v >= q1:
            mid[code] = ipos[code]
        else:
            lo[code] = ipos[code]
    return hi, mid, lo


def _m_day0_turnover(g):
    if "turnover_value" in g.columns:
        return float(g["turnover_value"].values[0])
    c = g["close"].values; v = g["volume"].values if "volume" in g else [0]
    return float(c[0] * v[0]) if len(c) else None


def _m_early_runup(g):
    c = g["close"].values
    if len(c) < 6 or not (c[0] > 0):
        return None
    return float(np.max(c[1:6]) / c[0] - 1) * 100  # 初日終値→最初の5日の最大上昇率(%)


def attention_analysis(ipos, quotes, listed):
    print("\n" + "=" * 78)
    print("◆ 注目度・初期過熱による分析（『初値が高く注目された株』の代理）")
    print("  ※公開価格はデータに無いため、初日売買代金＝注目度、初期5日最大上昇＝過熱度 で代用")
    print("=" * 78)

    # 1) 初期過熱度（初日終値→最初5日の最大上昇率）で3分割
    t1 = _tercile_split(ipos, _m_early_runup)
    if t1:
        hi, mid, lo = t1
        _print_forward("［初期過熱度で3分割］day0終値を起点に、その後のリターン",
                       [("過熱 高", hi), ("中", mid), ("低", lo)])

    # 2) 初日売買代金（注目度）で3分割
    t2 = _tercile_split(ipos, _m_day0_turnover)
    if t2:
        hi2, mid2, lo2 = t2
        _print_forward("［初日売買代金（注目度）で3分割］day0終値を起点に、その後のリターン",
                       [("注目 高", hi2), ("中", mid2), ("低", lo2)])

    # 3) 「注目された株（過熱度・上位1/3）」だけに各ルールを当てて検証
    if t1:
        hot = t1[0]
        print("\n" + "-" * 78)
        print(f"【注目株（初期過熱 上位1/3・{len(hot)}件）だけに各ルールを適用して検証】")
        hot_rules = [
            ("3日目で買う",          entry_on_day(3),       hot),
            ("5日目で買う",          entry_on_day(5),       hot),
            ("7日目で買う",          entry_on_day(7),       hot),
            ("高値-10%押し目",       entry_on_dip(10),      hot),
            ("高値-15%押し目",       entry_on_dip(15),      hot),
            ("初期5日高値ブレイク",    entry_on_breakout(5),  hot),
            ("5日ブレイク+出来高2倍",  entry_vol_breakout(5, 2.0), hot),
            ("3日連続上昇で買う",     entry_consec_up(3),    hot),
        ]
        compare_all(hot_rules, quotes)


# ---------- メイン ----------
def run(quotes, listed=None):
    print("新規上場（IPO）直後の値動き検証 — 上場3か月以内（初期60営業日）")
    quotes = quotes.copy()
    if "turnover_value" not in quotes.columns:
        v = quotes["volume"] if "volume" in quotes.columns else 0
        quotes["turnover_value"] = quotes["close"] * v
    ipos = detect_ipos(quotes)
    print(f"検出した新規上場銘柄: {len(ipos)} 件")
    if not ipos:
        print("新規上場銘柄が検出できませんでした。"); return
    liquid = filter_liquid(ipos, quotes, min_turnover=500_000_000, days=20)
    print(f"うち 売買代金5億以上（上場後20日平均）: {len(liquid)} 件")

    day_by_day(ipos)

    # 比較するルール一覧（ラベル, 入口関数, 対象ユニバース）
    rule_defs = [
        ("1日目で買う",             entry_on_day(1),        ipos),
        ("3日目で買う",             entry_on_day(3),        ipos),
        ("5日目で買う",             entry_on_day(5),        ipos),
        ("7日目で買う",             entry_on_day(7),        ipos),
        ("10日目で買う",            entry_on_day(10),       ipos),
        ("高値-10%押し目",          entry_on_dip(10),       ipos),
        ("高値-15%押し目",          entry_on_dip(15),       ipos),
        ("初期5日高値ブレイク",       entry_on_breakout(5),   ipos),
        ("初期10日高値ブレイク",      entry_on_breakout(10),  ipos),
        ("5日ブレイク+出来高2倍",     entry_vol_breakout(5, 2.0), ipos),
        ("3日連続上昇で買う",        entry_consec_up(3),     ipos),
        ("ギャップアップ+5%",        entry_gap_up(5),        ipos),
        ("5日移動平均 上抜け",       entry_ma_cross(5),      ipos),
        ("5日ブレイク(流動性5億)",    entry_on_breakout(5),   liquid),
        ("5日ブレイク+出来高(流動性)", entry_vol_breakout(5, 2.0), liquid),
        ("過熱+10% × 5日ブレイク",    entry_hot_breakout(10, 5), ipos),
        ("過熱+15% × 5日ブレイク",    entry_hot_breakout(15, 5), ipos),
        ("過熱+15% × 5日ブレイク(流動性)", entry_hot_breakout(15, 5), liquid),
        ("上場来高値更新",            entry_on_new_high(5),      ipos),
        ("上場来高値更新(流動性5億)",   entry_on_new_high(5),      liquid),
    ]
    passed = compare_all(rule_defs, quotes)

    # 保有日数で勝率がどう変わるか（有力ルール）
    holding_period_analysis([
        ("5日高値ブレイク+出来高2倍", entry_vol_breakout(5, 2.0), ipos),
        ("初期10日高値ブレイク",      entry_on_breakout(10),      ipos),
        ("過熱+15% × 5日ブレイク",    entry_hot_breakout(15, 5),  ipos),
        ("5日ブレイク+出来高(流動性5億)", entry_vol_breakout(5, 2.0), liquid),
    ], quotes)

    # 注目度・初期過熱による分析（「初値が高く注目された株」の代理）
    attention_analysis(ipos, quotes, listed)

    # 合格ルールだけ深掘り（TP/SL総当たり＋分割＋対相場＋業種）
    print("\n" + "=" * 74)
    if passed:
        print(f"◆ 採用候補（◎）の深掘り：{len(passed)}件")
        print("=" * 74)
        for label, fn, uni in passed:
            tp_sl_grid_for(uni, fn, label)
            split_report(uni, fn, label)
            market_excess(uni, quotes, fn, label)
            sector_breakdown(uni, listed, fn, label)
    else:
        print("◆ 両基準（対相場超過プラス＋分割YES）を満たすルールはありませんでした。")
        print("=" * 74)

    print("\n" + "=" * 74)
    print("判定：『対相場超過プラス』かつ『前半後半とも平均プラス』の両方(◎)だけ採用候補。")
    print("手数料・スリッページ未考慮。過去の傾向は将来を保証しません。")
    print("=" * 74)
