# -*- coding: utf-8 -*-
"""
指定銘柄の「過去同条件時点」を前半・後半に分けて検証する。

candidates-best で見た各銘柄の過去実績（2週間で+10%到達率・-5%到達率・
5日後平均）が、時期を問わず安定しているかを分割検証する。

事実として注意
--------------
  ・1銘柄の過去該当は40〜60件程度。分割すると各期間20〜30件。
    この母数では偶然の振れが残る。件数を明示して読むこと。
  ・candidates_best と同一の条件判定・到達判定を再利用（推測なし）。
  ・先読みなし。終値ベース判定。手数料・スリッページ未考慮。

使い方: py main.py validate-candidates 83080 85370 84110
        （コード未指定なら既定の3銘柄）
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

import config
import technical
from candidates_best import (
    _condition_met, _prep_fin_extended,
    FORWARD_DAYS, MIN_GAP_DAYS, HOLD_DAYS_2W, TARGET_UP, TARGET_DOWN, STOP_LOSS_HIST,
)

DEFAULT_CODES = ["23930", "93400", "83590"]   # 日本ケアサプライ・アソインターナショナル・八十二長野銀行


def _collect_events(code, cqi, sec_name, fin_by_code, fin_ext, require_sector=True):
    """過去に条件を満たした各時点の (日付, 5日後ret, 損切5%ret, +10到達, -5到達) を返す。"""
    cqi = cqi.sort_values("date").reset_index(drop=True)
    closes = cqi["close"].values
    dates = cqi["date"].values
    n = len(cqi)
    events = []
    last_idx = -MIN_GAP_DAYS - 1
    upper = n - max(FORWARD_DAYS, HOLD_DAYS_2W)
    for i in range(260, upper):
        if i - last_idx < MIN_GAP_DAYS:
            continue
        r = cqi.iloc[i].copy()
        r["sector33"] = sec_name
        if not _condition_met(r, code, pd.Timestamp(dates[i]), fin_by_code, fin_ext,
                              require_sector=require_sector):
            continue
        entry = closes[i]
        if not entry or entry <= 0:
            continue
        path5 = [(closes[i + k] / entry - 1.0) * 100 for k in range(1, FORWARD_DAYS + 1)]
        ret5 = path5[-1]
        stop5 = STOP_LOSS_HIST if any(pr <= STOP_LOSS_HIST for pr in path5) else path5[-1]
        path10 = [(closes[i + k] / entry - 1.0) * 100 for k in range(1, HOLD_DAYS_2W + 1)]
        up = any(pr >= TARGET_UP for pr in path10)
        down = any(pr <= TARGET_DOWN for pr in path10)
        events.append({"date": pd.Timestamp(dates[i]), "ret5": ret5,
                       "stop5": stop5, "up": up, "down": down})
        last_idx = i
    return events


def _stats(events):
    if not events:
        return None
    ret5 = np.array([e["ret5"] for e in events])
    stop5 = np.array([e["stop5"] for e in events])
    up = np.mean([e["up"] for e in events]) * 100
    down = np.mean([e["down"] for e in events]) * 100
    return {
        "n": len(events),
        "平均_損切なし%": round(float(ret5.mean()), 2),
        "平均_損切5%": round(float(stop5.mean()), 2),
        "2週+10%到達%": round(float(up), 1),
        "2週-5%到達%": round(float(down), 1),
    }


def run_strict(quotes, fin, listed, codes, require_sector=True):
    """指定銘柄を3分割（前期・中期・後期）で厳格に検証する。
    2分割より偶然が残りにくい。3期間すべてで平均(損切5%)プラスかを見る。"""
    from refine import HIGH_WIN_SECTORS
    qi = technical.add_indicators(quotes)
    g = qi.sort_values(["code", "date"]).groupby("code")
    h = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / h - 1.0

    sec_name, name_map = {}, {}
    if "sector33" in listed.columns:
        for _, r in listed.iterrows():
            sec_name[str(r["code"])] = r["sector33"]
    if "company_name" in listed.columns:
        for _, r in listed.iterrows():
            name_map[str(r["code"])] = r["company_name"]
    fin_ext = _prep_fin_extended(fin)
    fin_by_code = {}
    if "code" in fin.columns and "disclosed_date" in fin.columns:
        for cg, gg in fin.groupby("code"):
            fin_by_code[str(cg)] = sorted(pd.to_datetime(gg["disclosed_date"]).tolist())
    qi_by_code = {str(c): gg for c, gg in qi.groupby("code")}

    results = []
    for code in codes:
        code = str(code)
        sec = sec_name.get(code, "")
        is_hw = sec in HIGH_WIN_SECTORS
        rec = {"code": code, "name": name_map.get(code, "?"),
               "sector": sec, "high_win": is_hw}
        cqi = qi_by_code.get(code)
        if cqi is None:
            rec.update(whole=None, p1=None, p2=None, p3=None)
            results.append(rec); continue
        events = _collect_events(code, cqi, sec, fin_by_code, fin_ext,
                                 require_sector=require_sector)
        if len(events) < 12:
            rec.update(whole=_stats(events), p1=None, p2=None, p3=None)
            results.append(rec); continue
        ev = sorted(events, key=lambda e: e["date"])
        t = len(ev) // 3
        rec.update(whole=_stats(ev), p1=_stats(ev[:t]),
                   p2=_stats(ev[t:2 * t]), p3=_stats(ev[2 * t:]))
        results.append(rec)
    return results


def summarize_strict(results):
    L = ["=" * 84,
         "厳格再検証：3分割（前期・中期・後期）",
         "=" * 84, "",
         "2分割より偶然が残りにくい。3期間すべてで平均(損切5%)がプラスなら、",
         "その傾向は相対的に信頼できる（ただし各期間の件数はさらに少なくなる）。",
         "【業種】欄の ★ ＝ 高勝率業種（保険/銀行/証券/倉庫運輸/水産農林）。", ""]

    def _line(tag, s):
        if s is None:
            return f"    {tag}: （件数不足）"
        return (f"    {tag}: n{s['n']:>3}  平均(損切5%){s['平均_損切5%']:+5.2f}%  "
                f"+10%到達{s['2週+10%到達%']:4.1f}%  -5%到達{s['2週-5%到達%']:4.1f}%")

    passed = []
    for rec in results:
        code, name = rec["code"], rec["name"]
        whole, p1, p2, p3 = rec["whole"], rec["p1"], rec["p2"], rec["p3"]
        mark = "★" if rec["high_win"] else "　"
        L.append("=" * 84)
        L.append(f"【{name}（{code}）】 業種: {mark}{rec['sector']}")
        L.append(_line("全期間", whole))
        L.append(_line("前期 ", p1))
        L.append(_line("中期 ", p2))
        L.append(_line("後期 ", p3))
        if p1 and p2 and p3:
            allpos = p1["平均_損切5%"] > 0 and p2["平均_損切5%"] > 0 and p3["平均_損切5%"] > 0
            if allpos:
                L.append("    判定: ◎ 3期間すべて平均(損切5%)プラス（相対的に信頼できる）")
                passed.append(rec)
            else:
                negs = [t for t, s in zip(("前期", "中期", "後期"), (p1, p2, p3))
                        if s["平均_損切5%"] <= 0]
                L.append(f"    判定: × {'/'.join(negs)} でマイナス（3分割で崩れた＝偶然の可能性）")
        else:
            L.append("    判定: － 件数不足で3分割できず")
        L.append("")

    L += ["=" * 84, "総括（3分割を通った銘柄）", "=" * 84]
    if passed:
        # 全期間平均が高い順
        passed.sort(key=lambda r: -(r["whole"]["平均_損切5%"] if r["whole"] else -99))
        hw = [r for r in passed if r["high_win"]]
        non_hw = [r for r in passed if not r["high_win"]]
        L.append(f"◆ 高勝率業種（★）で通過: {len(hw)}件")
        for r in hw:
            L.append(f"  ◎ {r['name']}（{r['code']}）{r['sector']}: "
                     f"平均(損切5%){r['whole']['平均_損切5%']:+.2f}%")
        L.append(f"◆ それ以外の業種で通過: {len(non_hw)}件")
        for r in non_hw:
            L.append(f"  ◎ {r['name']}（{r['code']}）{r['sector']}: "
                     f"平均(損切5%){r['whole']['平均_損切5%']:+.2f}%")
        L.append("")
        L.append(f"  → {len(results)}銘柄中 {len(passed)}件 が3分割でも崩れなかった。")
        L.append("    高勝率業種かどうかで、通過の傾向に差があるかを見ること。")
        L.append("    ただし各期間の件数は少なく、これも将来を保証しない。")
    else:
        L.append("  該当なし。3分割にすると、どの銘柄も少なくとも1期間でマイナス。")
        L.append("  ＝2分割で残った数字は、期間の切り方に依存する偶然だった可能性が高い。")
    L += ["", "・3分割の各期間は十数件程度。母数が小さく、判定の精度にも限界がある。",
          "・過去の傾向であり将来を保証しない。終値ベース・手数料未考慮。"]
    return "\n".join(L)


def run(quotes, fin, listed, codes, require_sector=True, min_n_each=15):
    qi = technical.add_indicators(quotes)
    g = qi.sort_values(["code", "date"]).groupby("code")
    h = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / h - 1.0

    sec_name = {}
    name_map = {}
    if "sector33" in listed.columns:
        for _, r in listed.iterrows():
            sec_name[str(r["code"])] = r["sector33"]
    if "company_name" in listed.columns:
        for _, r in listed.iterrows():
            name_map[str(r["code"])] = r["company_name"]
    fin_ext = _prep_fin_extended(fin)
    fin_by_code = {}
    if "code" in fin.columns and "disclosed_date" in fin.columns:
        for cg, gg in fin.groupby("code"):
            fin_by_code[str(cg)] = sorted(pd.to_datetime(gg["disclosed_date"]).tolist())
    qi_by_code = {str(c): gg for c, gg in qi.groupby("code")}

    results = []
    for code in codes:
        code = str(code)
        cqi = qi_by_code.get(code)
        if cqi is None:
            results.append((code, name_map.get(code, "?"), None, None, None))
            continue
        events = _collect_events(code, cqi, sec_name.get(code), fin_by_code, fin_ext,
                                 require_sector=require_sector)
        if len(events) < max(4, min_n_each * 2):
            results.append((code, name_map.get(code, "?"), _stats(events), None, None))
            continue
        ev_sorted = sorted(events, key=lambda e: e["date"])
        mid = ev_sorted[len(ev_sorted) // 2]["date"]
        first = [e for e in ev_sorted if e["date"] < mid]
        second = [e for e in ev_sorted if e["date"] >= mid]
        results.append((code, name_map.get(code, "?"),
                        _stats(events), _stats(first), _stats(second)))
    return results


def summarize(results, only_passed=False):
    L = ["=" * 82,
         "指定銘柄の過去同条件を 前半・後半で分割検証",
         "=" * 82, ""]
    L.append("土台：押し目×決算回避×悪材料なし に該当した過去時点。")
    L.append("各期間で 平均・2週+10%到達率・2週-5%到達率 が安定しているかを見る。")
    if only_passed:
        L.append("※業種の絞りなし。分割検証を通った銘柄のみ表示（各期間n≧15）。")
    L.append("")

    def _line(tag, s):
        if s is None:
            return f"    {tag}: （件数不足）"
        return (f"    {tag}: n{s['n']:>3}  平均(損切なし){s['平均_損切なし%']:+5.2f}% "
                f"(損切5%){s['平均_損切5%']:+5.2f}%  "
                f"+10%到達{s['2週+10%到達%']:4.1f}%  -5%到達{s['2週-5%到達%']:4.1f}%")

    # 通過判定：平均(損切5%)が両期間プラス
    def _passed(first, second):
        return (first and second and
                first["平均_損切5%"] > 0 and second["平均_損切5%"] > 0)

    shown = 0
    for code, name, whole, first, second in results:
        if only_passed and not _passed(first, second):
            continue
        shown += 1
        L.append("=" * 82)
        L.append(f"【{name}（{code}）】")
        L.append(_line("全期間", whole))
        L.append(_line("前半 ", first))
        L.append(_line("後半 ", second))
        if first and second:
            checks = []
            if first["平均_損切5%"] > 0 and second["平均_損切5%"] > 0:
                checks.append("平均(損切5%)両期間＋")
            elif first["平均_損切5%"] > 0 or second["平均_損切5%"] > 0:
                checks.append("平均(損切5%)片方のみ＋")
            else:
                checks.append("平均(損切5%)両期間－")
            r1 = first["2週+10%到達%"] >= first["2週-5%到達%"]
            r2 = second["2週+10%到達%"] >= second["2週-5%到達%"]
            if r1 and r2:
                checks.append("+10%到達≧-5%到達 両期間成立")
            elif r1 or r2:
                checks.append("+10%到達≧-5%到達 片方のみ")
            else:
                checks.append("-5%到達の方が高い")
            L.append(f"    判定: {' / '.join(checks)}")
        L.append("")

    if only_passed:
        L.insert(6, f"→ 検証した銘柄のうち、平均(損切5%)が両期間プラスだったのは {shown} 件\n")

    L += ["=" * 82, "読み方", "=" * 82,
          "・平均(損切5%)が両期間プラスで、+10%到達≧-5%到達が両期間成立なら、",
          "  その銘柄の傾向は時期を問わず安定＝相対的に信頼できる。",
          "・片方の期間で崩れるなら、全期間の数字は偶然の可能性。",
          "・各期間20〜30件程度と少なく、偶然の振れは残る。件数(n)を必ず見ること。",
          "・過去の傾向であり将来を保証しない。終値ベース・手数料未考慮。"]

    # 総括：両条件を両期間で満たした銘柄を集計
    passed = []
    for code, name, whole, first, second in results:
        if first and second:
            avg_ok = first["平均_損切5%"] > 0 and second["平均_損切5%"] > 0
            hit_ok = (first["2週+10%到達%"] >= first["2週-5%到達%"] and
                      second["2週+10%到達%"] >= second["2週-5%到達%"])
            if avg_ok and hit_ok:
                passed.append((code, name, whole, first, second))
    L += ["", "=" * 82,
          f"総括：分割検証を通った銘柄（平均損切5%が両期間＋、かつ+10%到達≧-5%到達も両期間）",
          "=" * 82]
    if passed:
        # 全期間の平均_損切5%が高い順
        passed.sort(key=lambda x: -(x[2]["平均_損切5%"] if x[2] else -99))
        for code, name, whole, first, second in passed:
            L.append(f"  ◎ {name}（{code}）: 全期間 平均(損切5%){whole['平均_損切5%']:+.2f}% "
                     f"+10%到達{whole['2週+10%到達%']:.1f}% -5%到達{whole['2週-5%到達%']:.1f}% "
                     f"[前半n{first['n']}/後半n{second['n']}]")
        L.append("")
        L.append(f"  → 44件中 {len(passed)}件 が両期間で安定。ただし各期間の件数は少なく、")
        L.append("    これも『過去の傾向』であり将来を保証しない点は変わらない。")
    else:
        L.append("  該当なし。両期間で安定して条件を満たす銘柄は無かった。")
        L.append("  ＝どの銘柄も、時期によって傾向が変わる（偶然の域を出ない）。")
    return "\n".join(L)


def main(argv=None):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    argv = argv or []
    all_sectors = ("all" in argv) or ("--all-sectors" in argv) or ("all-sectors" in argv)
    strict = ("strict" in argv) or ("--strict" in argv)
    codes = [a for a in argv if a.isdigit()]
    from sources import JQuants, JST
    jq = JQuants()
    print("キャッシュからデータを読み込み中...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    fin = jq.financials(config.BACKTEST_LOOKBACK_DAYS)
    listed = jq.listed()
    print(f"  株価 {len(quotes):,}行 / 決算 {len(fin):,}行 / 銘柄 {len(listed):,}")

    require_sector = not all_sectors

    # strict：指定銘柄（なければ既定のベスト3候補）を3分割で厳格再検証
    # 名指しした銘柄をそのまま検証するため、業種フィルタは既定で外す
    # （高勝率業種でない銘柄も検証対象にできる）。sectorを効かせたい時だけ明示。
    if strict:
        if not codes:
            codes = DEFAULT_CODES
        strict_require_sector = ("sector" in argv)  # 既定オフ、"sector"指定時のみオン
        print(f"  【厳格再検証・3分割】対象: {codes}"
              f"（業種絞り: {'あり' if strict_require_sector else 'なし'}）")
        results = run_strict(quotes, fin, listed, codes, require_sector=strict_require_sector)
        report = summarize_strict(results)
        print("\n" + report)
        today = dt.datetime.now(JST).date().isoformat()
        with open(os.path.join(config.OUTPUT_DIR, f"{today}_validate_strict.txt"),
                  "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nレポート: output/{today}_validate_strict.txt")
        return

    if not codes:
        if all_sectors:
            # 業種を問わず、今日 押し目×決算回避×悪材料なし を満たす銘柄を対象に
            codes = _todays_pullback_codes(quotes, fin, listed)
            print(f"  【業種の絞りなし】今日の押し目候補 {len(codes)}件 を全部検証します")
            print("  （数百件を過去まで走査するため時間がかかります）")
        else:
            from candidates_best import find as find_best
            cand, as_of = find_best(quotes, fin, listed)
            if cand is None or cand.empty:
                print("本日、条件に該当する候補がないため検証対象がありません。")
                return
            codes = [str(c) for c in cand["code"].tolist()]
            print(f"  今日の候補 {len(codes)}件 を全部検証します（{as_of} 時点）")
    else:
        print(f"  対象銘柄: {codes}（業種絞り: {'なし' if all_sectors else 'あり'}）")

    results = run(quotes, fin, listed, codes, require_sector=require_sector)
    report = summarize(results, only_passed=all_sectors)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    suffix = "_allsectors" if all_sectors else ""
    with open(os.path.join(config.OUTPUT_DIR, f"{today}_validate_candidates{suffix}.txt"),
              "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nレポート: output/{today}_validate_candidates{suffix}.txt")


def _todays_pullback_codes(quotes, fin, listed):
    """業種を問わず、今日 押し目×決算回避×悪材料なし を満たす銘柄コードを返す。"""
    from candidates_best import (_f_pullback, _next_earnings_date, _biz_days_until,
                                 _latest_fin_before, _bad_flags, AVOID_EARNINGS_DAYS)
    qi = technical.add_indicators(quotes)
    g = qi.sort_values(["code", "date"]).groupby("code")
    h = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / h - 1.0
    as_of = qi["date"].max()
    snap = qi[qi["date"] == as_of].drop_duplicates("code", keep="last")
    fin_ext = _prep_fin_extended(fin)
    fin_by_code = {}
    if "code" in fin.columns and "disclosed_date" in fin.columns:
        for cg, gg in fin.groupby("code"):
            fin_by_code[str(cg)] = sorted(pd.to_datetime(gg["disclosed_date"]).tolist())
    out = []
    for _, r in snap.iterrows():
        code = str(r["code"])
        if not _f_pullback(r):
            continue
        earnings = _next_earnings_date(fin_by_code, code, pd.Timestamp(as_of))
        du = _biz_days_until(as_of, earnings)
        if du is not None and du <= AVOID_EARNINGS_DAYS:
            continue
        fr = _latest_fin_before(fin_ext, code, pd.Timestamp(as_of))
        fl = _bad_flags(fr)
        if fl["予想未達"] or fl["減益"] or fl["減収"]:
            continue
        out.append(code)
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
