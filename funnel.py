# -*- coding: utf-8 -*-
"""
候補の絞り込み過程を可視化する（ファネル表示）

これまで積み上げた全条件を、最新日のデータに1つずつ適用し、
・各段階で何株が残り、何株が脱落したか（ファネル）
・最終的に全条件を通過した株
・各条件で脱落した株（どこで落ちたか付き）
をすべて出力する。

「通過した株」だけでなく「脱落した株の経過」も見えるので、
条件が厳しすぎないか、なぜその株が残った/落ちたかを確認できる。

適用する条件（順番に絞る）
--------------------------
  1. ボックス反発シグナル
  2. 押し目ゾーン（52週高値から -15〜-5%）
  3. 高勝率業種（保険・銀行・証券・倉庫運輸・水産農林）
  4. 悪材料なし（予想未達・減益・減収でない）
  5. 決算5営業日以内
  6. 過去の決算前5日勝率70%以上

※これは買い推奨ではない。条件に該当する銘柄の提示まで。
  なぜ下げているか等は最終的に本人が確認する。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import config
import technical
from refine import _f_pullback, _f_high_win_sector, HIGH_WIN_SECTORS
from badnews import _latest_fin_before, _bad_flags, _prep_fin_extended
from preentry import (_next_earnings_date, _biz_days_until,
                      _pre_earnings_winrate)


# 絞り込みステップ: (名前, 判定関数)。判定関数は row(dict)を受けてbool。
def _build_steps(fin_ext, fin_by_code, by_code, as_of):
    def s_pullback(r):
        return _f_pullback(r)
    def s_sector(r):
        return _f_high_win_sector(r)
    def s_no_bad(r):
        fin_row = _latest_fin_before(fin_ext, r["code"], as_of)
        flags = _bad_flags(fin_row)
        return not (flags["予想未達"] or flags["減益"] or flags["減収"])
    def s_near_earn(r):
        ed = _next_earnings_date(fin_by_code, r["code"], as_of)
        du = _biz_days_until(as_of, ed)
        return du is not None and du <= 5
    def s_pre_wr(r):
        wr, n = _pre_earnings_winrate(by_code, fin_by_code, r["code"], as_of, window=5)
        return wr is not None and n >= 3 and wr >= 0.70
    return [
        ("押し目ゾーン(-15〜-5%)", s_pullback),
        ("高勝率業種", s_sector),
        ("悪材料なし", s_no_bad),
        ("決算5営業日以内", s_near_earn),
        ("過去決算前勝率70%以上", s_pre_wr),
    ]


def run(quotes, fin, listed):
    qi = technical.add_indicators(quotes)
    g = qi.groupby("code")
    qi["ret_1m"] = g["close"].transform(lambda s: s / s.shift(20) - 1.0)
    h = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / h - 1.0

    sec = (listed.set_index("code")["sector33"]
           if "sector33" in listed.columns else pd.Series(dtype=object))
    names = (listed.set_index("code")["company_name"]
             if "company_name" in listed.columns else pd.Series(dtype=object))

    as_of = qi["date"].max()
    snap = technical._snapshot(qi, as_of).set_index("code")
    by_code = technical._build_index(qi)
    fin_ext = _prep_fin_extended(fin)
    fin_by_code = {}
    for code, gg in fin.groupby("code"):
        fin_by_code[code] = sorted(pd.to_datetime(gg["disclosed_date"]).tolist())

    # 起点：ボックス反発シグナル
    snap_reset = snap.reset_index()
    start_codes = technical.entries_box_bottom(snap_reset)

    # 各銘柄の情報を dict 化
    def row_of(code):
        s = snap.loc[code]
        return {
            "code": code,
            "name": names.get(code, ""),
            "sector33": sec.get(code),
            "close": float(s["close"]),
            "pct_from_high": float(s.get("pct_from_high", np.nan)),
            "ret_1m": float(s.get("ret_1m", np.nan)) if pd.notna(s.get("ret_1m", np.nan)) else np.nan,
            "turnover_value": float(s.get("turnover_value", np.nan)),
        }

    current = [row_of(c) for c in start_codes if c in snap.index]
    steps = _build_steps(fin_ext, fin_by_code, by_code, as_of)

    # ファネル：各段階で通過/脱落を記録
    funnel = [("ボックス反発シグナル", len(current), 0)]
    dropped = []      # 脱落した銘柄（どの条件で落ちたか付き）
    for step_name, func in steps:
        passed, failed = [], []
        for r in current:
            if func(r):
                passed.append(r)
            else:
                r2 = dict(r); r2["脱落条件"] = step_name
                failed.append(r2)
        dropped.extend(failed)
        funnel.append((step_name, len(passed), len(failed)))
        current = passed

    return {
        "as_of": pd.Timestamp(as_of).date(),
        "funnel": funnel,
        "survivors": current,      # 全条件通過
        "dropped": dropped,        # 脱落（脱落条件付き）
    }


def _fmt_row(r, with_drop=False):
    base = (f"  {r['code']:>6}  {str(r['name'])[:16]:<16} {str(r.get('sector33',''))[:10]:<10} "
            f"{r['close']:>8.1f}  高値から{r['pct_from_high']*100:>6.1f}%  "
            f"直近1M{r['ret_1m']*100:>6.1f}%  代金{r['turnover_value']/1e8:>6.1f}億")
    if with_drop:
        base += f"  ← {r['脱落条件']}で脱落"
    return base


def summarize(result):
    L = []
    L.append("=" * 78)
    L.append(f"候補の絞り込み過程（ファネル）  ({result['as_of']} 時点)")
    L.append("=" * 78)
    L.append("")
    L.append("■ 各段階での通過・脱落")
    prev = None
    for i, (name, remain, dropped) in enumerate(result["funnel"]):
        if i == 0:
            L.append(f"  {name:<24} : {remain:>4} 株")
        else:
            bar = "█" * remain
            L.append(f"  └ {name:<22} : {remain:>4} 株  (−{dropped}株脱落) {bar}")
    L.append("")

    # 生き残り（全条件通過）
    surv = result["survivors"]
    L.append("=" * 78)
    L.append(f"■ 全条件を通過した株（最終候補）: {len(surv)} 株")
    L.append("=" * 78)
    if surv:
        for r in sorted(surv, key=lambda x: x["pct_from_high"]):
            L.append(_fmt_row(r))
    else:
        L.append("  なし（今日は全条件を通過する株がありません）")
    L.append("")

    # 脱落した株（どこで落ちたか）
    dropped = result["dropped"]
    L.append("=" * 78)
    L.append(f"■ 脱落した株の経過: {len(dropped)} 株（どの条件で落ちたか）")
    L.append("=" * 78)
    # 脱落条件ごとにまとめる
    if dropped:
        by_step = {}
        for r in dropped:
            by_step.setdefault(r["脱落条件"], []).append(r)
        # ファネルの順番で表示
        for name, _, _ in result["funnel"][1:]:
            group = by_step.get(name, [])
            if not group:
                continue
            L.append(f"\n  【{name} で脱落: {len(group)}株】")
            for r in sorted(group, key=lambda x: x["pct_from_high"])[:15]:
                L.append(_fmt_row(r))
            if len(group) > 15:
                L.append(f"    …ほか {len(group)-15} 株")
    else:
        L.append("  なし")

    L.append("")
    L.append("=" * 78)
    L.append("読み方・注意")
    L.append("=" * 78)
    L.append("・上から順に条件で絞り込む。各段階で何株が残り、何株が落ちたかが分かる。")
    L.append("・『全条件を通過した株』が最終候補。ただし買い推奨ではない。")
    L.append("・『脱落した株の経過』で、どの条件が厳しく効いているかが見える。")
    L.append("  多くがある条件で脱落しているなら、その条件がボトルネック。")
    L.append("・最終候補も、なぜ下げているか（悪材料か地合いか）は必ず自分で確認する。")
    L.append("・過去検証の超過リターンはわずかで、手数料を引くと消える水準。")
    L.append("・最終的な投資判断はご自身の責任で。")
    return "\n".join(L)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    from sources import JQuants, JST
    jq = JQuants()
    print("キャッシュからデータを読み込み中...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    fin = jq.financials(config.BACKTEST_LOOKBACK_DAYS)
    listed = jq.listed()
    print(f"  株価 {len(quotes):,}行 / 決算 {len(fin):,}行 / 銘柄 {len(listed):,}")

    result = run(quotes, fin, listed)
    report = summarize(result)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    # 生き残りと脱落をCSVでも保存
    if result["survivors"]:
        pd.DataFrame(result["survivors"]).to_csv(
            os.path.join(config.OUTPUT_DIR, f"{today}_funnel_survivors.csv"),
            index=False, encoding="utf-8-sig")
    if result["dropped"]:
        pd.DataFrame(result["dropped"]).to_csv(
            os.path.join(config.OUTPUT_DIR, f"{today}_funnel_dropped.csv"),
            index=False, encoding="utf-8-sig")
    print(f"\nCSV保存: output/{today}_funnel_survivors.csv / _dropped.csv")


if __name__ == "__main__":
    main()
