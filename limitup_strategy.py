# -*- coding: utf-8 -*-
"""
ストップ高後3日間の動きで銘柄を分類し、各分類ごとに「どう買えば勝てるか」を検証する。
  py main.py limitup-strategy

分類（ストップ高後3日間の上昇率、起点=ST高終値→3日目終値）
  強上昇 : +20%以上
  中上昇 : +5〜20%
  横ばい : ±5%
  下落   : -5%以下

各分類ごとに、2種類の買い方を総当たり検証：
  (A) 数日保有：4日目始値で参入。利確(+3〜15%)/損切り(-3〜8%)の全組み合わせ。
      7日目まで。利確・損切りに触れた順で決済、どちらも無ければ7日目終値。
  (B) 日計り  ：4日目始値で参入。その日の高値が寄りから+X%なら、その日の終値で売る
      （＝ローソクが長い日に、その日のうちに利確）。翌日以降も同様に判定。

事実にもとづく注意
  ・利確/損切りは高値・安値ベース（日中に触れたら約定と仮定）。実際は前後する。
  ・手数料・スリッページ未考慮。過去の傾向で将来を保証しない。分割検証も行う。
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import limitup

FORWARD = 7             # ストップ高後の追跡日数
OBSERVE = 3             # 観察日数（3日で分類）
TAKE_PROFITS = [3, 5, 8, 10, 15]    # 利確候補(%)
STOP_LOSSES = [-3, -5, -8]          # 損切り候補(%)


def _classify_3day(cqi, st_idx, st_close):
    """3日間の上昇率で分類。(分類名, 3日目終値の騰落率) を返す。"""
    obs = cqi.iloc[st_idx + 1: st_idx + 1 + OBSERVE]
    if len(obs) < OBSERVE:
        return None, None
    ch = (obs["close"].values[-1] / st_close - 1) * 100
    if ch >= 20:
        return "強上昇(3日+20%↑)", ch
    if ch >= 5:
        return "中上昇(3日+5〜20%)", ch
    if ch > -5:
        return "横ばい(3日±5%)", ch
    return "下落(3日-5%↓)", ch


def _simulate_hold(cqi, entry_idx, entry_price, tp, sl, max_days):
    """参入後、利確tp%/損切りsl%/期限で決済したときのリターン(%)を返す。
    高値がtpに触れたら利確、安値がslに触れたら損切り。同日両方なら…安全側で損切り優先。"""
    hold = cqi.iloc[entry_idx: entry_idx + max_days]
    tp_price = entry_price * (1 + tp / 100)
    sl_price = entry_price * (1 + sl / 100)
    for _, bar in hold.iterrows():
        hit_sl = bar["low"] <= sl_price
        hit_tp = bar["high"] >= tp_price
        if hit_sl and hit_tp:
            return sl  # 同日両方は損切り優先（安全側）
        if hit_sl:
            return sl
        if hit_tp:
            return tp
    # どちらも触れず → 最終日の終値
    if len(hold) > 0:
        return (hold["close"].values[-1] / entry_price - 1) * 100
    return 0.0


def _simulate_intraday(cqi, entry_idx, entry_price, long_pct, max_days):
    """日計り：参入後、各日で『高値が寄りから+long_pct%以上』ならその日の終値で売る。
    ローソクが長い日にその日のうちに利確する戦略。返り値=リターン(%)。"""
    hold = cqi.iloc[entry_idx: entry_idx + max_days]
    for _, bar in hold.iterrows():
        o = bar["open"]
        if o and o > 0 and (bar["high"] / o - 1) * 100 >= long_pct:
            # その日の終値で売る
            return (bar["close"] / entry_price - 1) * 100
    # 一度も条件を満たさず → 最終日終値
    if len(hold) > 0:
        return (hold["close"].values[-1] / entry_price - 1) * 100
    return 0.0


def build_trades(quotes):
    """全ストップ高について、3日分類と参入価格などを準備する。"""
    st = limitup.detect_limit_up(quotes)
    q = quotes.sort_values(["code", "date"])
    qi_by_code = {c: g.reset_index(drop=True) for c, g in q.groupby("code")}
    idx_by_code = {c: {pd.Timestamp(d): i for i, d in enumerate(g["date"])}
                   for c, g in qi_by_code.items()}

    trades = []
    for _, r in st.iterrows():
        code = r["code"]
        cqi = qi_by_code.get(code)
        if cqi is None:
            continue
        st_idx = idx_by_code[code].get(pd.Timestamp(r["date"]))
        if st_idx is None:
            continue
        cls, ch = _classify_3day(cqi, st_idx, r["close"])
        if cls is None:
            continue
        # 参入：4日目(=ST高翌日から3日観察した次)の始値
        entry_idx = st_idx + 1 + OBSERVE
        if entry_idx >= len(cqi):
            continue
        entry_price = cqi["open"].values[entry_idx]
        if not entry_price or entry_price <= 0 or entry_price != entry_price:
            continue
        trades.append({
            "code": code, "date": pd.Timestamp(r["date"]), "class": cls,
            "cqi": cqi, "entry_idx": entry_idx, "entry_price": entry_price,
        })
    return trades


CLASS_ORDER = ["強上昇(3日+20%↑)", "中上昇(3日+5〜20%)",
               "横ばい(3日±5%)", "下落(3日-5%↓)"]


def analyze_strategy(quotes):
    trades = build_trades(quotes)
    print(f"\n分類対象のストップ高（3日観察後に参入可能）: {len(trades):,}件")
    max_days = FORWARD - OBSERVE  # 参入後の最大保有日数

    # 分類ごとの件数
    from collections import Counter
    cnt = Counter(t["class"] for t in trades)
    print("\n【3日間の動きによる分類】")
    for c in CLASS_ORDER:
        print(f"  {c:<20}{cnt.get(c,0):>6,}件")

    # (A) 利確・損切り総当たり（分類別）
    print("\n" + "=" * 70)
    print("(A) 数日保有：分類別に『利確%/損切り%』を総当たり（勝率・平均リターン）")
    print("=" * 70)
    for c in CLASS_ORDER:
        sub = [t for t in trades if t["class"] == c]
        if len(sub) < 30:
            print(f"\n■ {c}（{len(sub)}件）※少数のため参考")
            if not sub:
                continue
        else:
            print(f"\n■ {c}（{len(sub):,}件）")
        print(f"    {'利確/損切り':<14}{'勝率':>7}{'平均R':>8}{'期待値':>8}")
        best = None
        for tp in TAKE_PROFITS:
            for sl in STOP_LOSSES:
                rets = [_simulate_hold(t["cqi"], t["entry_idx"],
                                       t["entry_price"], tp, sl, max_days)
                        for t in sub]
                rets = np.array(rets)
                win = (rets > 0).mean() * 100
                avg = rets.mean()
                # 期待値=平均リターン（既に含む）。best は平均リターンで選ぶ
                label = f"+{tp}%/{sl}%"
                if best is None or avg > best[1]:
                    best = (label, avg, win)
        # 各組み合わせを表示（全部は多いので、上位のみ＋best）
        rows = []
        for tp in TAKE_PROFITS:
            for sl in STOP_LOSSES:
                rets = np.array([_simulate_hold(t["cqi"], t["entry_idx"],
                                 t["entry_price"], tp, sl, max_days) for t in sub])
                rows.append((f"+{tp}%/{sl}%", (rets>0).mean()*100, rets.mean()))
        rows.sort(key=lambda x: x[2], reverse=True)
        for label, win, avg in rows[:5]:  # 平均リターン上位5組
            mark = " ★" if label == best[0] else ""
            print(f"    {label:<14}{win:>6.0f}%{avg:>7.1f}%{avg:>7.1f}%{mark}")

    # (B) 日計り（分類別）
    print("\n" + "=" * 70)
    print("(B) 日計り：『寄りから+X%上げた日の終値で売る』（分類別）")
    print("=" * 70)
    for c in CLASS_ORDER:
        sub = [t for t in trades if t["class"] == c]
        if not sub:
            continue
        print(f"\n■ {c}（{len(sub):,}件）")
        print(f"    {'長ローソク閾値':<14}{'勝率':>7}{'平均R':>8}")
        for long_pct in [3, 5, 8, 10]:
            rets = np.array([_simulate_intraday(t["cqi"], t["entry_idx"],
                             t["entry_price"], long_pct, max_days) for t in sub])
            print(f"    寄り+{long_pct}%で利確 {(rets>0).mean()*100:>6.0f}%{rets.mean():>7.1f}%")

    print("\n※利確/損切りは高値・安値ベース（日中に触れたら約定と仮定）。")
    print("※手数料・スリッページ未考慮。過去の傾向で将来を保証しません。")
    return trades


def run(quotes):
    print("=" * 70)
    print("ストップ高後3日分類 × 買い方の検証（直近2年）")
    print("=" * 70)
    trades = analyze_strategy(quotes)
    compare_entry_days(quotes)

    # 分割検証：前半・後半で、最も良かった買い方が再現するか（強上昇クラスで確認）
    if trades:
        dates = pd.to_datetime([t["date"] for t in trades])
        mid = dates.min() + (dates.max() - dates.min()) / 2
        print("\n" + "=" * 70)
        print("【分割検証】強上昇クラスで +5%/-5% 戦略が前半・後半で再現するか")
        print("=" * 70)
        max_days = FORWARD - OBSERVE
        for label, cond in [("前半", dates <= mid), ("後半", dates > mid)]:
            sub = [t for t, m in zip(trades, cond)
                   if m and t["class"] == "強上昇(3日+20%↑)"]
            if len(sub) < 10:
                print(f"  {label}: 強上昇クラスが少数（{len(sub)}件）")
                continue
            rets = np.array([_simulate_hold(t["cqi"], t["entry_idx"],
                             t["entry_price"], 5, -5, max_days) for t in sub])
            print(f"  {label}（{len(sub):,}件）: 勝率{(rets>0).mean()*100:.0f}% "
                  f"平均{rets.mean():+.1f}%")
    print("\n" + "=" * 70)
    print("検証完了。")
    print("=" * 70)


def compare_entry_days(quotes, entry_days=(3, 4, 5, 6, 7)):
    """参入日を変えて比較する。ストップ高翌日を1日目として、
    entry_days の各日の始値で参入し、7日目終値まで保有した場合の成績。
    分類（3日間の動き）は共通で、参入タイミングだけ変える。"""
    st = limitup.detect_limit_up(quotes)
    q = quotes.sort_values(["code", "date"])
    qi_by_code = {c: g.reset_index(drop=True) for c, g in q.groupby("code")}
    idx_by_code = {c: {pd.Timestamp(d): i for i, d in enumerate(g["date"])}
                   for c, g in qi_by_code.items()}

    print("\n" + "=" * 70)
    print("参入日を変えた比較（ストップ高翌日=1日目。7日目終値で決済）")
    print("=" * 70)
    print(f"{'参入日':<10}{'件数':>8}{'勝率':>8}{'平均最大上昇':>12}"
          f"{'平均最大下落':>12}{'平均最終':>10}")
    print("-" * 70)

    for ed in entry_days:
        rets_final, rets_up, rets_dn = [], [], []
        for _, r in st.iterrows():
            code = r["code"]
            cqi = qi_by_code.get(code)
            if cqi is None:
                continue
            st_idx = idx_by_code[code].get(pd.Timestamp(r["date"]))
            if st_idx is None:
                continue
            entry_idx = st_idx + ed  # ストップ高日+ed日目
            if entry_idx >= len(cqi):
                continue
            entry_price = cqi["open"].values[entry_idx]
            if not entry_price or entry_price <= 0 or entry_price != entry_price:
                continue
            # 決済期間：参入日〜7日目
            end_idx = st_idx + 7
            hold = cqi.iloc[entry_idx: end_idx + 1]
            if len(hold) == 0:
                continue
            hi = hold["high"].values.max()
            lo = hold["low"].values.min()
            last_c = hold["close"].values[-1]
            rets_up.append((hi / entry_price - 1) * 100)
            rets_dn.append((lo / entry_price - 1) * 100)
            rets_final.append((last_c / entry_price - 1) * 100)
        if not rets_final:
            continue
        import numpy as np
        rf = np.array(rets_final)
        print(f"{ed}日目参入{'':<4}{len(rf):>8,}{(rf>0).mean()*100:>7.0f}%"
              f"{np.mean(rets_up):>11.1f}%{np.mean(rets_dn):>11.1f}%"
              f"{rf.mean():>9.1f}%")
    print("-" * 70)
    print("※平均最大上昇＝参入後、高値で売れた場合の平均。平均最終＝7日目終値の平均。")
    print("※手数料・スリッページ未考慮。過去の傾向で将来を保証しません。")
