# -*- coding: utf-8 -*-
"""
決算前の上昇取り（条件の積み上げ）

これまでの検証で有効だった条件を、そのまま足し算する：

  土台1: ボックス反発のエントリー          （technical）
  土台2: 押し目ゾーン × 高勝率業種          （refine で有効確認）
  土台3: 決算1営業日前に手仕舞い            （preearnings で改善確認）
  ＋追加: 決算が N営業日以内に控えている銘柄だけに絞る  ← 今回の新条件

つまり「押し目で入った高勝率業種の銘柄のうち、決算が近いものだけを買い、
決算の直前に売る」。決算前に期待で上がりやすいなら、その上昇だけ取れる。

先読み防止
----------
銘柄選びに未来の株価は使わない。決算予定日は「直近開示+90日」で推定
（実運用では会社が事前公表するので分かる情報）。エントリー判定は
すべて as_of 時点で分かる情報のみ。

比較
----
  A: 押し目×高勝率業種 ＋ 決算前手仕舞い（決算の近さで絞らない）＝ 前回まで
  B: A ＋ 決算がN営業日以内に限定                         ＝ 今回の追加
Bの成績がAより良ければ、「決算前に絞る」ことに効果がある。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import backtest
import config
import technical
from refine import _f_pullback, _f_high_win_sector
from preearnings import (_next_earnings_date, _exit_box_with_earnings,
                         _forward_with_dates)
from badnews import _latest_fin_before, _bad_flags, _prep_fin_extended


# 決算の「近さ」の候補（営業日）。エントリー時点で決算までこの日数以内なら対象。
NEAR_EARNINGS_DAYS = [5, 10, 15, 20]


def _biz_days_until(as_of, earnings_date):
    """as_of から earnings_date までの概算営業日数。"""
    if earnings_date is None:
        return None
    cal_days = (earnings_date - pd.Timestamp(as_of)).days
    if cal_days < 0:
        return None
    return int(cal_days * 5 / 7)      # 暦日→営業日の概算


def _price_change_before(close_arr, dates_arr, target_date, window=5):
    """target_date の window営業日前 → 1営業日前 の騰落率を返す。

    決算「前」の値動きなので、決算当日は含めない。
    決算日の1営業日前の終値 ÷ window+1営業日前の終値 - 1。
    """
    pos = np.searchsorted(dates_arr, np.datetime64(pd.Timestamp(target_date)), side="left")
    # pos は決算日（またはそれ以降）の位置。決算前日は pos-1。
    end = pos - 1                      # 決算1営業日前
    start = end - window              # そのwindow営業日前
    if start < 0 or end <= start or end >= len(close_arr):
        return None
    p0, p1 = close_arr[start], close_arr[end]
    if not p0 or p0 <= 0:
        return None
    return p1 / p0 - 1.0


def _pre_earnings_winrate(by_code, fin_by_code, code, as_of, window=5):
    """as_of 以前に開示された決算について、決算前window日の上昇率(勝率)を返す。

    先読み防止：as_of より前の決算だけを使う。
    戻り値: (勝率, 母数)。母数が0なら (None, 0)。
    """
    arrs = by_code.get(code)
    disc = fin_by_code.get(code)
    if arrs is None or not disc:
        return None, 0
    close_arr, dates_arr = arrs[0], arrs[1]
    past_earnings = [d for d in disc if d < pd.Timestamp(as_of)]
    if not past_earnings:
        return None, 0
    ups, total = 0, 0
    for ed in past_earnings:
        ch = _price_change_before(close_arr, dates_arr, ed, window)
        if ch is None:
            continue
        total += 1
        if ch > 0:
            ups += 1
    if total == 0:
        return None, 0
    return ups / total, total


def run(quotes, fin, listed, step_days=7):
    qi = technical.add_indicators(quotes)
    g = qi.groupby("code")
    qi["ret_1m"] = g["close"].transform(lambda s: s / s.shift(20) - 1.0)
    h = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / h - 1.0
    sec = (listed.set_index("code")["sector33"]
           if "sector33" in listed.columns else pd.Series(dtype=object))

    by_code = technical._build_index(qi)
    all_dates = np.sort(qi["date"].unique())
    base_index = {c: (v[0], v[1]) for c, v in by_code.items()}

    fin_by_code = {}
    for code, gg in fin.groupby("code"):
        fin_by_code[code] = sorted(pd.to_datetime(gg["disclosed_date"]).tolist())

    # 悪材料判定用（前年同期比・会社予想つき）の決算データ
    fin_ext = _prep_fin_extended(fin)

    # 土台：ボックス反発 × 押し目 × 高勝率業種 のエントリーを集める
    # （決算前手仕舞いは全ルール共通。決算の近さだけを後で変える）
    print("  土台（ボックス反発×押し目×高勝率業種）のエントリーを収集中...")
    spec = technical.STRATEGIES["B_ボックス反発"]
    horizon = spec["horizon"]
    entries = []
    s_i, e_i = 80, len(all_dates) - horizon
    for as_of in all_dates[s_i:e_i:step_days]:
        snap = technical._snapshot(qi, as_of)
        if snap.empty:
            continue
        codes = technical.entries_box_bottom(snap)
        if not codes:
            continue
        snap_i = snap.set_index("code")
        for code in codes:
            if code not in snap_i.index:
                continue
            row = snap_i.loc[code].copy()
            row["sector33"] = sec.get(code)
            # 押し目は土台として固定（業種は評価段階で切り替えられるようフラグ化）
            if not _f_pullback(row):
                continue
            is_high_win_sector = _f_high_win_sector(row)
            path, path_dates = _forward_with_dates(by_code, code, as_of, horizon)
            if path is None:
                continue
            earnings = _next_earnings_date(fin_by_code, code, as_of)
            days_until = _biz_days_until(as_of, earnings)
            # 先読み防止：as_of以前の決算だけで、決算前5日の勝率を計算
            wr, wr_n = _pre_earnings_winrate(by_code, fin_by_code, code, as_of, window=5)
            # 悪材料判定（as_of以前の直近決算・先読みなし）
            fin_row = _latest_fin_before(fin_ext, code, as_of)
            flags = _bad_flags(fin_row)
            has_bad = flags["予想未達"] or flags["減益"] or flags["減収"]
            entries.append({"as_of": pd.Timestamp(as_of), "horizon": horizon,
                            "path": path, "path_dates": path_dates,
                            "earnings": earnings, "days_until": days_until,
                            "pre_winrate": wr, "pre_winrate_n": wr_n,
                            "has_bad_news": has_bad,
                            "high_win_sector": is_high_win_sector})
    print(f"  土台の候補（押し目のみ・業種は後で切替）: {len(entries):,} 件")

    base_cache = {}
    def _base(as_of, h):
        k = (as_of, h)
        if k not in base_cache:
            base_cache[k] = backtest.market_baseline(base_index, as_of, h)
        return base_cache[k]

    def _evaluate(near_days, min_pre_winrate=None, min_pre_n=3,
                  exclude_bad_news=False, require_sector=True,
                  avoid_near_days=None):
        """near_days: 決算までの営業日数上限（Noneなら絞らない・決算前に限定）。
        avoid_near_days: この営業日数以内に決算がある銘柄を除外（決算前を避ける）。
        min_pre_winrate: 過去の決算前5日勝率の下限（Noneなら絞らない）。
        min_pre_n: 勝率を信頼する最低母数（過去決算がこれ未満なら除外）。
        exclude_bad_news: Trueなら悪材料（予想未達/減益/減収）ありを除外。
        require_sector: Trueなら高勝率業種に限定、Falseなら業種を問わない。"""
        rets, period_map = [], {}
        for e in entries:
            if require_sector and not e.get("high_win_sector"):
                continue          # 高勝率業種でなければ除外
            if near_days is not None:
                du = e["days_until"]
                if du is None or du > near_days:
                    continue      # 決算が近くない銘柄は除外（決算前に限定）
            if avoid_near_days is not None:
                du = e["days_until"]
                if du is not None and du <= avoid_near_days:
                    continue      # 決算が近い銘柄を除外（決算前を避ける）
            if exclude_bad_news and e.get("has_bad_news"):
                continue          # 悪材料ありを除外
            if min_pre_winrate is not None:
                wr, wr_n = e["pre_winrate"], e["pre_winrate_n"]
                if wr is None or wr_n < min_pre_n:
                    continue      # 過去決算が少なく判定不能なら除外
                if wr < min_pre_winrate:
                    continue      # 過去の決算前勝率が基準未満なら除外
            r, _ = _exit_box_with_earnings(
                e["path_dates"], e["path"], 0.15, -0.07,
                earnings_date=e["earnings"])
            rets.append(r)
            period_map.setdefault((e["as_of"], e["horizon"]), []).append(r)
        if len(rets) < 30:
            return {"取引数": len(rets), "勝率%": None, "平均%": None,
                    "中央%": None, "対相場超過%": None}
        arr = np.array(rets)
        excess = []
        for (as_of, h), rs in period_map.items():
            b = _base(as_of, h)
            if pd.notna(b):
                excess.append(np.median(rs) - b)
        return {
            "取引数": len(arr),
            "勝率%": round((arr > 0).mean() * 100, 1),
            "平均%": round(arr.mean() * 100, 2),
            "中央%": round(np.median(arr) * 100, 2),
            "対相場超過%": round(np.median(excess) * 100, 2) if excess else None,
        }

    rows = []
    # 基準：押し目×高勝率業種（決算を問わない）＝今日の最良
    rows.append({"ルール": "押し目×高勝率業種(決算問わない・基準)",
                 **_evaluate(None, require_sector=True)})
    # 決算前を避ける：決算がN営業日以内の銘柄を除外
    rows.append({"ルール": "─── 決算前◯日以内を避ける ───",
                 "取引数": None, "勝率%": None, "平均%": None,
                 "中央%": None, "対相場超過%": None})
    for ad, label in [(5, "1週間"), (10, "2週間"), (20, "1か月"), (30, "1.5か月")]:
        rows.append({"ルール": f"決算前{ad}営業日({label})以内を避ける",
                     **_evaluate(None, require_sector=True, avoid_near_days=ad)})
    # 決算回避 + 悪材料なし も見る
    rows.append({"ルール": "─── 決算1か月回避 + 悪材料なし ───",
                 "取引数": None, "勝率%": None, "平均%": None,
                 "中央%": None, "対相場超過%": None})
    rows.append({"ルール": "決算前1か月回避 + 悪材料なし",
                 **_evaluate(None, require_sector=True, avoid_near_days=20,
                             exclude_bad_news=True)})
    return pd.DataFrame(rows)


def summarize(df):
    L = ["=" * 74,
         "決算前の上昇取り（押し目×高勝率業種×決算前手仕舞い ＋ 決算の近さ）",
         "=" * 74, ""]
    L.append(df.to_string(index=False))
    L += ["", "=" * 74, "読み方", "=" * 74,
          "・基準は『押し目×高勝率業種』（決算を問わない）＝これまでの最良。",
          "・『決算前◯日以内を避ける』が基準より対相場超過・勝率で上回れば、",
          "  決算が近い銘柄を避けることに効果がある（決算ギャンブルの回避が有効）。",
          "・下回れば、決算前を避けると良い機会も逃している、ということ。",
          "・避ける期間を1週間〜1.5か月で変えて、最適な回避幅を見る。",
          "・取引数は避けるほど減る。NaN=件数不足で評価対象外。",
          "・決算日は推定（直近開示+90日）。実際の予定日とはズレる。",
          "・有効な組み合わせは、分割検証で本物か確認する。",
          "・手数料・スリッページ未考慮。"]
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

    df = run(quotes, fin, listed)
    report = summarize(df)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    path = os.path.join(config.OUTPUT_DIR, f"{today}_preentry.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n比較表: {path}")


if __name__ == "__main__":
    main()
