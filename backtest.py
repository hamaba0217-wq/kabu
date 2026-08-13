# -*- coding: utf-8 -*-
"""
バックテスト

過去の各時点でスクリーニングを実行し、3か月後にどうなったかを集計します。

先読み（look-ahead bias）の防止
--------------------------------
これがバックテストで最も壊れやすい部分です。以下を徹底しています。

1. 株価は `date <= as_of` のみ使用
2. 決算は **開示日** `disclosed_date <= as_of` のみ使用
   （決算期末ではなく開示日で切るのが重要。期末日で切ると、
     まだ発表されていない数字を使ってしまいます）
3. 移動平均・52週高値はすべて後方参照のみ

「未来を知っているから儲かって見える」状態を作らないための措置です。

Freeプランでの制約
------------------
- データは12週間遅延・過去2年分
- 3か月の保有期間を確保する必要があるため、検証に使える起点は約1年半分
- そのうち **重複しない3か月区間は6つ程度**
  → 銘柄のサンプルは数十〜数百件取れますが、相場局面のサンプルは6個です。
     「たまたま良い相場だった」可能性を排除できません。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import screen

# Role 7 の執行ルール
STOP_LOSS = -0.25                       # ハード損切り
TRANCHES = [(0.50, 0.25),               # +50%で1/4売却
            (1.00, 0.25),               # +100%でさらに1/4
            (2.00, 0.25)]               # +200%（3倍）でさらに1/4
HOLD_DAYS = 60                          # 保有期限（営業日）


def simulate_position(path: np.ndarray) -> tuple[float, str]:
    """1銘柄の値動きに Role 7 の執行ルール（3倍狙い・分割利確）を当てる。"""
    remaining = 1.0
    realized = 0.0
    hit = [False] * len(TRANCHES)
    for px in path:
        ret = px - 1.0
        if ret <= STOP_LOSS:
            realized += remaining * STOP_LOSS
            return realized, "損切り"
        for i, (level, frac) in enumerate(TRANCHES):
            if not hit[i] and ret >= level:
                realized += frac * level
                remaining -= frac
                hit[i] = True
    if len(path):
        realized += remaining * (path[-1] - 1.0)
    reason = "3倍達成" if hit[-1] else ("期限切れ" if remaining > 0 else "全利確")
    return realized, reason


def simulate_target(path: np.ndarray, target: float,
                    stop_loss: float) -> tuple[float, str]:
    """目標到達で全部利確、損切りラインで撤退、期限まで未達なら成行手仕舞い。

    実戦に近いシンプルなルール。日次終値ベースで、
    先に触れた方（損切り or 目標）で決済する。同日に両方なら損切り優先（保守的）。

    target:    目標リターン（+0.5 = 1.5倍, +1.0 = 2倍）
    stop_loss: 損切りライン（-0.15 など、負の値）
    """
    for px in path:
        ret = px - 1.0
        if ret <= stop_loss:
            return stop_loss, "損切り"
        if ret >= target:
            return target, "目標達成"
    # 期限まで未達 → 最終終値で手仕舞い
    final = path[-1] - 1.0 if len(path) else 0.0
    return final, "期限切れ"


def build_index(q_rolled: pd.DataFrame) -> dict:
    """銘柄ごとに (close配列, date配列) を1度だけ作る。

    これを使い回すことで、起点ごとに全データを検索する無駄をなくす。
    backtest全体でこの索引を1回作れば十分。
    """
    idx = {}
    for code, g in q_rolled.sort_values("date").groupby("code"):
        idx[code] = (g["close"].to_numpy(), g["date"].to_numpy())
    return idx


def forward_stats(by_code: dict, codes: list[str], as_of,
                  horizon: int = HOLD_DAYS) -> pd.DataFrame:
    """as_of 以降の値動きから、各銘柄の結果を計算する（索引辞書ベース）。"""
    import numpy as _np
    as_of_ts = _np.datetime64(pd.Timestamp(as_of))
    rows = []
    for code in codes:
        arrs = by_code.get(code)
        if arrs is None:
            continue
        close_arr, dates_arr = arrs
        pos = _np.searchsorted(dates_arr, as_of_ts, side="right")
        if pos == 0:
            continue
        entry = close_arr[pos - 1]
        if not entry or entry <= 0:
            continue
        fut = close_arr[pos:pos + horizon]
        if len(fut) < horizon * 0.6:
            continue
        path = fut / entry
        ret_sim, reason = simulate_position(path)
        rows.append({
            "code": code,
            "entry": entry,
            "buy_hold_return": path[-1] - 1.0,
            "max_return": path.max() - 1.0,
            "min_return": path.min() - 1.0,
            "rule_return": ret_sim,
            "exit_reason": reason,
        })
    return pd.DataFrame(rows)


def market_baseline(by_code: dict, as_of, horizon: int = HOLD_DAYS) -> float:
    """同じ期間の全銘柄リターンの中央値（＝相場全体の水準・索引ベース）。

    これが無いと「+30%だった」が良い結果なのか判断できません。
    相場全体が+35%なら、その戦略は市場に負けています。
    """
    import numpy as _np
    as_of_ts = _np.datetime64(pd.Timestamp(as_of))
    rets = []
    for close_arr, dates_arr in by_code.values():
        pos = _np.searchsorted(dates_arr, as_of_ts, side="right")
        if pos == 0:
            continue
        entry = close_arr[pos - 1]
        fut = close_arr[pos:pos + horizon]
        if entry <= 0 or len(fut) == 0:
            continue
        rets.append(fut[-1] / entry - 1.0)
    if not rets:
        return float("nan")
    return float(_np.median(rets))


def _unused_market_baseline_old(q_rolled, as_of, horizon=HOLD_DAYS):
    hist = q_rolled[q_rolled["date"] <= pd.Timestamp(as_of)]
    if hist.empty:
        return float("nan")
    base = hist.groupby("code").tail(1)[["code", "close"]].rename(columns={"close": "entry"})
    fut = q_rolled[q_rolled["date"] > pd.Timestamp(as_of)]
    if fut.empty:
        return float("nan")
    end = (fut.sort_values("date").groupby("code").head(horizon)
              .groupby("code").tail(1)[["code", "close"]])
    m = base.merge(end, on="code")
    m = m[m["entry"] > 0]
    return float((m["close"] / m["entry"] - 1.0).median())


def run(quotes: pd.DataFrame, fin: pd.DataFrame, listed: pd.DataFrame,
        step_days: int = 21, horizon: int = HOLD_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """バックテスト本体。

    step_days: 何営業日ごとにスクリーニングを実行するか（21 ≒ 1か月）
    """
    q = screen.add_rolling(quotes, config.MA_WINDOW)
    all_dates = np.sort(q["date"].unique())

    # 【高速化】銘柄ごとの索引を1度だけ作り、全起点で使い回す
    by_code = build_index(q)

    # 起点として使える期間：移動平均に必要な日数 〜 最終日から horizon 手前まで
    start_idx = max(config.MA_WINDOW, 30)
    end_idx = len(all_dates) - horizon
    if end_idx <= start_idx:
        raise SystemExit(
            f"データが足りません。営業日{len(all_dates)}日分では"
            f"{horizon}営業日の検証ができません。"
        )
    as_of_list = all_dates[start_idx:end_idx:step_days]

    per_trade, per_period = [], []

    for as_of in as_of_list:
        prices = screen.snapshot_at(q, as_of)
        f_hist = fin[fin["disclosed_date"] <= pd.Timestamp(as_of)]
        yoy = screen.yoy_table(f_hist)
        if yoy.empty:
            continue

        picks, _ = screen.run_screen(prices, yoy, listed, pd.DataFrame())
        n = len(picks)
        base = market_baseline(by_code, as_of, horizon)

        if n == 0:
            per_period.append({"as_of": pd.Timestamp(as_of).date(), "n": 0,
                               "median": np.nan, "mean_rule": np.nan,
                               "win_rate": np.nan, "market": base, "excess": np.nan})
            continue

        res = forward_stats(by_code, picks["code"].tolist(), as_of, horizon)
        if res.empty:
            continue
        res = res.merge(picks[["code", "company_name", "op_yoy", "sales_yoy"]],
                        on="code", how="left")
        res["as_of"] = pd.Timestamp(as_of).date()
        per_trade.append(res)

        med = res["buy_hold_return"].median()
        per_period.append({
            "as_of": pd.Timestamp(as_of).date(),
            "n": len(res),
            "median": med,
            "mean_rule": res["rule_return"].mean(),
            "win_rate": (res["buy_hold_return"] > 0).mean(),
            "market": base,
            "excess": med - base if pd.notna(base) else np.nan,
        })

    trades = pd.concat(per_trade, ignore_index=True) if per_trade else pd.DataFrame()
    periods = pd.DataFrame(per_period)
    return trades, periods


def summarize(trades: pd.DataFrame, periods: pd.DataFrame) -> str:
    """結果を日本語のレポートにまとめる。"""
    L = []
    A = L.append

    A("=" * 62)
    A("バックテスト結果")
    A("=" * 62)

    if trades.empty:
        A("抽出銘柄が1件もありませんでした。条件が厳しすぎる可能性があります。")
        A("config.py の MIN_SALES_YOY から緩めてみてください。")
        return "\n".join(L)

    n_periods = len(periods)
    n_indep = max(1, int(n_periods * 21 / HOLD_DAYS))
    A(f"検証した起点            : {n_periods}回")
    A(f"うち重複しない3か月区間 : 約{n_indep}区間  ← ここが実質のサンプル数")
    A(f"延べ抽出銘柄            : {len(trades)}件")
    A(f"1回あたり平均抽出数     : {len(trades)/max(n_periods,1):.1f}銘柄")
    A("")

    bh = trades["buy_hold_return"]
    A("--- 3か月後リターン（単純保有した場合） ---")
    A(f"  中央値   : {bh.median():+.1%}")
    A(f"  平均     : {bh.mean():+.1%}")
    A(f"  最大     : {bh.max():+.1%}")
    A(f"  最小     : {bh.min():+.1%}")
    A(f"  勝率     : {(bh > 0).mean():.1%}")
    A("")

    mx = trades["max_return"]
    A("--- 期間中の最大上昇率（利確判断の材料） ---")
    A(f"  +50%以上に到達  : {(mx >= 0.50).mean():.1%}")
    A(f"  +100%以上に到達 : {(mx >= 1.00).mean():.1%}")
    A(f"  +200%以上（3倍）: {(mx >= 2.00).mean():.1%}  ← 本来の狙い")
    A("")

    rr = trades["rule_return"]
    A("--- Role 7 の執行ルールを適用した場合 ---")
    A("（-25%損切り / +50%・+100%・+200%で各1/4利確 / 60営業日で撤退）")
    A(f"  平均リターン : {rr.mean():+.1%}")
    A(f"  中央値       : {rr.median():+.1%}")
    A(f"  勝率         : {(rr > 0).mean():.1%}")
    counts = trades["exit_reason"].value_counts()
    for reason, c in counts.items():
        A(f"  {reason:8s} : {c}件 ({c/len(trades):.1%})")
    A("")

    if "excess" in periods and periods["excess"].notna().any():
        ex = periods["excess"].dropna()
        A("--- 相場全体との比較（最重要） ---")
        A(f"  超過リターンの中央値 : {ex.median():+.1%}")
        A(f"  相場に勝った区間     : {(ex > 0).sum()}/{len(ex)}")
        A("  ※ここがマイナスなら、単に相場が良かっただけで戦略に価値はありません")
        A("")

    A("--- 起点ごとの結果 ---")
    p = periods.copy()
    for c in ("median", "mean_rule", "win_rate", "market", "excess"):
        if c in p:
            p[c] = (p[c] * 100).round(1)
    A(p.to_string(index=False))
    A("")

    A("=" * 62)
    A("解釈の注意")
    A("=" * 62)
    A("・手数料とスリッページは含んでいません。実際の成績はこれより悪くなります")
    A("・上場廃止銘柄の扱い次第で、結果が実際より良く出ている可能性があります")
    A(f"・独立した相場局面のサンプルは約{n_indep}個しかありません")
    A("  → プラスでも「有効だと証明された」とは言えません")
    A("  → 全区間でマイナスなら「ダメだと判断してよい」とは言えます")
    A("・この検証は『撤退すべきかを早く知る』ためのものです")
    return "\n".join(L)
