# -*- coding: utf-8 -*-
"""
テクニカル戦略の一括検証

3つの手法を、同じ枠組み・同じ期間で検証し、横並びで比較します。

  A: 決算モメンタム順張り（PEAD）
     良い決算（会社予想比の上振れ／大幅増益）が出た翌日以降に買い、
     上昇トレンドに乗る。決算という「事実」が起点。

  B: ボックス下限の反発（逆張り）
     一定レンジで推移する銘柄が、下値支持線に接近したら買い、
     上限で売る。損切りは支持線割れで明確。

  C: ブレイクアウト（上値抜け）
     過去N日の高値を更新した瞬間に買う。出来高急増を伴うものに限定。

設計の共通ルール
----------------
・すべて「その日までの情報」だけでエントリー判定（先読みなし）
・エントリー後は各戦略のルールで利確／損切り／期限手仕舞い
・相場全体（同期間の全銘柄中央値）と比較して超過リターンを見る
・勝者だけでなく全トレードを集計（生存者バイアスなし）

重要な前提
----------
手数料・スリッページ・約定滑りは未考慮。実際の成績はこれより悪くなります。
特にブレイクアウトと決算翌日は、寄り付きのギャップで想定より不利な価格で
約定することが多く、この検証は楽観側に出ます。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import backtest
import config
import screen


HORIZON_DEFAULT = 40      # 保有上限（営業日）。トレンドが続けば途中で利確
STEP_DAYS = 10            # 起点をずらす間隔


# ---------------------------------------------------------------------------
# 技術指標（すべて後方参照のみ）
# ---------------------------------------------------------------------------

def add_indicators(quotes: pd.DataFrame) -> pd.DataFrame:
    """戦略判定に必要な指標を全期間分まとめて計算する。"""
    q = quotes.sort_values(["code", "date"]).copy()
    g = q.groupby("code")

    q["ma25"] = g["close"].transform(lambda s: s.rolling(25, min_periods=10).mean())
    q["ma75"] = g["close"].transform(lambda s: s.rolling(75, min_periods=20).mean())
    q["turnover_ma20"] = g["turnover_value"].transform(
        lambda s: s.rolling(20, min_periods=5).mean())
    q["turnover_spike"] = q["turnover_value"] / q["turnover_ma20"]

    # 直近の高値・安値（当日を除く過去N日）
    q["high_60"] = g["close"].transform(
        lambda s: s.shift(1).rolling(60, min_periods=20).max())
    q["low_60"] = g["close"].transform(
        lambda s: s.shift(1).rolling(60, min_periods=20).min())
    q["high_20"] = g["close"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).max())

    # レンジ幅（ボックス判定用）: 過去60日の高値と安値の乖離
    q["range_width"] = (q["high_60"] - q["low_60"]) / q["low_60"]

    # 前日終値（ギャップ判定用）
    q["prev_close"] = g["close"].shift(1)
    return q


# ---------------------------------------------------------------------------
# 各戦略のエントリー判定（as_of 時点のスナップショットに対して）
# ---------------------------------------------------------------------------

def _snapshot(qi: pd.DataFrame, as_of) -> pd.DataFrame:
    """as_of 時点で見えていた最新の1行を銘柄ごとに返す。"""
    s = qi[qi["date"] <= pd.Timestamp(as_of)]
    if s.empty:
        return s
    return s.groupby("code").tail(1)


def entries_breakout(snap: pd.DataFrame) -> list[str]:
    """C: 過去60日高値を上抜け かつ 出来高急増。"""
    cond = (
        (snap["close"] > snap["high_60"]) &
        (snap["turnover_spike"] >= 2.0) &
        (snap["close"] > snap["ma25"])          # 上昇基調の確認
    )
    return snap[cond.fillna(False)]["code"].tolist()


def entries_box_bottom(snap: pd.DataFrame) -> list[str]:
    """B: レンジ相場で、下値支持線の近くまで下げた銘柄。"""
    # レンジ幅が過大でない（=ボックス相場）＆ 下限に近い
    pos_in_range = (snap["close"] - snap["low_60"]) / (snap["high_60"] - snap["low_60"])
    cond = (
        (snap["range_width"] <= 0.40) &          # 高値と安値の差が40%以内=レンジ
        (snap["range_width"] >= 0.10) &          # 狭すぎるものは除外
        (pos_in_range <= 0.25) &                 # レンジ下から25%以内
        (snap["close"] > snap["low_60"])         # まだ支持線は割っていない
    )
    return snap[cond.fillna(False)]["code"].tolist()


def entries_earnings_momentum(snap: pd.DataFrame, fin_recent: pd.DataFrame,
                              as_of, lookback_days=5) -> list[str]:
    """A: 直近に good な決算を出した銘柄。

    「good」の判定（市場予想データが無いので近似）:
      - 直近 lookback_days 営業日以内に決算開示があった
      - かつ 営業利益が会社予想を上回る or 前年比で大幅増益
      - かつ 決算後に株価が上昇で反応している（ma25 の上）
    """
    recent = fin_recent[
        (fin_recent["disclosed_date"] <= pd.Timestamp(as_of)) &
        (fin_recent["disclosed_date"] >= pd.Timestamp(as_of) - pd.Timedelta(days=lookback_days * 2))
    ]
    if recent.empty:
        return []

    good_codes = []
    for code, g in recent.groupby("code"):
        latest = g.sort_values("disclosed_date").iloc[-1]
        op = latest.get("operating_profit")
        fc = latest.get("fc_operating_profit")
        op_prev = latest.get("op_prev")

        is_good = False
        # 会社予想比で上振れ
        if pd.notna(op) and pd.notna(fc) and fc and fc > 0:
            if op >= fc * 1.05:
                is_good = True
        # 前年比で大幅増益（30%以上）
        if pd.notna(op) and pd.notna(op_prev) and op_prev and op_prev > 0:
            if op / op_prev - 1 >= 0.30:
                is_good = True
        if is_good:
            good_codes.append(code)

    if not good_codes:
        return []

    # 決算後、株価が上昇反応している（ma25 の上にある）ものだけ
    sub = snap[snap["code"].isin(good_codes)]
    cond = (sub["close"] > sub["ma25"]).fillna(False)
    return sub[cond]["code"].tolist()


# ---------------------------------------------------------------------------
# 各戦略の手仕舞いルール
# ---------------------------------------------------------------------------

def _exit_trend(path, ma_path, stop=-0.10):
    """トレンド系（A・C）: 損切り、または上昇の勢いが切れたら（ma25割れ）手仕舞い。"""
    for i, px in enumerate(path):
        ret = px - 1.0
        if ret <= stop:
            return stop, "損切り"
        # ma25 を割ったらトレンド終了とみなす（初日は除く）
        if i >= 1 and ma_path is not None and i < len(ma_path):
            if not np.isnan(ma_path[i]) and px < ma_path[i]:
                return ret, "トレンド終了"
    return (path[-1] - 1.0 if len(path) else 0.0), "期限切れ"


def _exit_box(path, target=0.15, stop=-0.07):
    """ボックス反発（B）: 上限方向へ target 戻したら利確、支持線割れ（stop）で撤退。"""
    for px in path:
        ret = px - 1.0
        if ret <= stop:
            return stop, "支持線割れ"
        if ret >= target:
            return target, "反発利確"
    return (path[-1] - 1.0 if len(path) else 0.0), "期限切れ"


# ---------------------------------------------------------------------------
# バックテスト本体
# ---------------------------------------------------------------------------

def _forward(by_code, code, as_of, horizon):
    """as_of以降 horizon 営業日の (価格pathとma25 path) を entry 基準で返す。"""
    arrs = by_code.get(code)
    if arrs is None:
        return None, None
    close_arr, dates_arr, ma_arr = arrs
    pos = np.searchsorted(dates_arr, np.datetime64(pd.Timestamp(as_of)), side="right")
    if pos == 0:
        return None, None
    entry = close_arr[pos - 1]
    if not entry or entry <= 0:
        return None, None
    fut = close_arr[pos:pos + horizon]
    if len(fut) < horizon * 0.5:
        return None, None
    ma_fut = ma_arr[pos:pos + horizon] / entry
    return fut / entry, ma_fut


def _build_index(qi):
    """銘柄ごとに (close, date, ma25) 配列を作る。"""
    idx = {}
    for code, g in qi.sort_values("date").groupby("code"):
        idx[code] = (g["close"].to_numpy(), g["date"].to_numpy(),
                     g["ma25"].to_numpy())
    return idx


STRATEGIES = {
    "A_決算モメンタム": {"entry": "earnings", "exit": "trend", "horizon": 40},
    "B_ボックス反発":   {"entry": "box",      "exit": "box",   "horizon": 30},
    "C_ブレイクアウト": {"entry": "breakout", "exit": "trend", "horizon": 40},
}


def run(quotes, fin, listed, step_days=STEP_DAYS):
    qi = add_indicators(quotes)
    by_code = _build_index(qi)
    all_dates = np.sort(qi["date"].unique())

    # 決算に前年同期比・会社予想を付ける（Aで使う）
    fin2 = _prep_fin(fin)

    results = []
    for name, spec in STRATEGIES.items():
        horizon = spec["horizon"]
        print(f"\n{'='*56}\n▶ {name}\n{'='*56}")

        start_idx = 80
        end_idx = len(all_dates) - horizon
        if end_idx <= start_idx:
            print("  期間不足"); continue
        as_of_list = all_dates[start_idx:end_idx:step_days]

        rets, reasons, excess = [], [], []
        for as_of in as_of_list:
            snap = _snapshot(qi, as_of)
            if snap.empty:
                continue

            if spec["entry"] == "breakout":
                codes = entries_breakout(snap)
            elif spec["entry"] == "box":
                codes = entries_box_bottom(snap)
            else:
                codes = entries_earnings_momentum(snap, fin2, as_of)

            if not codes:
                continue

            period = []
            for code in codes:
                path, ma_path = _forward(by_code, code, as_of, horizon)
                if path is None:
                    continue
                if spec["exit"] == "box":
                    r, reason = _exit_box(path)
                else:
                    r, reason = _exit_trend(path, ma_path)
                rets.append(r); reasons.append(reason); period.append(r)

            base = backtest.market_baseline(
                {c: (v[0], v[1]) for c, v in by_code.items()}, as_of, horizon)
            if period and pd.notna(base):
                excess.append(np.median(period) - base)

        if not rets:
            results.append({"戦略": name, "取引数": 0})
            continue

        arr = np.array(rets); ex = np.array(excess)
        rc = pd.Series(reasons).value_counts()
        results.append({
            "戦略": name,
            "取引数": len(arr),
            "平均リターン%": round(arr.mean() * 100, 1),
            "中央値%": round(np.median(arr) * 100, 1),
            "勝率%": round((arr > 0).mean() * 100, 1),
            "対相場 超過%": round(np.median(ex) * 100, 1) if len(ex) else None,
            "相場に勝った回": f"{int((ex > 0).sum())}/{len(ex)}" if len(ex) else "—",
            "主な終了理由": ", ".join(f"{k}{v}" for k, v in rc.head(2).items()),
        })

    return pd.DataFrame(results)


def _prep_fin(fin):
    """決算に前年同期比・会社予想を付ける。"""
    import fundamentals
    f = fin.copy()
    # 会社予想（Aの good 判定に使う）
    if "fc_operating_profit" not in f.columns:
        f["fc_operating_profit"] = np.nan
    # 前年同期の営業利益（同じ period_type で1年前）
    f = f.sort_values(["code", "disclosed_date"])
    f["op_prev"] = f.groupby(["code", "period_type"])["operating_profit"].shift(1)
    return f


def summarize(df: pd.DataFrame) -> str:
    L = ["=" * 78,
         "テクニカル3戦略の比較（決算モメンタム / ボックス反発 / ブレイクアウト）",
         "=" * 78, ""]
    L.append(df.to_string(index=False))
    L += ["", "=" * 78, "読み方", "=" * 78,
          "・「対相場 超過%」が最重要。プラスなら市場平均に勝っている。",
          "・取引数が極端に少ない戦略は、条件が厳しすぎて偶然の可能性。",
          "・手数料・スリッページは未考慮。特にブレイクアウトと決算翌日は",
          "  寄り付きギャップで不利約定しやすく、実際はこれより悪くなる。",
          "・どれかが明確にプラスなら、そこを深掘りする価値がある。",
          "・全部マイナスでも『テクニカル単独では勝てない』という結論になる。"]
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

    # 会社予想（fc_operating_profit 等）は financials() のキャッシュに
    # 含まれるようになったため、raw_fin の再取得は不要。

    df = run(quotes, fin, listed)
    report = summarize(df)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    path = os.path.join(config.OUTPUT_DIR, f"{today}_technical.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n比較表: {path}")


if __name__ == "__main__":
    main()
