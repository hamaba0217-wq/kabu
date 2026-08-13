# -*- coding: utf-8 -*-
"""
Role 2: 一次スクリーニング

前年同期比の考え方
------------------
決算サマリーの売上・営業利益は「累計」で開示されます（1Q, 2Q累計, 3Q累計, 通期）。
そのため単純に前回開示と比べると期間がズレます。

ここでは **同じ期種どうしを1年前と比較** します。
  例) 今回が「2Q累計」なら、前年の「2Q累計」と比べる
これなら期間が揃うので、四半期を切り出す処理をせずに正しいYoYが出せます。
"""

from __future__ import annotations

import pandas as pd

import config


def add_rolling(quotes: pd.DataFrame, ma_window: int) -> pd.DataFrame:
    """移動平均などの後方参照のみの指標を、全期間分まとめて計算する。

    ここで計算する指標はすべて「その日までの情報」しか使いません（先読みなし）。
    一度計算しておけば、任意の時点のスナップショットを高速に切り出せます。
    """
    q = quotes.sort_values(["code", "date"]).copy()
    q["turnover_ma"] = (
        q.groupby("code")["turnover_value"]
         .transform(lambda s: s.rolling(ma_window, min_periods=5).mean())
    )
    q["high_52w"] = (
        q.groupby("code")["close"]
         .transform(lambda s: s.rolling(250, min_periods=20).max())
    )
    return q


SNAPSHOT_COLUMNS = ["code", "date", "close", "volume", "turnover_value",
                    "turnover_ma", "turnover_spike", "high_52w", "pct_from_high"]


def snapshot_at(q_rolled: pd.DataFrame, as_of) -> pd.DataFrame:
    """as_of 時点で「見えていた」最新の株価スナップショットを返す。"""
    q = q_rolled[q_rolled["date"] <= pd.Timestamp(as_of)]
    if q.empty:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    last = q.groupby("code").tail(1).copy()
    last["turnover_spike"] = last["turnover_value"] / last["turnover_ma"]
    last["pct_from_high"] = last["close"] / last["high_52w"] - 1.0
    return last[SNAPSHOT_COLUMNS]


def latest_prices(quotes: pd.DataFrame, ma_window: int) -> pd.DataFrame:
    """銘柄ごとの最新の株価・売買代金・移動平均を作る。"""
    q = add_rolling(quotes, ma_window)
    return snapshot_at(q, q["date"].max())


def yoy_table(fin: pd.DataFrame) -> pd.DataFrame:
    """銘柄ごとに、最新開示とその1年前の同じ期種を突き合わせてYoYを出す。"""
    if fin.empty:
        return pd.DataFrame()

    f = fin.sort_values(["code", "disclosed_date"]).copy()
    records = []

    for code, g in f.groupby("code"):
        cur = g.iloc[-1]
        ptype = cur.get("period_type")

        # 同じ期種で、1年前後（330〜400日前）に開示されたものを探す
        same = g[g["period_type"] == ptype]
        prev = None
        for _, row in same.iloc[:-1][::-1].iterrows():
            gap = (cur["disclosed_date"] - row["disclosed_date"]).days
            if 300 <= gap <= 430:
                prev = row
                break
        if prev is None:
            continue

        def ratio(now, before):
            if pd.isna(now) or pd.isna(before) or before is None:
                return None
            if before > 0:
                return now / before - 1.0
            # 前年が赤字/ゼロ: 黒字転換なら別枠で判定するのでNoneを返す
            return None

        op_now, op_prev = cur.get("operating_profit"), prev.get("operating_profit")
        turnaround = (
            pd.notna(op_now) and pd.notna(op_prev)
            and op_prev is not None and op_now is not None
            and op_prev <= 0 < op_now
        )

        records.append({
            "code": code,
            "period_type": ptype,
            "disclosed_date": cur["disclosed_date"],
            "prev_disclosed_date": prev["disclosed_date"],
            "net_sales": cur.get("net_sales"),
            "operating_profit": op_now,
            "sales_yoy": ratio(cur.get("net_sales"), prev.get("net_sales")),
            "op_yoy": ratio(op_now, op_prev),
            "turnaround": turnaround,
            "shares_outstanding": cur.get("shares_outstanding"),
        })

    return pd.DataFrame(records)


def run_screen(prices: pd.DataFrame, yoy: pd.DataFrame,
               listed: pd.DataFrame, holdings: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """全条件を適用し、通過銘柄と各段階の残数ログを返す。"""
    log = []

    df = prices.merge(yoy, on="code", how="inner").merge(listed, on="code", how="left")
    log.append(f"株価×決算の突合後: {len(df)}銘柄")

    # 時価総額（終値 × 発行済株式数）
    if "shares_outstanding" in df.columns:
        df["market_cap"] = df["close"] * df["shares_outstanding"]
    else:
        df["market_cap"] = pd.NA

    def keep(mask, name):
        nonlocal df
        before = len(df)
        df = df[mask]
        log.append(f"{name}: {before} → {len(df)}")

    keep(df["close"] >= config.MIN_PRICE, f"株価 {config.MIN_PRICE}円以上")

    if getattr(config, "MAX_PRICE", None):
        keep(df["close"] <= config.MAX_PRICE, f"株価 {config.MAX_PRICE}円以下")

    if df["market_cap"].notna().any():
        keep(
            df["market_cap"].between(config.MIN_MARKET_CAP, config.MAX_MARKET_CAP),
            f"時価総額 {config.MIN_MARKET_CAP/1e8:.0f}〜{config.MAX_MARKET_CAP/1e8:.0f}億円",
        )
    else:
        log.append("時価総額: 発行済株式数が取得できずスキップ")

    keep(df["turnover_value"] >= config.MIN_TURNOVER_VALUE,
         f"売買代金 {config.MIN_TURNOVER_VALUE/1e8:.1f}億円以上")

    # 業績条件（主軸）
    op_ok = df["op_yoy"] >= config.MIN_OP_PROFIT_YOY
    if config.ALLOW_TURNAROUND:
        op_ok = op_ok | df["turnaround"].fillna(False)
    keep(op_ok.fillna(False), f"営業利益YoY +{config.MIN_OP_PROFIT_YOY:.0%}以上 or 黒字転換")

    keep((df["sales_yoy"] >= config.MIN_SALES_YOY).fillna(False),
         f"売上YoY +{config.MIN_SALES_YOY:.0%}以上")

    if config.MIN_TURNOVER_SPIKE_RATIO > 0:
        keep((df["turnover_spike"] >= config.MIN_TURNOVER_SPIKE_RATIO).fillna(False),
             f"売買代金が20日平均の{config.MIN_TURNOVER_SPIKE_RATIO}倍以上")

    if config.TARGET_MARKETS and "market" in df.columns:
        pat = "|".join(config.TARGET_MARKETS)
        keep(df["market"].fillna("").str.contains(pat), f"対象市場 {config.TARGET_MARKETS}")

    if config.EXCLUDE_SECTORS and "sector33" in df.columns:
        pat = "|".join(config.EXCLUDE_SECTORS)
        keep(~df["sector33"].fillna("").str.contains(pat), "除外業種を除く")

    # 大量保有報告書フラグ（除外条件ではなく、参考情報）
    if holdings is not None and not holdings.empty:
        recent = set(holdings["code"].astype(str))
        df["large_holding_filed"] = df["code"].str[:4].isin(recent)
    else:
        df["large_holding_filed"] = False

    df = df.sort_values("op_yoy", ascending=False)
    return df, log


OUTPUT_COLUMNS = [
    "code", "company_name", "market", "sector33", "date", "close",
    "market_cap", "turnover_value", "turnover_spike",
    "sales_yoy", "op_yoy", "turnaround", "period_type", "disclosed_date",
    "pct_from_high", "large_holding_filed",
]


def to_output(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    out = df[cols].copy()
    # J-Quantsは5桁コードを返すので、見やすい4桁も併記する
    from sources import code4
    out.insert(0, "code4", code4(out["code"]))
    if "market_cap" in out:
        out["market_cap_oku"] = (out["market_cap"] / 1e8).round(1)
    if "turnover_value" in out:
        out["turnover_oku"] = (out["turnover_value"] / 1e8).round(2)
    for c in ("sales_yoy", "op_yoy", "pct_from_high"):
        if c in out:
            out[c] = (out[c] * 100).round(1)
    return out.drop(columns=[c for c in ("market_cap", "turnover_value") if c in out])
