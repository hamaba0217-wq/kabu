# -*- coding: utf-8 -*-
"""
rule_lab.py — 「1か月以内に +10% 到達」の勝率を上げる売買ルールの自動探索

勝ちの定義（利用者指定）
  ・シグナル日 t の終値で判定 → 翌営業日 t+1 の始値で買う
  ・t+1〜t+20 営業日（約1か月）のどこかで高値が 買値×1.10 以上 → 勝ち（+10%で利確）
  ・届かなければ t+20 の終値で売る（時間切れ）
  ・同じ銘柄は保有期間中に重ねて買わない（重複シグナルは除外）

採用基準（過剰最適化を避ける）
  ・取引数 150 以上、期間を3分割して各期間 30 以上
  ・全体の勝率が現チャンピオンより +1pt 以上高い
  ・3期間のうち最も悪い期間の勝率も、現チャンピオンの最悪期間より高い
  ・3期間すべてで平均損益（+10%利確／時間切れ売り）がプラス
  手数料・スリッページ・税は未考慮。投資助言ではありません。

使い方
  python rule_lab.py            # データ差分更新 → 未検証ルールを一定数テスト → 採用なら HTML/MD 更新
  python rule_lab.py --no-fetch # データ更新なし
  python rule_lab.py --push     # 採用があれば git commit & push
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

STATE = os.path.join("rules", "state.json")
RULES_MD = os.path.join("rules", "RULES.md")
RULES_HTML = os.path.join("docs", "rules.html")

HOLD = 20          # 約1か月（営業日）
TP = 0.10          # 勝ち = +10%
MIN_N = 150
MIN_N_SPLIT = 30
BATCH = int(os.environ.get("RULE_LAB_BATCH", "400"))

# --------------------------------------------------------------------------
# データ
# --------------------------------------------------------------------------

def _adjust(q: pd.DataFrame) -> pd.DataFrame:
    """全期間を通して分割・併合を調整（後の調整係数の累積積を過去の価格に掛ける）"""
    if "adj_factor" not in q.columns or q["adj_factor"].isna().all():
        return q
    f = q["adj_factor"].fillna(1.0).replace(0, 1.0)
    # 各行より「後」の係数の積
    logf = np.log(f)
    rev = logf[::-1].groupby(q["code"][::-1]).cumsum()[::-1]
    cum = np.exp(rev - logf)
    for col in ("open", "high", "low", "close"):
        q[col] = q[col] * cum
    q["volume"] = q["volume"] / cum
    return q


def load_data(fetch: bool) -> pd.DataFrame:
    import cache
    if fetch:
        subprocess.run([sys.executable, "-u", "fetch_data.py"])
    status = open(os.path.join("data", "fetch_status.txt")).read().strip()         if os.path.exists(os.path.join("data", "fetch_status.txt")) else "none"
    q = cache.load("raw_quotes")
    if q is None or status != "ready":
        print(f"RESULT skipped=データ取得途中（{status}）。次回の実行で続きを取得します")
        raise SystemExit(0)
    cols = [c for c in ["code", "date", "open", "high", "low", "close", "volume", "turnover_value", "adj_factor"]
            if c in q.columns]
    q = q[cols].copy()
    q["date"] = pd.to_datetime(q["date"])
    q = q.dropna(subset=["open", "high", "low", "close"])
    q = q[(q["close"] > 0) & (q["open"] > 0)]
    q = q.sort_values(["code", "date"]).reset_index(drop=True)
    q = _adjust(q)
    # 業種
    lst = cache.load("listed")
    if lst is not None:
        sc = next((c for c in ["sector33", "sector17", "Sector17CodeName", "S17Nm", "sector"] if c in lst.columns), None)
        cc = next((c for c in ["code", "Code"] if c in lst.columns), None)
        if sc and cc:
            m = lst[[cc, sc]].drop_duplicates(cc)
            m.columns = ["code", "sector"]
            m["code"] = m["code"].astype(str)
            q = q.merge(m, on="code", how="left")
    if "sector" not in q.columns:
        q["sector"] = "不明"
    q["sector"] = q["sector"].fillna("不明")
    # 仮説(2026-11-21 18回目)：市場区分（1=プライム・2=スタンダード・3=グロース。現在の区分を使う）
    q["mkt_seg"] = np.nan
    if lst is not None and "market" in lst.columns and "code" in lst.columns:
        ms = lst[["code", "market"]].drop_duplicates("code").copy()
        ms["code"] = ms["code"].astype(str)
        ms["mkt_seg"] = ms["market"].map({"プライム": 1.0, "スタンダード": 2.0, "グロース": 3.0})
        q = q.drop(columns=["mkt_seg"]).merge(ms[["code", "mkt_seg"]], on="code", how="left")
    return q


def _etf(q, codes):
    """J-Quants 株価に含まれるETFの終値・売買代金（日付→値）"""
    s = q[q["code"].isin(codes)]
    if s.empty:
        return None
    code = s["code"].value_counts().index[0]
    s = s[s["code"] == code].set_index("date")
    tv = s["turnover_value"] if "turnover_value" in s.columns else s["close"] * s["volume"]
    return pd.DataFrame({"close": s["close"], "tv": tv})


def add_context(q: pd.DataFrame) -> pd.DataFrame:
    """相場全体・投資家心理・外部要因（金/国債/金利/為替/米株/原油/VIX）・季節性・決算タイミング"""
    dates = pd.DatetimeIndex(sorted(q["date"].unique()))
    ctx = pd.DataFrame(index=dates)
    # --- 相場の内部（その日の引けで確定する値） ---
    up = (q["ret1"] > 0).groupby(q["date"]).sum()
    dn = (q["ret1"] < 0).groupby(q["date"]).sum()
    ctx["breadth"] = up / (up + dn)                                   # 値上がり銘柄の割合
    ctx["adr25"] = up.rolling(25).sum() / dn.rolling(25).sum() * 100  # 騰落レシオ(25日)
    # --- ETFから見る他の資産・個人投資家の心理 ---
    for name, codes in {"gold": ["1540", "15400", "1328", "13280"],
                        "bond": ["2510", "25100"],
                        "nk": ["1321", "13210"],
                        "reit": ["1343", "13430"]}.items():
        e = _etf(q, codes)
        if e is not None:
            ctx[f"{name}_ret20"] = e["close"].reindex(dates).pct_change(20)
    lev, inv = _etf(q, ["1570", "15700"]), _etf(q, ["1357", "13570"])
    if lev is not None and inv is not None:
        r = (lev["tv"] / inv["tv"]).reindex(dates)
        lr = np.log(r.replace(0, np.nan))
        ctx["lev_z"] = (lr - lr.rolling(60).mean()) / lr.rolling(60).std()  # 強気(レバ)/弱気(インバ)売買代金比の偏り
    # --- 外部指標（1営業日ずらし＝判定時点で確実に分かっている値だけ使う） ---
    try:
        import macro
        m = macro.load()
        if m is None or os.environ.get("RULE_LAB_FETCH_MACRO", "1") == "1":
            m = macro.fetch()
    except Exception as e:
        print(f"[警告] 外部指標なし: {e}")
        m = None
    if m is not None and len(m):
        m = m.sort_index().ffill()
        x = pd.DataFrame(index=m.index)
        x["vix"] = m["vix"]
        x["vix_chg5"] = m["vix"] - m["vix"].shift(5)
        x["us10y_chg20"] = m["us10y"] - m["us10y"].shift(20)
        x["jgb10y_chg20"] = m["jgb10y"] - m["jgb10y"].shift(20)
        x["jgb_slope"] = m["jgb10y"] - m["jgb2y"]
        x["usdjpy_ret20"] = m["usdjpy"].pct_change(20)
        x["wti_ret20"] = m["wti"].pct_change(20)
        x["sp500_ret1"] = m["sp500"].pct_change(1)
        x["sp500_ret20"] = m["sp500"].pct_change(20)
        x["nasdaq_ret5"] = m["nasdaq"].pct_change(5)
        x["us_dd60"] = m["sp500"] / m["sp500"].rolling(60, min_periods=40).max() - 1   # 仮説(14回目)：米国株の60日高値からの下落
        x["usdjpy_ret5"] = m["usdjpy"].pct_change(5)                    # 仮説(17回目)：為替の短期の急変（円高・円安ショック）
        x["usdjpy_ret1"] = m["usdjpy"].pct_change(1)                    # 為替感応度の計算用（GRIDには入れない）
        x["vix_rel60"] = m["vix"] / m["vix"].rolling(60, min_periods=40).mean()   # 仮説(18回目)：VIXの60日平均に対する水準
        x["wti_ret5"] = m["wti"].pct_change(5)                             # 仮説(20回目2周目)：原油の短期の急変
        x["sp500_ret5"] = m["sp500"].pct_change(5)                         # 日本株の相対の弱さの計算用（GRIDには入れない）
        x["wti_ret1"] = m["wti"].pct_change(1)                             # 原油感応度の計算用（GRIDには入れない）
        x["vix_chg1"] = m["vix"].diff()                                    # 仮説(20回目)：VIXの1日変化（急騰後の反落＝不安のピークアウト）
        # 仮説(22回目2周目)：米10年金利の短期の急変・VIXの20日最大に対する比（市場全体の不安のピークアウト）
        x["us10y_chg5"] = m["us10y"] - m["us10y"].shift(5)
        x["vix_pk20"] = m["vix"] / m["vix"].rolling(20, min_periods=15).max()
        x["spread_chg20"] = (m["us10y"] - m["jgb10y"]) - (m["us10y"] - m["jgb10y"]).shift(20)   # 仮説(19回目)：日米10年金利差の20日変化
        # 仮説(2026-11-22 29回目)：ドル円の20日ボラ（為替の不安定さ）・日本2年金利の20日変化（日銀の利上げ観測）
        x["fx_vol20"] = m["usdjpy"].pct_change().rolling(20, min_periods=15).std()
        x["jgb2y_chg20"] = m["jgb2y"] - m["jgb2y"].shift(20)
        # 日付 t には「t より前の日付」の値を使う
        x.index = x.index + pd.Timedelta(days=1)
        x = x[~x.index.duplicated(keep="last")]
        al = x.reindex(x.index.union(dates)).ffill().reindex(dates)
        ctx = ctx.join(al)
    ctx["month"] = ctx.index.month
    ctx["dow"] = ctx.index.dayofweek
    ctx["dom"] = ctx.index.day                                         # 仮説(2026-11-21 2回目)：月末・月初効果
    # 仮説(2026-11-21 9回目)：連休前＝次の営業日までの暦日数（取引所の休日は事前に公表されているので先読みではない）
    ds = pd.Series(dates, index=dates)
    gapd = (ds.shift(-1) - ds).dt.days
    ctx["hol_next"] = ((gapd >= 4) | ((ctx.index.dayofweek < 4) & (gapd >= 2))).astype(float).where(gapd.notna())
    ctx.index.name = "date"
    q = q.merge(ctx.reset_index(), on="date", how="left")
    # --- 業種内での相対的な強さ ---
    sec = q.groupby(["date", "sector"])["ret20"].transform("median")
    q["rel_sec20"] = q["ret20"] - sec
    # --- 決算発表からの日数（発表日当日を含む過去の発表のみ） ---
    try:
        import cache
        f = cache.load("fin")
        if f is not None and "disclosed_date" in f.columns:
            f = f[["code", "disclosed_date"]].dropna().copy()
            f["code"] = f["code"].astype(str)
            f["disclosed_date"] = pd.to_datetime(f["disclosed_date"])
            f = f.drop_duplicates().sort_values("disclosed_date")
            q = q.sort_values("date")
            q = pd.merge_asof(q, f.rename(columns={"disclosed_date": "last_disc"}),
                              left_on="date", right_on="last_disc", by="code", direction="backward")
            q["since_disc"] = (q["date"] - q["last_disc"]).dt.days
            # 仮説(2026-11-21 14回目)：決算発表後の株価反応＝発表日前日の終値から t 終値までの騰落（発表から20日以内のみ）
            q = q.sort_values(["code", "date"]).reset_index(drop=True)
            pc = q.groupby("code", sort=False)["close"].shift(1)
            q["_pc_disc"] = pc.where(q["date"] == q["last_disc"])
            q["_pc_disc"] = q.groupby(["code", "last_disc"])["_pc_disc"].transform("first")
            q["disc_ret"] = (q["close"] / q["_pc_disc"] - 1).where(q["since_disc"] <= 20)
            q = q.drop(columns=["_pc_disc"])
            q = q.drop(columns=["last_disc"])
    except Exception as e:
        print(f"[警告] 決算日データなし: {e}")
    # 仮説(2026-11-21 7回目)：PBR・会社予想（営業利益）の修正率（発表日当日までに開示された値のみ）
    try:
        import cache
        f = cache.load("fin")
        if f is not None and {"bps", "fc_operating_profit", "period_type"} <= set(f.columns):
            f = f[["code", "disclosed_date", "period_type", "bps", "fc_operating_profit"]].copy()
            f["code"] = f["code"].astype(str)
            f["disclosed_date"] = pd.to_datetime(f["disclosed_date"])
            f = f.sort_values(["code", "disclosed_date"])
            b = f.dropna(subset=["bps"])
            b = b[b["bps"] > 0][["code", "disclosed_date", "bps"]].drop_duplicates(["code", "disclosed_date"], keep="last")
            fc = f.dropna(subset=["fc_operating_profit"]).drop_duplicates(["code", "disclosed_date"], keep="last").copy()
            prev = fc.groupby("code")["fc_operating_profit"].shift(1)
            pgap = (fc["disclosed_date"] - fc.groupby("code")["disclosed_date"].shift(1)).dt.days
            ok = (fc["period_type"] != "FY") & (pgap <= 200) & (prev.abs() > 0)   # FYの発表は翌期の予想に切り替わるので除外
            fc["fc_rev"] = np.where(ok, (fc["fc_operating_profit"] - prev) / prev.abs(), np.nan)
            fc = fc.dropna(subset=["fc_rev"])[["code", "disclosed_date", "fc_rev"]]
            q = q.sort_values("date")
            q = pd.merge_asof(q, b.sort_values("disclosed_date").rename(columns={"disclosed_date": "d_bps"}),
                              left_on="date", right_on="d_bps", by="code", direction="backward")
            q["pbr"] = q["close"] / q["bps"]
            q = pd.merge_asof(q, fc.sort_values("disclosed_date").rename(columns={"disclosed_date": "d_fc"}),
                              left_on="date", right_on="d_fc", by="code", direction="backward")
            q.loc[(q["date"] - q["d_fc"]).dt.days > 100, "fc_rev"] = np.nan   # 古い修正は使わない
            q = q.drop(columns=["d_bps", "bps", "d_fc"])
            # 仮説(2026-11-21 8回目)：会社予想に対する営業利益の進捗の上振れ・時価総額
            g0 = cache.load("fin")
            g0_ns = g0["fc_net_sales"].values if "fc_net_sales" in g0.columns else None
            g0 = g0[["code", "disclosed_date", "period_type", "operating_profit", "fc_operating_profit", "shares_outstanding"]
                    + (["net_sales"] if "net_sales" in g0.columns else [])].copy()
            g0["code"] = g0["code"].astype(str)
            g0["disclosed_date"] = pd.to_datetime(g0["disclosed_date"])
            exp = g0["period_type"].map({"1Q": 0.25, "2Q": 0.5, "3Q": 0.75})
            g0["prog_ex"] = np.where(g0["fc_operating_profit"] > 0,
                                     g0["operating_profit"] / g0["fc_operating_profit"] - exp, np.nan)
            pe = g0.dropna(subset=["prog_ex"])[["code", "disclosed_date", "prog_ex"]] \
                .drop_duplicates(["code", "disclosed_date"], keep="last").sort_values("disclosed_date")
            q = pd.merge_asof(q.sort_values("date"), pe.rename(columns={"disclosed_date": "d_pe"}),
                              left_on="date", right_on="d_pe", by="code", direction="backward")
            q.loc[(q["date"] - q["d_pe"]).dt.days > 100, "prog_ex"] = np.nan
            so = g0[g0["shares_outstanding"] > 0][["code", "disclosed_date", "shares_outstanding"]] \
                .drop_duplicates(["code", "disclosed_date"], keep="last").sort_values("disclosed_date")
            # 仮説(2026-11-22 35回目)：発行済株式数の前回の決算からの変化（自己株式の消却＝減少、増資・転換＝増加）。分割(±50%超)は除外
            so = so.sort_values(["code", "disclosed_date"])
            so["sh_chg"] = so.groupby("code")["shares_outstanding"].pct_change()
            so.loc[so["sh_chg"].abs() > 0.5, "sh_chg"] = np.nan
            so = so.sort_values("disclosed_date")
            q = pd.merge_asof(q.sort_values("date"), so.rename(columns={"disclosed_date": "d_so"}),
                              left_on="date", right_on="d_so", by="code", direction="backward")
            q["mcap"] = q["close"] * q["shares_outstanding"]
            q["turn20"] = q["tv20"] / q["mcap"].replace(0, np.nan)   # 仮説(14回目)：売買の回転率（20日平均売買代金/時価総額）
            q = q.drop(columns=["d_pe", "d_so", "shares_outstanding"])
            # 仮説(2026-11-21 12回目)：予想営業利益の利回り（会社予想の営業利益/時価総額）・予想営業利益率（営業利益/売上）
            fz = g0[["code", "disclosed_date", "fc_operating_profit"]].copy()
            fz["fc_ns"] = g0_ns if g0_ns is not None else np.nan
            fz = fz.dropna(subset=["fc_operating_profit"]).drop_duplicates(["code", "disclosed_date"], keep="last")                 .sort_values("disclosed_date").rename(columns={"disclosed_date": "d_fz", "fc_operating_profit": "fc_op_now"})
            q = pd.merge_asof(q.sort_values("date"), fz, left_on="date", right_on="d_fz", by="code", direction="backward")
            old = (q["date"] - q["d_fz"]).dt.days > 200
            q["op_yield"] = (q["fc_op_now"] / q["mcap"].replace(0, np.nan)).where(~old)
            q["fc_margin"] = (q["fc_op_now"] / q["fc_ns"].where(q["fc_ns"] > 0)).where(~old)
            q = q.drop(columns=["d_fz", "fc_op_now", "fc_ns"])
            # 仮説(2026-11-21 15回目)：今期見通しの伸び＝直近の会社予想営業利益/直近の本決算(FY)の営業利益実績 − 1
            fya = g0[(g0["period_type"] == "FY") & (g0["operating_profit"] > 0)][["code", "disclosed_date", "operating_profit"]]                 .drop_duplicates(["code", "disclosed_date"], keep="last").sort_values("disclosed_date")                 .rename(columns={"disclosed_date": "d_fya", "operating_profit": "fy_op"})
            fcz = g0.dropna(subset=["fc_operating_profit"])[["code", "disclosed_date", "fc_operating_profit"]]                 .drop_duplicates(["code", "disclosed_date"], keep="last").sort_values("disclosed_date")                 .rename(columns={"disclosed_date": "d_fcz", "fc_operating_profit": "fc_op_g"})
            q = pd.merge_asof(q.sort_values("date"), fya, left_on="date", right_on="d_fya", by="code", direction="backward")
            q = pd.merge_asof(q.sort_values("date"), fcz, left_on="date", right_on="d_fcz", by="code", direction="backward")
            okg = ((q["date"] - q["d_fya"]).dt.days <= 300) & ((q["date"] - q["d_fcz"]).dt.days <= 200) & (q["d_fcz"] >= q["d_fya"] - pd.Timedelta(days=45))
            q["fc_growth"] = (q["fc_op_g"] / q["fy_op"] - 1).where(okg)
            q = q.drop(columns=["d_fya", "fy_op", "d_fcz", "fc_op_g"])
            # 仮説(2026-11-21 13回目)：売上の進捗率の上振れ＝累計売上/会社予想の売上 − 標準進捗（利益より振れが小さい）
            gs = cache.load("fin")
            if {"net_sales", "fc_net_sales"} <= set(gs.columns):
                gs = gs[["code", "disclosed_date", "period_type", "net_sales", "fc_net_sales"]].copy()
                gs["code"] = gs["code"].astype(str)
                gs["disclosed_date"] = pd.to_datetime(gs["disclosed_date"])
                exs = gs["period_type"].map({"1Q": 0.25, "2Q": 0.5, "3Q": 0.75})
                gs["sales_prog"] = np.where(gs["fc_net_sales"] > 0, gs["net_sales"] / gs["fc_net_sales"] - exs, np.nan)
                sp = gs.dropna(subset=["sales_prog"])[["code", "disclosed_date", "sales_prog"]]                     .drop_duplicates(["code", "disclosed_date"], keep="last").sort_values("disclosed_date")
                q = pd.merge_asof(q.sort_values("date"), sp.rename(columns={"disclosed_date": "d_sp"}),
                                  left_on="date", right_on="d_sp", by="code", direction="backward")
                q.loc[(q["date"] - q["d_sp"]).dt.days > 100, "sales_prog"] = np.nan
                q = q.drop(columns=["d_sp"])
            # 仮説(2026-11-21 19回目2周目)：営業利益の前年同期比（同じ四半期区分の約1年前の開示と比較、発表日当日までの値のみ）
            gy = g0[["code", "disclosed_date", "period_type", "operating_profit"]].dropna(subset=["operating_profit"])                 .drop_duplicates(["code", "disclosed_date"], keep="last").sort_values("disclosed_date").copy()
            gy["d_prev"] = gy["disclosed_date"] - pd.Timedelta(days=365)
            py_ = gy[["code", "period_type", "disclosed_date", "operating_profit"]].rename(
                columns={"disclosed_date": "d_py", "operating_profit": "op_py"}).sort_values("d_py")
            gy = pd.merge_asof(gy.sort_values("d_prev"), py_, left_on="d_prev", right_on="d_py",
                               by=["code", "period_type"], direction="nearest", tolerance=pd.Timedelta(days=60))
            gy["op_yoy"] = np.where(gy["op_py"] > 0, gy["operating_profit"] / gy["op_py"] - 1, np.nan)
            gy = gy.dropna(subset=["op_yoy"])[["code", "disclosed_date", "op_yoy"]].sort_values("disclosed_date")
            q = pd.merge_asof(q.sort_values("date"), gy.rename(columns={"disclosed_date": "d_yoy"}),
                              left_on="date", right_on="d_yoy", by="code", direction="backward")
            q.loc[(q["date"] - q["d_yoy"]).dt.days > 100, "op_yoy"] = np.nan
            q = q.drop(columns=["d_yoy"])
            # 仮説(2026-11-21 22回目3周目)：売上高の前年同期比（営業利益と同じ方法。発表日当日までの値のみ）
            if "net_sales" not in g0.columns:
                g0["net_sales"] = np.nan
            gs2 = g0[["code", "disclosed_date", "period_type", "net_sales"]].dropna(subset=["net_sales"])                 .drop_duplicates(["code", "disclosed_date"], keep="last").sort_values("disclosed_date").copy()
            gs2["d_prev"] = gs2["disclosed_date"] - pd.Timedelta(days=365)
            ps_ = gs2[["code", "period_type", "disclosed_date", "net_sales"]].rename(
                columns={"disclosed_date": "d_py", "net_sales": "ns_py"}).sort_values("d_py")
            gs2 = pd.merge_asof(gs2.sort_values("d_prev"), ps_, left_on="d_prev", right_on="d_py",
                                by=["code", "period_type"], direction="nearest", tolerance=pd.Timedelta(days=60))
            gs2["sales_yoy"] = np.where(gs2["ns_py"] > 0, gs2["net_sales"] / gs2["ns_py"] - 1, np.nan)
            gs2 = gs2.dropna(subset=["sales_yoy"])[["code", "disclosed_date", "sales_yoy"]].sort_values("disclosed_date")
            q = pd.merge_asof(q.sort_values("date"), gs2.rename(columns={"disclosed_date": "d_syoy"}),
                              left_on="date", right_on="d_syoy", by="code", direction="backward")
            q.loc[(q["date"] - q["d_syoy"]).dt.days > 100, "sales_yoy"] = np.nan
            q = q.drop(columns=["d_syoy"])
            # 仮説(2026-11-22 29回目)：営業利益率の前年同期比の変化（%pt。同じ四半期区分の約1年前の開示と比較、発表日当日までの値のみ）
            gm = g0[["code", "disclosed_date", "period_type", "operating_profit", "net_sales"]].dropna(subset=["operating_profit", "net_sales"])                 .drop_duplicates(["code", "disclosed_date"], keep="last").sort_values("disclosed_date").copy()
            gm = gm[gm["net_sales"] > 0]
            gm["opm"] = gm["operating_profit"] / gm["net_sales"]
            gm["d_prev"] = gm["disclosed_date"] - pd.Timedelta(days=365)
            pm_ = gm[["code", "period_type", "disclosed_date", "opm"]].rename(
                columns={"disclosed_date": "d_pm", "opm": "opm_py"}).sort_values("d_pm")
            gm = pd.merge_asof(gm.sort_values("d_prev"), pm_, left_on="d_prev", right_on="d_pm",
                               by=["code", "period_type"], direction="nearest", tolerance=pd.Timedelta(days=60))
            gm["opm_chg"] = gm["opm"] - gm["opm_py"]
            gm = gm.dropna(subset=["opm_chg"])[["code", "disclosed_date", "opm_chg"]].sort_values("disclosed_date")
            q = pd.merge_asof(q.sort_values("date"), gm.rename(columns={"disclosed_date": "d_opm"}),
                              left_on="date", right_on="d_opm", by="code", direction="backward")
            q.loc[(q["date"] - q["d_opm"]).dt.days > 100, "opm_chg"] = np.nan
            q = q.drop(columns=["d_opm"])
    except Exception as e:
        print(f"[警告] PBR・予想修正なし: {e}")
    # 仮説(2026-11-21 7回目)：業種ローテーション＝業種全体の20日騰落（中央値）
    q["sec20"] = q.groupby(["date", "sector"])["ret20"].transform("median")
    # 仮説(2026-11-21 8回目)：業種全体の5日騰落（売られた業種の短期の戻り）
    q["sec5"] = q.groupby(["date", "sector"])["ret5"].transform("median")
    # 仮説(2026-11-21 10回目)：業種内の20日騰落のばらつき・業種の勢いの変化（20日前の業種騰落との差）
    q["sec_disp20"] = q.groupby(["date", "sector"])["ret20"].transform("std")
    sd = q.groupby(["sector", "date"])["sec20"].first().groupby(level=0).shift(20).rename("sec20_prev").reset_index()
    q = q.merge(sd, on=["sector", "date"], how="left")
    q["sec_turn"] = q["sec20"] - q["sec20_prev"]
    # 仮説(2026-11-21 20回目)：業種ぐるみの急落＝同じ業種で20日騰落が-15%以下の銘柄の割合
    # 仮説(2026-11-21 20回目2周目)：業種全体のその日の騰落（中央値）＝業種の反発初日・続落
    q["sec_ret1"] = q.groupby(["date", "sector"])["ret1"].transform("median")
    # 仮説(2026-11-21 20回目3周目)：業種全体・市場全体の出来高の急増（出来高倍率の中央値）＝投げ売りの集中
    q["sec_vratio"] = q.groupby(["date", "sector"])["vratio"].transform("median")
    q["mkt_vratio"] = q.groupby("date")["vratio"].transform("median")
    q["sec_crash"] = (q["ret20"] <= -0.15).astype(float).where(q["ret20"].notna()).groupby([q["date"], q["sector"]]).transform("mean")
    # 仮説(2026-11-21 22回目)：業種ぐるみの急落の進み方＝同業種の急落割合の5日変化、20日の最大に対する現在の比（ピークアウト）
    scs = q.groupby(["sector", "date"])["sec_crash"].first().sort_index()
    scg = scs.groupby(level=0)
    scmax = scg.transform(lambda z: z.rolling(20, min_periods=10).max())
    sc = pd.DataFrame({"sec_crash_chg5": scs - scg.shift(5),
                       "sec_crash_pk": scs / scmax.replace(0, np.nan)}).reset_index()
    q = q.merge(sc, on=["sector", "date"], how="left")
    # 仮説(2026-11-21 22回目4周目)：急落が広がった業種の割合（同業種の急落割合≥15%の業種数/全業種数）＝相場全体の売られ方の広さ
    q["sec_wide"] = (scs >= 0.15).astype(float).where(scs.notna()).groupby(level=1).mean().reindex(q["date"]).values
    # 仮説(2026-11-21 22回目)：業種の大型株（時価総額上位3銘柄）の5日騰落の平均＝業種の主力株が先に戻り始めたか
    if "mcap" in q.columns:
        rk = q.groupby(["date", "sector"])["mcap"].rank(ascending=False, method="first")
        q["sec_big5"] = q["ret5"].where(rk <= 3).groupby([q["date"], q["sector"]]).transform("mean")
        q["big_gap5"] = q["sec_big5"] - q["ret5"]
    q = q.drop(columns=["sec20_prev"])
    # 仮説(2026-11-21 9回目)：市場との連動（60日ベータ）・為替感応度（円安と株価の60日相関）
    q = q.sort_values(["code", "date"]).reset_index(drop=True)
    gc = q["code"]

    def _roll(s):
        return s.groupby(gc).transform(lambda z: z.rolling(60, min_periods=40).mean())
    x1 = q["ret1"]
    m1 = q.groupby("date")["ret1"].transform("median")
    cov = _roll(x1 * m1) - _roll(x1) * _roll(m1)
    var = _roll(m1 * m1) - _roll(m1) ** 2
    q["beta60"] = cov / var.replace(0, np.nan)
    if "usdjpy_ret1" in q.columns:
        ok = x1.notna() & q["usdjpy_ret1"].notna()
        xa, fa = x1.where(ok), q["usdjpy_ret1"].where(ok)
        cxy = _roll(xa * fa) - _roll(xa) * _roll(fa)
        sx = (_roll(xa * xa) - _roll(xa) ** 2).clip(lower=0) ** 0.5
        sf = (_roll(fa * fa) - _roll(fa) ** 2).clip(lower=0) ** 0.5
        q["fx_corr60"] = cxy / (sx * sf).replace(0, np.nan)
    # 仮説(2026-11-22 33回目)：前夜の米国株(S&P500)との60日相関（外需・米国連動株か）。sp500_ret1 は前日までの確定値
    if "sp500_ret1" in q.columns:
        ok = x1.notna() & q["sp500_ret1"].notna()
        xa, fa = x1.where(ok), q["sp500_ret1"].where(ok)
        cxy = _roll(xa * fa) - _roll(xa) * _roll(fa)
        sx = (_roll(xa * xa) - _roll(xa) ** 2).clip(lower=0) ** 0.5
        sf = (_roll(fa * fa) - _roll(fa) ** 2).clip(lower=0) ** 0.5
        q["us_corr60"] = cxy / (sx * sf).replace(0, np.nan)
    else:
        q["us_corr60"] = np.nan
    # 仮説(2026-11-22 38回目)：前日の原油(WTI)との60日相関（原油高で売られる/買われる銘柄か）。wti_ret1 は前日までの確定値
    if "wti_ret1" in q.columns:
        ok = x1.notna() & q["wti_ret1"].notna()
        xa, fa = x1.where(ok), q["wti_ret1"].where(ok)
        cxy = _roll(xa * fa) - _roll(xa) * _roll(fa)
        sx = (_roll(xa * xa) - _roll(xa) ** 2).clip(lower=0) ** 0.5
        sf = (_roll(fa * fa) - _roll(fa) ** 2).clip(lower=0) ** 0.5
        q["oil_corr60"] = cxy / (sx * sf).replace(0, np.nan)
    else:
        q["oil_corr60"] = np.nan
    # 仮説(2026-11-21 19回目)：市場要因を除いた20日騰落＝20日騰落 − 60日ベータ×市場全体の20日騰落（個別の悪材料による下げ）
    q["idio20"] = q["ret20"] - q["beta60"].clip(-1, 3) * q["mkt20"]
    # 仮説(2026-11-21 19回目)：市場全体の投げ売り度＝20日騰落が-20%以下の銘柄の割合（t 終値で確定）
    q["frac_crash"] = (q["ret20"] <= -0.2).astype(float).where(q["ret20"].notna()).groupby(q["date"]).transform("mean")
    return q.sort_values(["code", "date"]).reset_index(drop=True)


def add_features(q: pd.DataFrame) -> pd.DataFrame:
    g = q.groupby("code", sort=False)
    c, h, l, o, v = q["close"], q["high"], q["low"], q["open"], q["volume"]
    prev = g["close"].shift(1)
    q["ret1"] = c / prev - 1
    for n in (5, 20, 60):
        q[f"ret{n}"] = c / g["close"].shift(n) - 1
    for n in (5, 25, 75):
        q[f"ma{n}"] = g["close"].transform(lambda s, n=n: s.rolling(n, min_periods=n).mean())
    q["dev25"] = c / q["ma25"] - 1
    q["dev75"] = c / q["ma75"] - 1
    q["ma_up"] = (q["ma5"] > q["ma25"]).astype(float)
    q["hi250"] = g["high"].transform(lambda s: s.rolling(250, min_periods=120).max())
    q["from_hi"] = c / q["hi250"] - 1
    q["lo60"] = g["low"].transform(lambda s: s.rolling(60, min_periods=40).min())
    q["from_lo60"] = c / q["lo60"] - 1
    q["vol20"] = g["ret1"].transform(lambda s: s.rolling(20, min_periods=15).std())
    tv = q["turnover_value"] if "turnover_value" in q.columns else c * v
    q["tv20"] = tv.groupby(q["code"]).transform(lambda s: s.rolling(20, min_periods=15).mean())
    q["vratio"] = v / v.groupby(q["code"]).transform(lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    d = q["ret1"].fillna(0)
    up = d.clip(lower=0).groupby(q["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    dn = (-d.clip(upper=0)).groupby(q["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    q["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    rng = (h - l).replace(0, np.nan)
    q["clpos"] = (c - l) / rng
    q["gap"] = o / prev - 1
    q["body"] = (c - o) / o
    # 仮説(2026-11-21)：連続陽線・出来高の枯れ・52週高値更新（すべて t 終値時点で確定）
    bull = (c > o).astype(int)
    grp = (bull == 0).groupby(q["code"]).cumsum()
    q["up_streak"] = bull.groupby([q["code"], grp]).cumsum().astype(float)
    v5 = v.groupby(q["code"]).transform(lambda s: s.rolling(5, min_periods=5).mean())
    v60 = v.groupby(q["code"]).transform(lambda s: s.shift(5).rolling(60, min_periods=40).mean())
    q["vdry"] = v5 / v60.replace(0, np.nan)
    phi = g["high"].transform(lambda s: s.shift(1).rolling(250, min_periods=120).max())
    q["new_hi"] = (c > phi).astype(float).where(phi.notna())
    # 仮説(2026-11-21 2回目)：値幅の収縮・75日線の傾き（t 終値時点で確定）
    h20 = g["high"].transform(lambda s: s.rolling(20, min_periods=20).max())
    l20 = g["low"].transform(lambda s: s.rolling(20, min_periods=20).min())
    q["rng20"] = (h20 - l20) / c
    q["ma75_slope"] = q["ma75"] / q.groupby("code", sort=False)["ma75"].shift(20) - 1
    # 仮説(2026-11-21 3回目)：下ヒゲ・直近20日の最大日次上昇・ゴールデンクロス直後（t 終値時点で確定）
    q["shadow_lo"] = (np.minimum(o, c) - l) / c
    q["max1_20"] = q.groupby("code", sort=False)["ret1"].transform(lambda s: s.rolling(20, min_periods=15).max())
    g2 = q.groupby("code", sort=False)
    cross = (q["ma25"] > q["ma75"]) & (g2["ma25"].shift(1) <= g2["ma75"].shift(1))
    q["gc5"] = cross.astype(float).groupby(q["code"]).transform(lambda s: s.rolling(5, min_periods=1).max()).where(q["ma75"].notna())
    # 仮説(2026-11-21 10回目)：直近20日のうち下落した日の割合（じり安＝売りが続いた銘柄）
    q["down_frac20"] = (q["ret1"] < 0).astype(float).where(q["ret1"].notna())         .groupby(q["code"]).transform(lambda s: s.rolling(20, min_periods=20).mean())
    # 仮説(2026-11-21 12回目)：終値ベースの連続下落日数（売りが途切れない銘柄の反発）
    dnb = (q["ret1"] < 0).astype(int)
    grp2 = (dnb == 0).groupby(q["code"]).cumsum()
    q["dn_streak"] = dnb.groupby([q["code"], grp2]).cumsum().astype(float)
    # 仮説(2026-11-21 13回目)：ボラの急拡大＝直近5日の日次ボラ/前60日の日次ボラ（t 終値時点で確定）
    sd5 = q.groupby("code", sort=False)["ret1"].transform(lambda s: s.rolling(5, min_periods=5).std())
    sd60 = q.groupby("code", sort=False)["ret1"].transform(lambda s: s.shift(5).rolling(60, min_periods=40).std())
    q["vol_exp"] = sd5 / sd60.replace(0, np.nan)
    # 仮説(2026-11-21 15回目)：夜間(前日終値→当日始値)と日中(始値→終値)の20日累計騰落の分解（t 終値時点で確定）
    lg_on = np.log(o / prev)
    lg_in = np.log(c / o)
    q["ovn20"] = np.expm1(lg_on.groupby(q["code"]).transform(lambda s: s.rolling(20, min_periods=20).sum()))
    q["intra20"] = np.expm1(lg_in.groupby(q["code"]).transform(lambda s: s.rolling(20, min_periods=20).sum()))
    # 仮説(2026-11-21 16回目)：夜間騰落の短期版（5日累計）。直近の寄り付きの売り込みを捉える（t 終値時点で確定）
    q["ovn5"] = np.expm1(lg_on.groupby(q["code"]).transform(lambda s: s.rolling(5, min_periods=5).sum()))
    # 仮説(2026-11-21 17回目)：上向きの反転足＝前日が下落し、当日の終値が前日の高値を上回る（t 終値時点で確定）
    ph = g["high"].shift(1)
    q["rev_up"] = ((c > ph) & (g["close"].shift(1) < g["close"].shift(2))).astype(float).where(ph.notna())
    # 仮説(2026-11-21 15回目)：市場全体の荒れ具合＝その日の全銘柄の騰落のばらつき（標準偏差）の5日平均
    xd = q.groupby("date")["ret1"].std().sort_index().rolling(5, min_periods=5).mean().rename("xs_disp5")
    q = q.merge(xd, on="date", how="left")
    # 仮説(2026-11-21 18回目)：個別の窓＝その銘柄の当日の窓 − 全銘柄の窓の中央値（市場要因を除いた寄り付きの売られ方）
    q["xs_gap"] = q["gap"] - q.groupby("date")["gap"].transform("median")
    # 仮説(2026-11-21 19回目2周目)：市場全体の窓（全銘柄の窓の中央値＝寄り付きの一斉の売り/買い）・52週安値を終値で割り込み
    q["mkt_gap"] = q.groupby("date")["gap"].transform("median")
    # 仮説(2026-11-21 19回目4周目)：製造業（景気・為替・海外株に敏感な業種）かどうか（東証33業種で判定）
    MFG = {"食料品", "繊維製品", "パルプ･紙", "化学", "医薬品", "石油･石炭製品", "ゴム製品", "ガラス･土石製品", "鉄鋼",
           "非鉄金属", "金属製品", "機械", "電気機器", "輸送用機器", "精密機器", "その他製品"}
    q["mfg"] = q["sector"].isin(MFG).astype(float).where(q["sector"].notna()) if "sector" in q.columns else np.nan
    # 仮説(2026-11-21 20回目)：20日高値からの経過日数（下げの速さ。少ないほど急な下げ）
    q["days_hi20"] = g["high"].transform(lambda s: s.rolling(20, min_periods=20).apply(lambda a: 19 - np.argmax(a), raw=True))
    # 仮説(2026-11-21 21回目)：長期の勢い（250日騰落）・直近20日の最大の日次下落（一発の悪材料か、じり安か）・
    # 上昇日と下落日の出来高の比（20日。売り物の出尽くし／買い集め）。すべて t 終値時点で確定
    q["ret250"] = c / g["close"].shift(250) - 1
    q["min1_20"] = q.groupby("code", sort=False)["ret1"].transform(lambda s: s.rolling(20, min_periods=15).min())
    vu = v.where(q["ret1"] > 0, 0.0).groupby(q["code"]).transform(lambda s: s.rolling(20, min_periods=15).sum())
    vd = v.where(q["ret1"] < 0, 0.0).groupby(q["code"]).transform(lambda s: s.rolling(20, min_periods=15).sum())
    q["ud_vol20"] = vu / vd.replace(0, np.nan)
    plo = g["low"].transform(lambda s: s.shift(1).rolling(250, min_periods=120).min())
    q["new_lo"] = (c < plo).astype(float).where(plo.notna())
    # 仮説(2026-11-22 24回目)：安値からの戻り始め（終値/直近5日安値）・下げの集中度（5日騰落/20日騰落）・
    # 出来高のクライマックス（当日の出来高/前60日の最大）。すべて t 終値時点で確定
    l5 = g["low"].transform(lambda s: s.rolling(5, min_periods=5).min())
    q["bounce5"] = c / l5 - 1
    q["crash_speed"] = (q["ret5"] / q["ret20"]).where(q["ret20"] <= -0.05)
    vmx = v.groupby(q["code"]).transform(lambda s: s.shift(1).rolling(60, min_periods=40).max())
    q["vol_climax"] = v / vmx.replace(0, np.nan)
    # 仮説(2026-11-22 25回目)：日次騰落の歪度（60日。宝くじのような上振れ銘柄か、下振れが溜まった銘柄か）・
    # 非流動性（Amihud：|日次騰落|/売買代金(億円) の20日平均）・市場全体の5日騰落の歪み（全銘柄の歪度＝一部の銘柄だけの急落か）
    q["skew60"] = q.groupby("code", sort=False)["ret1"].transform(lambda s: s.rolling(60, min_periods=40).skew())
    am = (q["ret1"].abs() / (tv / 1e8).replace(0, np.nan))
    q["illiq20"] = am.groupby(q["code"]).transform(lambda s: s.rolling(20, min_periods=15).mean())
    xsk = q.groupby("date")["ret5"].skew().rename("xs_skew5")
    q = q.merge(xsk, on="date", how="left")
    # 仮説(2026-11-22 26回目)：ボラで割った下げの深さ（20日騰落/(60日の日次ボラ×√20)＝その銘柄にとって異例の下げか）・
    # 窓を開けて下げた日の日中の戻り（(終値-始値)/(前日終値-始値)、窓≤-2%のみ）・60日安値からの経過日数。すべて t 終値時点で確定
    sd60b = q.groupby("code", sort=False)["ret1"].transform(lambda s: s.rolling(60, min_periods=40).std())
    q["z_drop20"] = q["ret20"] / (sd60b * np.sqrt(20)).replace(0, np.nan)
    pc = q.groupby("code", sort=False)["close"].shift(1)
    q["gap_rec"] = ((q["close"] - q["open"]) / (pc - q["open"]).replace(0, np.nan)).where(q["gap"] <= -0.02)
    q["lo_days60"] = q.groupby("code", sort=False)["low"].transform(lambda s: s.rolling(60, min_periods=40).apply(lambda a: len(a) - 1 - np.argmin(a), raw=True))
    # 市場全体（全銘柄の20日騰落の中央値＝相場の地合い）
    mk = q.groupby("date")["ret20"].median().rename("mkt20")
    q = q.merge(mk, on="date", how="left")
    # 仮説(2026-11-21 13回目)：相場全体の短期の急落（全銘柄の5日騰落の中央値）
    mk5 = q.groupby("date")["ret5"].median().rename("mkt5")
    q = q.merge(mk5, on="date", how="left")
    q = add_context(q)
    # 仮説(2026-11-21 20回目4周目)：日本株だけが売られた度合い＝市場全体の5日騰落(中央値) − 米国株(S&P500)の5日騰落(前日)
    q["jp_rel5"] = q["mkt5"] - q["sp500_ret5"] if "sp500_ret5" in q.columns else np.nan
    # 仮説(2026-11-22 26回目2周目)：市場全体の連続下落日数（全銘柄の前日比の中央値がマイナスの日の連続）・
    # 60日安値をつけた銘柄の割合（市場全体の投げ売り）・業種の5日騰落−市場全体の5日騰落（業種のローテーション）。t 終値時点で確定
    md1 = q.groupby("date")["ret1"].median().sort_index()
    dn_m = (md1 < 0).astype(int)
    q = q.merge(dn_m.groupby((dn_m == 0).cumsum()).cumsum().astype(float).rename("mkt_dn_streak"), on="date", how="left")
    atlo = (q["low"] <= q["lo60"]).astype(float).where(q["lo60"].notna())
    q = q.merge(atlo.groupby(q["date"]).mean().rename("mkt_lo60_frac"), on="date", how="left")
    q["sec_rel5"] = q["sec5"] - q["mkt5"] if "sec5" in q.columns else np.nan
    # 仮説(2026-11-22 27回目)：日本の市場全体の60日高値からの下落（全銘柄の前日比の中央値を積み上げた指数）・
    # 業種内での20日騰落の順位（0=業種で最も下げた銘柄）・直近5日で下ヒゲ≥2%をつけた日数（下値での買い支え）。t 終値時点で確定
    mix = (1 + md1.fillna(0)).cumprod()
    q = q.merge((mix / mix.rolling(60, min_periods=40).max() - 1).rename("mkt_dd60"), on="date", how="left")
    q["sec_rank20"] = q.groupby(["date", "sector"])["ret20"].rank(pct=True) if "sector" in q.columns else np.nan
    q["hammer5"] = (q["shadow_lo"] >= 0.02).astype(float).where(q["shadow_lo"].notna()).groupby(q["code"]).transform(lambda s: s.rolling(5, min_periods=5).sum())
    # 仮説(2026-11-22 28回目)：日次リターンの20日の1次自己相関（負＝行き過ぎが戻りやすい銘柄）・株価の水準（低位株は%で動きやすい）・
    # 直近5日の引け位置（当日レンジ内の終値位置）の平均（引けにかけて買われる日が続くか）。すべて t 終値時点で確定
    r1p = q.groupby("code", sort=False)["ret1"].shift(1)
    xy = (q["ret1"] * r1p).groupby(q["code"]).transform(lambda s: s.rolling(20, min_periods=15).mean())
    mx_ = q.groupby("code", sort=False)["ret1"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    sx_ = q.groupby("code", sort=False)["ret1"].transform(lambda s: s.rolling(20, min_periods=15).std())
    q["ac20"] = (xy - mx_ ** 2) / (sx_ ** 2).replace(0, np.nan)
    q["px_lvl"] = q["close"]
    q["clpos5"] = q.groupby("code", sort=False)["clpos"].transform(lambda s: s.rolling(5, min_periods=4).mean())
    # 仮説(2026-11-22 30回目)：急落前の地合い＝t-20 から t-60 の騰落（急落前は強かった銘柄か）・その業種の中央値・
    # 業種との60日相関（下げが業種要因か個別要因か）。すべて t 終値時点で確定
    gc_ = q.groupby("code", sort=False)["close"]
    q["pre_trend"] = gc_.shift(20) / gc_.shift(60) - 1
    q["sec_pre"] = q.groupby(["date", "sector"])["pre_trend"].transform("median") if "sector" in q.columns else np.nan
    if "sector" in q.columns:
        smr = q.groupby(["date", "sector"])["ret1"].transform("median")
        rr = lambda s: s.groupby(q["code"]).transform(lambda x: x.rolling(60, min_periods=40).mean())
        cov_ = rr(q["ret1"] * smr) - rr(q["ret1"]) * rr(smr)
        v1 = rr(q["ret1"] ** 2) - rr(q["ret1"]) ** 2
        v2 = rr(smr ** 2) - rr(smr) ** 2
        q["sec_corr60"] = cov_ / np.sqrt((v1 * v2).where(v1 * v2 > 0))
    else:
        q["sec_corr60"] = np.nan
    # 仮説(2026-11-22 31回目)：60日安値からの戻りの業種内での先行度（銘柄の戻り−業種の中央値）・業種内の売買代金の順位（業種の主力株か）・
    # 当日に上げた業種内の銘柄の割合（業種ぐるみの反転の日か）。すべて t 終値時点で確定
    if "sector" in q.columns:
        q["bounce_rel"] = q["from_lo60"] - q.groupby(["date", "sector"])["from_lo60"].transform("median")
        q["sec_tv_rank"] = q.groupby(["date", "sector"])["tv20"].rank(pct=True)
        q["sec_up_frac"] = (q["ret1"] > 0).astype(float).where(q["ret1"].notna()).groupby([q["date"], q["sector"]]).transform("mean")
    else:
        q["bounce_rel"] = q["sec_tv_rank"] = q["sec_up_frac"] = np.nan
    # 仮説(2026-11-22 31回目2周目)：直近20日で最大の日次下落からの経過日数（ショックが新しいか古いか）・
    # 売買代金の5日平均/60日平均（注目度が1週間単位で高まっているか）。すべて t 終値時点で確定
    q["min1_days"] = q.groupby("code", sort=False)["ret1"].transform(
        lambda s: s.rolling(20, min_periods=20).apply(lambda a: 19 - np.argmin(a), raw=True))
    tv_ = q["turnover_value"] if "turnover_value" in q.columns else q["close"] * q["volume"]
    gtv = tv_.groupby(q["code"])
    q["tv5_60"] = gtv.transform(lambda s: s.rolling(5, min_periods=5).mean()) / gtv.transform(lambda s: s.rolling(60, min_periods=40).mean()).replace(0, np.nan)
    # 仮説(2026-11-22 32回目)：値幅の収縮＝直近5日の(高値最大-安値最小)/直近20日の同値（急落後に値動きが落ち着いたか＝売りの一巡）・
    # 5日線乖離（短期の戻り始めか、まだ下げの途中か）。すべて t 終値時点で確定
    gh_ = q.groupby("code", sort=False)["high"]; gl_ = q.groupby("code", sort=False)["low"]
    r5_ = gh_.transform(lambda s: s.rolling(5, min_periods=5).max()) - gl_.transform(lambda s: s.rolling(5, min_periods=5).min())
    r20_ = gh_.transform(lambda s: s.rolling(20, min_periods=20).max()) - gl_.transform(lambda s: s.rolling(20, min_periods=20).min())
    q["rng5_20"] = r5_ / r20_.replace(0, np.nan)
    q["dev5"] = q["close"] / q.groupby("code", sort=False)["close"].transform(lambda s: s.rolling(5, min_periods=5).mean()) - 1
    # 仮説(2026-11-22 33回目)：業種内・市場全体で25日線より上にある銘柄の割合（業種・市場の地合いの広がり）。t 終値で確定
    ab25 = (q["dev25"] > 0).astype(float).where(q["dev25"].notna())
    q["mkt_above25"] = ab25.groupby(q["date"]).transform("mean")
    q["sec_above25"] = ab25.groupby([q["date"], q["sector"]]).transform("mean") if "sector" in q.columns else np.nan
    # 仮説(2026-11-22 34回目)：当日の騰落の業種差・直近20日で60日安値を更新した日数・業種内の時価総額の順位。t 終値で確定
    q["rel_sec1"] = q["ret1"] - q["sec_ret1"] if "sec_ret1" in q.columns else np.nan
    lo60c = q.groupby("code", sort=False)["close"].transform(lambda s: s.rolling(60, min_periods=40).min())
    nlo = (q["close"] <= lo60c).astype(float).where(lo60c.notna())
    q["lo_cnt20"] = nlo.groupby(q["code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).sum())
    if "mcap" in q.columns and "sector" in q.columns:
        q["sec_mcap_rank"] = q.groupby(["date", "sector"])["mcap"].rank(pct=True)
    else:
        q["sec_mcap_rank"] = np.nan
    # 仮説(2026-11-22 35回目)：個別の20日ボラ/業種の20日ボラ(中央値)＝業種の中で特に荒れているか・
    # 業種の60日騰落の業種間の順位（0=最も下げた業種）＝長めの業種ローテーション。t 終値で確定
    if "sector" in q.columns:
        q["vol_rel_sec"] = q["vol20"] / q.groupby(["date", "sector"])["vol20"].transform("median").replace(0, np.nan)
        s60 = q.groupby(["date", "sector"])["ret60"].median()
        q["sec60_rank"] = s60.groupby(level=0).rank(pct=True).reindex(pd.MultiIndex.from_arrays([q["date"], q["sector"]])).values
    else:
        q["vol_rel_sec"] = q["sec60_rank"] = np.nan
    if "sh_chg" not in q.columns:
        q["sh_chg"] = np.nan
    # 仮説(2026-11-22 37回目)：業種内のPBRの順位（業種の中で割安か）・市場全体で25日線より上の割合の5日変化（地合いの改善の始まり）・
    # 直近20日で窓≤-2%を開けて下げた日数（夜間のショックの繰り返し）。t 終値で確定
    if "pbr" in q.columns and "sector" in q.columns:
        q["sec_pbr_rank"] = q["pbr"].where(q["pbr"] > 0).groupby([q["date"], q["sector"]]).rank(pct=True)
    else:
        q["sec_pbr_rank"] = np.nan
    ma25d = q.groupby("date")["mkt_above25"].first().sort_index()
    q = q.merge((ma25d - ma25d.shift(5)).rename("mkt_ab25_chg5"), on="date", how="left")
    gdn = (q["gap"] <= -0.02).astype(float).where(q["gap"].notna())
    q["gap_dn_cnt20"] = gdn.groupby(q["code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).sum())
    # 仮説(2026-11-22 38回目)：業種内で25日線より上の割合の5日変化（業種の地合いが戻り始めたか）・
    # 日本株と米国株の60日高値からの下落の差（日本固有の下げか）。t 終値・前夜までの確定値
    if "sector" in q.columns:
        sab = q.groupby(["date", "sector"])["sec_above25"].first().sort_index()
        sch = (sab - sab.groupby(level=1).shift(5)).rename("sec_ab25_chg5").reset_index()
        q = q.merge(sch, on=["date", "sector"], how="left")
    else:
        q["sec_ab25_chg5"] = np.nan
    q["jp_us_dd"] = q["mkt_dd60"] - q["us_dd60"] if "us_dd60" in q.columns else np.nan
    # 仮説(2026-11-22 39回目)：直近20日で出来高が最大だった日の騰落（投げ売りの日か、買い集めの日か）・その日からの経過日数・
    # 10日騰落（20日より短い下げの速さ）。すべて t 終値時点で確定
    vol_ = q["volume"].astype(float)
    vidx = vol_.groupby(q["code"], sort=False).transform(
        lambda s: s.rolling(20, min_periods=20).apply(lambda a: 19 - np.argmax(a), raw=True))
    q["vmax_days"] = vidx
    r1s = {k: q.groupby("code", sort=False)["ret1"].shift(k) for k in range(20)}
    vr = pd.Series(np.nan, index=q.index)
    for k in range(20):
        vr = vr.where(vidx != k, r1s[k])
    q["vmax_ret"] = vr
    q["ret10"] = q.groupby("code", sort=False)["close"].transform(lambda s: s / s.shift(10) - 1)
    # 仮説(2026-11-22 40回目)：物色の向き（小型株-大型株の5日騰落の差＝規模のローテーション、低PBR-高PBRの20日騰落の差＝割安株の
    # ローテーション）と、20日騰落の全銘柄内の順位（その日の相場に対する相対的な下げの深さ）。すべて t 終値時点で確定
    def _spread(col, key, lo=0.3, hi=0.7):
        if key not in q.columns:
            return pd.Series(np.nan, index=q.index)
        k = q[key].where(q[key] > 0)
        r = k.groupby(q["date"]).rank(pct=True)
        a_ = q[col].where(r <= lo).groupby(q["date"]).transform("median")
        b_ = q[col].where(r >= hi).groupby(q["date"]).transform("median")
        return a_ - b_
    q["size_sp5"] = _spread("ret5", "mcap")
    q["val_sp20"] = _spread("ret20", "pbr")
    q["ret20_xr"] = q.groupby("date")["ret20"].rank(pct=True)
    cnum =pd.to_numeric(q["code"].str[:4], errors="coerce")
    q["etf"] = cnum.between(1300, 1699) | cnum.between(2500, 2599)
    # 結果（未来）: 翌日始値で買い、t+1..t+20 の高値最大
    g = q.groupby("code", sort=False)
    q["buy"] = g["open"].shift(-1)
    q["fmax"] = g["high"].transform(lambda s: s[::-1].rolling(HOLD, min_periods=HOLD).max()[::-1].shift(-1))
    q["fmin"] = g["low"].transform(lambda s: s[::-1].rolling(HOLD, min_periods=HOLD).min()[::-1].shift(-1))
    q["fclose"] = g["close"].shift(-HOLD)
    q["win"] = (q["fmax"] >= q["buy"] * (1 + TP)).astype(float)
    q["pnl"] = np.where(q["win"] == 1, TP, q["fclose"] / q["buy"] - 1)
    q["dd10"] = (q["fmin"] <= q["buy"] * 0.90).astype(float)
    q.loc[q["fclose"].isna() | q["buy"].isna(), ["win", "pnl", "dd10"]] = np.nan
    # 仮説(2026-11-21 17回目)：その銘柄の過去の+10%到達のしやすさ＝t-20 日以前に買った場合の勝ち（t 終値までに確定済み）の60日平均
    q["hit_hist"] = q.groupby("code", sort=False)["win"].transform(lambda s: s.shift(HOLD).rolling(60, min_periods=40).mean())
    return q


# --------------------------------------------------------------------------
# ルール
# --------------------------------------------------------------------------
BASE = [["tv20", ">=", 5e8], ["close", ">=", 200]]   # 流動性（既存 config と同じ水準）

GRID = {
    "ret1":      [(">=", 0.03), (">=", 0.05), (">=", 0.08), ("<=", -0.03), ("<=", -0.05)],
    "ret5":      [(">=", 0.05), (">=", 0.10), ("<=", -0.05), ("<=", -0.10)],
    "ret20":     [(">=", 0.10), (">=", 0.20), ("<=", -0.10), ("<=", -0.20)],
    "ret60":     [(">=", 0.20), (">=", 0.40), ("<=", -0.15)],
    "dev25":     [(">=", 0.05), (">=", 0.10), ("<=", -0.05), ("<=", -0.10)],
    "dev75":     [(">=", 0.10), ("<=", -0.10)],
    "ma_up":     [(">=", 1)],
    "from_hi":   [(">=", -0.03), (">=", -0.10), ("<=", -0.30), ("<=", -0.50)],
    "from_lo60": [("<=", 0.05), (">=", 0.30)],
    "vol20":     [(">=", 0.03), (">=", 0.04), (">=", 0.05), ("<=", 0.02)],
    "tv20":      [(">=", 2e9), ("<=", 2e9)],
    "vratio":    [(">=", 2), (">=", 3), (">=", 5), ("<=", 0.7)],
    "rsi14":     [("<=", 30), ("<=", 40), (">=", 60), (">=", 70)],
    "clpos":     [(">=", 0.8), ("<=", 0.2)],
    "gap":       [(">=", 0.02), ("<=", -0.02)],
    "body":      [(">=", 0.03), ("<=", -0.03)],
    "mkt20":     [(">=", 0.0), ("<=", 0.0), (">=", 0.03)],
    "close":     [("<=", 1000), (">=", 3000)],
    # 相場全体・投資家心理
    "breadth":   [(">=", 0.6), ("<=", 0.3)],
    "adr25":     [(">=", 120), ("<=", 80), ("<=", 70)],
    "lev_z":     [(">=", 1), ("<=", -1)],
    "rel_sec20": [(">=", 0.05), (">=", 0.10), ("<=", -0.05)],
    # 他の資産（金・国債・REIT）
    "gold_ret20": [(">=", 0.03), ("<=", -0.03)],
    "bond_ret20": [(">=", 0.005), ("<=", -0.005)],
    "reit_ret20": [(">=", 0.03), ("<=", -0.03)],
    "nk_ret20":  [(">=", 0.05), ("<=", -0.05)],
    # 金利・為替・海外
    "vix":       [(">=", 25), ("<=", 15), (">=", 20)],
    "vix_chg5":  [(">=", 3), ("<=", -3)],
    "us10y_chg20": [(">=", 0.2), ("<=", -0.2)],
    "jgb10y_chg20": [(">=", 0.1), ("<=", -0.1)],
    "jgb_slope": [(">=", 1.0), ("<=", 0.5)],
    "usdjpy_ret20": [(">=", 0.02), ("<=", -0.02)],
    "wti_ret20": [(">=", 0.08), ("<=", -0.08)],
    "sp500_ret1": [(">=", 0.01), ("<=", -0.01), ("<=", -0.02)],
    "sp500_ret20": [(">=", 0.03), ("<=", -0.03)],
    "nasdaq_ret5": [(">=", 0.03), ("<=", -0.03)],
    # 季節性・決算タイミング
    "month":     [("==", m) for m in range(1, 13)],
    "dow":       [("==", d) for d in range(5)],
    "since_disc": [("<=", 3), ("<=", 10), (">=", 60)],
    # 仮説(2026-11-21)
    "up_streak": [(">=", 3), (">=", 5)],
    "vdry":      [("<=", 0.5), ("<=", 0.7), (">=", 1.5)],
    "new_hi":    [(">=", 1)],
    # 仮説(2026-11-21 2回目)
    "rng20":     [("<=", 0.08), ("<=", 0.12), (">=", 0.40)],
    "ma75_slope": [(">=", 0.03), (">=", 0.08), ("<=", -0.03)],
    "dom":       [("<=", 5), (">=", 25)],
    # 仮説(2026-11-21 3回目)
    "shadow_lo": [(">=", 0.02), (">=", 0.04)],
    "max1_20":   [(">=", 0.08), (">=", 0.15), ("<=", 0.02)],
    "gc5":       [(">=", 1)],
    # 仮説(2026-11-21 7回目)
    "pbr":       [("<=", 0.7), ("<=", 1.0), (">=", 3.0)],
    "fc_rev":    [(">=", 0.05), (">=", 0.15), ("<=", -0.05)],
    "sec20":     [(">=", 0.05), ("<=", -0.05), ("<=", -0.10)],
    # 仮説(2026-11-21 8回目)
    "prog_ex":   [(">=", 0.10), (">=", 0.20), ("<=", -0.10)],
    "mcap":      [("<=", 3e10), ("<=", 1e11), (">=", 5e11)],
    "sec5":      [(">=", 0.03), ("<=", -0.03)],
    # 仮説(2026-11-21 9回目)
    "hol_next":  [(">=", 1)],
    "beta60":    [(">=", 1.5), ("<=", 0.5)],
    "fx_corr60": [(">=", 0.3), ("<=", -0.1)],
    # 仮説(2026-11-21 10回目)
    "down_frac20": [(">=", 0.65), ("<=", 0.35)],
    "sec_disp20": [(">=", 0.15), ("<=", 0.05)],
    "sec_turn":  [(">=", 0.08), ("<=", -0.08)],
    # 仮説(2026-11-21 12回目)
    "op_yield":  [(">=", 0.10), (">=", 0.15), ("<=", 0.03)],
    "fc_margin": [(">=", 0.10), (">=", 0.15), ("<=", 0.03)],
    "dn_streak": [(">=", 4), (">=", 6)],
    "sales_prog": [(">=", 0.05), (">=", -0.02), ("<=", -0.05)],
    "vol_exp":   [(">=", 2.0), ("<=", 0.6)],
    "mkt5":      [("<=", -0.02), (">=", 0.02)],
    "disc_ret":  [("<=", -0.10), ("<=", -0.05), (">=", 0.05)],
    "us_dd60":   [("<=", -0.03), ("<=", -0.06)],
    "turn20":    [(">=", 0.01), (">=", 0.02), ("<=", 0.002)],
    # 仮説(2026-11-21 15回目)：夜間・日中の騰落の分解、市場全体の荒れ具合、来期見通しの伸び
    "ovn20":     [("<=", -0.10), ("<=", -0.08), ("<=", -0.04), (">=", 0.05)],
    "intra20":   [("<=", -0.15), ("<=", -0.08), (">=", 0.10)],
    "ovn5":      [("<=", -0.06), ("<=", -0.04), ("<=", -0.025)],
    "xs_disp5":  [(">=", 0.03), ("<=", 0.018)],
    "fc_growth": [(">=", 0.20), (">=", 0.10), ("<=", -0.10)],
    # 仮説(2026-11-21 17回目)
    "usdjpy_ret5": [(">=", 0.015), ("<=", -0.015), ("<=", -0.01), ("<=", 0.0)],
    "rev_up":    [(">=", 1)],
    "hit_hist":  [(">=", 0.5), (">=", 0.3), ("<=", 0.05)],
    # 仮説(2026-11-21 18回目)：個別の窓・VIXの相対水準・市場区分
    "xs_gap":    [("<=", -0.05), ("<=", -0.03), (">=", 0.03)],
    "vix_rel60": [(">=", 1.3), (">=", 1.15), ("<=", 0.85)],
    "mkt_seg":   [("==", 1), ("==", 2), ("==", 3), ("<=", 2)],
    # 仮説(2026-11-21 19回目)：市場要因を除いた下げ・市場全体の投げ売り度・日米金利差
    "idio20":    [("<=", -0.25), ("<=", -0.15), (">=", 0.15)],
    "frac_crash": [(">=", 0.10), (">=", 0.05), ("<=", 0.01)],
    "spread_chg20": [(">=", 0.2), ("<=", -0.2)],
    "mkt_gap":   [("<=", -0.015), ("<=", -0.008), (">=", 0.01)],
    "new_lo":    [(">=", 1)],
    "op_yoy":    [(">=", 0.30), (">=", 0.10), ("<=", -0.20)],
    "mfg":       [(">=", 1), ("<=", 0)],
    # 仮説(2026-11-21 20回目)：VIXのピークアウト・業種ぐるみの急落・下げの速さ
    "vix_chg1":  [("<=", -1), ("<=", -2), (">=", 2)],
    "sec_crash": [(">=", 0.30), (">=", 0.15), ("<=", 0.02)],
    "days_hi20": [("<=", 5), ("<=", 10), (">=", 15)],
    "sec_ret1":  [(">=", 0.02), (">=", 0.01), ("<=", -0.02)],
    "wti_ret5":  [(">=", 0.05), ("<=", -0.05)],
    "sec_vratio": [(">=", 1.5), (">=", 1.2), ("<=", 0.8)],
    "mkt_vratio": [(">=", 1.3), (">=", 1.1), ("<=", 0.8)],
    "jp_rel5":   [("<=", -0.03), ("<=", -0.015), (">=", 0.02)],
    # 仮説(2026-11-21 21回目)：長期の勢い・最大の日次下落・上昇日/下落日の出来高比
    "ret250":    [(">=", 0.30), (">=", 0.0), ("<=", -0.20)],
    "min1_20":   [("<=", -0.10), ("<=", -0.06), (">=", -0.025)],
    "ud_vol20":  [(">=", 1.5), ("<=", 0.6), ("<=", 0.4)],
    # 仮説(2026-11-21 22回目)：業種ぐるみの急落の進み方・業種の大型株の戻り
    "sec_crash_chg5": [(">=", 0.10), (">=", 0.05), ("<=", -0.05)],
    "sec_crash_pk":   [("<=", 0.5), ("<=", 0.8), (">=", 1)],
    "sec_big5":  [(">=", 0.03), ("<=", -0.03), ("<=", -0.06)],
    "big_gap5":  [(">=", 0.08), (">=", 0.04), ("<=", -0.03)],
    # 仮説(2026-11-21 22回目2周目)：米金利の短期の急変・VIXのピークアウト・前日の為替の急変
    "us10y_chg5": [(">=", 0.15), ("<=", -0.15)],
    "vix_pk20":  [(">=", 0.95), ("<=", 0.8), ("<=", 0.65)],
    "usdjpy_ret1": [(">=", 0.007), ("<=", -0.007)],
    # 仮説(2026-11-21 22回目3周目)：売上高の前年同期比
    "sales_yoy": [(">=", 0.15), (">=", 0.05), ("<=", -0.05)],
    "sec_wide":  [(">=", 0.5), (">=", 0.25), ("<=", 0.1)],
    # 仮説(2026-11-22 24回目)：安値からの戻り始め・下げの集中度・出来高のクライマックス
    "bounce5":   [(">=", 0.05), (">=", 0.02), ("<=", 0.005)],
    "crash_speed": [(">=", 0.7), ("<=", 0.3), ("<=", 0.0)],
    "vol_climax": [(">=", 1.0), (">=", 0.7), ("<=", 0.2)],
    # 仮説(2026-11-22 25回目)：歪度・非流動性・市場全体の騰落の歪み
    "skew60":    [(">=", 1.0), (">=", 0.3), ("<=", -0.5)],
    "illiq20":   [(">=", 0.005), (">=", 0.002), ("<=", 0.0005)],
    "xs_skew5":  [(">=", 1.0), ("<=", -1.0), ("<=", 0.0)],
    # 仮説(2026-11-22 26回目)：ボラで割った下げの深さ・窓を開けて下げた日の日中の戻り・60日安値からの経過日数
    "z_drop20":  [("<=", -2.5), ("<=", -1.5), (">=", 0.0)],
    "gap_rec":   [(">=", 1.0), (">=", 0.5), ("<=", 0.0)],
    "lo_days60": [("<=", 0), ("<=", 3), (">=", 20)],
    # 仮説(2026-11-22 26回目2周目)：市場全体の連続下落日数・60日安値をつけた銘柄の割合・業種の市場に対する5日騰落の差
    "mkt_dn_streak": [(">=", 4), (">=", 2), ("<=", 0)],
    "mkt_lo60_frac": [(">=", 0.2), (">=", 0.1), ("<=", 0.03)],
    "sec_rel5":  [("<=", -0.05), ("<=", -0.02), (">=", 0.03)],
    # 仮説(2026-11-22 27回目)：日本の市場全体の60日高値からの下落・業種内の20日騰落の順位・直近5日の下ヒゲの日数
    "mkt_dd60":  [("<=", -0.10), ("<=", -0.05), (">=", -0.01)],
    "sec_rank20": [("<=", 0.1), ("<=", 0.25), (">=", 0.75)],
    "hammer5":   [(">=", 3), (">=", 2), ("<=", 0)],
    # 仮説(2026-11-22 28回目)：日次リターンの自己相関・株価の水準・直近5日の引け位置の平均
    "ac20":      [("<=", -0.3), ("<=", -0.15), (">=", 0.15)],
    "px_lvl":    [("<=", 300), ("<=", 1000), (">=", 5000)],
    "clpos5":    [(">=", 0.65), ("<=", 0.35), ("<=", 0.25)],
    # 仮説(2026-11-22 29回目)：営業利益率の前年同期比の変化・ドル円の20日ボラ・日本2年金利の20日変化
    "opm_chg":   [(">=", 0.03), (">=", 0.01), ("<=", -0.02)],
    "fx_vol20":  [(">=", 0.008), (">=", 0.006), ("<=", 0.004)],
    "jgb2y_chg20": [(">=", 0.1), (">=", 0.05), ("<=", 0.0)],
    # 仮説(2026-11-22 30回目)：急落前の地合い（個別・業種）・業種との60日相関
    "pre_trend": [(">=", 0.15), (">=", 0.05), ("<=", -0.10)],
    "sec_pre":   [(">=", 0.10), (">=", 0.03), ("<=", -0.05)],
    "sec_corr60": [(">=", 0.6), (">=", 0.4), ("<=", 0.2)],
    # 仮説(2026-11-22 31回目)：業種内の戻りの先行度・業種内の売買代金の順位・当日に上げた業種内の割合
    "bounce_rel": [(">=", 0.03), (">=", 0.0), ("<=", -0.03)],
    "sec_tv_rank": [(">=", 0.8), (">=", 0.5), ("<=", 0.3)],
    "sec_up_frac": [(">=", 0.7), (">=", 0.5), ("<=", 0.3)],
    "min1_days": [("<=", 2), ("<=", 5), (">=", 12)],
    "tv5_60":    [(">=", 2.0), (">=", 1.5), ("<=", 0.8)],
    "rng5_20":   [("<=", 0.3), ("<=", 0.45), (">=", 0.8)],
    "dev5":      [(">=", 0.03), (">=", 0.0), ("<=", -0.03)],
    # 仮説(2026-11-22 33回目)：米国株との60日相関・業種内/市場全体で25日線より上の割合
    "us_corr60": [(">=", 0.3), (">=", 0.15), ("<=", 0.0)],
    "sec_above25": [("<=", 0.1), ("<=", 0.25), (">=", 0.6)],
    "mkt_above25": [("<=", 0.2), ("<=", 0.35), (">=", 0.6)],
    # 仮説(2026-11-22 34回目)：当日の騰落の業種差・60日安値の更新回数・業種内の時価総額の順位
    "rel_sec1":  [("<=", -0.03), ("<=", -0.015), (">=", 0.02)],
    "lo_cnt20":  [(">=", 8), (">=", 4), ("<=", 1)],
    "sec_mcap_rank": [(">=", 0.8), (">=", 0.5), ("<=", 0.3)],
    # 仮説(2026-11-22 35回目)：発行済株式数の変化・業種内でのボラの相対・業種の60日騰落の業種間順位
    "sh_chg":    [("<=", -0.01), (">=", 0.005), (">=", 0.02)],
    "vol_rel_sec": [(">=", 1.5), (">=", 1.2), ("<=", 0.8)],
    "sec60_rank": [("<=", 0.15), ("<=", 0.3), (">=", 0.85)],
    # 仮説(2026-11-22 37回目)：業種内のPBRの順位・市場全体の25日線より上の割合の5日変化・窓を開けて下げた日数
    "sec_pbr_rank": [("<=", 0.2), ("<=", 0.4), (">=", 0.8)],
    "mkt_ab25_chg5": [(">=", 0.15), (">=", 0.05), ("<=", -0.15)],
    "gap_dn_cnt20": [(">=", 4), (">=", 2), ("<=", 0)],
    "oil_corr60": [(">=", 0.15), (">=", 0.05), ("<=", -0.1)],
    "sec_ab25_chg5": [(">=", 0.2), (">=", 0.05), ("<=", -0.2)],
    "jp_us_dd": [("<=", -0.08), ("<=", -0.04), (">=", 0.02)],
    "vmax_ret": [("<=", -0.05), ("<=", -0.02), (">=", 0.02)],
    "vmax_days": [("<=", 2), ("<=", 6), (">=", 12)],
    "ret10": [("<=", -0.15), ("<=", -0.08), (">=", 0.05)],
    "size_sp5": [("<=", -0.02), ("<=", -0.01), (">=", 0.01)],
    "val_sp20": [("<=", -0.03), (">=", 0.02), (">=", 0.04)],
    "ret20_xr": [("<=", 0.02), ("<=", 0.05), (">=", 0.95)],
}

LABEL = {
    "ret1": "前日比", "ret5": "5日騰落", "ret20": "20日騰落", "ret60": "60日騰落",
    "dev25": "25日線乖離", "dev75": "75日線乖離", "ma_up": "5日線>25日線",
    "from_hi": "52週高値からの位置", "from_lo60": "60日安値からの上昇", "vol20": "20日ボラ(日次)",
    "tv20": "20日平均売買代金", "vratio": "出来高倍率(20日平均比)", "rsi14": "RSI14",
    "clpos": "当日レンジ内の終値位置", "gap": "当日の窓", "body": "当日の陽線/陰線幅",
    "mkt20": "市場全体の20日騰落(中央値)", "close": "株価",
    "breadth": "値上がり銘柄の割合", "adr25": "騰落レシオ(25日)",
    "lev_z": "個人の強気度(レバETF/インバETF売買代金の偏差)", "rel_sec20": "業種平均に対する20日騰落の差",
    "gold_ret20": "金(ETF)の20日騰落", "bond_ret20": "国債ETFの20日騰落", "reit_ret20": "REIT(ETF)の20日騰落",
    "nk_ret20": "日経225(ETF)の20日騰落", "vix": "VIX恐怖指数(前日)", "vix_chg5": "VIXの5日変化(ポイント)",
    "us10y_chg20": "米10年金利の20日変化(%pt)", "jgb10y_chg20": "日本10年国債金利の20日変化(%pt)",
    "jgb_slope": "日本国債 10年-2年金利差(%pt)", "usdjpy_ret20": "ドル円の20日変化",
    "wti_ret20": "原油(WTI)の20日騰落", "sp500_ret1": "前夜のS&P500", "sp500_ret20": "S&P500の20日騰落",
    "nasdaq_ret5": "NASDAQの5日騰落", "month": "月", "dow": "曜日", "since_disc": "直近の決算発表からの日数",
    "up_streak": "連続陽線の日数", "vdry": "出来高の枯れ(5日平均/前60日平均)", "new_hi": "52週高値を終値で更新",
    "rng20": "20日の値幅(高値-安値)/株価", "ma75_slope": "75日線の20日間の傾き", "dom": "日付(月の何日)",
    "shadow_lo": "下ヒゲの長さ/株価", "max1_20": "直近20日の最大の日次上昇率", "gc5": "25日線が75日線を上抜けて5日以内",
    "pbr": "PBR(株価/直近BPS)", "fc_rev": "会社予想(営業利益)の直近の修正率", "sec20": "業種全体の20日騰落(中央値)",
    "prog_ex": "営業利益の進捗率-標準進捗(会社予想比)", "mcap": "時価総額", "sec5": "業種全体の5日騰落(中央値)",
    "hol_next": "連休前(次の営業日まで休日を挟む)", "beta60": "市場に対する60日ベータ", "fx_corr60": "円安との60日相関(為替感応度)",
    "down_frac20": "直近20日の下落日の割合", "sec_disp20": "業種内の20日騰落のばらつき", "sec_turn": "業種の20日騰落の変化(20日前比)",
    "op_yield": "予想営業利益の利回り(会社予想/時価総額)", "fc_margin": "予想営業利益率(会社予想)", "dn_streak": "終値の連続下落日数",
    "sales_prog": "売上の進捗率-標準進捗(会社予想比)", "vol_exp": "ボラの急拡大(5日/前60日)", "mkt5": "市場全体の5日騰落(中央値)",
    "disc_ret": "決算発表後の株価反応(発表前日比・20日以内)", "us_dd60": "S&P500の60日高値からの下落(前日)", "turn20": "売買の回転率(20日平均売買代金/時価総額)",
    "ovn20": "夜間(前日終値→始値)の20日累計騰落", "intra20": "日中(始値→終値)の20日累計騰落", "ovn5": "夜間(前日終値→始値)の5日累計騰落",
    "xs_disp5": "市場全体の騰落のばらつき(5日平均)", "fc_growth": "今期見通しの伸び(会社予想営業利益/前期実績)",
    "usdjpy_ret5": "ドル円の5日変化(前日)", "rev_up": "上向きの反転足(前日下落・終値が前日高値超え)", "hit_hist": "過去の+10%到達率(確定済みの直近60回)",
    "xs_gap": "個別の窓(当日の窓-全銘柄の窓の中央値)", "vix_rel60": "VIXの60日平均に対する倍率(前日)", "mkt_seg": "市場区分(1=プライム 2=スタンダード 3=グロース)",
    "idio20": "市場要因を除いた20日騰落(ベータ調整)", "frac_crash": "20日騰落-20%以下の銘柄の割合(市場全体)", "spread_chg20": "日米10年金利差の20日変化(%pt)",
    "mkt_gap": "市場全体の窓(全銘柄の窓の中央値)", "new_lo": "52週安値を終値で割り込み", "op_yoy": "営業利益の前年同期比(直近の決算)",
    "mfg": "製造業(東証33業種)",
    "vix_chg1": "VIXの1日変化(前日・ポイント)", "sec_crash": "同業種で20日騰落-15%以下の銘柄の割合", "days_hi20": "20日高値からの経過日数",
    "sec_ret1": "業種全体の当日の騰落(中央値)", "wti_ret5": "原油(WTI)の5日騰落(前日)",
    "sec_vratio": "業種全体の出来高倍率(中央値)", "mkt_vratio": "市場全体の出来高倍率(中央値)",
    "jp_rel5": "日本株の米国株に対する5日騰落の差",
    "ret250": "250日騰落(長期の勢い)", "min1_20": "直近20日の最大の日次下落率", "ud_vol20": "上昇日/下落日の出来高比(20日)",
    "sec_crash_chg5": "同業種の急落割合の5日変化", "sec_crash_pk": "同業種の急落割合の20日最大に対する比",
    "sec_big5": "業種の大型株(時価総額上位3)の5日騰落", "big_gap5": "業種の大型株の5日騰落-その銘柄の5日騰落",
    "us10y_chg5": "米10年金利の5日変化(%pt・前日)", "vix_pk20": "VIXの20日最大に対する比(前日)", "usdjpy_ret1": "ドル円の1日変化(前日)",
    "sales_yoy": "売上高の前年同期比(直近の決算)", "sec_wide": "急落が広がった業種の割合(市場全体)",
    "bounce5": "直近5日の安値からの戻り", "crash_speed": "下げの集中度(5日騰落/20日騰落)",
    "vol_climax": "出来高のクライマックス(当日/前60日の最大)",
    "skew60": "日次騰落の歪度(60日)", "illiq20": "非流動性(|日次騰落|/売買代金億円, 20日平均)",
    "xs_skew5": "市場全体の5日騰落の歪度(全銘柄)",
    "z_drop20": "ボラで割った20日騰落(20日騰落/(60日ボラ×√20))", "gap_rec": "窓を開けて下げた日の日中の戻り(窓≤-2%)",
    "lo_days60": "60日安値からの経過日数",
    "mkt_dn_streak": "市場全体の連続下落日数(前日比の中央値)", "mkt_lo60_frac": "60日安値をつけた銘柄の割合(市場全体)",
    "sec_rel5": "業種の5日騰落-市場全体の5日騰落",
    "mkt_dd60": "日本の市場全体の60日高値からの下落(中央値指数)", "sec_rank20": "業種内の20日騰落の順位(0=最も下げた)",
    "hammer5": "直近5日で下ヒゲ≥2%の日数",
    "ac20": "日次リターンの20日自己相関", "px_lvl": "株価の水準(円)", "clpos5": "直近5日の引け位置の平均",
    "opm_chg": "営業利益率の前年同期比の変化(直近の決算)", "fx_vol20": "ドル円の20日ボラ(日次・前日)",
    "jgb2y_chg20": "日本2年金利の20日変化(%pt・前日)",
    "pre_trend": "急落前の騰落(60日前→20日前)", "sec_pre": "業種の急落前の騰落(60日前→20日前・中央値)",
    "sec_corr60": "業種との60日相関(日次騰落)",
    "bounce_rel": "60日安値からの戻り−業種の中央値", "sec_tv_rank": "業種内の売買代金の順位(1=最大)",
    "sec_up_frac": "当日に上げた業種内の銘柄の割合",
    "min1_days": "直近20日の最大の日次下落からの経過日数", "tv5_60": "売買代金の5日平均/60日平均",
    "rng5_20": "直近5日の値幅/直近20日の値幅", "dev5": "5日線乖離",
    "us_corr60": "前夜の米国株(S&P500)との60日相関", "sec_above25": "業種内で25日線より上の銘柄の割合",
    "mkt_above25": "市場全体で25日線より上の銘柄の割合",
    "rel_sec1": "当日の騰落−業種の当日の騰落(中央値)", "lo_cnt20": "直近20日で60日安値(終値)を更新した日数",
    "sec_mcap_rank": "業種内の時価総額の順位(1=最大)",
    "sh_chg": "発行済株式数の前回決算からの変化", "vol_rel_sec": "20日ボラ/業種の20日ボラ(中央値)",
    "sec60_rank": "業種の60日騰落の業種間の順位(0=最も下げた)",
    "sec_pbr_rank": "業種内のPBRの順位(0=最も割安)", "mkt_ab25_chg5": "市場全体で25日線より上の割合の5日変化",
    "gap_dn_cnt20": "直近20日で窓≤-2%を開けて下げた日数",
    "oil_corr60": "前日の原油(WTI)との60日相関", "sec_ab25_chg5": "業種内で25日線より上の割合の5日変化",
    "jp_us_dd": "日本株と米国株の60日高値からの下落の差",
    "vmax_ret": "直近20日で出来高最大の日の騰落", "vmax_days": "直近20日で出来高最大の日からの経過日数",
    "ret10": "10日騰落",
    "size_sp5": "小型株-大型株の5日騰落の差(時価総額の下位3割-上位3割)", "val_sp20": "低PBR株-高PBR株の20日騰落の差(下位3割-上位3割)",
    "ret20_xr": "20日騰落の全銘柄内の順位(0=最も下げた)",
}
POINT_FEATS = {"jgb2y_chg20", "us10y_chg5", "vix_chg1", "spread_chg20", "vix", "vix_chg5", "us10y_chg20", "jgb10y_chg20", "jgb_slope", "adr25", "lev_z"}
DOW = "月火水木金"


def fmt_cond(f, op, th):
    lab = LABEL.get(f, f)
    if f in ("ma_up", "new_hi", "gc5", "hol_next"):
        return lab
    if f in ("up_streak", "dn_streak"):
        return f"{lab} {op} {th:g}日"
    if f in ("tv20", "mcap"):
        return f"{lab} {op} {th/1e8:g}億円"
    if f in ("close",):
        return f"{lab} {op} {th:g}円"
    if f == "month":
        return f"{int(th)}月"
    if f == "dow":
        return f"{DOW[int(th)]}曜日"
    if f == "dom":
        return f"{lab} {op} {th:g}日"
    if f in ("days_hi20", "lo_days60", "mkt_dn_streak", "min1_days", "gap_dn_cnt20", "vmax_days"):
        return f"{lab} {op} {th:g}日"
    if f == "since_disc":
        return f"{lab} {op} {th:g}日"
    if f == "breadth":
        return f"{lab} {op} {th*100:g}%"
    if f in POINT_FEATS:
        return f"{lab} {op} {th:g}"
    if f == "mkt_seg":
        if op == "<=" and th == 2:
            return "グロース以外(プライム・スタンダード)"
        return {1: "プライム", 2: "スタンダード", 3: "グロース"}.get(int(th), lab) if op == "==" else f"{lab} {op} {th:g}"
    if f == "mfg":
        return "製造業" if op == ">=" else "非製造業"
    if f in ("sec_vratio", "mkt_vratio", "vratio", "rsi14", "clpos", "vdry", "pbr", "vol_exp", "vix_rel60", "sec_crash_pk", "vix_pk20", "sec_wide", "hammer5", "ac20", "px_lvl", "clpos5", "sec_corr60", "sec_tv_rank", "sec_up_frac", "tv5_60", "rng5_20", "us_corr60", "sec_above25", "mkt_above25", "lo_cnt20", "sec_mcap_rank", "oil_corr60", "ret20_xr"):
        return f"{lab} {op} {th:g}"
    return f"{lab} {op} {th*100:+g}%"


def rule_key(conds):
    s = json.dumps(sorted([list(map(str, c)) for c in conds]))
    return hashlib.md5(s.encode()).hexdigest()[:12]


def apply(q, conds):
    m = pd.Series(True, index=q.index)
    m &= ~q["etf"]
    for f, op, th in BASE + conds:
        if f not in q.columns:
            return m & False
        if op == ">=":
            m &= q[f] >= th
        elif op == "<=":
            m &= q[f] <= th
        else:
            m &= q[f] == th
    return m


def dedup_trades(q, mask):
    """同一銘柄は保有期間(HOLD日)中に重ねて買わない"""
    sel = q.loc[mask & q["win"].notna(), ["code", "date", "win", "pnl", "dd10"]]
    if sel.empty:
        return sel
    sel = sel.copy()
    sel["idx"] = sel.index  # 行番号は銘柄内で日付順に連続
    keep = []
    last_code, last_idx = None, -10**9
    for code, idx in zip(sel["code"].values, sel["idx"].values):
        if code != last_code:
            last_code, last_idx = code, -10**9
        if idx - last_idx > HOLD:
            keep.append(True)
            last_idx = idx
        else:
            keep.append(False)
    return sel[np.array(keep)]


def evaluate(q, conds, edges):
    t = dedup_trades(q, apply(q, conds))
    res = {"n": int(len(t))}
    if len(t) == 0:
        return res
    res.update(win=float(t["win"].mean()), pnl=float(t["pnl"].mean()), dd10=float(t["dd10"].mean()))
    splits = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = t[(t["date"] >= a) & (t["date"] < b)]
        splits.append({"n": int(len(s)),
                       "win": float(s["win"].mean()) if len(s) else None,
                       "pnl": float(s["pnl"].mean()) if len(s) else None})
    res["splits"] = splits
    return res


def passes(res, champ):
    if res.get("n", 0) < MIN_N:
        return False
    sp = res["splits"]
    if any(s["n"] < MIN_N_SPLIT for s in sp):
        return False
    if any(s["pnl"] is None or s["pnl"] <= 0 for s in sp):
        return False
    if res["win"] < champ["win"] + 0.01:
        return False
    if min(s["win"] for s in sp) <= min(s["win"] for s in champ["splits"]):
        return False
    return True


def candidates(champ_conds, tested):
    """チャンピオン + 条件1つ追加 / 条件1つ差し替え / 条件1つ削除 / 単独・2条件の新規"""
    singles = [[f, op, th] for f, lst in GRID.items() for op, th in lst]
    out = []
    # 研究担当が手元データの速報で有望と見た組み合わせ（rules/seeds.json）を最優先で検証
    try:
        for s in json.load(open(os.path.join("rules", "seeds.json"), encoding="utf-8")).get("seeds", []):
            cs = [list(c) for c in s.get("conds", [])]
            out.append(cs)
            if champ_conds:
                out.append(champ_conds + [c for c in cs if c[0] not in {x[0] for x in champ_conds}])
    except Exception:
        pass
    used = {c[0] for c in champ_conds}
    for s in singles:                      # 1つ追加
        if s[0] not in used:
            out.append(champ_conds + [s])
    for i, c in enumerate(champ_conds):    # 1つ差し替え・削除
        for s in singles:
            if s[0] == c[0] and s != c:
                out.append(champ_conds[:i] + [s] + champ_conds[i + 1:])
        out.append(champ_conds[:i] + champ_conds[i + 1:])
    for s in singles:                      # 単独
        out.append([s])
    rng = np.random.default_rng(len(tested) + int(dt.datetime.now().strftime("%Y%m%d%H")))
    for _ in range(3000):                  # ランダム2〜3条件（探索の幅）
        k = rng.integers(2, 4)
        pick = rng.choice(len(singles), size=k, replace=False)
        cs = [singles[i] for i in pick]
        if len({c[0] for c in cs}) == k:
            out.append(cs)
    seen, uniq = set(), []
    for cs in out:
        key = rule_key(cs)
        if key not in seen and key not in tested:
            seen.add(key)
            uniq.append(cs)
    return uniq


# --------------------------------------------------------------------------
# 出力
# --------------------------------------------------------------------------

def pct(x, d=1):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.{d}f}%"


def today_signals(q, conds, last_date):
    m = apply(q, conds) & (q["date"] == last_date)
    s = q.loc[m, ["code", "sector", "close", "ret1", "ret20", "vratio", "tv20"]].copy()
    return s.sort_values("tv20", ascending=False).head(30)


def single_report(q, edges, base):
    """要因1つずつの成績（研究用）。基準（条件なし）との差を記録"""
    out = []
    for f, lst in GRID.items():
        for op, th in lst:
            r = evaluate(q, [[f, op, th]], edges)
            if not r.get("n"):
                continue
            sp = [x["win"] for x in r["splits"] if x["win"] is not None]
            out.append({"cond": [f, op, th], "label": fmt_cond(f, op, th), "n": r["n"],
                        "win": r["win"], "pnl": r["pnl"], "lift": r["win"] - base["win"],
                        "min_split_win": min(sp) if sp else None})
    out.sort(key=lambda x: -x["lift"])
    return out


def write_feature_report(state):
    rows = state.get("singles") or []
    md = ["# 要因別の成績（1か月以内+10%到達）", "",
          f"更新 {state['updated']} ／ 基準（流動性条件のみ）勝率 {pct(state['baseline'].get('win'))}", "",
          "| 条件 | 取引数 | 勝率 | 基準との差 | 最悪期間の勝率 | 平均損益 |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['label']} | {r['n']} | {pct(r['win'])} | {r['lift']*100:+.1f}pt | "
                  f"{pct(r['min_split_win'])} | {pct(r['pnl'],2)} |")
    open(os.path.join("rules", "feature_report.md"), "w", encoding="utf-8").write("\n".join(md))


def write_outputs(state, q, last_date):
    os.makedirs("rules", exist_ok=True)
    champ = state["champion"]
    hist = state["history"]
    # Markdown
    md = ["# 売買ルール（1か月以内に +10% 到達の確率を上げる）", "",
          "自動探索 `rule_lab.py` が採用したルールの記録です。投資助言ではありません。",
          "手数料・スリッページ・税は未考慮。", "",
          "## 勝ちの定義", f"- シグナル日の終値で判定 → 翌営業日の始値で買う",
          f"- {HOLD}営業日以内に高値が買値+10%に届けば勝ち（+10%で利確）",
          f"- 届かなければ{HOLD}営業日目の終値で売る", "",
          "## 現在のルール", ""]
    md += [f"- {fmt_cond(*c)}" for c in BASE + champ["conds"]]
    md += ["", f"勝率 {pct(champ['win'])} / 平均損益 {pct(champ['pnl'],2)} / 取引数 {champ['n']} / "
           f"期間中に-10%まで下がった割合 {pct(champ.get('dd10'))}", "", "## 採用履歴", ""]
    for h in reversed(hist):
        md.append(f"### {h['date']}  勝率 {pct(h['win'])}（前 {pct(h['prev_win'])}）")
        md += [f"- {fmt_cond(*c)}" for c in h["conds"]]
        md += [f"- 説明: {h['explain']}", ""]
    open(RULES_MD, "w", encoding="utf-8").write("\n".join(md))

    # HTML（既存サイトのナビ・デザインを流用）
    try:
        from web_report import _page
    except Exception:
        _page = None
    sig = today_signals(q, champ["conds"], last_date)
    def rows_split(sp):
        return "".join(f"<tr><td>期間{i+1}</td><td>{s['n']}</td><td>{pct(s['win'])}</td><td>{pct(s['pnl'],2)}</td></tr>"
                       for i, s in enumerate(sp))
    hist_html = ""
    for h in reversed(hist):
        hist_html += (f"<div class='card'><div class='name'>{h['date']} 採用 "
                      f"<span class='pill up'>勝率 {pct(h['win'])}</span></div>"
                      f"<div class='stat'>前のルール: 勝率 {pct(h['prev_win'])} → {pct(h['win'])}"
                      f"（取引数 {h['n']}、平均損益 {pct(h['pnl'],2)}）</div>"
                      f"<ul>{''.join('<li>'+fmt_cond(*c)+'</li>' for c in h['conds'])}</ul>"
                      f"<div class='stat'>{h['explain']}</div></div>")
    sig_rows = "".join(
        f"<tr><td>{r.code}</td><td>{r.sector}</td><td>{r.close:,.0f}</td><td>{pct(r.ret1)}</td>"
        f"<td>{pct(r.ret20)}</td><td>{r.vratio:.1f}</td></tr>" for r in sig.itertuples())
    base = state.get("baseline", {})
    body = f"""
<h1>🎯 勝率改善ルール（1か月で+10%）</h1>
<div class="meta">最終更新 {state['updated']} ／ データ最終日 {last_date.date()} ／ 検証ルール数 累計 {len(state['tested'])}</div>
<div class="card strong">
  <div class="name">現在のルール <span class="pill up">勝率 {pct(champ['win'])}</span></div>
  <ul>{''.join('<li>'+fmt_cond(*c)+'</li>' for c in BASE + champ['conds'])}</ul>
  <div class="stat">取引数 {champ['n']} ／ 平均損益 {pct(champ['pnl'],2)} ／ 1か月内に一時-10%まで下落した割合 {pct(champ.get('dd10'))}<br>
  比較：条件なし（流動性のみ）の勝率 {pct(base.get('win'))}、平均損益 {pct(base.get('pnl'),2)}</div>
  <div class="tablewrap"><table><thead><tr><th>期間(3分割)</th><th>取引数</th><th>勝率</th><th>平均損益</th></tr></thead>
  <tbody>{rows_split(champ['splits'])}</tbody></table></div>
</div>
<h2>勝ちの定義と売買方法</h2>
<div class="card"><ul>
<li>シグナル日の<b>終値</b>で条件を判定し、<b>翌営業日の始値</b>で買う</li>
<li>買値の <b>+10% に指値</b>（売り）を置く。{HOLD}営業日（約1か月）以内に届けば勝ち</li>
<li>届かなければ <b>{HOLD}営業日目の終値で売却</b>（必ず1か月以内に手仕舞い）</li>
<li>同じ銘柄は保有中に買い増ししない</li>
</ul></div>
<h2>直近データ日（{last_date.date()}）にルールを満たした銘柄</h2>
<div class="tablewrap"><table><thead><tr><th>コード</th><th>業種</th><th>終値</th><th>前日比</th><th>20日騰落</th><th>出来高倍率</th></tr></thead>
<tbody>{sig_rows or '<tr><td colspan=6>該当なし</td></tr>'}</tbody></table></div>
<h2>採用履歴（新しい順）</h2>
{hist_html or '<div class="none">まだ初期ルールから更新されていません。</div>'}
<div class="note">採用基準：取引数{MIN_N}以上・3分割の各期間{MIN_N_SPLIT}以上、全体勝率が現ルール+1pt以上、
最悪期間の勝率も現ルールより上、3期間すべてで平均損益プラス。多数のルールを試すため、偶然の当たりが
混ざる可能性は残ります。手数料・スリッページ・税は未考慮。データの遅延（J-Quantsのプラン）にご注意ください。
本ページは過去データの統計であり、投資助言ではありません。売買の判断と結果責任は利用者にあります。</div>
"""
    html = _page("勝率改善ルール", "rules", body) if _page else f"<!doctype html><meta charset=utf-8><body>{body}"
    open(RULES_HTML, "w", encoding="utf-8").write(html)


def explain(prev_conds, new_conds):
    p = {rule_key([c]): c for c in prev_conds}
    n = {rule_key([c]): c for c in new_conds}
    add = [fmt_cond(*n[k]) for k in n if k not in p]
    rem = [fmt_cond(*p[k]) for k in p if k not in n]
    parts = []
    if add:
        parts.append("追加: " + "、".join(add))
    if rem:
        parts.append("削除: " + "、".join(rem))
    return " ／ ".join(parts) or "変更なし"


def git_push(msg):
    def run(*a):
        r = subprocess.run(["git", *a], capture_output=True, text=True)
        print("git", " ".join(a), "->", r.returncode, (r.stdout + r.stderr).strip()[:300])
        return r.returncode
    run("add", *[p for p in (RULES_MD, RULES_HTML, STATE, os.path.join("rules", "feature_report.md"))
                 if os.path.exists(p)])
    if run("commit", "-m", msg) != 0:
        return
    run("pull", "--rebase", "-X", "theirs", "origin", "main")
    run("push", "origin", "HEAD:main")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--force-write", action="store_true")
    a = ap.parse_args()

    q = add_features(load_data(not a.no_fetch))
    valid = q[q["win"].notna()]
    d0, d1 = valid["date"].min(), valid["date"].max()
    edges = list(pd.date_range(d0, d1 + pd.Timedelta(days=1), periods=4))
    last_date = q["date"].max()
    print(f"データ {q['code'].nunique()}銘柄 {q['date'].min().date()}〜{last_date.date()} / 検証期間 {d0.date()}〜{d1.date()}")

    state = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else None
    base = evaluate(q, [], edges)
    if state is None:
        state = {"champion": {"conds": [], **base}, "history": [], "tested": {}, "baseline": base}
    state["baseline"] = base
    # チャンピオンは最新データで再評価
    champ = {"conds": state["champion"]["conds"], **evaluate(q, state["champion"]["conds"], edges)}
    state["champion"] = champ
    print(f"基準(条件なし) 勝率 {pct(base['win'])} n={base['n']} / 現ルール 勝率 {pct(champ['win'])} n={champ['n']}")

    if state.get("singles_date") != str(last_date.date()):
        state["singles"] = single_report(q, edges, base)
        state["singles_date"] = str(last_date.date())
        state["updated"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        os.makedirs("rules", exist_ok=True)
        write_feature_report(state)
        print(f"要因別レポート更新（{len(state['singles'])}条件）")
    cands = candidates(champ["conds"], state["tested"])[:BATCH]
    print(f"今回テスト: {len(cands)} ルール")
    improved = False
    for i, cs in enumerate(cands):
        r = evaluate(q, cs, edges)
        state["tested"][rule_key(cs)] = round(r.get("win", 0) or 0, 4)
        if passes(r, champ):
            ex = explain(champ["conds"], cs)
            print(f"  ★採用 勝率 {pct(champ['win'])} → {pct(r['win'])} n={r['n']}  {ex}")
            state["history"].append({"date": dt.date.today().isoformat(), "conds": cs, "win": r["win"],
                                     "prev_win": champ["win"], "n": r["n"], "pnl": r["pnl"], "explain": ex})
            champ = {"conds": cs, **r}
            state["champion"] = champ
            improved = True
    state["updated"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    os.makedirs("rules", exist_ok=True)
    json.dump(state, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    if improved or a.force_write or not os.path.exists(RULES_HTML):
        write_outputs(state, q, last_date)
        print("rules/RULES.md・docs/rules.html を更新")
    if a.push:
        if improved:
            git_push(f"ルール更新: 1か月+10%勝率 {pct(champ['win'])}（n={champ['n']}）")
        else:
            git_push(f"ルール探索: 改善なし（累計 {len(state['tested'])} 件検証）")
    print(f"RESULT improved={improved} win={pct(champ['win'])} n={champ['n']} tested_total={len(state['tested'])}")


if __name__ == "__main__":
    main()
