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


def _shrink(q: pd.DataFrame) -> pd.DataFrame:
    """メモリ節約：小数の列を float32 に（5年分×多数の要因で GitHub のサーバーのメモリ16GBを超えないように）"""
    for c in q.columns[q.dtypes == "float64"]:
        q[c] = q[c].astype("float32")
    return q


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
            if name == "gold":   # 仮説(53回目14周目 担当1)：金との相関の計算用（当日のETF終値＝t 終値で確定。GRIDには入れない）
                ctx["gold_ret1"] = e["close"].reindex(dates).pct_change(1)
            if name == "nk":     # 仮説(53回目19周目 担当1)：日経225(ETF)の20日実現ボラ(年率%)。t 終値で確定（VIXとの差の計算用）
                ctx["nk_rv20"] = e["close"].reindex(dates).pct_change(1).rolling(20, min_periods=15).std() * np.sqrt(252) * 100
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
        for col in ("vix", "us10y", "usdjpy", "wti", "sp500", "nasdaq", "jgb10y", "jgb2y"):
            if col not in m.columns:            # 取得できなかった指標は空欄（その要因の条件は該当なしになる）
                m[col] = np.nan
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
        x["jgb10y_chg1"] = m["jgb10y"].diff()                              # 仮説(53回目2周目 担当1)：金利感応度の計算用（GRIDには入れない）
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
        # 仮説(2026-11-22 43回目)：米国のグロース偏重（ナスダック-S&P500の5日騰落の差）・ドル円の60日平均からの乖離（円高・円安の水準）
        x["ndx_spx5"] = m["nasdaq"].pct_change(5) - m["sp500"].pct_change(5)
        x["usdjpy_dev60"] = m["usdjpy"] / m["usdjpy"].rolling(60, min_periods=40).mean() - 1
        # 仮説(2026-11-22 45回目)：日米10年金利差の水準（前日）＝円キャリーの巻き戻しが起きやすい局面か
        x["rate_sp"] = m["us10y"] - m["jgb10y"]
        # 仮説(53回目7周目 担当1)：リスクオフの総合点（5日で 円高・VIX上昇・米10年金利低下・S&P500下落 の同時発生数 0〜4）と
        # その和らぎ（当日の点−直近5日の最大。マイナス=リスクオフが和らぎつつある）。前日までの確定値
        _mf = m[["usdjpy", "vix", "us10y", "sp500"]].ffill()
        _ro = ((_mf["usdjpy"].pct_change(5) < 0).astype(float) + (_mf["vix"].diff(5) > 0).astype(float)
               + (_mf["us10y"].diff(5) < 0).astype(float) + (_mf["sp500"].pct_change(5) < 0).astype(float))
        _ro = _ro.where(_mf.notna().all(axis=1) & _mf.shift(5).notna().all(axis=1))
        x["riskoff5"] = _ro
        x["riskoff_off"] = _ro - _ro.rolling(5, min_periods=3).max()
        # 日付 t には「t より前の日付」の値を使う
        x.index = x.index + pd.Timedelta(days=1)
        x = x[~x.index.duplicated(keep="last")]
        al = x.reindex(x.index.union(dates)).ffill().reindex(dates)
        ctx = ctx.join(al)
    # 仮説(53回目19周目 担当1)：恐怖の割高度＝VIX(前日) − 日経225の20日実現ボラ。プラス=米国の不安が日本の実際の値動きより大きい、
    # マイナス=日本だけ実際の値動きが荒れている（日本固有の投げ）
    ctx["vix_rv_gap"] = (ctx["vix"] - ctx["nk_rv20"]) if {"vix", "nk_rv20"} <= set(ctx.columns) else np.nan
    ctx["month"] = ctx.index.month
    ctx["dow"] = ctx.index.dayofweek
    ctx["dom"] = ctx.index.day                                         # 仮説(2026-11-21 2回目)：月末・月初効果
    # 仮説(2026-11-21 9回目)：連休前＝次の営業日までの暦日数（取引所の休日は事前に公表されているので先読みではない）
    ds = pd.Series(dates, index=dates)
    gapd = (ds.shift(-1) - ds).dt.days
    ctx["hol_next"] = ((gapd >= 4) | ((ctx.index.dayofweek < 4) & (gapd >= 2))).astype(float).where(gapd.notna())
    # 仮説(2026-11-22 41回目)：SQ（毎月第2金曜）までの暦日数＝需給の節目、3月・9月末（配当の権利付き最終日）までの暦日数＝
    # 保有期間中に権利落ちを含むか。どちらも暦で事前に決まっているので先読みではない
    def _sq(y, mth):
        d = pd.Timestamp(year=y, month=mth, day=1)
        return d + pd.Timedelta(days=(4 - d.dayofweek) % 7 + 7)
    sqd, dvd = [], []
    for t in ctx.index:
        nx = _sq(t.year, t.month)
        if nx < t.normalize():
            y2, m2 = (t.year + 1, 1) if t.month == 12 else (t.year, t.month + 1)
            nx = _sq(y2, m2)
        sqd.append((nx - t.normalize()).days)
        cands = [pd.Timestamp(year=t.year + k, month=mm, day=1) + pd.offsets.MonthEnd(0) for k in (0, 1) for mm in (3, 9)]
        dvd.append(min((c_ - t.normalize()).days for c_ in cands if c_ >= t.normalize()))
    ctx["sq_days"] = sqd
    ctx["div_days"] = dvd
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
            # 仮説(2026-11-22 46回目)：BPSの前回の決算からの変化（純資産の積み上がり＝利益の蓄積・自社株買い）。分割(±50%超)は除外
            b = b.sort_values(["code", "disclosed_date"])
            b["bps_chg"] = b.groupby("code")["bps"].pct_change()
            b.loc[b["bps_chg"].abs() > 0.5, "bps_chg"] = np.nan
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
            # 仮説(2026-11-22 52回目)：予想売上高の利回り（会社予想の売上高/時価総額＝PSRの逆数。赤字でも測れる割安さ）
            q["sales_yld"] = (q["fc_ns"].where(q["fc_ns"] > 0) / q["mcap"].replace(0, np.nan)).where(~old)
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
    # 仮説(53回目19周目 担当1)：上げの日・下げの日に分けたベータ（市場の中央値が上げた日/下げた日だけで回帰、60日・各15日以上）と
    # その差（プラス=下げには鈍く上げには敏感＝反発の恩恵を受けやすい）、直近20日の市場が上げた日の超過騰落の平均（戻る日の強さ）
    def _rollc(s, w=60, mp=15):
        return s.groupby(gc).transform(lambda z: z.rolling(w, min_periods=mp).mean())
    for _nm, _mk in (("beta_up60", m1 > 0), ("beta_dn60", m1 < 0)):
        _ok = _mk & x1.notna()
        _xa, _ma = x1.where(_ok), m1.where(_ok)
        _cv = _rollc(_xa * _ma) - _rollc(_xa) * _rollc(_ma)
        _vr = _rollc(_ma * _ma) - _rollc(_ma) ** 2
        q[_nm] = (_cv / _vr.where(_vr > 1e-8)).clip(-3, 5)
    q["beta_asym"] = q["beta_up60"] - q["beta_dn60"]
    q["up_day_xs20"] = _rollc((x1 - m1).where((m1 > 0) & x1.notna()), 20, 5)
    # 仮説(53回目20周目 担当1)：下げた日の超過騰落の平均（下げの日に余計に売られたか）、上げの日−下げの日の差、
    # 業種内の up_day_xs20 の中央値（業種ぐるみで戻っていないか）と銘柄−業種の差
    q["dn_day_xs20"] = _rollc((x1 - m1).where((m1 < 0) & x1.notna()), 20, 5)
    q["updn_xs20"] = q["up_day_xs20"] - q["dn_day_xs20"]
    q["sec_up_xs20"] = q.groupby(["date", "sector"])["up_day_xs20"].transform("median")
    q["up_xs_rel"] = q["up_day_xs20"] - q["sec_up_xs20"]
    if "usdjpy_ret1" in q.columns:
        ok = x1.notna() & q["usdjpy_ret1"].notna()
        xa, fa = x1.where(ok), q["usdjpy_ret1"].where(ok)
        cxy = _roll(xa * fa) - _roll(xa) * _roll(fa)
        sx = (_roll(xa * xa) - _roll(xa) ** 2).clip(lower=0) ** 0.5
        sf = (_roll(fa * fa) - _roll(fa) ** 2).clip(lower=0) ** 0.5
        q["fx_corr60"] = cxy / (sx * sf).replace(0, np.nan)
        # 仮説(2026-11-22 53回目2周目 担当1)：為替の追い風・向かい風＝為替感応度×ドル円の20日変化（マイナス=為替の動きで売られた）
        if "usdjpy_ret20" in q.columns:
            q["fx_tail20"] = q["fx_corr60"] * q["usdjpy_ret20"]
        # 仮説(53回目3周目 担当1)：短期(5日)の為替の追い風
        if "usdjpy_ret5" in q.columns:
            q["fx_tail5"] = q["fx_corr60"] * q["usdjpy_ret5"]
    # 仮説(2026-11-22 53回目14周目 担当1)：金の追い風＝金(ETF)との60日相関×金の20日騰落（安全資産と連動する銘柄か）
    if {"gold_ret1", "gold_ret20"} <= set(q.columns):
        ok = x1.notna() & q["gold_ret1"].notna()
        xa, fa = x1.where(ok), q["gold_ret1"].where(ok)
        cxy = _roll(xa * fa) - _roll(xa) * _roll(fa)
        sx = (_roll(xa * xa) - _roll(xa) ** 2).clip(lower=0) ** 0.5
        sf = (_roll(fa * fa) - _roll(fa) ** 2).clip(lower=0) ** 0.5
        q["gold_corr60"] = cxy / (sx * sf).replace(0, np.nan)
        q["gold_tail20"] = q["gold_corr60"] * q["gold_ret20"]
    else:
        q["gold_corr60"] = q["gold_tail20"] = np.nan
    # 仮説(2026-11-22 53回目2周目 担当1)：金利感応度（日本10年金利の前日の変化との60日相関）と金利の追い風＝感応度×10年金利の20日変化
    if "jgb10y_chg1" in q.columns:
        ok = x1.notna() & q["jgb10y_chg1"].notna()
        xa, fa = x1.where(ok), q["jgb10y_chg1"].where(ok)
        cxy = _roll(xa * fa) - _roll(xa) * _roll(fa)
        sx = (_roll(xa * xa) - _roll(xa) ** 2).clip(lower=0) ** 0.5
        sf = (_roll(fa * fa) - _roll(fa) ** 2).clip(lower=0) ** 0.5
        q["rate_corr60"] = cxy / (sx * sf).replace(0, np.nan)
        if "jgb10y_chg20" in q.columns:
            q["rate_tail20"] = q["rate_corr60"] * q["jgb10y_chg20"]
    # 仮説(2026-11-22 33回目)：前夜の米国株(S&P500)との60日相関（外需・米国連動株か）。sp500_ret1 は前日までの確定値
    if "sp500_ret1" in q.columns:
        ok = x1.notna() & q["sp500_ret1"].notna()
        xa, fa = x1.where(ok), q["sp500_ret1"].where(ok)
        cxy = _roll(xa * fa) - _roll(xa) * _roll(fa)
        sx = (_roll(xa * xa) - _roll(xa) ** 2).clip(lower=0) ** 0.5
        sf = (_roll(fa * fa) - _roll(fa) ** 2).clip(lower=0) ** 0.5
        q["us_corr60"] = cxy / (sx * sf).replace(0, np.nan)
        # 仮説(53回目5周目 担当1)：米国株の追い風＝米国株との相関×S&P500の20日騰落（前日までの確定値）
        q["us_tail20"] = q["us_corr60"] * q["sp500_ret20"] if "sp500_ret20" in q.columns else np.nan
    else:
        q["us_corr60"] = np.nan
        q["us_tail20"] = np.nan
    # 仮説(2026-11-22 38回目)：前日の原油(WTI)との60日相関（原油高で売られる/買われる銘柄か）。wti_ret1 は前日までの確定値
    if "wti_ret1" in q.columns:
        ok = x1.notna() & q["wti_ret1"].notna()
        xa, fa = x1.where(ok), q["wti_ret1"].where(ok)
        cxy = _roll(xa * fa) - _roll(xa) * _roll(fa)
        sx = (_roll(xa * xa) - _roll(xa) ** 2).clip(lower=0) ** 0.5
        sf = (_roll(fa * fa) - _roll(fa) ** 2).clip(lower=0) ** 0.5
        q["oil_corr60"] = cxy / (sx * sf).replace(0, np.nan)
        # 仮説(53回目3周目 担当1)：原油の追い風＝原油との相関×WTIの20日騰落（前日までの確定値）
        q["oil_tail20"] = q["oil_corr60"] * q["wti_ret20"] if "wti_ret20" in q.columns else np.nan
    else:
        q["oil_corr60"] = np.nan
        q["oil_tail20"] = np.nan
    # 仮説(2026-11-21 19回目)：市場要因を除いた20日騰落＝20日騰落 − 60日ベータ×市場全体の20日騰落（個別の悪材料による下げ）
    q["idio20"] = q["ret20"] - q["beta60"].clip(-1, 3) * q["mkt20"]
    # 仮説(53回目9周目 担当1)：下げの原因の割合＝20日の下げ(≤-5%の銘柄のみ)のうち 市場(ベータ×市場の20日騰落)・業種(業種の20日騰落の中央値)で説明できる割合
    _dn20 = q["ret20"].where(q["ret20"] <= -0.05)
    q["mkt_share"] = (q["beta60"].clip(-1, 3) * q["mkt20"] / _dn20).clip(-1, 2)
    q["sec_share"] = ((q["ret20"] - q["rel_sec20"]) / _dn20).clip(-1, 2) if "rel_sec20" in q.columns else np.nan
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
    # 仮説(2026-11-22 担当1 23周目)：200日線乖離（長期のトレンドに対する位置）と、市場全体の200日線乖離の中央値
    # （長期の下げ相場か短期の調整か）。t 終値で確定。200日分の履歴が無い銘柄は欠損
    q["dev200"] = c / g["close"].transform(lambda s: s.rolling(200, min_periods=150).mean()) - 1
    q["mkt_dev200"] = q["dev200"].groupby(q["date"]).transform("median")
    q["dev200_rel"] = q["dev200"] - q["mkt_dev200"]
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
    q = _shrink(q)
    q = _shrink(add_context(q))
    # 仮説(2026-11-21 20回目4周目)：日本株だけが売られた度合い＝市場全体の5日騰落(中央値) − 米国株(S&P500)の5日騰落(前日)
    q["jp_rel5"] = q["mkt5"] - q["sp500_ret5"] if "sp500_ret5" in q.columns else np.nan
    # 仮説(2026-11-22 26回目2周目)：市場全体の連続下落日数（全銘柄の前日比の中央値がマイナスの日の連続）・
    # 60日安値をつけた銘柄の割合（市場全体の投げ売り）・業種の5日騰落−市場全体の5日騰落（業種のローテーション）。t 終値時点で確定
    md1 = q.groupby("date")["ret1"].median().sort_index()
    dn_m = (md1 < 0).astype(int)
    q = q.merge(dn_m.groupby((dn_m == 0).cumsum()).cumsum().astype(float).rename("mkt_dn_streak"), on="date", how="left")
    atlo = (q["low"] <= q["lo60"]).astype(float).where(q["lo60"].notna())
    q = q.merge(atlo.groupby(q["date"]).mean().rename("mkt_lo60_frac"), on="date", how="left")
    # 仮説(2026-11-22 46回目)：市場全体の60日安値割合の5日変化（投げ売りが収まりつつあるか）。t 終値で確定
    _ml = q.groupby("date")["mkt_lo60_frac"].first().sort_index()
    q = q.merge((_ml - _ml.shift(5)).rename("mkt_lo60_chg5"), on="date", how="left")
    q["sec_rel5"] = q["sec5"] - q["mkt5"] if "sec5" in q.columns else np.nan
    # 仮説(2026-11-22 27回目)：日本の市場全体の60日高値からの下落（全銘柄の前日比の中央値を積み上げた指数）・
    # 業種内での20日騰落の順位（0=業種で最も下げた銘柄）・直近5日で下ヒゲ≥2%をつけた日数（下値での買い支え）。t 終値時点で確定
    mix = (1 + md1.fillna(0)).cumprod()
    q = q.merge((mix / mix.rolling(60, min_periods=40).max() - 1).rename("mkt_dd60"), on="date", how="left")
    # 仮説(2026-11-22 担当2)：市場全体の下げが続いている長さ＝同じ指数の120日高値からの経過営業日数と120日高値からの下落
    # （数日の急落か、何か月も続く下げ相場か。採用ルールの弱点＝長い下げ相場が検証期間にほぼ無い点の確認用）。t 終値時点で確定
    _amax = mix.rolling(120, min_periods=60).apply(lambda a: len(a) - 1 - int(np.argmax(a)), raw=True)
    q = q.merge(_amax.rename("mkt_hi_days"), on="date", how="left")
    q = q.merge((mix / mix.rolling(120, min_periods=60).max() - 1).rename("mkt_dd120"), on="date", how="left")
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
    # 仮説(2026-11-22 41回目)：業種全体の52週高値からの位置（業種の中央値）と、銘柄-業種の差＝業種ごと深く下げたのか銘柄だけか
    q["sec_from_hi"] = q.groupby(["date", "sector"])["from_hi"].transform("median")
    q["from_hi_rel"] = q["from_hi"] - q["sec_from_hi"]
    # 仮説(2026-11-22 42回目)：2日RSI（短期の売られすぎ）・RSIの強気ダイバージェンス（20日安値圏なのにRSI14が20日の最低から戻っている）・
    # その日の市場全体の騰落（中央値、全面安の日に買うか）。すべて t 終値時点で確定
    d2 = q["ret1"].fillna(0)
    up2 = d2.clip(lower=0).groupby(q["code"], sort=False).transform(lambda s: s.rolling(2, min_periods=2).mean())
    dn2 = (-d2.clip(upper=0)).groupby(q["code"], sort=False).transform(lambda s: s.rolling(2, min_periods=2).mean())
    q["rsi2"] = (100 * up2 / (up2 + dn2).replace(0, np.nan))
    lo20c = q.groupby("code", sort=False)["close"].transform(lambda s: s.rolling(20, min_periods=20).min())
    rmin = q.groupby("code", sort=False)["rsi14"].transform(lambda s: s.rolling(20, min_periods=20).min())
    q["rsi_div"] = (q["rsi14"] - rmin).where(q["close"] <= lo20c * 1.03)
    q = q.merge(q.groupby("date")["ret1"].median().rename("mkt_ret1"), on="date", how="left")
    # 仮説(2026-11-22 43回目)：業種内の2日RSIの中央値（業種ぐるみの短期の売られすぎ）。t 終値時点で確定
    q["sec_rsi2"] = q.groupby(["date", "sector"])["rsi2"].transform("median")
    # 仮説(2026-11-22 44回目)：業種の連続下落日数（業種の前日比の中央値がマイナスの日の連続）・当日の騰落を20日ボラで割った値
    # （その日のショックの大きさ）・業種内で60日安値をつけた銘柄の割合（業種ぐるみの投げ売り）。すべて t 終値時点で確定
    if "sector" in q.columns and "sec_ret1" in q.columns:
        sr1 = q.groupby(["date", "sector"])["sec_ret1"].first().sort_index()
        sdn = (sr1 < 0).astype(int)
        sgrp = (sdn == 0).groupby(level=1).cumsum()
        sstk = sdn.groupby([sdn.index.get_level_values(1), sgrp]).cumsum().astype(float).rename("sec_dn_streak").reset_index()
        q = q.merge(sstk, on=["date", "sector"], how="left")
        q["sec_lo60_frac"] = (q["low"] <= q["lo60"]).astype(float).where(q["lo60"].notna()).groupby([q["date"], q["sector"]]).transform("mean")
    else:
        q["sec_dn_streak"] = q["sec_lo60_frac"] = np.nan
    q["ret1_z"] = q["ret1"] / q["vol20"].replace(0, np.nan)
    # 仮説(2026-11-22 45回目)：業種の急落状態（業種内の20日騰落-15%以下の割合≥15%）の連続日数＝業種ぐるみの急落の初期か長期化か、
    # 業種内の5日騰落の順位（0=最も下げた）＝業種の中でも直近さらに売られたか。どちらも t 終値時点で確定
    if "sector" in q.columns:
        scs2 = q.groupby(["date", "sector"])["sec_crash"].first().sort_index()
        son = (scs2 >= 0.15).astype(int)
        sg2 = (son == 0).groupby(level=1).cumsum()
        scd = son.groupby([son.index.get_level_values(1), sg2]).cumsum().astype(float).where(scs2.notna()).rename("sec_crash_days").reset_index()
        q = q.merge(scd, on=["date", "sector"], how="left")
        q["sec_rank5"] = q.groupby(["date", "sector"])["ret5"].rank(pct=True)
    else:
        q["sec_crash_days"] = q["sec_rank5"] = np.nan
    # 仮説(2026-11-22 46回目)：業種内の営業利益の前年同期比の中央値（業種の業績の向き）。発表済みの決算のみ
    q["sec_op_yoy"] = q.groupby(["date", "sector"])["op_yoy"].transform("median") if "op_yoy" in q.columns else np.nan
    # 仮説(2026-11-22 46回目2周目)：業種の5日騰落の業種間の順位（0=最も下げた業種、短期の業種ローテーション）・
    # 20日高値（終値）からの下落率（直近の戻り高値からの深さ）。t 終値時点で確定
    if "sector" in q.columns:
        s5r = q.groupby(["date", "sector"])["sec5"].first()
        q["sec5_rank"] = s5r.groupby(level=0).rank(pct=True).reindex(pd.MultiIndex.from_arrays([q["date"], q["sector"]])).values
    else:
        q["sec5_rank"] = np.nan
    q["dd20"] = q["close"] / q.groupby("code", sort=False)["close"].transform(lambda s: s.rolling(20, min_periods=20).max()) - 1
    # 仮説(53回目22周目 担当1)：業種のローテーション＝ディフェンシブ業種（食料品・医薬品・電気ガス・陸運）への逃避度。
    # ディフェンシブ−それ以外の20日・5日騰落の差（中央値）。5日の差がマイナス＝逃避が巻き戻り始めた。t 終値で確定
    if "sector" in q.columns:
        _df = q["sector"].isin(["食料品", "医薬品", "電気･ガス業", "陸運業"])
        q["is_def"] = _df.astype(float).where(q["sector"].notna() & (q["sector"] != "不明"))
        for _w in (20, 5):
            _a = q[f"ret{_w}"].where(q["is_def"] == 1).groupby(q["date"]).transform("median")
            _b = q[f"ret{_w}"].where(q["is_def"] == 0).groupby(q["date"]).transform("median")
            q[f"def_sp{_w}"] = _a - _b
    else:
        q["is_def"] = q["def_sp20"] = q["def_sp5"] = np.nan
    # 仮説(2026-11-22 47回目)：製造業-非製造業の20日騰落の差（景気敏感株のローテーション）・業種の20日騰落の業種間のばらつき
    # （業種ローテーションの激しさ）・業種内のボラ拡大(5日/60日)の中央値（業種ぐるみの変動の拡大）。すべて t 終値時点で確定
    if "sector" in q.columns and "mfg" in q.columns:
        _m = q["ret20"].where(q["mfg"] == 1).groupby(q["date"]).transform("median")
        _n = q["ret20"].where(q["mfg"] == 0).groupby(q["date"]).transform("median")
        q["mfg_sp20"] = _m - _n
        _sd = q.groupby(["date", "sector"])["sec20"].first().groupby(level=0).std().rename("xsec_disp20")
        q = q.merge(_sd, on="date", how="left")
        q["sec_vol_exp"] = q.groupby(["date", "sector"])["vol_exp"].transform("median")
        # 仮説(2026-11-22 担当2)：業種間のばらつき（採用ルールの柱）の派生。5日での広がり・60日平均に対する倍率
        # （ばらつきが急に広がった日か）・その業種の20日騰落の業種間の順位（0=最も下げた業種）。t 終値時点で確定
        _sd = _sd.sort_index()
        q["xsec_disp_chg5"] = q["date"].map(_sd - _sd.shift(5))
        q["xsec_disp_rel60"] = q["date"].map(_sd / _sd.rolling(60, min_periods=40).mean())
        _rk = q.groupby(["date", "sector"])["sec20"].first().groupby(level=0).rank(pct=True).rename("sec20_rank")
        q = q.merge(_rk.reset_index(), on=["date", "sector"], how="left")
    else:
        q["mfg_sp20"] = q["xsec_disp20"] = q["sec_vol_exp"] = np.nan
        q["xsec_disp_chg5"] = q["xsec_disp_rel60"] = q["sec20_rank"] = np.nan
    # 仮説(2026-11-22 48回目)：陽の包み足（前日陰線を当日陽線が包む＝売りの一巡）・安値の連続切り下げ日数・
    # 直下の価格の節目（100円・1000円などの区切り）からの距離（節目が下値の支えになるか）。すべて t 終値時点で確定
    _g = q.groupby("code", sort=False)
    _po, _pc, _pl = _g["open"].shift(1), _g["close"].shift(1), _g["low"].shift(1)
    q["engulf"] = ((_pc < _po) & (q["close"] > q["open"]) & (q["close"] >= _po) & (q["open"] <= _pc)).astype(float)
    q.loc[_pc.isna(), "engulf"] = np.nan
    _ll = (q["low"] < _pl).astype(int)
    q["ll_streak"] = _ll.groupby([q["code"], (_ll == 0).groupby(q["code"]).cumsum()]).cumsum()
    _unit = 10 ** np.floor(np.log10(q["close"].where(q["close"] > 0)))
    q["rnd_up"] = q["close"] / (np.floor(q["close"] / _unit) * _unit) - 1
    # 仮説(2026-11-22 49回目)：業種内の大型株（時価総額上位3割）の5日騰落の中央値-業種全体の中央値（業種のリーダーが先に戻っているか）・
    # 2度目の急落か（t-60〜t-21 の間に20日騰落-15%以下の日があったか）。どちらも t 終値時点で確定
    if "sec_mcap_rank" in q.columns and "sector" in q.columns and q["sec_mcap_rank"].notna().any():
        _big5 = q["ret5"].where(q["sec_mcap_rank"] >= 0.7).groupby([q["date"], q["sector"]]).transform("median")
        q["sec_lead5"] = _big5 - q.groupby(["date", "sector"])["ret5"].transform("median")
    else:
        q["sec_lead5"] = np.nan
    _cr = (q["ret20"] <= -0.15).astype(float).where(q["ret20"].notna())
    q["prior_crash"] = _cr.groupby(q["code"], sort=False).transform(lambda s: s.shift(20).rolling(40, min_periods=20).max())
    # 仮説(2026-11-22 50回目)：20日の値動きの効率（|20日騰落|/日次騰落の絶対値の合計。1に近い=一方向に整然と下げた、0に近い=上下に振れながら）・
    # 20日の出来高加重平均価格(VWAP)からの乖離（直近20日に買った人の平均取得価格からの距離＝含み損の大きさ）。どちらも t 終値時点で確定
    _gc = q.groupby("code", sort=False)
    _r1a = _gc["close"].pct_change().abs()
    _sabs = _r1a.groupby(q["code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).sum())
    _lr20 = np.log(q["close"] / _gc["close"].shift(20)).abs()
    q["eff20"] = (_lr20 / _sabs).where(_sabs > 0)
    _pv = (q["close"] * q["volume"]).groupby(q["code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).sum())
    _vv = q["volume"].groupby(q["code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).sum())
    q["vwap_dev20"] = (q["close"] / (_pv / _vv) - 1).where(_vv > 0)
    # 仮説(2026-11-22 51回目)：業種の投げ売りのピーク越え（業種内の60日安値割合−直近5日の最大値。マイナス=投げ売りが峠を越えた）・
    # 市場全体のボラ拡大（全銘柄の前日比の中央値の5日標準偏差/60日標準偏差）・高値の連続切り下げ日数。すべて t 終値時点で確定
    if "sec_lo60_frac" in q.columns and q["sec_lo60_frac"].notna().any():
        _st = q[["date", "sector", "sec_lo60_frac"]].dropna().drop_duplicates(["date", "sector"]).sort_values("date")
        _st["_pk5"] = _st.groupby("sector")["sec_lo60_frac"].transform(lambda s: s.rolling(5, min_periods=3).max())
        _st = (_st["sec_lo60_frac"] - _st["_pk5"]).set_axis(pd.MultiIndex.from_frame(_st[["date", "sector"]]))
        q["sec_lo60_off"] = pd.MultiIndex.from_frame(q[["date", "sector"]]).map(_st).astype(float)
    else:
        q["sec_lo60_off"] = np.nan
    _m1 = q.groupby("date")["ret1"].median().sort_index()
    _rvx = _m1.rolling(5, min_periods=5).std() / _m1.rolling(60, min_periods=40).std()
    q["mkt_rv_exp"] = q["date"].map(_rvx)
    _ph = q.groupby("code", sort=False)["high"].shift(1)
    _lh = (q["high"] < _ph).astype(int)
    q["lh_streak"] = _lh.groupby([q["code"], (_lh == 0).groupby(q["code"]).cumsum()]).cumsum()
    # 仮説(2026-11-22 51回目3周目)：市場全体のボラ拡大の峠越え（当日の値−直近5日の最大値。マイナス=乱高下が収まりつつある）・
    # 市場全体の下げの速さ（5日騰落の中央値/20日騰落の中央値。市場が20日で2%以上下げている日のみ。1に近い=直近5日で一気に下げた）
    _rvo = _rvx - _rvx.rolling(5, min_periods=3).max()
    q["mkt_rv_off"] = q["date"].map(_rvo)
    if {"mkt5", "mkt20"} <= set(q.columns):
        q["mkt_speed"] = (q["mkt5"] / q["mkt20"]).where(q["mkt20"] <= -0.02)
    else:
        q["mkt_speed"] = np.nan
    # 仮説(2026-11-22 52回目)：市場全体の売買代金のうち下落銘柄が占める割合（件数でなく金額で見た売り圧力）の当日値と5日平均
    _tv = q["turnover_value"] if "turnover_value" in q.columns else q["close"] * q["volume"]
    _dn = _tv.where(q["ret1"] < 0, 0.0).groupby(q["date"]).sum() / _tv.groupby(q["date"]).sum().replace(0, np.nan)
    _dn = _dn.sort_index()
    q["mkt_dn_tv"] = q["date"].map(_dn)
    q["mkt_dn_tv5"] = q["date"].map(_dn.rolling(5, min_periods=4).mean())
    # 仮説(2026-11-22 担当2)：市場との連動の変化。市場全体の日次騰落(中央値)との20日相関・60日相関の差
    # （急落で個別株が市場と一緒に売られた＝連れ安か）と、全銘柄の20日相関の平均（相場全体が一斉に動いている度合い）。t 終値時点で確定
    if "mkt_ret1" in q.columns:
        _mr = q["mkt_ret1"]
        def _cr(w, mp):
            rr = lambda s: s.groupby(q["code"]).transform(lambda x: x.rolling(w, min_periods=mp).mean())
            c = rr(q["ret1"] * _mr) - rr(q["ret1"]) * rr(_mr)
            v = (rr(q["ret1"] ** 2) - rr(q["ret1"]) ** 2) * (rr(_mr ** 2) - rr(_mr) ** 2)
            return c / np.sqrt(v.where(v > 0))
        q["mkt_corr20"] = _cr(20, 15)
        q["mkt_corr_chg"] = q["mkt_corr20"] - _cr(60, 40)
        q["mkt_corr_avg"] = q.groupby("date")["mkt_corr20"].transform("mean")
    else:
        q["mkt_corr20"] = q["mkt_corr_chg"] = q["mkt_corr_avg"] = np.nan
    # 仮説(2026-11-22 53回目 担当1)：季節性・投資家心理
    #  disc_season 市場全体で直近10日以内に決算を発表した銘柄の割合（決算シーズンの最中か）
    #  mkt_disc_ret 直近10日以内に発表した銘柄の決算反応(disc_ret)の中央値（その決算シーズンの地合い。20銘柄以上の日のみ）
    #  round_up 上の切りのよい株価（桁の半分単位：1234円→1500円）までの距離（+10%の手前に節目があるか）
    if "since_disc" in q.columns:
        _rec = (q["since_disc"] <= 10)
        q["disc_season"] = q["date"].map(_rec.groupby(q["date"]).mean())
        if "disc_ret" in q.columns:
            _dr = q["disc_ret"].where(_rec)
            _cnt = _dr.groupby(q["date"]).count()
            q["mkt_disc_ret"] = q["date"].map(_dr.groupby(q["date"]).median().where(_cnt >= 20))
            # 仮説(53回目6周目 担当1)：同業他社の決算反応の波及＝同じ業種で直近10日に発表した銘柄の決算反応の中央値（3銘柄以上）
            _g = [q["date"], q["sector"]]
            _sc = _dr.groupby(_g).transform("count")
            q["sec_disc_ret"] = _dr.groupby(_g).transform("median").where(_sc >= 3)
            q["sec_disc_ret"] = q.groupby(["date", "sector"])["sec_disc_ret"].transform("max")
        else:
            q["mkt_disc_ret"] = np.nan
            q["sec_disc_ret"] = np.nan
    else:
        q["sec_disc_ret"] = np.nan
        q["disc_season"] = np.nan
        q["mkt_disc_ret"] = np.nan
    # 仮説(53回目4周目 担当1)：業種全体の為替の追い風（業種内の fx_tail20 の中央値）と、銘柄の追い風−業種の中央値
    if "fx_tail20" in q.columns:
        q["sec_fx_tail20"] = q.groupby(["date", "sector"])["fx_tail20"].transform("median")
        q["sec_us_tail20"] = q.groupby(["date", "sector"])["us_tail20"].transform("median") if "us_tail20" in q.columns else np.nan
        q["fx_tail_rel"] = q["fx_tail20"] - q["sec_fx_tail20"]
    else:
        q["sec_fx_tail20"] = q["fx_tail_rel"] = q["sec_us_tail20"] = np.nan
    # 仮説(53回目10周目 担当1)：月替わりの季節性＝前月の月間騰落（前月末/前々月末の終値。当月に入れば確定済み）と月初来騰落（前月末→t）
    _ym = q["date"].dt.year * 12 + q["date"].dt.month
    _me = q.assign(_ym=_ym).groupby(["code", "_ym"], sort=False)["close"].last().rename("_mc").reset_index()
    _me = _me.sort_values(["code", "_ym"])
    _me["_pm_ret"] = _me.groupby("code")["_mc"].pct_change()
    _me["_ym"] = _me["_ym"] + 1      # 翌月から使う
    _mm = q[["code"]].assign(_ym=_ym).merge(_me[["code", "_ym", "_mc", "_pm_ret"]], on=["code", "_ym"], how="left")
    q["pm_ret"] = _mm["_pm_ret"].values
    q["mtd_ret"] = (q["close"] / _mm["_mc"].values - 1)
    # 仮説(53回目12周目 担当1)：値幅制限（東証の制限値幅表）＝前日終値の価格帯ごとの制限値幅の95%以上動いて引けた日数（直近20日）。
    # 分割の調整後の株価なので厳密ではない（目安）。t 終値時点で確定
    _pc = q.groupby("code", sort=False)["close"].shift(1)
    _br = [100, 200, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000, 50000, 70000, 100000]
    _lw = [30, 50, 80, 100, 150, 300, 400, 500, 700, 1000, 1500, 3000, 4000, 5000, 7000, 10000, 15000, 30000]
    _lim = pd.Series(np.array(_lw, dtype=float)[np.searchsorted(_br, _pc.fillna(0).values, side="right")], index=q.index).where(_pc > 0)
    _d = q["close"] - _pc
    _ldn = (_d <= -0.95 * _lim).astype(float).where(_lim.notna())
    _lup = (_d >= 0.95 * _lim).astype(float).where(_lim.notna())
    q["lim_dn20"] = _ldn.groupby(q["code"]).transform(lambda x: x.rolling(20, min_periods=15).sum())
    q["lim_up20"] = _lup.groupby(q["code"]).transform(lambda x: x.rolling(20, min_periods=15).sum())
    # 仮説(53回目25周目 担当1)：下げの形＝直近20日の大陰線（前日比-5%以下）の日数と、20日の下げ（≤-5%の銘柄）のうち大陰線の日の下げの割合
    # （数回の急落で下げたか、じわじわ下げたか）。t 終値で確定
    _bd = q["ret1"].where(q["ret1"] <= -0.05, 0.0).where(q["ret1"].notna())
    q["big_dn20"] = (_bd < 0).astype(float).where(_bd.notna()).groupby(q["code"]).transform(lambda x: x.rolling(20, min_periods=15).sum())
    _bds = _bd.groupby(q["code"]).transform(lambda x: x.rolling(20, min_periods=15).sum())
    q["big_dn_share"] = (_bds / q["ret20"].where(q["ret20"] <= -0.05)).clip(0, 2)
    _c = q["close"].where(q["close"] > 0)
    _unit = (10.0 ** np.floor(np.log10(_c))) / 2
    q["round_up"] = (np.floor(_c / _unit) + 1) * _unit / _c - 1
    # 仮説(2026-11-22 担当2)：値幅制限（東証の制限値幅表）いっぱいまで下げて安値引け＝ストップ安（需給の投げ）・ストップ高。
    # 前日終値（分割調整後のため、分割前の期間は近似）から制限値幅を求め、前日比が制限値幅の9割以上で安値(高値)引けの日。t 終値で確定
    _pc = q.groupby("code", sort=False)["close"].shift(1)
    _edges = [100, 200, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000, 50000, 70000, 100000, 150000, 200000, 300000, 500000]
    _lims = [30, 50, 80, 100, 150, 300, 400, 500, 700, 1000, 1500, 3000, 4000, 5000, 7000, 10000, 15000, 30000, 40000, 50000, 70000, 100000]
    _lim = pd.Series(np.array(_lims, dtype=float)[np.searchsorted(_edges, _pc.fillna(0).to_numpy(), side="right")], index=q.index).where(_pc.notna())
    _sl = ((q["close"] - _pc <= -0.9 * _lim) & (q["close"] <= q["low"] * 1.001)).astype(float).where(_pc.notna())
    _sh = ((q["close"] - _pc >= 0.9 * _lim) & (q["close"] >= q["high"] * 0.999)).astype(float).where(_pc.notna())
    q["stop_lo5"] = _sl.groupby(q["code"]).transform(lambda x: x.rolling(5, min_periods=5).sum())
    q["stop_hi20"] = _sh.groupby(q["code"]).transform(lambda x: x.rolling(20, min_periods=20).sum())
    q["mkt_stop_lo"] = _sl.groupby(q["date"]).transform("mean")
    # 仮説(2026-11-22 担当2)：20日の下げのうち決算発表後の反応が占める割合（決算が原因の下げか、地合い・需給の下げか）。
    # 直近20日以内に発表が無い銘柄は0。disc_ret・ret20 とも t 終値で確定
    if "disc_ret" in q.columns:
        _ds = (q["disc_ret"] / q["ret20"]).where((q["ret20"] < -0.05) & q["disc_ret"].notna())
        q["disc_share"] = _ds.clip(-1, 2).where(q["disc_ret"].notna(), 0.0).where(q["ret20"].notna())
    else:
        q["disc_share"] = np.nan
    # 仮説(2026-11-22 担当2)：日本市場全体（日次騰落の中央値）と前夜の米国株(S&P500)の20日相関とその60日相関からの変化
    # （日本だけの下げ＝米国と切り離された下げか、世界同時の下げか）。sp500_ret1 は前日までの確定値
    if "sp500_ret1" in q.columns and "mkt_ret1" in q.columns:
        _dd = q.groupby("date")[["mkt_ret1", "sp500_ret1"]].first().sort_index()
        _c20 = _dd["mkt_ret1"].rolling(20, min_periods=15).corr(_dd["sp500_ret1"])
        _c60 = _dd["mkt_ret1"].rolling(60, min_periods=40).corr(_dd["sp500_ret1"])
        q["mkt_us_corr20"] = q["date"].map(_c20)
        q["mkt_us_corr_chg"] = q["date"].map(_c20 - _c60)
    else:
        q["mkt_us_corr20"] = q["mkt_us_corr_chg"] = np.nan
    cnum =pd.to_numeric(q["code"].str[:4], errors="coerce")
    q["etf"] = cnum.between(1300, 1699) | cnum.between(2500, 2599)
    # 仮説(53回目21周目 担当1)：PBR1倍割れ（東証の資本効率の改善要請で下値の目安になりやすい）。急落で新たに1倍を割ったか
    # （20日前のPBR>1・当日≤1 → 1）、PBRの20日前との比、市場全体（ETF除く）でPBR1倍割れの銘柄の割合（割安感の広がり）。t 終値で確定
    if "pbr" in q.columns:
        _pb = q["pbr"].where(q["pbr"] > 0)
        _pb20 = _pb.groupby(q["code"], sort=False).shift(20)
        q["pbr_cross1"] = ((_pb <= 1.0) & (_pb20 > 1.0)).astype(float).where(_pb.notna() & _pb20.notna())
        q["pbr_vs1"] = _pb - 1.0
        q["mkt_pbr1_frac"] = (_pb <= 1.0).astype(float).where(_pb.notna() & ~q["etf"]).groupby(q["date"]).transform("mean")
        _mf = q.groupby("date")["mkt_pbr1_frac"].first().sort_index()
        q["mkt_pbr1_chg20"] = q["date"].map(_mf - _mf.shift(20))
    else:
        q["pbr_cross1"] = q["pbr_vs1"] = q["mkt_pbr1_frac"] = q["mkt_pbr1_chg20"] = np.nan
    # 仮説(53回目23周目 担当1)：直近の株式分割（調整係数<1 の日＝分割の効力発生日。事前に公表されるので先読みではない）。
    # 分割で最低投資金額が下がり個人の買いが入りやすい。直近60日・120日以内に分割があったか、株式併合（係数>1）があったか
    if "adj_factor" in q.columns:
        _af = q["adj_factor"].fillna(1.0)
        _sp = (_af < 0.999).astype(float)
        _rs = (_af > 1.001).astype(float)
        _g = q["code"]
        q["split60"] = _sp.groupby(_g, sort=False).transform(lambda x: x.rolling(60, min_periods=1).max())
        q["split120"] = _sp.groupby(_g, sort=False).transform(lambda x: x.rolling(120, min_periods=1).max())
        q["rsplit120"] = _rs.groupby(_g, sort=False).transform(lambda x: x.rolling(120, min_periods=1).max())
    else:
        q["split60"] = q["split120"] = q["rsplit120"] = np.nan
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
    # 仮説(53回目24周目 担当1)：確定済みの過去の結果（t-20 日以前に買った分＝t 終値までに確定）から
    # その銘柄の-10%下落の起きやすさ・平均損益、その銘柄が急落(20日騰落≤-15%)した日に買った場合の勝ち（直近120日）、
    # 市場全体で最近の急落買い（20日騰落≤-15%・売買代金5億円以上）が勝てているか（確定済みの直近20営業日の平均）＝押し目買いの地合い
    _gc = q["code"]
    q["dd_hist"] = q.groupby(_gc, sort=False)["dd10"].transform(lambda s: s.shift(HOLD).rolling(60, min_periods=40).mean())
    q["pnl_hist"] = q.groupby(_gc, sort=False)["pnl"].transform(lambda s: s.shift(HOLD).rolling(60, min_periods=40).mean())
    _cw = q["win"].where(q["ret20"] <= -0.15)
    q["crash_hit_hist"] = _cw.groupby(_gc, sort=False).transform(lambda s: s.shift(HOLD).rolling(120, min_periods=5).mean())
    _ok = (q["ret20"] <= -0.15) & ~q["etf"] & (q["tv20"] >= 5e8)
    _dw = q["win"].where(_ok).groupby(q["date"]).mean().sort_index()
    _dn = q["win"].where(_ok).groupby(q["date"]).count().sort_index()
    _dw = _dw.where(_dn >= 5)
    q["mkt_crash_hit"] = q["date"].map(_dw.shift(HOLD).rolling(20, min_periods=5).mean())
    return _shrink(q)


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
    "dev200":    [("<=", -0.30), ("<=", -0.15), (">=", 0.0)],
    "mkt_dev200": [("<=", -0.05), ("<=", 0.0), (">=", 0.05)],
    "dev200_rel": [("<=", -0.2), (">=", 0.0)],
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
    "mkt_hi_days": [("<=", 10), ("<=", 30), (">=", 60)],
    "mkt_dd120": [("<=", -0.15), ("<=", -0.10), (">=", -0.03)],
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
    "sq_days": [("<=", 4), ("<=", 10), (">=", 20)],
    "div_days": [("<=", 10), ("<=", 30), (">=", 90)],
    "sec_from_hi": [("<=", -0.30), ("<=", -0.20), (">=", -0.08)],
    "from_hi_rel": [("<=", -0.20), ("<=", -0.10), (">=", 0.0)],
    "rsi2": [("<=", 5), ("<=", 15), (">=", 90)],
    "rsi_div": [(">=", 10), (">=", 5), ("<=", 1)],
    "mkt_ret1": [("<=", -0.02), ("<=", -0.01), (">=", 0.01)],
    "mkt_corr20": [(">=", 0.7), (">=", 0.5), ("<=", 0.2)],
    "mkt_corr_chg": [(">=", 0.2), (">=", 0.1), ("<=", -0.1)],
    "mkt_corr_avg": [(">=", 0.45), (">=", 0.4), ("<=", 0.3)],
    "ndx_spx5": [("<=", -0.015), ("<=", -0.005), (">=", 0.01)],
    "usdjpy_dev60": [("<=", -0.03), ("<=", -0.01), (">=", 0.02)],
    "sec_rsi2": [("<=", 10), ("<=", 25), (">=", 75)],
    "sec_dn_streak": [(">=", 4), (">=", 2), ("<=", 0)],
    "ret1_z": [("<=", -2), ("<=", -1), (">=", 1)],
    "sec_lo60_frac": [(">=", 0.3), (">=", 0.15), ("<=", 0.03)],
    "rate_sp": [(">=", 2.6), (">=", 2.3), ("<=", 2.0)],
    "sec_crash_days": [("<=", 3), ("<=", 8), (">=", 15)],
    "sec_rank5": [("<=", 0.1), ("<=", 0.25), (">=", 0.75)],
    "bps_chg": [(">=", 0.03), (">=", 0.01), ("<=", -0.01)],
    "mkt_lo60_chg5": [("<=", -0.1), ("<=", -0.03), (">=", 0.05)],
    "sec_op_yoy": [(">=", 0.15), (">=", 0.0), ("<=", -0.15)],
    "sec5_rank": [("<=", 0.1), ("<=", 0.25), (">=", 0.8)],
    "dd20": [("<=", -0.20), ("<=", -0.12), (">=", -0.03)],
    "mfg_sp20": [("<=", -0.03), ("<=", -0.01), (">=", 0.02)],
    "xsec_disp20": [(">=", 0.06), (">=", 0.04), ("<=", 0.025)],
    "mkt_us_corr20": [(">=", 0.5), (">=", 0.3), ("<=", 0.1)],
    "mkt_us_corr_chg": [(">=", 0.2), ("<=", -0.2)],
    "disc_share": [(">=", 0.5), (">=", 0.2), ("<=", 0.0)],
    "stop_lo5": [(">=", 1), (">=", 2)],
    "stop_hi20": [(">=", 1), (">=", 2)],
    "mkt_stop_lo": [(">=", 0.001), (">=", 0.002)],
    "xsec_disp_chg5": [(">=", 0.01), (">=", 0.005), ("<=", -0.005)],
    "xsec_disp_rel60": [(">=", 1.5), (">=", 1.2), ("<=", 0.8)],
    "sec20_rank": [("<=", 0.1), ("<=", 0.25), (">=", 0.75)],
    "sec_vol_exp": [(">=", 1.5), (">=", 1.2), ("<=", 0.8)],
    "engulf": [(">=", 1)],
    "ll_streak": [(">=", 5), (">=", 3), ("<=", 0)],
    "rnd_up": [("<=", 0.02), ("<=", 0.05), (">=", 0.5)],
    "sec_lead5": [(">=", 0.02), (">=", 0.0), ("<=", -0.02)],
    "prior_crash": [(">=", 1), ("<=", 0)],
    "eff20": [(">=", 0.5), (">=", 0.3), ("<=", 0.15)],
    "vwap_dev20": [("<=", -0.15), ("<=", -0.10), (">=", 0.05)],
    "sec_lo60_off": [("<=", -0.1), ("<=", -0.05), (">=", 0)],
    "mkt_rv_exp": [(">=", 1.5), (">=", 1.2), ("<=", 0.8)],
    "lh_streak": [(">=", 5), (">=", 3), ("<=", 0)],
    "mkt_rv_off": [("<=", -0.3), ("<=", -0.1), (">=", 0)],
    "mkt_speed": [(">=", 0.8), (">=", 0.5), ("<=", 0.2)],
    "sales_yld": [(">=", 2.0), (">=", 1.0), ("<=", 0.3)],
    "mkt_dn_tv": [(">=", 0.8), (">=", 0.7), ("<=", 0.3)],
    "mkt_dn_tv5": [(">=", 0.65), (">=", 0.6), ("<=", 0.4)],
    "disc_season": [(">=", 0.3), (">=", 0.15), ("<=", 0.03)],
    "mkt_disc_ret": [(">=", 0.02), (">=", 0.0), ("<=", -0.02)],
    "round_up": [("<=", 0.03), ("<=", 0.08), (">=", 0.2)],
    "fx_tail20": [("<=", -0.01), ("<=", -0.005), (">=", 0.005)],
    "rate_corr60": [(">=", 0.2), ("<=", -0.2)],
    "rate_tail20": [("<=", -0.02), (">=", 0.02)],
    "fx_tail5": [("<=", -0.003), (">=", 0.0), (">=", 0.003)],
    "oil_tail20": [("<=", -0.01), (">=", 0.0), (">=", 0.01)],
    "sec_fx_tail20": [("<=", -0.003), (">=", 0.0), (">=", 0.003)],
    "fx_tail_rel": [("<=", -0.003), (">=", 0.0), (">=", 0.003)],
    "us_tail20": [("<=", -0.005), (">=", 0.0), (">=", 0.01)],
    "sec_us_tail20": [("<=", -0.003), (">=", 0.0), (">=", 0.005)],
    "sec_disc_ret": [("<=", -0.05), ("<=", -0.02), (">=", 0.02)],
    "riskoff5": [(">=", 4), (">=", 3), ("<=", 0)],
    "riskoff_off": [("<=", -2), ("<=", -1), (">=", 0)],
    "mkt_share": [(">=", 0.5), (">=", 0.25), ("<=", 0.0)],
    "sec_share": [(">=", 0.7), (">=", 0.4), ("<=", 0.1)],
    "pm_ret": [("<=", -0.15), ("<=", -0.08), (">=", 0.15)],
    "mtd_ret": [("<=", -0.15), ("<=", -0.08), (">=", 0.1)],
    "lim_dn20": [(">=", 1), (">=", 2)],
    "lim_up20": [(">=", 1), (">=", 2)],
    "gold_corr60": [(">=", 0.3), ("<=", -0.1)],
    "gold_tail20": [("<=", -0.005), (">=", 0.0), (">=", 0.01)],
    "beta_up60": [(">=", 1.5), (">=", 1.0), ("<=", 0.5)],
    "beta_dn60": [(">=", 1.5), ("<=", 0.5)],
    "beta_asym": [(">=", 0.3), (">=", 0.0), ("<=", -0.3)],
    "up_day_xs20": [(">=", 0.005), (">=", 0.0), ("<=", -0.005)],
    "vix_rv_gap": [(">=", 5), ("<=", 0), ("<=", -5)],
    "dn_day_xs20": [("<=", -0.005), ("<=", 0.0), (">=", 0.005)],
    "updn_xs20": [(">=", 0.005), ("<=", 0.0), ("<=", -0.01)],
    "sec_up_xs20": [("<=", -0.002), ("<=", 0.0), (">=", 0.002)],
    "up_xs_rel": [("<=", -0.003), ("<=", 0.0), (">=", 0.003)],
    "big_dn20": [(">=", 3), (">=", 1), ("<=", 0)],
    "big_dn_share": [(">=", 0.7), ("<=", 0.3)],
    "pbr_cross1": [(">=", 1)],
    "pbr_vs1": [("<=", -0.2), ("<=", 0.0), (">=", 0.0), (">=", 0.5)],
    "mkt_pbr1_frac": [(">=", 0.4), ("<=", 0.3)],
    "mkt_pbr1_chg20": [(">=", 0.05), (">=", 0.02), ("<=", -0.02)],
    "is_def": [(">=", 1), ("<=", 0)],
    "def_sp20": [(">=", 0.05), (">=", 0.02), ("<=", 0.0), ("<=", -0.02)],
    "def_sp5": [(">=", 0.01), ("<=", 0.0), ("<=", -0.01)],
    "split60": [(">=", 1), ("<=", 0)],
    "split120": [(">=", 1)],
    "rsplit120": [(">=", 1)],
    "dd_hist": [(">=", 0.3), ("<=", 0.05)],
    "pnl_hist": [(">=", 0.02), ("<=", -0.02)],
    "crash_hit_hist": [(">=", 0.5), ("<=", 0.0)],
    "mkt_crash_hit": [(">=", 0.6), (">=", 0.4), ("<=", 0.2)],
}

LABEL = {
    "ret1": "前日比", "ret5": "5日騰落", "ret20": "20日騰落", "ret60": "60日騰落",
    "dev25": "25日線乖離", "dev75": "75日線乖離", "dev200": "200日線乖離", "mkt_dev200": "市場全体の200日線乖離(中央値)", "dev200_rel": "200日線乖離-市場の中央値", "ma_up": "5日線>25日線",
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
    "mkt_dd60": "日本の市場全体の60日高値からの下落(中央値指数)", "mkt_hi_days": "市場全体の120日高値からの経過日数(下げ相場の長さ)", "mkt_dd120": "市場全体の120日高値からの下落(中央値指数)", "sec_rank20": "業種内の20日騰落の順位(0=最も下げた)",
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
    "ret20_xr": "20日騰落の全銘柄内の順位(0=最も下げた)", "sq_days": "次のSQ(第2金曜)までの暦日数", "div_days": "3月・9月末(配当の権利)までの暦日数",
    "sec_from_hi": "業種の52週高値からの位置(中央値)", "from_hi_rel": "52週高値からの位置-業種の中央値",
    "rsi2": "2日RSI", "rsi_div": "20日安値圏でのRSI14の20日最低からの戻り(強気ダイバージェンス)", "mkt_ret1": "市場全体のその日の騰落(中央値)", "mkt_corr20": "市場全体(中央値)との20日相関", "mkt_corr_chg": "市場との20日相関−60日相関(連れ安の強まり)", "mkt_corr_avg": "全銘柄の市場との20日相関の平均(一斉に動く度合い)",
    "ndx_spx5": "ナスダック-S&P500の5日騰落の差(前日)", "usdjpy_dev60": "ドル円の60日平均からの乖離(前日)", "sec_rsi2": "業種内の2日RSIの中央値",
    "sec_dn_streak": "業種の連続下落日数(業種の前日比の中央値)", "ret1_z": "当日の騰落/20日ボラ", "sec_lo60_frac": "業種内で60日安値をつけた銘柄の割合",
    "rate_sp": "日米10年金利差の水準(%pt・前日)", "sec_crash_days": "業種の急落状態(急落割合≥15%)の連続日数", "sec_rank5": "業種内の5日騰落の順位(0=最も下げた)",
    "bps_chg": "BPSの前回決算からの変化", "mkt_lo60_chg5": "市場全体の60日安値割合の5日変化", "sec_op_yoy": "業種内の営業利益の前年同期比(中央値)",
    "sec5_rank": "業種の5日騰落の業種間の順位(0=最も下げた)", "dd20": "20日高値(終値)からの下落率",
    "mfg_sp20": "製造業-非製造業の20日騰落の差(中央値)", "xsec_disp20": "業種の20日騰落の業種間のばらつき(標準偏差)", "mkt_us_corr20": "日本市場全体と前夜の米国株の20日相関", "mkt_us_corr_chg": "日本市場と米国株の20日相関−60日相関", "disc_share": "20日の下げのうち決算発表後の反応の割合(発表なし=0)", "stop_lo5": "直近5日のストップ安(制限値幅いっぱいで安値引け)の日数", "stop_hi20": "直近20日のストップ高の日数", "mkt_stop_lo": "当日ストップ安の銘柄の割合(市場全体)", "xsec_disp_chg5": "業種間のばらつきの5日変化", "xsec_disp_rel60": "業種間のばらつきの60日平均に対する倍率", "sec20_rank": "業種の20日騰落の業種間の順位(0=最も下げた)",
    "sec_vol_exp": "業種内のボラ拡大(5日/60日)の中央値",
    "engulf": "陽の包み足(前日陰線を当日陽線が包む)", "ll_streak": "安値の連続切り下げ日数", "rnd_up": "直下の価格の節目(100円・1000円など)からの距離",
    "sec_lead5": "業種内の大型株(時価総額上位3割)の5日騰落-業種全体(中央値)", "prior_crash": "2度目の急落(t-60〜t-21に20日騰落-15%以下の日あり=1)",
    "eff20": "20日の値動きの効率(|20日騰落|/日次騰落の絶対値の合計)", "vwap_dev20": "20日の出来高加重平均価格(VWAP)からの乖離", "sec_lo60_off": "業種内の60日安値割合-直近5日の最大値(投げ売りのピーク越え)", "mkt_rv_exp": "市場全体のボラ拡大(前日比の中央値の5日/60日標準偏差)", "lh_streak": "高値の連続切り下げ日数", "mkt_rv_off": "市場全体のボラ拡大-直近5日の最大値(乱高下の峠越え)", "mkt_speed": "市場全体の下げの速さ(5日騰落/20日騰落の中央値)", "sales_yld": "予想売上高/時価総額(PSRの逆数)", "mkt_dn_tv": "市場全体の売買代金のうち下落銘柄の割合", "mkt_dn_tv5": "市場全体の売買代金のうち下落銘柄の割合(5日平均)", "disc_season": "直近10日以内に決算発表した銘柄の割合(決算シーズン)", "mkt_disc_ret": "直近10日の決算発表銘柄の株価反応(中央値)", "round_up": "上の切りのよい株価(桁の半分単位)までの距離", "fx_tail20": "為替の追い風(為替感応度×ドル円の20日変化)", "rate_corr60": "日本10年金利の上昇との60日相関(金利感応度)", "rate_tail20": "金利の追い風(金利感応度×10年金利の20日変化)", "fx_tail5": "為替の短期の追い風(為替感応度×ドル円の5日変化)", "oil_tail20": "原油の追い風(原油との相関×WTIの20日騰落)", "sec_fx_tail20": "業種全体の為替の追い風(中央値)", "fx_tail_rel": "為替の追い風-業種の中央値", "us_tail20": "米国株の追い風(米国株との相関×S&P500の20日騰落)", "sec_us_tail20": "業種全体の米国株の追い風(中央値)", "sec_disc_ret": "同業種で直近10日に決算発表した銘柄の株価反応(中央値)", "riskoff5": "リスクオフの総合点(5日で円高・VIX上昇・米金利低下・米株安の数)", "riskoff_off": "リスクオフの和らぎ(総合点-直近5日の最大)", "mkt_share": "20日の下げのうち市場全体で説明できる割合(ベータ×市場の騰落)", "sec_share": "20日の下げのうち業種全体で説明できる割合", "pm_ret": "前月の月間騰落", "mtd_ret": "月初来の騰落(前月末から)", "lim_dn20": "直近20日でストップ安付近で引けた日数", "lim_up20": "直近20日でストップ高付近で引けた日数", "gold_corr60": "金(ETF)との60日相関", "gold_tail20": "金の追い風(金との相関×金の20日騰落)", "beta_up60": "市場が上げた日だけの60日ベータ", "beta_dn60": "市場が下げた日だけの60日ベータ", "beta_asym": "上げの日のベータ−下げの日のベータ", "up_day_xs20": "直近20日の市場が上げた日の超過騰落(平均)", "vix_rv_gap": "VIX(前日)−日経225の20日実現ボラ", "dn_day_xs20": "直近20日の市場が下げた日の超過騰落(平均)", "updn_xs20": "上げた日の超過騰落−下げた日の超過騰落", "sec_up_xs20": "業種内の市場が上げた日の超過騰落(中央値)", "up_xs_rel": "市場が上げた日の超過騰落−業種の中央値", "pbr_cross1": "急落でPBR1倍を新たに割った(20日前は1倍超)", "pbr_vs1": "PBR−1", "mkt_pbr1_frac": "市場全体でPBR1倍割れの銘柄の割合", "mkt_pbr1_chg20": "PBR1倍割れの銘柄の割合の20日変化", "is_def": "ディフェンシブ業種(食料品・医薬品・電気ガス・陸運)", "def_sp20": "ディフェンシブ業種−それ以外の20日騰落の差(中央値)", "def_sp5": "ディフェンシブ業種−それ以外の5日騰落の差(中央値)", "split60": "直近60日以内に株式分割", "split120": "直近120日以内に株式分割", "rsplit120": "直近120日以内に株式併合", "dd_hist": "過去の-10%下落の起きやすさ(確定済みの直近60回)", "big_dn20": "直近20日の大陰線(前日比-5%以下)の日数", "big_dn_share": "20日の下げのうち大陰線の日の下げの割合", "pnl_hist": "過去の平均損益(確定済みの直近60回)", "crash_hit_hist": "その銘柄の急落時(20日騰落≤-15%)の過去の勝率(確定済み・120日)", "mkt_crash_hit": "市場全体の最近の急落買いの勝率(確定済み・20営業日)",
}
POINT_FEATS = {"rate_sp", "jgb2y_chg20", "us10y_chg5", "vix_chg1", "spread_chg20", "vix", "vix_chg5", "us10y_chg20", "jgb10y_chg20", "jgb_slope", "adr25", "lev_z"}
DOW = "月火水木金"


def fmt_cond(f, op, th):
    lab = LABEL.get(f, f)
    if f in ("ma_up", "new_hi", "gc5", "hol_next", "engulf"):
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
    if f in ("days_hi20", "lo_days60", "mkt_dn_streak", "min1_days", "gap_dn_cnt20", "vmax_days", "sq_days", "div_days", "sec_dn_streak", "sec_crash_days", "ll_streak", "lh_streak", "stop_lo5", "stop_hi20", "mkt_hi_days"):
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
    if f == "is_def":
        return "ディフェンシブ業種" if op == ">=" else "ディフェンシブ業種以外"
    if f in ("sec_vratio", "mkt_vratio", "vratio", "rsi14", "clpos", "vdry", "pbr", "vol_exp", "vix_rel60", "sec_crash_pk", "vix_pk20", "sec_wide", "hammer5", "ac20", "px_lvl", "clpos5", "sec_corr60", "sec_tv_rank", "sec_up_frac", "tv5_60", "rng5_20", "us_corr60", "sec_above25", "mkt_above25", "lo_cnt20", "sec_mcap_rank", "oil_corr60", "ret20_xr", "rsi2", "rsi_div", "sec_rsi2", "ret1_z", "sec_lo60_frac", "sec_rank5", "prior_crash", "eff20", "sec_lo60_off", "mkt_rv_exp", "mkt_rv_off", "mkt_speed", "sales_yld", "mkt_dn_tv", "mkt_dn_tv5", "disc_season", "fx_tail20", "rate_corr60", "rate_tail20", "fx_tail5", "oil_tail20", "sec_fx_tail20", "fx_tail_rel", "us_tail20", "sec_us_tail20", "riskoff5", "riskoff_off", "lim_dn20", "lim_up20", "gold_corr60", "gold_tail20", "beta_up60", "beta_dn60", "beta_asym", "vix_rv_gap", "pbr_cross1", "pbr_vs1", "split60", "split120", "rsplit120", "big_dn20", "big_dn_share"):
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
<div class="card"><div class="name">このルールは偶然の当たりではないか？</div>
<div class="stat">年別・局面ごとの成績、閾値や勝ちの定義をずらしたとき、条件を外したときの変化などの点検結果：
<a href="robustness.html">採用ルールの点検</a></div></div>
<h2>採用履歴（新しい順）</h2>
{hist_html or '<div class="none">まだ初期ルールから更新されていません。</div>'}
<div class="note">採用基準：取引数{MIN_N}以上・3分割の各期間{MIN_N_SPLIT}以上、全体勝率が現ルール+1pt以上、
最悪期間の勝率も現ルールより上、3期間すべてで平均損益プラス。多数のルールを試すため、偶然の当たりが
混ざる可能性は残ります。手数料・スリッページ・税は未考慮。データの遅延（J-Quantsのプラン）にご注意ください。
本ページは過去データの統計であり、投資助言ではありません。売買の判断と結果責任は利用者にあります。</div>
"""
    html = _page("勝率改善ルール", "rules", body) if _page else f"<!doctype html><meta charset=utf-8><body>{body}"
    open(RULES_HTML, "w", encoding="utf-8").write(html)


ROBUST_MD = os.path.join("rules", "robustness.md")


def _shift_variants(f, op, th):
    """閾値を少しずらした版（情報用）。整数っぽい条件は±1、それ以外は|th|の±10%・±20%"""
    if op == "==" or f in {"mfg", "mkt_seg"}:
        return []
    if float(th).is_integer() and abs(th) <= 10:
        vals = [th - 1, th + 1]
    else:
        d = abs(th) if th != 0 else 0.01
        vals = [th - 0.2 * d, th - 0.1 * d, th + 0.1 * d, th + 0.2 * d]
    return [round(v, 6) for v in vals if v != th]


def robustness_report(q, conds, edges, base, path=None, singles=None):
    """担当2の点検：採用ルールが偶然の当たりでないか（情報として出すだけ。採用判定には使わない）"""
    path = path or ROBUST_MD
    t = dedup_trades(q, apply(q, conds))
    md = ["# 採用ルールの点検（偶然の当たりでないか）", "",
          "情報として出すだけの点検結果です（採用基準 passes() には使っていません）。投資助言ではありません。", "",
          f"更新 {dt.datetime.now():%Y-%m-%d %H:%M} ／ 基準（流動性のみ）勝率 {pct(base.get('win'))} n={base.get('n')}", "",
          "## 対象のルール", ""] + [f"- {fmt_cond(*c)}" for c in BASE + conds]
    if len(t) == 0:
        md += ["", "取引0件"]
        _write_keep_manual(path, md)
        return
    t = t.copy()
    t["date"] = pd.to_datetime(t["date"])
    md += ["", f"取引数 {len(t)} ／ 勝率 {pct(t['win'].mean())} ／ 平均損益 {pct(t['pnl'].mean(), 2)}", "",
           "## 年別", "", "| 年 | 取引数 | 勝率 | 平均損益 |", "|---|---|---|---|"]
    for y, g in t.groupby(t["date"].dt.year):
        md.append(f"| {y} | {len(g)} | {pct(g['win'].mean())} | {pct(g['pnl'].mean(), 2)} |")
    # 時期の集中
    per = t["date"].dt.to_period("M")
    mon = t.groupby(per).agg(n=("win", "size"), win=("win", "mean"), pnl=("pnl", "mean"))
    mon = mon.sort_values("n", ascending=False)
    wk = t.groupby(t["date"].dt.to_period("W"))["win"].size().sort_values(ascending=False)
    md += ["", "## 時期の集中", "",
           f"- 取引のあった日数 {t['date'].nunique()} ／ 月数 {len(mon)} ／ 週数 {len(wk)}",
           f"- 最多の1か月に {mon['n'].iloc[0]/len(t)*100:.0f}% ／ 上位3か月に {mon['n'].iloc[:3].sum()/len(t)*100:.0f}% ／ "
           f"上位3週に {wk.iloc[:3].sum()/len(t)*100:.0f}%", "",
           "| 月 | 取引数 | 勝率 | 平均損益 |", "|---|---|---|---|"]
    for p, r in mon.head(10).iterrows():
        md.append(f"| {p} | {int(r['n'])} | {pct(r['win'])} | {pct(r['pnl'], 2)} |")
    md.append("")
    for k in (1, 3):
        rest = t[~per.isin(set(mon.index[:k]))]
        md.append(f"- 上位{k}か月を除くと：取引数 {len(rest)} 勝率 {pct(rest['win'].mean() if len(rest) else None)} "
                  f"平均損益 {pct(rest['pnl'].mean() if len(rest) else None, 2)}")
    # 買うのが遅れたら：d 営業日後の始値で買い、そこから HOLD 日（d=1 が本来のルール）。反発の1日に依存していないか
    md += ["", "## 買う日をずらしたとき（d=1 が本来のルール：翌営業日の始値）", "",
           "| 買う日 | 取引数 | 勝率 | 平均損益 |", "|---|---|---|---|"]
    o, h, c, cd = (q[k].to_numpy() for k in ("open", "high", "close", "code"))

    def sim(tt, d=1, hold=HOLD, tp=TP):
        w_, p_ = [], []
        for i, code in zip(tt.index.to_numpy(), tt["code"].to_numpy()):
            a, b = i + d, i + d + hold - 1
            if b >= len(q) or cd[b] != code or not np.isfinite(o[a]) or o[a] <= 0:
                continue
            win_ = np.nanmax(h[a:b + 1]) >= o[a] * (1 + tp)
            w_.append(win_)
            p_.append(tp if win_ else c[b] / o[a] - 1)
        return len(w_), (float(np.mean(w_)) if w_ else None), (float(np.mean(p_)) if p_ else None)
    for d in (1, 2, 3, 5):
        n_, w_, p_ = sim(t, d=d)
        md.append(f"| {d}営業日後 | {n_} | {pct(w_)} | {pct(p_, 2)} |")
    # 勝ちの定義を少し変えたとき（情報用。採用に使う定義は +10%・20営業日のまま）。基準（流動性のみ）も同じ定義で比べる
    bt = dedup_trades(q, apply(q, []))
    md += ["", "## 勝ちの定義を少し変えたとき（情報用。本来は +10%・20営業日）", "",
           "| 利確 | 保有日数 | 取引数 | 勝率 | 平均損益 | 基準の勝率 | 基準との差 |", "|---|---|---|---|---|---|---|"]
    for tp_, hold_ in ((0.10, 20), (0.08, 20), (0.12, 20), (0.15, 20), (0.10, 10), (0.10, 15), (0.10, 25)):
        n_, w_, p_ = sim(t, hold=hold_, tp=tp_)
        _, bw, _ = sim(bt, hold=hold_, tp=tp_)
        md.append(f"| +{tp_*100:.0f}% | {hold_}日 | {n_} | {pct(w_)} | {pct(p_, 2)} | {pct(bw)} | "
                  f"{(w_ - bw) * 100:+.1f}pt |" if w_ is not None and bw is not None else f"| +{tp_*100:.0f}% | {hold_}日 | {n_} | - | - | - | - |")
    # 日付の効果と銘柄選びの効果：信号が出た日と同じ日に、条件を減らして買った場合（その日に買うこと自体の勝率）
    sd_ = set(pd.to_datetime(t["date"]).dt.normalize())
    md += ["", "## 信号が出た日と同じ日に買った場合（日付の効果と銘柄選びの効果の分離）", "",
           "| 同じ日に買う対象 | 取引数 | 勝率 | 平均損益 |", "|---|---|---|---|"]
    for lbl_, cs_ in (("流動性条件のみ（全銘柄）", []), ("20日騰落 <= -12% のみ", [["ret20", "<=", -0.12]]),
                      ("製造業のみ", [["mfg", ">=", 1]]), ("採用ルール", conds)):
        tt_ = dedup_trades(q, apply(q, cs_) & pd.to_datetime(q["date"]).dt.normalize().isin(sd_))
        if len(tt_):
            md.append(f"| {lbl_} | {len(tt_)} | {pct(tt_['win'].mean())} | {pct(tt_['pnl'].mean(), 2)} |")
    # 仮説(担当2 27周目)：同じ日の取引はまとめて勝ち負けする（独立ではない）。信号日を単位にしたブートストラップと
    # 1日ずつ外したとき（ジャックナイフ）で、取引数から見た見かけより不確かさが大きくないかを見る（情報用）
    dd_ = t["date"].dt.normalize()
    gd_ = t.groupby(dd_).agg(n=("win", "size"), w=("win", "sum"), p=("pnl", "sum"))
    md += ["", "## 信号日を単位にした不確かさ（同じ日の取引はまとめて当たり外れする）", ""]
    if len(gd_) >= 3:
        rng_ = np.random.default_rng(0)
        nn_, ww_, pp_ = gd_["n"].to_numpy(float), gd_["w"].to_numpy(float), gd_["p"].to_numpy(float)
        bs_w, bs_p = [], []
        for _ in range(2000):
            ix_ = rng_.integers(0, len(gd_), len(gd_))
            bs_w.append(ww_[ix_].sum() / nn_[ix_].sum())
            bs_p.append(pp_[ix_].sum() / nn_[ix_].sum())
        jk_ = [(ww_.sum() - ww_[k]) / (nn_.sum() - nn_[k]) for k in range(len(gd_)) if nn_.sum() > nn_[k]]
        wr_ = float(t["win"].mean())
        se_iid = (wr_ * (1 - wr_) / len(t)) ** 0.5
        md += [f"- 信号日 {len(gd_)} 日・取引 {len(t)} 件。1日あたり最大 {int(nn_.max())} 件（全体の {nn_.max()/nn_.sum()*100:.0f}%）",
               f"- 取引を独立とみなした場合の勝率の95%範囲：{pct(max(0.0, wr_ - 1.96 * se_iid))}〜{pct(min(1.0, wr_ + 1.96 * se_iid))}",
               f"- **信号日を単位にしたブートストラップ（2000回）の勝率**：5%点 {pct(float(np.percentile(bs_w, 5)))}・"
               f"2.5%点 {pct(float(np.percentile(bs_w, 2.5)))}・中央 {pct(float(np.median(bs_w)))}",
               f"- 同 平均損益：5%点 {pct(float(np.percentile(bs_p, 5)), 2)}・中央 {pct(float(np.median(bs_p)), 2)}",
               f"- 1日ずつ外したとき（ジャックナイフ）の勝率：最低 {pct(min(jk_)) if jk_ else '-'}・最高 {pct(max(jk_)) if jk_ else '-'}",
               f"- 信号日を等しく数えた勝率 {pct(float((ww_ / nn_).mean()))}・勝率が8割未満の日 {int(((ww_ / nn_) < 0.8).sum())} 日"]
        ep_ = (pd.Series(gd_.index).diff().dt.days > 30).cumsum().to_numpy()   # 信号日の間が30日より空いたら別の局面
        if ep_.max() >= 2:
            ge_ = pd.DataFrame({"e": ep_, "n": nn_, "w": ww_}).groupby("e").sum()
            en_, ew_ = ge_["n"].to_numpy(float), ge_["w"].to_numpy(float)
            bs_e = []
            for _ in range(2000):
                ix_ = rng_.integers(0, len(ge_), len(ge_))
                bs_e.append(ew_[ix_].sum() / en_[ix_].sum())
            md.append(f"- **局面（{len(ge_)}回）を単位にしたブートストラップの勝率**：5%点 {pct(float(np.percentile(bs_e, 5)))}・"
                      f"中央 {pct(float(np.median(bs_e)))}（局面が少ないほど幅は広い）")
        md += ["",
               "| 信号日 | 取引数 | 勝率 | 平均損益 |", "|---|---|---|---|"]
        for d_, r_ in gd_.sort_values("n", ascending=False).head(25).iterrows():
            md.append(f"| {d_:%Y-%m-%d} | {int(r_['n'])} | {pct(r_['w'] / r_['n'])} | {pct(r_['p'] / r_['n'], 2)} |")
    else:
        md.append("- 信号日が3日未満のため計算しない")
    # 相場の状態別：市場全体の下げの長さ・深さで分けた成績（長い下げ相場でも効くか）
    md += ["", "## 相場の状態別（市場全体の下げの長さ・深さ）", "", "| 区分 | 取引数 | 勝率 | 平均損益 |", "|---|---|---|---|"]
    for f_, cuts, unit in (("mkt_hi_days", [-1, 10, 30, 60, 1e9], "日"), ("mkt_dd120", [-1, -0.2, -0.1, -0.05, 1], ""),
                           ("mkt_dd60", [-1, -0.15, -0.10, -0.05, 1], "")):
        if f_ not in q.columns:
            continue
        v_ = q.loc[t.index, f_]
        for lo_, hi_ in zip(cuts[:-1], cuts[1:]):
            g_ = t[(v_ > lo_) & (v_ <= hi_)]
            if len(g_):
                rng_ = (f"{lo_ + 1:g}〜{hi_:g}{unit}" if unit else f"{lo_*100:+.0f}%〜{hi_*100:+.0f}%").replace("1e+09", "")
                md.append(f"| {LABEL.get(f_, f_)} {rng_} | {len(g_)} | {pct(g_['win'].mean())} | {pct(g_['pnl'].mean(), 2)} |")
    # 独立した局面ごと：信号日の間が30暦日より空いたら別の局面とみなす（同じ局面の取引は独立な試行ではない）
    ds = t.sort_values("date")
    ep = (ds["date"].diff().dt.days.fillna(0) > 30).cumsum()
    eps = ds.groupby(ep).agg(start=("date", "min"), end=("date", "max"), n=("win", "size"), win=("win", "mean"), pnl=("pnl", "mean"))
    md += ["", "## 独立した局面ごと（信号日の間が30日より空いたら別の局面）", "",
           f"- 局面の数 {len(eps)} ／ 局面を等しく数えた勝率 {pct(eps['win'].mean())} ／ 平均損益 {pct(eps['pnl'].mean(), 2)} ／ "
           f"勝率50%未満の局面 {int((eps['win'] < 0.5).sum())}", "",
           "| 開始 | 終了 | 取引数 | 勝率 | 平均損益 |", "|---|---|---|---|---|"]
    for r in eps.itertuples():
        md.append(f"| {r.start.date()} | {r.end.date()} | {r.n} | {pct(r.win)} | {pct(r.pnl, 2)} |")
    # 銘柄の集中
    cc = t["code"].value_counts()
    md += ["", "## 銘柄の集中", "",
           f"- 銘柄数 {len(cc)} ／ 最多銘柄 {cc.index[0]} が {cc.iloc[0]} 回（{cc.iloc[0]/len(t)*100:.0f}%）／ "
           f"上位10銘柄で {cc.iloc[:10].sum()/len(t)*100:.0f}%"]
    if "sector" in q.columns:
        sec = q.loc[t.index, "sector"].value_counts()
        md.append("- 業種の上位：" + "、".join(f"{s} {n}回" for s, n in sec.head(5).items()))

    def row(lbl, r):
        sp = r.get("splits") or []
        return (f"| {lbl} | {r.get('n', 0)} | {pct(r.get('win'))} | {pct(r.get('pnl'), 2)} | "
                + " / ".join(f"{s['n']}:{pct(s['win'])}" for s in sp) + " |")
    md += ["", "## 条件を1つずつ外したとき", "",
           "| 外した条件 | 取引数 | 勝率 | 平均損益 | 各期間(取引数:勝率) |", "|---|---|---|---|---|",
           row("（外さない）", evaluate(q, conds, edges))]
    for i, c in enumerate(conds):
        md.append(row(fmt_cond(*c), evaluate(q, conds[:i] + conds[i + 1:], edges)))
    # 担当2：条件の効き目が「少数の取引の入れ替わり」だけで出ていないか（同一銘柄の重複除外で買う日がずれる分も含む）
    try:
        def _keys(d):
            return list(zip(d["code"].astype(str), pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")))
        kt = set(_keys(t))
        md += ["", "## 条件ごとの取引の入れ替わり（その条件を外した版との差）", "",
               "条件を外すと増える取引（＝その条件が除いた取引）と、外すと消える取引（＝同一銘柄の重複除外で買う日がずれて"
               "その条件があるときだけ現れる取引）。入れ替わりが数件だけで勝率差が大きい条件は、少数の取引による偶然の疑い。", "",
               "| 条件 | 除いた取引 | その勝率 | その平均損益 | 条件があるときだけの取引 | その勝率 | 外した版との勝率差 |",
               "|---|---|---|---|---|---|---|"]
        small = []
        for i, c in enumerate(conds):
            t2 = dedup_trades(q, apply(q, conds[:i] + conds[i + 1:]))
            k2 = set(_keys(t2))
            rem = t2[[k not in kt for k in _keys(t2)]]
            add = t[[k not in k2 for k in _keys(t)]]
            dwin = (t["win"].mean() - t2["win"].mean()) * 100 if len(t2) else float("nan")
            md.append(f"| {fmt_cond(*c)} | {len(rem)} | {pct(rem['win'].mean()) if len(rem) else '-'} | "
                      f"{pct(rem['pnl'].mean(), 2) if len(rem) else '-'} | {len(add)} | "
                      f"{pct(add['win'].mean()) if len(add) else '-'} | {dwin:+.1f}pt |")
            if 0 < len(rem) + len(add) <= 12:
                small.append((c, rem, add))
        for c, rem, add in small:
            md += ["", f"- 入れ替わりが少ない条件：{fmt_cond(*c)}（除いた {len(rem)} 件・現れた {len(add)} 件）", "",
                   "| 区分 | 信号日 | 銘柄 | 勝ち | 損益 |", "|---|---|---|---|---|"]
            for lbl, d in (("除いた", rem), ("現れた", add)):
                for _, r in d.sort_values("date").iterrows():
                    md.append(f"| {lbl} | {pd.Timestamp(r['date']).date()} | {r['code']} | {int(r['win'])} | {pct(r['pnl'], 2)} |")
    except Exception as e:  # 情報用の出力なので失敗しても本体は止めない
        md += ["", f"（取引の入れ替わりの点検は失敗：{e}）"]
    md += ["", "## 閾値を少しずらしたとき（崖のように急に悪化するなら偶然の疑い）", "",
           "| 条件 → ずらした値 | 取引数 | 勝率 | 平均損益 | 各期間(取引数:勝率) |", "|---|---|---|---|---|"]
    for i, (f, op, th) in enumerate(conds):
        for v in _shift_variants(f, op, th):
            cs = conds[:i] + [[f, op, v]] + conds[i + 1:]
            md.append(row(f"{fmt_cond(f, op, th)} → {fmt_cond(f, op, v)}", evaluate(q, cs, edges)))
    # 条件を減らした簡単な版：外しても勝率が最も下がらない条件から順に1つずつ外していく（どの条件が勝率を支えているか）
    def weeks(cs):
        tt = dedup_trades(q, apply(q, cs))
        return int(pd.to_datetime(tt["date"]).dt.to_period("W").nunique()) if len(tt) else 0
    md += ["", "## 条件を減らした簡単な版（勝率への影響が小さい条件から順に外す）", "",
           "| 外した条件（累積） | 残りの条件数 | 取引数 | 取引のあった週 | 勝率 | 平均損益 | 各期間(取引数:勝率) |",
           "|---|---|---|---|---|---|---|"]
    cur = list(conds)
    core3 = None
    r0 = evaluate(q, cur, edges)
    md.append(f"| （元のルール） | {len(cur)} | {r0.get('n', 0)} | {weeks(cur)} | {pct(r0.get('win'))} | {pct(r0.get('pnl'), 2)} | "
              + " / ".join(f"{s['n']}:{pct(s['win'])}" for s in r0.get("splits") or []) + " |")
    while len(cur) > 1:
        best = None
        for i in range(len(cur)):
            cs = cur[:i] + cur[i + 1:]
            r = evaluate(q, cs, edges)
            if best is None or (r.get("win") or 0) > (best[1].get("win") or 0):
                best = (i, r, cs)
        i, r, cs = best
        md.append(f"| {fmt_cond(*cur[i])} | {len(cs)} | {r.get('n', 0)} | {weeks(cs)} | {pct(r.get('win'))} | {pct(r.get('pnl'), 2)} | "
                  + " / ".join(f"{s['n']}:{pct(s['win'])}" for s in r.get("splits") or []) + " |")
        cur = cs
        if len(cur) == 3:
            core3 = list(cur)
    # 簡単な3条件版に1条件ずつ足したとき：局面（週）を減らさずに勝率を上げる条件はどれか（要因別の成績の上位150条件から）
    if singles and core3:
        rc = evaluate(q, core3, edges)
        wc = weeks(core3)
        used = {c[0] for c in core3}
        rows3 = []
        for sg in [x["cond"] for x in singles[:150]]:
            if sg[0] in used:
                continue
            cs = core3 + [list(sg)]
            r = evaluate(q, cs, edges)
            sp = r.get("splits") or []
            if r.get("n", 0) >= 300 and all(x["n"] >= 30 and (x["pnl"] or -1) > 0 for x in sp):
                rows3.append((r["win"], r["n"], weeks(cs), min(x["win"] for x in sp), fmt_cond(*sg)))
        rows3.sort(key=lambda x: -x[0])
        md += ["", f"## 簡単な3条件版に1条件ずつ足したとき（取引300件以上・各期間30件以上・損益プラスのもの、上位30）", "",
               f"3条件版：{' かつ '.join(fmt_cond(*c) for c in core3)} → 勝率 {pct(rc.get('win'))} 取引数 {rc.get('n', 0)} 週 {wc}", "",
               "| 足した条件 | 勝率 | 取引数 | 取引のあった週 | 最悪期間の勝率 |", "|---|---|---|---|---|"]
        md += [f"| {lbl} | {pct(w)} | {n} | {wk_} | {pct(mn)} |" for w, n, wk_, mn, lbl in rows3[:30]]
    # seeds の局面集中（勝率が高くても少数の局面に偏っていないか）
    try:
        seeds = json.load(open(os.path.join("rules", "seeds.json"), encoding="utf-8")).get("seeds", [])
    except Exception:
        seeds = []
    srows = []
    for s in seeds:
        cs = [list(c) for c in s.get("conds", [])]
        tt = dedup_trades(q, apply(q, cs))
        if len(tt) < 30:
            continue
        dd = pd.to_datetime(tt["date"]).sort_values()
        epn = int((dd.diff().dt.days.fillna(0) > 30).sum() + 1)
        wk = dd.dt.to_period("W").value_counts()
        qw = tt.groupby(pd.to_datetime(tt["date"]).dt.to_period("Q"))["win"].agg(["mean", "size"])
        qw5 = qw[qw["size"] >= 5]
        # 仮説(担当2 30周目)：信号日を単位にしたブートストラップ（500回）の勝率の5%点＝同じ日の重なりを考えた悪い側の勝率
        gdw = tt.groupby(pd.to_datetime(tt["date"]).dt.normalize())["win"].agg(["sum", "size"])
        b5 = None
        if len(gdw) >= 3:
            rs_ = np.random.default_rng(0)
            sw_, sn_ = gdw["sum"].to_numpy(float), gdw["size"].to_numpy(float)
            ix_ = rs_.integers(0, len(gdw), (500, len(gdw)))
            b5 = float(np.percentile(sw_[ix_].sum(1) / sn_[ix_].sum(1), 5))
        srows.append((tt["win"].mean(), len(tt), tt["pnl"].mean(), wk.size, wk.iloc[:3].sum() / len(tt), epn,
                      " かつ ".join(fmt_cond(*c) for c in cs), qw["mean"].mean(), len(qw),
                      (qw5["mean"].min() if len(qw5) else None), b5))
    if srows:
        srows.sort(key=lambda x: -x[0])
        md += ["", f"## seeds の局面集中（取引30件以上の {len(srows)} 個、勝率の高い順）", "",
               "（四半期の勝率＝取引のあった四半期を1つずつ等しく数えた勝率、最悪の四半期＝取引5件以上の四半期で最も低い勝率、"
               "信号日単位5%点＝信号日を単位にしたブートストラップ500回の勝率の5%点）", "",
               "| 勝率 | 取引数 | 平均損益 | 週数 | 上位3週の割合 | 局面の数 | 四半期の勝率 | 四半期数 | 最悪の四半期 | 信号日単位5%点 | 条件 |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {pct(w)} | {n} | {pct(p, 2)} | {wkn} | {t3*100:.0f}% | {e} | {pct(qm)} | {qn} | {pct(qmin)} | {pct(b5)} | {lbl} |"
               for w, n, p, wkn, t3, e, lbl, qm, qn, qmin, b5 in srows]
    # 担当2が検証中の条件について、seeds からその条件を外した版との差（その条件の上乗せが5年分でもあるか）
    CHECK = ("disc_share", "mkt_us_corr20", "mkt_us_corr_chg", "mkt_corr_chg", "stop_lo5", "mkt_stop_lo", "xsec_disp_rel60")
    crow = []
    for s in seeds:
        cs = [list(c) for c in s.get("conds", [])]
        for j, cnd in enumerate(cs):
            if cnd[0] in CHECK and len(cs) > 1:
                rw, ro = evaluate(q, cs, edges), evaluate(q, cs[:j] + cs[j + 1:], edges)
                if rw.get("n") and ro.get("n"):
                    crow.append(f"| {fmt_cond(*cnd)} | {pct(ro['win'])} (n={ro['n']}) | {pct(rw['win'])} (n={rw['n']}) | "
                                f"{(rw['win'] - ro['win']) * 100:+.1f}pt | {min(x['win'] or 0 for x in ro['splits'])*100:.0f}% → "
                                f"{min(x['win'] or 0 for x in rw['splits'])*100:.0f}% | {' かつ '.join(fmt_cond(*c) for c in cs[:j] + cs[j + 1:])} |")
    if crow:
        md += ["", "## 検証中の条件の上乗せ（seeds からその条件を外した版との比較）", "",
               "| 条件 | 外した版の勝率 | 足した版の勝率 | 差 | 最悪期間の勝率 | 元の条件 |", "|---|---|---|---|---|---|"] + crow
    _write_keep_manual(path, md)


ROBUST_HTML = os.path.join("docs", "robustness.html")


def _md_to_html(text):
    """robustness.md をサイト用の簡単な HTML にする（見出し・箇条書き・表・太字のみ）"""
    import html as _h
    import re as _re
    def inl(x):
        return _re.sub(r"\*\*(.+?)\*\*", r"<b></b>", _h.escape(x))
    out, i, lines = [], 0, text.splitlines()
    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.startswith("<!--") or not ln.strip():
            i += 1
            continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                th = "".join(f"<th>{inl(c)}</th>" for c in rows[0])
                tb = "".join("<tr>" + "".join(f"<td>{inl(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
                out.append(f"<div class='tablewrap'><table><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table></div>")
            continue
        if ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                items.append(f"<li>{inl(lines[i].lstrip()[2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        if ln.startswith("# "):
            out.append(f"<h1>{inl(ln[2:])}</h1>")
        elif ln.startswith("## "):
            out.append(f"<h2>{inl(ln[3:])}</h2>")
        else:
            out.append(f"<p>{inl(ln)}</p>")
        i += 1
    return "\n".join(out)


def write_robust_html():
    if not os.path.exists(ROBUST_MD):
        return
    try:
        from web_report import _page
    except Exception:
        _page = None
    body = _md_to_html(open(ROBUST_MD, encoding="utf-8").read())
    html = _page("採用ルールの点検", "rules", body) if _page else f"<!doctype html><meta charset=utf-8><body>{body}"
    os.makedirs("docs", exist_ok=True)
    open(ROBUST_HTML, "w", encoding="utf-8").write(html)


def _write_keep_manual(path, md):
    """研究担当が手で書いた考察（AUTO 印より前）は残し、自動部分だけ書き換える"""
    mark = "<!-- AUTO: ここから下は rule_lab.py が毎回書き換える -->"
    head = ""
    if os.path.exists(path):
        old = open(path, encoding="utf-8").read()
        if mark in old:
            head = old.split(mark)[0]
    open(path, "w", encoding="utf-8").write(head + mark + "\n\n" + "\n".join(md) + "\n")


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
    run("add", *[p for p in (RULES_MD, RULES_HTML, STATE, os.path.join("rules", "feature_report.md"), ROBUST_MD, ROBUST_HTML)
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
    try:   # 担当2の点検（情報用。採用判定には使わない）
        robustness_report(q, champ["conds"], edges, base, singles=state.get("singles"))
        write_robust_html()
        print(f"{ROBUST_MD}・{ROBUST_HTML} を更新")
    except Exception as e:
        print("点検レポートの作成に失敗:", e)
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
