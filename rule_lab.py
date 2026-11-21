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

def load_data(fetch: bool) -> pd.DataFrame:
    import cache
    if fetch and cache.load("quotes") is None:
        raise SystemExit("data/quotes.pkl がまだありません（初回取得中の可能性）。今回はスキップ")
    if fetch:
        try:
            from sources import JQuants
            JQuants().quotes(1100)
        except Exception as e:
            print(f"[警告] 差分取得に失敗（キャッシュで続行）: {e}")
    q = cache.load("quotes")
    if q is None:
        raise SystemExit("data/quotes.pkl がありません。python fetch_data.py を先に実行してください")
    cols = [c for c in ["code", "date", "open", "high", "low", "close", "volume", "turnover_value"]
            if c in q.columns]
    q = q[cols].copy()
    q["date"] = pd.to_datetime(q["date"])
    q = q.dropna(subset=["open", "high", "low", "close"])
    q = q[(q["close"] > 0) & (q["open"] > 0)]
    q = q.sort_values(["code", "date"]).reset_index(drop=True)
    # 業種
    lst = cache.load("listed")
    if lst is not None:
        sc = next((c for c in ["sector17", "Sector17CodeName", "S17Nm", "sector"] if c in lst.columns), None)
        cc = next((c for c in ["code", "Code"] if c in lst.columns), None)
        if sc and cc:
            m = lst[[cc, sc]].drop_duplicates(cc)
            m.columns = ["code", "sector"]
            m["code"] = m["code"].astype(str)
            q = q.merge(m, on="code", how="left")
    if "sector" not in q.columns:
        q["sector"] = "不明"
    q["sector"] = q["sector"].fillna("不明")
    return q


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
    # 市場全体（全銘柄の20日騰落の中央値＝相場の地合い）
    mk = q.groupby("date")["ret20"].median().rename("mkt20")
    q = q.merge(mk, on="date", how="left")
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
}

LABEL = {
    "ret1": "前日比", "ret5": "5日騰落", "ret20": "20日騰落", "ret60": "60日騰落",
    "dev25": "25日線乖離", "dev75": "75日線乖離", "ma_up": "5日線>25日線",
    "from_hi": "52週高値からの位置", "from_lo60": "60日安値からの上昇", "vol20": "20日ボラ(日次)",
    "tv20": "20日平均売買代金", "vratio": "出来高倍率(20日平均比)", "rsi14": "RSI14",
    "clpos": "当日レンジ内の終値位置", "gap": "当日の窓", "body": "当日の陽線/陰線幅",
    "mkt20": "市場全体の20日騰落(中央値)", "close": "株価",
}


def fmt_cond(f, op, th):
    lab = LABEL.get(f, f)
    if f == "ma_up":
        return lab
    if f == "tv20":
        return f"{lab} {op} {th/1e8:g}億円"
    if f in ("close",):
        return f"{lab} {op} {th:g}円"
    if f in ("vratio", "rsi14", "clpos"):
        return f"{lab} {op} {th:g}"
    return f"{lab} {op} {th*100:+g}%"


def rule_key(conds):
    s = json.dumps(sorted([list(map(str, c)) for c in conds]))
    return hashlib.md5(s.encode()).hexdigest()[:12]


def apply(q, conds):
    m = pd.Series(True, index=q.index)
    for f, op, th in BASE + conds:
        m &= (q[f] >= th) if op == ">=" else (q[f] <= th)
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
    run("add", *[p for p in (RULES_MD, RULES_HTML, STATE) if os.path.exists(p)])
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
