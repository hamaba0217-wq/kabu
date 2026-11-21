"""
PCの手元データ（data/）で、仮説をすぐに確かめるための速報ツール（研究担当のAI用）。

- 判定・勝ちの定義・取引の重複除外・3分割は rule_lab.py と完全に同じ関数を使う。
- 手元のデータは期間が短い（例：約8か月）ため、ここでの結果は「速報」。
  ルールの採用は GitHub Actions（長い期間のデータ）の rule_lab.py だけが行う。
- 使い方：
    py quick_lab.py                 要因1つずつの成績 → rules/quick_report.md
    py quick_lab.py --only up_streak,vdry   指定した要因だけ表示（書き出しなし）
    py quick_lab.py --combo "ret20<=-0.2;vdry<=0.7"   条件の組み合わせを評価（書き出しなし）
- 特徴量は output/quick_features.pkl にキャッシュする（rule_lab.py・データが変わると作り直す）。
"""
import argparse
import datetime as dt
import hashlib
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
import rule_lab as L  # noqa: E402

CACHE = os.path.join("output", "quick_features.pkl")
REPORT = os.path.join("rules", "quick_report.md")


def _sig():
    h = hashlib.md5(open("rule_lab.py", "rb").read())
    for f in ("raw_quotes.pkl", "fin.pkl", "macro.pkl", "listed.pkl"):
        p = os.path.join("data", f)
        if os.path.exists(p):
            h.update(f"{f}{os.path.getmtime(p)}{os.path.getsize(p)}".encode())
    return h.hexdigest()


def features():
    sig = _sig()
    if os.path.exists(CACHE):
        try:
            obj = pd.read_pickle(CACHE)
            if obj.get("sig") == sig:
                return obj["q"]
        except Exception:
            pass
    q = L.add_features(L.load_data(False))
    os.makedirs("output", exist_ok=True)
    pd.to_pickle({"sig": sig, "q": q}, CACHE)
    return q


def parse_combo(s):
    out = []
    for part in s.split(";"):
        part = part.strip()
        for op in (">=", "<=", "=="):
            if op in part:
                f, th = part.split(op)
                out.append([f.strip(), op, float(th)])
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--combo", default="")
    a = ap.parse_args()

    q = features()
    valid = q[q["win"].notna()]
    d0, d1 = valid["date"].min(), valid["date"].max()
    edges = list(pd.date_range(d0, d1 + pd.Timedelta(days=1), periods=4))
    base = L.evaluate(q, [], edges)
    head = (f"手元データ {q['code'].nunique()}銘柄 {q['date'].min().date()}〜{q['date'].max().date()} ／ "
            f"検証期間 {d0.date()}〜{d1.date()} ／ 基準（流動性のみ）勝率 {L.pct(base['win'])} "
            f"n={base['n']} 平均損益 {L.pct(base['pnl'], 2)}")
    print(head)

    if a.combo:
        cs = parse_combo(a.combo)
        r = L.evaluate(q, cs, edges)
        print(" かつ ".join(L.fmt_cond(*c) for c in cs))
        print(f"  n={r['n']} 勝率 {L.pct(r.get('win'))} 平均損益 {L.pct(r.get('pnl'), 2)} "
              f"各期間 {[(s['n'], L.pct(s['win']), L.pct(s['pnl'], 2)) for s in r.get('splits', [])]}")
        return

    only = {x.strip() for x in a.only.split(",") if x.strip()}
    if only:
        keep = {k: v for k, v in L.GRID.items() if k in only}
        L.GRID, saved = keep, L.GRID
    rows = L.single_report(q, edges, base)
    for r in rows:
        print(f"{r['lift']*100:+6.1f}pt 勝率 {L.pct(r['win'])} 最悪期間 {L.pct(r['min_split_win'])} "
              f"損益 {L.pct(r['pnl'], 2)} n={r['n']}  {r['label']}")
    if only:
        return

    md = ["# 速報：要因別の成績（PCの手元データ）", "",
          "**注意：期間が短いデータでの速報です。ルールの採用には使いません**"
          "（採用は GitHub Actions の長期間データで rule_lab.py が判定）。", "",
          f"更新 {dt.datetime.now():%Y-%m-%d %H:%M} ／ {head}", "",
          "| 条件 | 取引数 | 勝率 | 基準との差 | 最悪期間の勝率 | 平均損益 |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['label']} | {r['n']} | {L.pct(r['win'])} | {r['lift']*100:+.1f}pt | "
                  f"{L.pct(r['min_split_win'])} | {L.pct(r['pnl'], 2)} |")
    os.makedirs("rules", exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"→ {REPORT}（{len(rows)}条件）")


if __name__ == "__main__":
    main()
