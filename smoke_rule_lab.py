# -*- coding: utf-8 -*-
"""rule_lab.py の動作確認（架空データ・APIキー不要・数十秒）。研究で rule_lab を変更したら必ず実行。
   python smoke_rule_lab.py  → 最後に SMOKE OK が出れば合格"""
import os
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault("RULE_LAB_FETCH_MACRO", "0")
import rule_lab as R

tmp = tempfile.mkdtemp()
R.RULES_MD = os.path.join(tmp, "t.md")
R.RULES_HTML = os.path.join(tmp, "t.html")
R.SIG_HTML = os.path.join(tmp, "sig.html")      # 本物の docs/ を書き換えないよう一時フォルダへ
R.INDEX_HTML = os.path.join(tmp, "index.html")
open(R.INDEX_HTML, "w", encoding="utf-8").write(
    '<div class="nav"><a href="index.html">今日の候補</a></div><h1>t</h1><div class="date-badge">基準日</div><p>x</p>')
rng = np.random.default_rng(0)
dates = pd.bdate_range("2023-01-02", periods=420)
codes = [f"{1000 + c}0" for c in range(150)] + ["15400", "25100", "15700", "13570", "13210", "13430"]
rows = []
for c in codes:
    p = 1000 * np.exp(np.cumsum(rng.normal(0, 0.025, len(dates))))
    rows.append(pd.DataFrame(dict(code=c, date=dates, open=p, high=p * 1.03, low=p * 0.97, close=p,
                                  volume=1e6, turnover_value=p * 1e6, adj_factor=1.0, sector="テスト")))
q = R.add_features(R._adjust(pd.concat(rows, ignore_index=True)))
v = q[q["win"].notna()]
edges = list(pd.date_range(v["date"].min(), v["date"].max() + pd.Timedelta(days=1), periods=4))
base = R.evaluate(q, [], edges)
assert base["n"] > 0, "取引が0件"
n_ok = 0
for f, lst in R.GRID.items():
    assert f in R.LABEL, f"LABEL に {f} がありません"
    for op, th in lst:
        R.fmt_cond(f, op, th)
        n_ok += 1
cands = R.candidates([], {})
for cs in cands[:30]:
    R.evaluate(q, cs, edges)
st = {"champion": {"conds": [], **base}, "history": [], "tested": {}, "baseline": base,
      "updated": "test", "singles": R.single_report(q, edges, base)}
R.write_outputs(st, q, q["date"].max())
R.write_outputs(st, q, q["date"].max())   # 2回目は欄の差し替え（重複しないこと）
_ix = open(R.INDEX_HTML, encoding="utf-8").read()
assert _ix.count(R.SIG_START) == 1 and 'href="rules.html"' in _ix, "トップページの欄・メニューの差し込みに失敗"
R.robustness_report(q, [["ret20", "<=", -0.1], ["hammer5", ">=", 1]], edges, base, path=os.path.join(tmp, "rb.md"))
print(open(os.path.join(tmp, "rb.md"), encoding="utf-8").read()[:600])
missing = [f for f in R.GRID if f not in q.columns]
print("列が無い要因（本番データでのみ計算されるものは可）:", missing)
print(f"SMOKE OK 条件{n_ok}個 候補{len(cands)}件 基準勝率{base['win']:.3f}")
