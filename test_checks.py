# -*- coding: utf-8 -*-
"""チェックリストを合成データで検証する（LLM・API不使用）。"""
import datetime as dt
import pandas as pd
import checks, config, fundamentals, jobs

def fin_of(sh_now, sh_past, op=[900,1100,1400,1500], fc=[1000,1200,1300,1400]):
    rows=[]
    for i,(d,o,f) in enumerate(zip(
            [dt.datetime(2022,5,10),dt.datetime(2023,5,10),
             dt.datetime(2024,5,10),dt.datetime(2025,5,10)], op, fc)):
        rows.append({"code":"11110","disclosed_date":d,"period_type":"FY",
                     "period_end":d-dt.timedelta(days=40),"operating_profit":o,
                     "fc_operating_profit":f,"net_sales":o*8,
                     "shares_outstanding": sh_past if i<3 else sh_now})
    return pd.DataFrame(rows)

def quotes_of(mult, n=80):
    d = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame({"code":"11110","date":d,
                         "close":[1000*(mult**(i/(n-1))) for i in range(n)],
                         "volume":1000,"turnover_value":5e8})

print("=== 各チェックが式を返すか ===")
for r in [checks.check_dilution(fin_of(13_000_000,10_000_000)),
          checks.check_already_run_up(quotes_of(4.0)),
          checks.check_psr(5e10, 2e9, "FY"),
          checks.check_turnover_vs_fundamentals(5.0, 10),
          checks.check_management(fundamentals.management_score(fin_of(1e7,1e7)))]:
    print(f"\n [{r['項目']}] {r['判定']}")
    print(f"   式  : {r['式']}")
    print(f"   閾値: {r['閾値']}")
    print(f"   出典: {r['出典']}")
    assert r["式"] and r["式"] != "計算不可（データ欠損）", r

print("\n=== 「調べて問題なし」は適合になるか ===")
ok = checks.check_dilution(fin_of(10_050_000,10_000_000))
print(f"  希薄化 +0.5% → {ok['判定']}  （{ok['式']}）")
assert ok["判定"] == checks.PASS

print("\n=== 「調べられなかった」は不明→不合格か ===")
mgmt = fundamentals.management_score(fin_of(1e7,1e7))
row = {"code":"11110","turnover_spike":1.2,"op_yoy":50.0,"market_cap":1e10,
       "net_sales":5e9,"period_type":"FY","large_holding_filed":False,"close":1200,
       "market_cap_oku":100.0,"turnover_oku":5.0,"sales_yoy":20.0,"pct_from_high":-5.0}
res = checks.run_all(row, quotes_of(1.2),
                     fin_of(1e7,1e7).drop(columns=["shares_outstanding"]),
                     mgmt, pd.DataFrame(columns=["code","date"]), dt.date(2026,7,23))
print(f"  不明{res['不明']}件 / 不適合{res['不適合']}件 → 全項目通過={res['全項目通過']}")
assert res["全項目通過"] is False

print("\n=== スクリーニング通過理由（Role 2）の式 ===")
for b in checks.screening_basis(row, config):
    print(f"  {b['項目']:16s} {b['式']}")
assert len(checks.screening_basis(row, config)) >= 4

print("\n=== HTML生成（式・出典・開示一覧を含むか）===")
docs = pd.DataFrame([{"submit_date":"2026-07-10","doc_type":"臨時報告書",
                      "doc_description":"臨時報告書","filer_name":"テスト社"}])
html = jobs._check_html("1111","テスト社", checks.screening_basis(row, config),
                        mgmt, fundamentals.execution_plan(1200.0), res, docs)
for must in ["計算式","出典","EDINET原文","見ていない失敗要因","なぜ抽出されたか"]:
    assert must in html, must
print(f"  OK（{len(html)}文字・必須項目すべて含む）")
print("\nすべて期待通りです。")
