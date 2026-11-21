# -*- coding: utf-8 -*-
"""勝率分析用の株価・決算データを data/ にキャッシュ（差分取得）"""
import sys
from sources import JQuants
days = int(sys.argv[1]) if len(sys.argv) > 1 else 1100
jq = JQuants()
q = jq.quotes(days)
print("quotes", q.shape, q["date"].min(), q["date"].max())
try:
    f = jq.financials(days)
    print("fin", f.shape)
except Exception as e:
    print("fin error", e)
try:
    l = jq.listed()
    print("listed", l.shape)
except Exception as e:
    print("listed error", e)
