# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import numpy as np

def _bad_flags(fin_row):
    res = {"予想未達": False, "減益": False, "減収": False}
    if fin_row is None or len(fin_row) == 0:
        return res
    
    op = fin_row.get("operating_profit")
    fc_op = fin_row.get("fc_operating_profit")
    if pd.notna(op) and pd.notna(fc_op) and fc_op > 0:
        if op < fc_op:
            res["予想未達"] = True

    op_prev = fin_row.get("op_prev")
    if pd.notna(op) and pd.notna(op_prev) and op_prev > 0:
        if op < op_prev:
            res["減益"] = True

    sales = fin_row.get("net_sales")
    sales_prev = fin_row.get("sales_prev")
    if pd.notna(sales) and pd.notna(sales_prev) and sales_prev > 0:
        if sales < sales_prev:
            res["減収"] = True

    return res

def _latest_fin_before(fin, code, as_of):
    if fin is None or fin.empty:
        return None
    sub = fin[(fin["code"].astype(str) == str(code)) & (fin["disclosed_date"] <= pd.Timestamp(as_of))]
    if sub.empty:
        return None
    return sub.sort_values("disclosed_date").iloc[-1]

def _prep_fin_extended(fin):
    if fin is None or fin.empty:
        return pd.DataFrame()
    f = fin.copy()
    f = f.sort_values(["code", "disclosed_date"])
    if "operating_profit" in f.columns:
        f["op_prev"] = f.groupby(["code", "period_type"])["operating_profit"].shift(1)
    if "net_sales" in f.columns:
        f["sales_prev"] = f.groupby(["code", "period_type"])["net_sales"].shift(1)
    return f
