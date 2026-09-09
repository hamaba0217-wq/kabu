# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import numpy as np

def _bad_flags(fin):
    return set()

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
    return fin.copy()
