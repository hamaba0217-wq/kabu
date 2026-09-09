# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd

CACHE_DIR = "data"

def _path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.pkl")

def load(name: str):
    p = _path(name)
    if os.path.exists(p):
        try:
            return pd.read_pickle(p)
        except Exception:
            pass
    return None

def save(name: str, obj) -> bool:
    try:
        p = _path(name)
        pd.to_pickle(obj, p)
        return True
    except Exception:
        return False
