# -*- coding: utf-8 -*-
"""
ローカルキャッシュ（差分取得）

一度取得した株価・決算データをローカルに保存し、
次回以降は「保存済みの最終日の翌日 〜 今日」の差分だけを取りに行きます。

保存形式について
----------------
pandas標準の to_pickle / read_pickle を使います。
Parquet（pyarrow）と違い、**追加ライブラリのインストールが不要**です。
pandas さえ入っていれば動くので、環境構築の失敗要因を1つ減らせます。
圧縮も指定でき、CSVより小さく・速く読み書きできます。

保存場所: data/ フォルダ
  data/quotes.pkl   株価日足
  data/fin.pkl      決算サマリー
  data/listed.pkl   銘柄一覧
"""

from __future__ import annotations

import datetime as dt
import os

import pandas as pd

CACHE_DIR = "data"
JST = dt.timezone(dt.timedelta(hours=9))


def _path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.pkl")


def load(name: str) -> pd.DataFrame | None:
    """キャッシュを読む。無ければ None。壊れていても None を返して再取得させる。"""
    p = _path(name)
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_pickle(p, compression="gzip")
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return None
        return df
    except Exception as e:
        print(f"  [警告] キャッシュ {p} の読み込みに失敗（無視して再取得します）: {e}")
        return None


def save(name: str, df: pd.DataFrame) -> bool:
    """キャッシュを保存。成功なら True。失敗しても処理は止めない。"""
    if df is None or len(df) == 0:
        return False
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = _path(name) + ".tmp"
    try:
        # 一時ファイルに書いてから置き換える（書き込み中の破損を防ぐ）
        df.to_pickle(tmp, compression="gzip")
        os.replace(tmp, _path(name))
        return True
    except Exception as e:
        print(f"  [警告] キャッシュ {name} の保存に失敗（処理は続行します）: {e}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False


def last_date(df: pd.DataFrame | None, date_col: str = "date") -> dt.date | None:
    """キャッシュ済みデータの最終日を返す。"""
    if df is None or len(df) == 0 or date_col not in df.columns:
        return None
    s = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if len(s) == 0:
        return None
    return s.max().date()


def merge(old: pd.DataFrame | None, new: pd.DataFrame,
          keys: list[str]) -> pd.DataFrame:
    """既存と新規を結合し、重複（同じ日・同じ銘柄）を除く。"""
    if old is None or len(old) == 0:
        combined = new
    elif new is None or len(new) == 0:
        combined = old
    else:
        combined = pd.concat([old, new], ignore_index=True)
    if len(combined) and all(k in combined.columns for k in keys):
        combined = combined.drop_duplicates(subset=keys, keep="last")
    return combined.reset_index(drop=True)
