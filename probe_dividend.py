# -*- coding: utf-8 -*-
"""
J-Quantsの配当データが取れるか、どんな列があるかを確認する診断スクリプト。
  py probe_dividend.py
出力をチャットに貼ってください。それを見て、配当列の実装を決めます。
"""

from __future__ import annotations
import os
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def main():
    import config
    from sources import JQuants
    jq = JQuants()
    client = jq.cli  # 内部のjquants-api-clientを使う

    print("=" * 70)
    print("J-Quants 配当データの取得を試します")
    print("=" * 70)

    # まず、配当らしきメソッドを全部探す
    cand = [m for m in dir(client)
            if ("div" in m.lower() or "fin" in m.lower()) and m.startswith("get")]
    print("\n配当・財務らしきメソッド一覧:")
    for m in cand:
        print(f"    {m}")

    # 配当API（get_fin_dividend など）を試す。トヨタ(72030)で確認。
    code = "72030"
    got = False
    for method in ["get_fin_dividend", "get_fins_dividend", "get_dividend",
                   "get_eq_dividend", "get_fin_dividends"]:
        if hasattr(client, method):
            print(f"\nメソッド `{method}` を発見。{code} で取得を試みます…")
            try:
                df = getattr(client, method)(code=code)
                got = True
                print(f"  → 取得成功: {len(df)}行")
                if len(df) > 0:
                    print(f"\n【列名一覧】（これが最も重要）:")
                    for c in df.columns:
                        print(f"    {c}")
                    print(f"\n【最新3件の中身】:")
                    print(df.tail(3).to_string(index=False))
                break
            except Exception as e:
                print(f"  → 失敗: {type(e).__name__}: {e}")
    if not got:
        print("\n配当メソッドが見つからないか、全て失敗しました。")
        print("上の『配当・財務らしきメソッド一覧』を、チャットに貼ってください。")


if __name__ == "__main__":
    main()
