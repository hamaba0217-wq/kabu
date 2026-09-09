# -*- coding: utf-8 -*-
"""
機関投資家の空売り残高（0.5%以上の報告義務分）が、Standardプランで取れるか診断する。
  py probe_short.py
出力をチャットに貼ってください。取れる列を見て、実装を判断します。

背景（事実）
------------
  ・今キャッシュにあるのは「業種別空売り比率」（個別銘柄ではなく業種単位）。
  ・「機関の空売り残」は、空売り残高報告（大口の空売りポジション報告）で、別API。
  ・これが Standard プランで取れるか（配当のように 403 で弾かれないか）を確認する。
"""

from __future__ import annotations
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def main():
    from sources import JQuants
    jq = JQuants()
    client = jq.cli

    print("=" * 70)
    print("空売り残高（機関の報告分）が取れるか診断")
    print("=" * 70)

    # 空売り関連らしきメソッドを全部探す
    cand = [m for m in dir(client)
            if ("short" in m.lower() or "position" in m.lower()) and m.startswith("get")]
    print("\n空売り・ポジション関連のメソッド一覧:")
    for m in cand:
        print(f"    {m}")

    # 空売り残高報告APIを試す（メソッド名の候補を順に）
    code = "72030"  # トヨタ
    got = False
    for method in ["get_mkt_short_positions", "get_mkt_short_selling_positions",
                   "get_short_positions", "get_mkt_short_position",
                   "get_fin_short_positions"]:
        if hasattr(client, method):
            print(f"\nメソッド `{method}` を発見。{code} で取得を試みます…")
            try:
                fn = getattr(client, method)
                try:
                    df = fn(code=code)
                except TypeError:
                    # code引数を取らないAPIなら日付で
                    df = fn(start_dt="2026-08-01", end_dt="2026-08-19")
                got = True
                print(f"  → 取得成功: {len(df)}行")
                if len(df) > 0:
                    print(f"\n【列名一覧】:")
                    for c in df.columns:
                        print(f"    {c}")
                    print(f"\n【最新3件】:")
                    print(df.tail(3).to_string(index=False))
                break
            except Exception as e:
                print(f"  → 失敗: {type(e).__name__}: {str(e)[:200]}")
    if not got:
        print("\n空売り残高報告のメソッドが見つからないか、全て失敗しました。")
        print("上の『空売り・ポジション関連のメソッド一覧』を、チャットに貼ってください。")


if __name__ == "__main__":
    main()
