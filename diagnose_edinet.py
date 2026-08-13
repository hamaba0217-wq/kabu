# -*- coding: utf-8 -*-
"""
EDINET大株主データ(get_edinet_major_shareholders)の中身を診断する。
5倍株の前兆分析（大株主の異動）を正しく作るために、実データの構造を確認する。
"""

from __future__ import annotations

import pandas as pd


def main():
    from sources import JQuants
    jq = JQuants()
    print("=" * 72)
    print("EDINET大株主データの構造診断")
    print("=" * 72)

    m = "get_edinet_major_shareholders"
    fn = getattr(jq.cli, m)

    # トヨタで取得して中身を見る
    print("\nトヨタ(72030)の大株主データを取得します...\n")
    df = fn(code="72030")
    print(f"取得行数: {len(df)}")
    print(f"全列: {list(df.columns)}\n")

    print("-" * 72)
    print("1件目の全項目（各列に何が入っているか）:")
    print("-" * 72)
    if len(df) > 0:
        row = df.iloc[0]
        for col in df.columns:
            val = row[col]
            s = str(val)
            if len(s) > 100:
                s = s[:100] + " …(省略)"
            print(f"  {col}: {s}")

    # SubDate（提出日）の分布を見る（時系列で追えるか）
    print("\n" + "-" * 72)
    date_col = None
    for c in df.columns:
        if "date" in c.lower() or "Date" in c:
            date_col = c
            break
    if date_col:
        print(f"日付列『{date_col}』の値: {sorted(df[date_col].unique())}")
    else:
        print("日付らしき列が見つかりません。列名を確認してください。")

    # 複数銘柄で、日付範囲を確認（5倍株の起点前に取れるか）
    print("\n" + "-" * 72)
    print("別の銘柄でも日付範囲を確認（キオクシア285A、メタプラネット33500）:")
    for code in ["2850", "33500", "285A0"]:
        try:
            d = fn(code=code)
            if len(d) > 0 and date_col:
                dates = sorted(d[date_col].unique())
                print(f"  {code}: {len(d)}件, 日付 {dates[0]}〜{dates[-1]}")
            else:
                print(f"  {code}: {len(d)}件")
        except Exception as e:
            print(f"  {code}: エラー {e}")

    print("\n※この構造を見て、大株主の『異動』を時系列で捉える分析を設計します。")


if __name__ == "__main__":
    main()
