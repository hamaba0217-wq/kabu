# -*- coding: utf-8 -*-
"""
決算データの会社予想(fc_operating_profit)が実際どう入っているか診断する。
上方修正の判定を正しく作るために、まず実データの中身を確認する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def main():
    from sources import JQuants
    jq = JQuants()
    print("キャッシュから決算データを読み込み中...")
    fin = jq.financials(config.BACKTEST_LOOKBACK_DAYS)
    print(f"  決算 {len(fin):,}行\n")

    print("=" * 72)
    print("決算データの会社予想（fc_operating_profit）の実態診断")
    print("=" * 72)

    # 列の存在
    cols = list(fin.columns)
    print(f"\n列一覧: {cols}\n")

    # fc_operating_profit の充填率
    if "fc_operating_profit" in fin.columns:
        n = len(fin)
        filled = fin["fc_operating_profit"].notna().sum()
        print(f"fc_operating_profit（会社予想営業利益）:")
        print(f"  埋まっている: {filled:,} / {n:,} 行 ({filled/n*100:.1f}%)")
    else:
        print("[!] fc_operating_profit 列がありません")
        return

    # period_type の種類
    if "period_type" in fin.columns:
        print(f"\nperiod_type（決算の種類）の内訳:")
        for pt, cnt in fin["period_type"].value_counts().items():
            print(f"  {pt}: {cnt:,}件")

    # ある銘柄の予想の時系列を見る（トヨタ=72030 など主要銘柄で例示）
    print(f"\n{'-'*72}")
    print("予想営業利益の時系列の例（同じ銘柄で予想がどう動くか）:")
    for code in ["72030", "67580", "99840", "83060"]:
        g = fin[fin["code"] == code].sort_values("disclosed_date")
        if g.empty:
            continue
        print(f"\n  銘柄 {code}:")
        for _, r in g.tail(6).iterrows():
            d = r["disclosed_date"]
            pt = r.get("period_type", "?")
            fc = r.get("fc_operating_profit")
            fc_s = f"{fc:,.0f}" if pd.notna(fc) else "（空）"
            print(f"    {pd.Timestamp(d).date()} [{pt}] 予想営業利益={fc_s}")

    # 上方修正らしきものが実際どれくらいあるか（銘柄内で予想が前回より上がった回数）
    print(f"\n{'-'*72}")
    print("予想が前回開示より上がった回数（単純にshift比較した場合）:")
    f = fin.sort_values(["code", "disclosed_date"]).copy()
    f["fc_prev"] = f.groupby("code")["fc_operating_profit"].shift(1)
    both = f[(f["fc_operating_profit"].notna()) & (f["fc_prev"].notna())]
    up = both[both["fc_operating_profit"] > both["fc_prev"] * 1.03]
    print(f"  比較できた開示: {len(both):,}件")
    print(f"  うち予想3%超上昇（上方修正らしき）: {len(up):,}件 "
          f"({len(up)/len(both)*100:.1f}%)" if len(both) else "  比較できた開示なし")
    print("\n  ※これが0件なら、shift比較では上方修正を捉えられない（期の扱いが原因）。")
    print("    period_type を考慮した比較が必要と分かる。")


if __name__ == "__main__":
    main()
