# -*- coding: utf-8 -*-
"""
空売り比率が高い業種 × 売られすぎ の候補抽出（実運用向け）

分割検証（validate-shortratio）を両期間プラスで通った条件で、
今日エントリー候補になる銘柄を抽出する。

条件（検証で確定したもの）
--------------------------
  ・25日移動平均から -20〜-10% 下方乖離（売られすぎ）
  ・空売り比率が高い業種（中央値以上）に属する
運用の目安（検証結果）
  ・利確なし〜高め / 保有15日前後 / 損切り-10%
  ・平均+5.38%・勝率73.9%（全期間）。前半+1.39%/後半+1.77%（対相場超過）
  ・ただし件数は少なく（各100件程度）、手数料・スリッページ未考慮

先読み防止：最新営業日時点の指標で判定。空売り比率は7日前までの値。

※本スクリプトは分析補助です。投資助言ではありません。
  実際の売買判断と結果責任は利用者にあります。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import config
import technical
from oversold import add_oversold_indicators
import supply_demand2 as sd2


DEV_MIN, DEV_MAX = -20.0, -10.0
LAG_DAYS = 7


def find(quotes, listed, short_ratio):
    qi = add_oversold_indicators(quotes)
    as_of = qi["date"].max()
    snap = qi[qi["date"] == as_of].copy()

    # 業種コード対応
    sec_name = {}
    sec_code = {}
    if "sector33" in listed.columns:
        for _, r in listed.iterrows():
            sec_name[str(r["code"])] = r["sector33"]
    if "sector33_code" in listed.columns:
        for _, r in listed.iterrows():
            sec_code[str(r["code"])] = str(r["sector33_code"])
    name_map = {}
    if "company_name" in listed.columns:
        for _, r in listed.iterrows():
            name_map[str(r["code"])] = r["company_name"]

    # 空売り比率：業種コード→直近（7日前まで）の比率
    sr = short_ratio.copy()
    sr["date"] = pd.to_datetime(sr["date"])
    sr = sr.sort_values(["sector33", "date"])
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=LAG_DAYS)
    sr_recent = sr[sr["date"] <= cutoff]
    latest_sr = {}
    for s, g in sr_recent.groupby("sector33"):
        latest_sr[str(s)] = g.iloc[-1]["short_ratio"]
    # 高い業種の閾値（比率の中央値）
    vals = [v for v in latest_sr.values() if pd.notna(v)]
    med = float(np.median(vals)) if vals else np.nan

    # 売られすぎ銘柄を抽出
    rows = []
    for _, s in snap.iterrows():
        dev = s["ma25_dev"]
        if pd.isna(dev) or not (DEV_MIN <= dev <= DEV_MAX):
            continue
        code = str(s["code"])
        scode = sec_code.get(code)
        ratio = latest_sr.get(str(scode)) if scode is not None else np.nan
        is_high = pd.notna(ratio) and pd.notna(med) and ratio >= med
        rows.append({
            "code": code,
            "銘柄名": name_map.get(code, ""),
            "業種": sec_name.get(code, ""),
            "移動平均乖離%": round(dev, 1),
            "空売り比率": round(ratio, 3) if pd.notna(ratio) else None,
            "空売り高業種": "◎" if is_high else "",
            "終値": round(float(s["close"]), 1),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        # 空売り高業種を上に、乖離が深い順
        df = df.sort_values(["空売り高業種", "移動平均乖離%"],
                            ascending=[False, True]).reset_index(drop=True)
    return df, pd.Timestamp(as_of).date(), med


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    from sources import JQuants, JST
    jq = JQuants()
    print("キャッシュからデータを読み込み中...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    listed = jq.listed()
    short_ratio = jq.short_ratio(config.BACKTEST_LOOKBACK_DAYS)
    print(f"  株価 {len(quotes):,}行 / 銘柄 {len(listed):,} / 空売り比率 {len(short_ratio):,}行")

    df, as_of, med = find(quotes, listed, short_ratio)
    print(f"\n{'='*70}")
    print(f"空売り比率が高い業種 × 売られすぎ の候補  （{as_of} 時点）")
    print(f"{'='*70}")
    print(f"空売り比率の業種中央値: {med:.3f}（これ以上を『高い業種』とする）\n")
    if df.empty:
        print("該当する売られすぎ銘柄は今日ありませんでした。")
        return
    high = df[df["空売り高業種"] == "◎"]
    print(f"売られすぎ銘柄: {len(df)}件 / うち空売り高業種(◎): {len(high)}件\n")
    print("◎ = 検証で両期間プラスだった『空売り比率が高い業種』の売られすぎ銘柄")
    print(df.to_string(index=False))
    print(f"\n{'-'*70}")
    print("運用の目安（検証結果・手数料未考慮）:")
    print("  利確なし〜高め / 保有15日前後 / 損切り-10%")
    print("  ◎銘柄: 全期間 平均+5.38%・勝率73.9%、前半+1.39%/後半+1.77%（対相場超過）")
    print("  ※件数が少なく偶然の振れが残る。投資助言ではありません。")

    today = dt.datetime.now(JST).date().isoformat()
    path = os.path.join(config.OUTPUT_DIR, f"{today}_candidates_sr.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nCSV: {path}")


if __name__ == "__main__":
    main()
