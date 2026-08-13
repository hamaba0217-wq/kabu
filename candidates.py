# -*- coding: utf-8 -*-
"""
今日エントリー可能な候補銘柄の抽出

分割検証で本物と確認された条件を、最新日のデータに適用し、
「今この条件に該当する銘柄」を実際にリストする。

適用する条件（validate で前半・後半どちらも超過プラスだったもの）
------------------------------------------------------------------
  土台: 3戦略（決算モメンタム / ボックス反発 / ブレイクアウト）のいずれかで
        エントリーシグナルが出ている
  ＋ 押し目ゾーン       … 52週高値から -15〜-5%
  ＋ 高勝率業種         … 保険・銀行・証券・倉庫運輸・水産農林
  ＋（任意）直近1M下げ  … 直近20営業日 -10〜0%

重要な位置づけ
--------------
これは「買い推奨」ではありません。過去の傾向から、この条件に合う銘柄が
統計的にやや勝ちやすかった、というだけ。個別銘柄が上がる保証はなく、
最終判断は必ず本人が行うこと。上位に出た銘柄も、なぜ下げているのか
（悪材料か、地合いか）は別途自分で確認する必要がある。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import config
import technical
from refine import _f_pullback, _f_high_win_sector, _f_mild_down_1m, HIGH_WIN_SECTORS


def find(quotes, fin, listed, require_mild_down=False):
    qi = technical.add_indicators(quotes)
    g = qi.groupby("code")
    qi["ret_1m"] = g["close"].transform(lambda s: s / s.shift(20) - 1.0)
    h = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / h - 1.0

    sec = (listed.set_index("code")["sector33"]
           if "sector33" in listed.columns else pd.Series(dtype=object))
    names = (listed.set_index("code")["company_name"]
             if "company_name" in listed.columns else pd.Series(dtype=object))

    as_of = qi["date"].max()
    snap = technical._snapshot(qi, as_of).set_index("code")
    fin2 = technical._prep_fin(fin)

    # どの戦略でシグナルが出ているか
    snap_reset = snap.reset_index()
    sig = {}
    for code in technical.entries_breakout(snap_reset):
        sig.setdefault(code, []).append("ブレイクアウト")
    for code in technical.entries_box_bottom(snap_reset):
        sig.setdefault(code, []).append("ボックス反発")
    for code in technical.entries_earnings_momentum(snap_reset, fin2, as_of):
        sig.setdefault(code, []).append("決算モメンタム")

    rows = []
    for code, strategies in sig.items():
        if code not in snap.index:
            continue
        row = snap.loc[code].copy()
        row["sector33"] = sec.get(code)
        # 条件適用
        if not _f_pullback(row):
            continue
        if not _f_high_win_sector(row):
            continue
        if require_mild_down and not _f_mild_down_1m(row):
            continue

        rows.append({
            "コード": code,
            "銘柄名": names.get(code, ""),
            "業種": row.get("sector33"),
            "株価": round(float(row["close"]), 1),
            "高値から%": round(float(row["pct_from_high"]) * 100, 1),
            "直近1M%": round(float(row["ret_1m"]) * 100, 1)
                if pd.notna(row.get("ret_1m")) else None,
            "売買代金_億": round(float(row["turnover_value"]) / 1e8, 1),
            "シグナル": "/".join(strategies),
        })

    df = pd.DataFrame(rows)
    if len(df):
        # 高値からの位置が深い順（より押している順）
        df = df.sort_values("高値から%").reset_index(drop=True)
    return df, pd.Timestamp(as_of).date()


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    from sources import JQuants, JST
    jq = JQuants()
    print("キャッシュからデータを読み込み中...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    fin = jq.financials(config.BACKTEST_LOOKBACK_DAYS)
    listed = jq.listed()
    print(f"  株価 {len(quotes):,}行 / 決算 {len(fin):,}行 / 銘柄 {len(listed):,}")

    for require, label in [(False, "押し目+高勝率業種"),
                           (True, "押し目+高勝率業種+直近1M下げ")]:
        df, as_of = find(quotes, fin, listed, require_mild_down=require)
        print("\n" + "=" * 66)
        print(f"【{label}】に該当する銘柄  （{as_of} 時点）")
        print("=" * 66)
        if df.empty:
            print("  該当銘柄なし（今日はこの条件に合う銘柄がありません）")
        else:
            print(f"  {len(df)} 銘柄が該当\n")
            print(df.to_string(index=False))
            today = dt.datetime.now(JST).date().isoformat()
            tag = "with_down" if require else "base"
            path = os.path.join(config.OUTPUT_DIR, f"{today}_candidates_{tag}.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"\n  保存: {path}")

    print("\n" + "=" * 66)
    print("重要な注意")
    print("=" * 66)
    print("・これは買い推奨ではありません。過去の傾向で『やや勝ちやすかった』")
    print("  条件に合う銘柄というだけです。個別に上がる保証はありません。")
    print("・各銘柄がなぜ高値から下げているか（悪材料か地合いか）は、")
    print("  必ずご自身で確認してください。悪材料での下落は押し目ではありません。")
    print("・過去検証の超過リターンはわずかで、手数料を引くと消える水準です。")
    print("・最終的な投資判断はご自身の責任で行ってください。")


if __name__ == "__main__":
    main()
