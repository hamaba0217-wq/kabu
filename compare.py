# -*- coding: utf-8 -*-
"""
条件プリセットの比較検証

前回のベースライン条件に「1つだけ」変更を加えた複数パターンを、
同じ期間・同じデータで順にバックテストし、結果を並べて比較します。

重要な原則
----------
各プリセットは、ベースラインから **1項目だけ** 変えています。
複数を同時に変えないので、「どの変更が効いたか」が明確に分かります。
（2つ以上変えると、改善しても何が効いたのか永久に分かりません）

比較の見方
----------
最重要は「相場全体との超過リターン中央値」。
ベースラインは前回 -6.6% でした。これがプラス方向に動いた条件が、
逆算分析で見えた仮説の裏付けになります。
逆に変わらない/悪化するなら、その仮説は偶然だったということです。
"""

from __future__ import annotations

import copy
import datetime as dt
import os

import pandas as pd

import backtest
import config


# ベースライン = 現在の config の値
def _baseline() -> dict:
    return {
        "MAX_MARKET_CAP": config.MAX_MARKET_CAP,
        "MIN_MARKET_CAP": config.MIN_MARKET_CAP,
        "MIN_TURNOVER_VALUE": config.MIN_TURNOVER_VALUE,
        "MIN_OP_PROFIT_YOY": config.MIN_OP_PROFIT_YOY,
        "MIN_SALES_YOY": config.MIN_SALES_YOY,
        "MIN_PRICE": config.MIN_PRICE,
        "MIN_TURNOVER_SPIKE_RATIO": config.MIN_TURNOVER_SPIKE_RATIO,
        "MAX_PRICE": getattr(config, "MAX_PRICE", None),
    }


# 各プリセット: ベースラインから1項目だけ変更
def presets() -> list[tuple[str, dict]]:
    base = _baseline()
    out = [("0. ベースライン（前回条件）", dict(base))]

    # 仮説1: 低位株重視（株価500円以下）。逆算でリフト最大だった帯。
    #   下限200円は残しつつ、上限500円を新設 → 200〜500円に絞る
    p1 = dict(base); p1["MAX_PRICE"] = 500
    out.append(("1. 低位株（株価200-500円）", p1))

    # 仮説2: 小型株さらに限定（時価総額50億以下）。リフト1.78。
    p2 = dict(base); p2["MAX_MARKET_CAP"] = 5_000_000_000
    out.append(("2. 小型株限定（時価総額50億以下）", p2))

    # 仮説3: 売買代金の急増（20日平均比5倍以上）。リフト1.95。
    p3 = dict(base); p3["MIN_TURNOVER_SPIKE_RATIO"] = 5.0
    out.append(("3. 売買代金急増（平均比5倍以上）", p3))

    return out


def _apply(preset: dict) -> None:
    """プリセットの値を config に反映する。"""
    for k, v in preset.items():
        setattr(config, k, v)


def run(quotes, fin, listed, step_days=21):
    """各プリセットで順にバックテストし、サマリー行を集める。"""
    saved = _baseline()          # 元に戻すため保存
    results = []

    try:
        for name, preset in presets():
            _apply(preset)
            print(f"\n{'='*60}\n▶ {name}\n{'='*60}")

            trades, periods = backtest.run(quotes, fin, listed, step_days=step_days)

            if trades.empty or periods.empty:
                print("  抽出ゼロ。この条件は厳しすぎます。")
                results.append({
                    "条件": name, "抽出数": 0, "勝率%": None,
                    "中央値%": None, "超過中央値%": None, "3倍到達%": None,
                })
                continue

            bh = trades["buy_hold_return"]
            ex = periods["excess"].dropna() if "excess" in periods else pd.Series(dtype=float)
            n_periods_traded = int((periods["n"] > 0).sum()) if "n" in periods else 0
            beat = int((ex > 0).sum())

            results.append({
                "条件": name,
                "抽出数": len(trades),
                "勝率%": round((bh > 0).mean() * 100, 1),
                "中央値%": round(bh.median() * 100, 1),
                "超過中央値%": round(ex.median() * 100, 1) if len(ex) else None,
                "相場に勝った回": f"{beat}/{len(ex)}" if len(ex) else "—",
                "3倍到達%": round((trades["max_return"] >= 2.0).mean() * 100, 1),
            })
    finally:
        _apply(saved)            # 必ず元に戻す

    return pd.DataFrame(results)


def summarize(df: pd.DataFrame) -> str:
    L = ["=" * 72,
         "条件プリセット比較（各条件はベースラインから1項目だけ変更）",
         "=" * 72, ""]
    L.append(df.to_string(index=False))
    L += ["", "=" * 72, "読み方", "=" * 72,
          "・最重要は「超過中央値%」= 相場平均をどれだけ上回ったか。",
          "  ベースラインより大きくプラス方向に動いた条件が、有望な仮説。",
          "・「相場に勝った回」が増えているかも重要。",
          "・抽出数が極端に減った条件は、勝率が上がっても",
          "  『たまたま少数が当たっただけ』の可能性。件数も見ること。",
          "・改善が見られても、これは同じ2年間での結果。",
          "  本採用の前に、別期間でも再現するか確認するのが理想です。"]
    return "\n".join(L)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    from sources import JQuants, JST
    jq = JQuants()
    print("キャッシュからデータを読み込み中...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    fin = jq.financials(config.BACKTEST_LOOKBACK_DAYS)
    listed = jq.listed()
    print(f"  株価 {len(quotes):,}行 / 決算 {len(fin):,}行 / 銘柄 {len(listed):,}")

    df = run(quotes, fin, listed)
    report = summarize(df)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    path = os.path.join(config.OUTPUT_DIR, f"{today}_compare.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n比較表: {path}")


if __name__ == "__main__":
    main()
