# -*- coding: utf-8 -*-
"""API接続なしで、バックテスト処理を合成データで検証する。

ランダムな値動きなので「成績」に意味はありません。
確認するのは、処理が最後まで通ること・先読みが起きていないことです。
"""

import datetime as dt

import numpy as np
import pandas as pd

import backtest
import config
import screen

rng = np.random.default_rng(42)

N_STOCKS = 120
N_DAYS = 500


def make_data():
    codes = [f"{1000+i}" for i in range(N_STOCKS)]
    dates = pd.bdate_range("2024-08-01", periods=N_DAYS)

    qrows = []
    for code in codes:
        px = rng.uniform(300, 3000)
        drift = rng.normal(0.0004, 0.0010)
        vol = rng.uniform(0.015, 0.045)
        turnover = rng.uniform(1e8, 3e9)
        for d in dates:
            px *= float(np.exp(rng.normal(drift, vol)))
            qrows.append({"code": code, "date": d, "close": px,
                          "volume": turnover / px,
                          "turnover_value": turnover * float(np.exp(rng.normal(0, 0.3)))})
    quotes = pd.DataFrame(qrows)
    quotes.attrs["turnover_is_estimated"] = False

    frows = []
    for code in codes:
        shares = rng.integers(5_000_000, 60_000_000)
        base_sales = rng.uniform(2e9, 3e10)
        base_op = base_sales * rng.uniform(0.03, 0.15)
        # 4半期ぶんの開示を2年分
        for k, disc in enumerate(pd.date_range("2024-08-10", periods=9, freq="91D")):
            growth = 1 + rng.normal(0.10, 0.25)
            frows.append({
                "code": code, "disclosed_date": disc,
                "period_type": ["1Q", "2Q", "3Q", "FY"][k % 4],
                "period_end": disc - dt.timedelta(days=40),
                "net_sales": base_sales * (growth ** k),
                "operating_profit": base_op * (growth ** k),
                "shares_outstanding": shares,
            })
    fin = pd.DataFrame(frows)

    listed = pd.DataFrame([{"code": c, "company_name": f"テスト{c}",
                            "market": "グロース", "sector33": "サービス業"}
                           for c in codes])
    return quotes, fin, listed


def test_no_lookahead(quotes):
    """snapshot_at が as_of より後のデータを混ぜていないことを確認する。"""
    q = screen.add_rolling(quotes, config.MA_WINDOW)
    as_of = q["date"].unique()[200]
    snap = screen.snapshot_at(q, as_of)
    assert (snap["date"] <= as_of).all(), "先読みが発生しています"
    print(f"  先読みチェック: OK（as_of={pd.Timestamp(as_of).date()} 以前のみ使用）")


def test_simulate():
    """執行ルールのシミュレーションが正しく動くか。"""
    # 一直線に -30% まで下落 → 損切りで -25%
    path = np.linspace(1.0, 0.70, 60)
    r, reason = backtest.simulate_position(path)
    assert reason == "損切り" and abs(r - (-0.25)) < 1e-9, (r, reason)
    print(f"  損切りケース: {r:+.1%} ({reason})")

    # 3倍まで上昇 → 各段階で1/4ずつ利確、残り1/4は最終値
    path = np.linspace(1.0, 3.0, 60)
    r, reason = backtest.simulate_position(path)
    expected = 0.25*0.5 + 0.25*1.0 + 0.25*2.0 + 0.25*2.0
    assert abs(r - expected) < 1e-9, (r, expected)
    print(f"  3倍ケース  : {r:+.1%} ({reason})")

    # 横ばい → ほぼゼロ
    path = np.ones(60)
    r, reason = backtest.simulate_position(path)
    assert abs(r) < 1e-9
    print(f"  横ばいケース: {r:+.1%} ({reason})")


def main():
    print("合成データを生成中...")
    quotes, fin, listed = make_data()
    print(f"  株価 {len(quotes)}行 / 決算 {len(fin)}行 / {len(listed)}銘柄\n")

    print("単体チェック")
    test_simulate()
    test_no_lookahead(quotes)

    print("\nバックテスト実行...")
    trades, periods = backtest.run(quotes, fin, listed, step_days=21)
    print(backtest.summarize(trades, periods))


if __name__ == "__main__":
    main()
