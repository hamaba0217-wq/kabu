# -*- coding: utf-8 -*-
"""API接続なしで、スクリーニングのロジックだけを合成データで検証する。"""

import datetime as dt

import numpy as np
import pandas as pd

import config
import screen


def make_quotes():
    rows = []
    base = dt.datetime(2026, 5, 1)
    specs = {
        "1111": (1500, 500_000_000),   # 条件クリア想定
        "2222": (150,   800_000_000),  # 株価が安すぎ → 落ちる
        "3333": (2000,   50_000_000),  # 売買代金不足 → 落ちる
        "4444": (900,  1_200_000_000), # 業績で落ちる想定
    }
    for code, (px, tv) in specs.items():
        for i in range(60):
            d = base + dt.timedelta(days=i)
            rows.append({
                "code": code, "date": d,
                "close": px * (1 + 0.001 * i),
                "volume": tv / px,
                "turnover_value": tv,
            })
    df = pd.DataFrame(rows)
    df.attrs["turnover_is_estimated"] = False
    return df


def make_fin():
    rows = []
    # (code, 今期営業利益, 前年同期営業利益, 今期売上, 前年売上, 発行済株式数)
    specs = {
        "1111": (600, 400, 8000, 6500, 20_000_000),   # 営業+50% 売上+23% → 通過
        "2222": (300, 200, 5000, 4000, 30_000_000),
        "3333": (900, 600, 9000, 7000, 10_000_000),
        "4444": (500, 480, 7000, 6900, 25_000_000),   # 伸び率不足 → 落ちる
        "5555": (100, -50, 3000, 2400, 15_000_000),   # 黒字転換 → 通過ルート確認
    }
    for code, (op, op_p, sales, sales_p, sh) in specs.items():
        rows.append({"code": code, "disclosed_date": dt.datetime(2025, 8, 10),
                     "period_type": "2Q", "period_end": dt.datetime(2025, 6, 30),
                     "net_sales": sales_p * 1e6, "operating_profit": op_p * 1e6,
                     "shares_outstanding": sh})
        rows.append({"code": code, "disclosed_date": dt.datetime(2026, 8, 10),
                     "period_type": "2Q", "period_end": dt.datetime(2026, 6, 30),
                     "net_sales": sales * 1e6, "operating_profit": op * 1e6,
                     "shares_outstanding": sh})
    return pd.DataFrame(rows)


def make_listed():
    return pd.DataFrame([
        {"code": c, "company_name": f"テスト{c}", "market": "グロース",
         "sector33": "情報・通信業"}
        for c in ["1111", "2222", "3333", "4444", "5555"]
    ])


def main():
    quotes, fin, listed = make_quotes(), make_fin(), make_listed()

    prices = screen.latest_prices(quotes, config.MA_WINDOW)
    print("=== 最新株価 ===")
    print(prices.to_string(index=False))

    yoy = screen.yoy_table(fin)
    print("\n=== 前年同期比 ===")
    print(yoy[["code", "period_type", "sales_yoy", "op_yoy", "turnaround"]].to_string(index=False))

    result, log = screen.run_screen(prices, yoy, listed, pd.DataFrame())
    print("\n=== 絞り込み経過 ===")
    for line in log:
        print("  " + line)

    print("\n=== 通過銘柄 ===")
    out = screen.to_output(result)
    print(out.to_string(index=False) if len(out) else "(なし)")

    # 検証
    passed = set(result["code"])
    print("\n=== 検証 ===")
    assert "1111" in passed, "1111 は通過するはず"
    assert "2222" not in passed, "2222 は株価200円未満で落ちるはず"
    assert "3333" not in passed, "3333 は売買代金3億円未満で落ちるはず"
    assert "4444" not in passed, "4444 は業績の伸び率不足で落ちるはず"
    print("すべて期待通りです。")


if __name__ == "__main__":
    main()
