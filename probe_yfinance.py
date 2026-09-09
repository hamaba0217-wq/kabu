# -*- coding: utf-8 -*-
"""
yfinance で日本株の配当日程（権利落ち日）が取れるか確認する診断。
  py probe_yfinance.py
出力をチャットに貼ってください。取れるか・正確かを見て、実装を判断します。

※ 権利確定日と権利落ち日の関係：
   権利落ち日（ex-dividend date）の1営業日前が、権利付き最終日。
   権利確定日は、権利付き最終日に約定した分が確定する日で、通常は権利落ち日の前営業日〜数日内。
   yfinanceが返すのは通常 ex-dividend date（権利落ち日）。
"""

from __future__ import annotations
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# あなたの候補に実際に出た銘柄で試す
TARGETS = {
    "8630": "SOMPOホールディングス",
    "8766": "東京海上ホールディングス",
    "8411": "みずほFG",
    "8308": "りそなHD",
}


def main():
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance がインストールされていません。")
        print("  py -m pip install yfinance を実行してください。")
        return

    print("=" * 70)
    print("yfinance で日本株の配当日程が取れるか確認")
    print("=" * 70)

    for code, name in TARGETS.items():
        ticker = f"{code}.T"
        print(f"\n■ {name}（{ticker}）")
        try:
            t = yf.Ticker(ticker)
            info = t.info
            ex_date = info.get("exDividendDate")
            div_rate = info.get("dividendRate")
            last_div = info.get("lastDividendValue")
            last_div_date = info.get("lastDividendDate")

            # UNIXタイムスタンプを日付に変換
            def to_date(v):
                if v is None:
                    return "なし"
                try:
                    return str(pd.Timestamp(v, unit="s").date())
                except Exception:
                    return str(v)

            print(f"   配当落ち日(exDividendDate): {to_date(ex_date)}")
            print(f"   1株配当(dividendRate): {div_rate}")
            print(f"   直近配当額(lastDividendValue): {last_div}")
            print(f"   直近配当日(lastDividendDate): {to_date(last_div_date)}")

            # 配当履歴（最新3件）
            d = t.dividends
            if d is not None and len(d) > 0:
                print(f"   配当履歴(最新3件):")
                for dt, amt in d.tail(3).items():
                    print(f"      {dt.date()} : {amt}円")
            else:
                print(f"   配当履歴: なし")
        except Exception as e:
            print(f"   取得失敗: {type(e).__name__}: {e}")

    print("\n" + "=" * 70)
    print("この結果を、そのままチャットに貼ってください。")
    print("配当落ち日が正しく取れているか（各社の実際の権利確定月と合うか）を確認します。")


if __name__ == "__main__":
    main()
