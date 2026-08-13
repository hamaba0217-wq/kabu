# -*- coding: utf-8 -*-
"""
日次スクリーニング実行スクリプト

使い方
------
  python main.py doctor      # 実行前の環境チェック（困ったらまずこれ）
  python main.py inspect     # APIが返す列名を確認
  python main.py backtest    # 過去データで条件を検証
  python main.py reverse     # 急騰銘柄の特徴を逆算分析
  python main.py compare     # 条件プリセットを並べて比較検証
  python main.py targets     # 目標倍率×保有期間の6パターン比較
  python main.py technical   # テクニカル3戦略の一括検証
  python main.py analyze-trades  # 3戦略の勝敗を特徴別に分析
  python main.py filters     # エントリーフィルターを1つずつ検証
  python main.py refine      # 押し目ゾーン×絞り込み軸の探索
  python main.py validate    # 前半後半に分けて本物か検証
  python main.py candidates  # 今日該当する候補銘柄を抽出
  python main.py funnel      # 絞り込み過程と脱落株を可視化
  python main.py bigwin      # 大勝ち(+10%)に効く条件を探索
  python main.py bigwin-refine  # 高勝率業種×大勝ち条件の探索
  python main.py run         # スクリーニングを実行してCSVを出力
  python main.py daily-mail  # 日次レポートをメール送信
  python main.py weekly-mail # 週次レポートをメール送信

出力
----
  output/YYYY-MM-DD_candidates.csv   通過銘柄
  output/YYYY-MM-DD_paste.md         チャットに貼り付ける用のテキスト
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import pandas as pd

import analyze_trades
import candidates
import bigwin
import bigwin_refine
import backtest
import preearnings
import preentry
import oversold
import quality
import supply_demand
import supply_demand2
import validate_shortratio
import candidates_sr
import candidates_best
import web_report
import fivebagger
import precursor
import probe_tdnet
import diagnose_forecast
import diagnose_edinet
import validate_candidates
import profit_targets
import selltiming
import badnews
import compare
import config
import exitgrid
import filters
import funnel
import jobs
import reasons
import refine
import reverse
import screen
import targets
import technical
import validate
import validate_oversold
from sources import JQuants, fetch_large_holdings, JST


def cmd_inspect() -> None:
    """APIが実際に返す列名を表示する。列名の不一致はここで潰します。"""
    jq = JQuants()
    end = jq._end_date()   # Freeプランの遅延を見込んだ終端（今日ではない）
    print(f"（データ取得の終端: {end.date()} … 遅延を考慮）")

    print("=" * 70)
    print("[1] 銘柄一覧 get_list()")
    df = jq.raw_listed()
    print(f"  行数: {len(df)}")
    print(f"  列名: {list(df.columns)}")

    print("\n[2] 株価日足 get_eq_bars_daily_range()")
    df = jq.raw_quotes(end - dt.timedelta(days=3), end)   # レート制限に配慮し最小限
    print(f"  行数: {len(df)}")
    print(f"  列名: {list(df.columns)}")
    if len(df):
        print(f"  最新日付: {df.iloc[-1].to_dict().get('Date')}")
        print("  先頭1行:")
        print(df.head(1).T.to_string())

    print("\n[3] 決算サマリー get_fin_summary_range()")
    df = jq.raw_fin(end - dt.timedelta(days=3), end)      # レート制限に配慮し最小限
    print(f"  行数: {len(df)}")
    print(f"  列名: {list(df.columns)}")
    if len(df):
        print("  先頭1行:")
        print(df.head(1).T.to_string())

    print("=" * 70)
    print("列名が sources.py の候補リストと違う場合は、候補リストに追加してください。")
    print("※ 行数0でも異常ではありません（Freeプランは12週間遅延のため）")


def cmd_doctor() -> None:
    """実行前の自己診断。よくある失敗を先に検出する。"""
    print("=" * 60)
    print("環境チェック")
    print("=" * 60)
    ok = True

    # 1. 必要なライブラリ
    print("\n[1] 必要なライブラリ")
    for mod, pip_name in [("pandas","pandas"), ("numpy","numpy"),
                          ("requests","requests"), ("jquantsapi","jquants-api-client"),
                          ("tabulate","tabulate")]:
        try:
            __import__(mod)
            print(f"  OK  {mod}")
        except ImportError:
            print(f"  NG  {mod} が無い → py -m pip install {pip_name}")
            ok = False

    # 2. APIキー
    print("\n[2] APIキー")
    if os.environ.get("JQUANTS_API_KEY"):
        print("  OK  JQUANTS_API_KEY 設定済み")
    else:
        print("  NG  JQUANTS_API_KEY が未設定 → setx JQUANTS_API_KEY \"キー\"")
        ok = False
    if os.environ.get("EDINET_API_KEY"):
        print("  OK  EDINET_API_KEY 設定済み")
    else:
        print("  --  EDINET_API_KEY 未設定（開示書類の取得はスキップされます）")

    # 3. モジュールの相互参照
    print("\n[3] プログラムの整合性")
    try:
        import config, sources, screen, backtest, reverse, checks
        import fundamentals, cache, compare, jobs, regime, mailer
        print("  OK  全モジュールが正常に読み込めます")
    except Exception as e:
        print(f"  NG  モジュール読み込みエラー: {e}")
        ok = False

    # 3b. main.py が呼ぶモジュールが全て import 済みか（import漏れ検出）
    import re as _re
    src = open(__file__, encoding="utf-8").read()
    imported = set(_re.findall(r"^import (\w+)", src, _re.MULTILINE))
    called = set(_re.findall(r"\b(\w+)\.main\(\)", src))
    local_mods = {f[:-3] for f in os.listdir(".") if f.endswith(".py")}
    missing = (called & local_mods) - imported
    if missing:
        print(f"  NG  main.py が使うのに import されていない: {missing}")
        print(f"      → main.py の先頭に import 文を追加してください")
        ok = False
    else:
        print("  OK  全コマンドの依存モジュールが import 済み")

    # 4. キャッシュの読み書き
    print("\n[4] キャッシュの読み書き")
    try:
        import cache, pandas as pd
        t = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "code": ["1"]})
        if cache.save("_doctor_test", t):
            back = cache.load("_doctor_test")
            if back is not None and len(back) == 1:
                print("  OK  data/ フォルダへの保存・読み込みができます")
            os.remove(cache._path("_doctor_test"))
        else:
            print("  NG  キャッシュを保存できません")
            ok = False
    except Exception as e:
        print(f"  NG  キャッシュエラー: {e}")
        ok = False

    # 5. 設定値
    print("\n[5] 設定（config.py）")
    import config
    print(f"  レート制限     : {config.RATE_LIMIT_PER_MIN}件/分"
          + ("  ← Lightは60" if config.RATE_LIMIT_PER_MIN < 60 else ""))
    print(f"  データ遅延     : {config.DATA_LAG_DAYS}日"
          + ("  ← Lightは0" if config.DATA_LAG_DAYS > 0 else ""))
    print(f"  検証期間       : {config.BACKTEST_LOOKBACK_DAYS}日")

    print("\n" + "=" * 60)
    print("診断結果: " + ("✅ 実行できます" if ok else "❌ 上のNG項目を直してください"))
    print("=" * 60)


def cmd_run() -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    jq = JQuants()

    print("株価を取得中...")
    quotes = jq.quotes(config.PRICE_LOOKBACK_DAYS)
    if quotes.empty:
        print("株価が0件でした。無料プランは12週間遅延のため、"
              "取得期間を config.PRICE_LOOKBACK_DAYS で広げてください。")
        sys.exit(1)
    data_date = quotes["date"].max().date()
    print(f"  {len(quotes)}行 / データ最終日: {data_date}")
    if quotes.attrs.get("turnover_is_estimated"):
        print("  [注意] 売買代金の列が無いため 終値×出来高 で概算しています。")

    print("決算サマリーを取得中...")
    fin = jq.financials(config.FIN_LOOKBACK_DAYS)
    print(f"  {len(fin)}行")

    print("銘柄一覧を取得中...")
    listed = jq.listed()
    print(f"  {len(listed)}銘柄")

    print("大量保有報告書を取得中（EDINET）...")
    holdings = fetch_large_holdings(config.EDINET_LOOKBACK_DAYS)
    print(f"  {len(holdings)}件")

    print("\n集計中...")
    prices = screen.latest_prices(quotes, config.MA_WINDOW)
    yoy = screen.yoy_table(fin)
    print(f"  前年同期比を計算できた銘柄: {len(yoy)}")

    result, log = screen.run_screen(prices, yoy, listed, holdings)

    print("\n--- 絞り込みの経過 ---")
    for line in log:
        print("  " + line)
    print(f"\n最終通過: {len(result)}銘柄")

    out = screen.to_output(result)
    today = dt.datetime.now(JST).date().isoformat()
    csv_path = os.path.join(config.OUTPUT_DIR, f"{today}_candidates.csv")
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV: {csv_path}")

    md_path = os.path.join(config.OUTPUT_DIR, f"{today}_paste.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 日次スクリーニング結果 {today}\n\n")
        f.write(f"- データ最終日: {data_date}\n")
        f.write(f"- 通過銘柄数: {len(result)}\n\n")
        f.write("## 絞り込みの経過\n")
        for line in log:
            f.write(f"- {line}\n")
        f.write("\n## 通過銘柄\n\n")
        if len(out):
            f.write(out.to_markdown(index=False))
        else:
            f.write("本日は条件を満たす銘柄がありませんでした。\n")
        f.write("\n")
    print(f"貼り付け用: {md_path}")

    if len(result) == 0:
        print("\n候補ゼロです。これは正常な結果です。"
              "無理に条件を緩めず、まず数日分の推移を見てください。")


def cmd_backtest() -> None:
    """過去データでスクリーニング条件を検証する。"""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    jq = JQuants()

    print("株価の履歴を取得中（2年分・数分かかります）...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    print(f"  {len(quotes)}行 / "
          f"{quotes['date'].min().date()} 〜 {quotes['date'].max().date()}")

    print("決算サマリーを取得中...")
    fin = jq.financials(config.BACKTEST_LOOKBACK_DAYS)
    print(f"  {len(fin)}行")

    print("銘柄一覧を取得中...")
    listed = jq.listed()
    print(f"  {len(listed)}銘柄\n")

    print("バックテスト実行中...")
    trades, periods = backtest.run(quotes, fin, listed,
                                   step_days=config.BACKTEST_STEP_DAYS)

    report = backtest.summarize(trades, periods)
    print("\n" + report)

    today = dt.datetime.now(JST).date().isoformat()
    if not trades.empty:
        p = os.path.join(config.OUTPUT_DIR, f"{today}_backtest_trades.csv")
        trades.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"\n個別結果: {p}")
    p = os.path.join(config.OUTPUT_DIR, f"{today}_backtest_report.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("```\n" + report + "\n```\n")
    print(f"レポート: {p}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "doctor":
        cmd_doctor()
    elif cmd == "inspect":
        cmd_inspect()
    elif cmd == "run":
        cmd_run()
    elif cmd == "backtest":
        cmd_backtest()
    elif cmd == "reverse":
        reverse.main()
    elif cmd == "compare":
        compare.main()
    elif cmd == "targets":
        targets.main()
    elif cmd == "technical":
        technical.main()
    elif cmd == "analyze-trades":
        analyze_trades.main()
    elif cmd == "filters":
        filters.main()
    elif cmd == "refine":
        refine.main()
    elif cmd == "validate":
        validate.main()
    elif cmd == "candidates":
        candidates.main()
    elif cmd == "funnel":
        funnel.main()
    elif cmd == "bigwin":
        bigwin.main()
    elif cmd == "bigwin-refine":
        bigwin_refine.main()
    elif cmd == "reasons":
        reasons.main()
    elif cmd == "exitgrid":
        exitgrid.main()
    elif cmd == "badnews":
        badnews.main()
    elif cmd == "preearnings":
        preearnings.main()
    elif cmd == "preentry":
        preentry.main()
    elif cmd == "oversold":
        oversold.main()
    elif cmd == "profit-targets":
        profit_targets.main()
    elif cmd == "selltiming":
        selltiming.main()
    elif cmd == "validate-oversold":
        validate_oversold.main()
    elif cmd == "quality":
        quality.main()
    elif cmd == "supply-demand":
        supply_demand.main()
    elif cmd == "supply-demand2":
        supply_demand2.main()
    elif cmd == "validate-shortratio":
        validate_shortratio.main()
    elif cmd == "candidates-sr":
        candidates_sr.main()
    elif cmd == "candidates-best":
        candidates_best.main(sys.argv[2:])
    elif cmd == "web-report":
        web_report.main()
    elif cmd == "fivebagger":
        fivebagger.main()
    elif cmd == "precursor":
        precursor.main()
    elif cmd == "probe-tdnet":
        probe_tdnet.main()
    elif cmd == "diagnose-forecast":
        diagnose_forecast.main()
    elif cmd == "diagnose-edinet":
        diagnose_edinet.main()
    elif cmd == "validate-candidates":
        validate_candidates.main(sys.argv[2:])
    elif cmd == "daily-mail":
        jobs.daily()
    elif cmd == "weekly-mail":
        jobs.weekly()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
