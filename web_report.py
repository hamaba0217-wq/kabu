# -*- coding: utf-8 -*-
"""
web_report.py
1. daily_morning_routine を実行（0件でも必ず最新日付でCSVを生成）
2. docs/index.html を生成・更新
"""
from __future__ import annotations
import os
import sys
import glob
import datetime as dt
import pandas as pd
import numpy as np

import config
from daily_morning_routine import run_morning_routine

def generate_web_report():
    print("=" * 76)
    print("Webレポート（docs/index.html）のHTML生成中...")
    print("=" * 76)

    pattern = os.path.join(config.OUTPUT_DIR, "*_morning_orders.csv")
    files = glob.glob(pattern)

    df = pd.DataFrame()
    target_date = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date().isoformat()

    if files:
        latest_file = max(files, key=os.path.getmtime)
        try:
            target_date = os.path.basename(latest_file).split("_")[0]
            df = pd.read_csv(latest_file)
            print(f"  最新の指値CSVを読み込みました: {latest_file} (件数: {len(df)})")
        except Exception as e:
            print(f"  CSV読み込みエラー: {e}")

    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>今日の株式運用候補 ＆ 指値プラン・検証手法別おすすめ</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #f8f9fa; color: #333; line-height: 1.6; }}
        h1 {{ color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 30px; font-size: 1.3em; border-left: 4px solid #1a73e8; padding-left: 10px; }}
        .date-badge {{ background: #e8f0fe; color: #1a73e8; padding: 5px 12px; border-radius: 4px; font-weight: bold; display: inline-block; margin-bottom: 20px; }}
        
        .strategy-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .strategy-card {{ background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); padding: 20px; border-top: 4px solid #1a73e8; }}
        .strategy-card.alt {{ border-top-color: #28a745; }}
        .strategy-card h3 {{ margin-top: 0; color: #1a73e8; font-size: 1.1em; }}
        .strategy-card.alt h3 {{ color: #28a745; }}
        .strategy-card p {{ font-size: 0.9em; margin-bottom: 10px; }}
        .strategy-card ul {{ font-size: 0.85em; padding-left: 20px; color: #555; }}
        
        table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.08); margin-top: 15px; font-size: 0.9em; }}
        th, td {{ padding: 12px 15px; text-align: center; border-bottom: 1px solid #eee; }}
        th {{ background: #1a73e8; color: #fff; font-weight: 600; white-space: nowrap; }}
        tr:hover {{ background: #f1f3f4; }}
        
        .badge-main {{ background: #e8f0fe; color: #1a73e8; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }}
        .badge-sub {{ background: #e6f4ea; color: #137333; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }}
        
        .price-buy {{ font-weight: bold; color: #2c3e50; }}
        .price-tp {{ font-weight: bold; color: #137333; }}
        .price-sl {{ font-weight: bold; color: #d93025; }}
        .no-data {{ text-align: center; padding: 40px; color: #666; background: #fff; border-radius: 8px; }}
        footer {{ margin-top: 40px; text-align: center; font-size: 0.8em; color: #888; }}
    </style>
</head>
<body>

    <h1>今日の株式運用候補 ＆ 指値プラン</h1>
    <div class="date-badge">基準日: {target_date}</div>

    <h2>💡 検証方法・アプローチ別の特徴とおすすめ</h2>
    <div class="strategy-container">
        <div class="strategy-card">
            <h3>📈 【本命セクター重視】アプローチ</h3>
            <p><strong>考え方:</strong> 過去のバックテストデータから「勝率55%以上」が実証されている高勝率セクター（海運、銀行、パルプ・紙など）に資金を集中させる方法。</p>
            <p><strong>こんな方におすすめ:</strong></p>
            <ul>
                <li>相場全体のトレンドに逆らわず、確率的に優位な業種から手堅くリターンを狙いたい方</li>
                <li>セクターローテーションの波に乗って安定感を重視したい方</li>
            </ul>
        </div>
        <div class="strategy-card alt">
            <h3>⚡ 【高スコア・モメンタム急増】アプローチ</h3>
            <p><strong>考え方:</strong> 52週高値から大幅に下落した大底圏（-30%以下）にあり、かつ本日の出来高が急増している銘柄をスコアリングで多角的に抽出する方法。</p>
            <p><strong>こんな方におすすめ:</strong></p>
            <ul>
                <li>底打ちからの急反発（リバウンド）による大きな値幅や短期的な利益を狙いたい方</li>
                <li>話題性や出来高の勢い（モメンタム）を重視したダイナミックなトレードが好きな方</li>
            </ul>
        </div>
    </div>

    <h2>📋 本日のスクリーニング＆指値結果一覧</h2>
"""

    if not df.empty:
        html_content += """
    <table>
        <thead>
            <tr>
                <th>銘柄コード</th>
                <th>企業名</th>
                <th>業種</th>
                <th>検証区分</th>
                <th>おすすめ度(%)</th>
                <th>買い指値</th>
                <th>利確目標 (TP+10%)</th>
                <th>損切ライン (SL-5%)</th>
                <th>出来高倍率</th>
            </tr>
        </thead>
        <tbody>
"""
        for _, row in df.iterrows():
            badge_class = "badge-main" if "本命" in str(row["区分"]) else "badge-sub"
            html_content += f"""
            <tr>
                <td><strong>{row['銘柄コード']}</strong></td>
                <td><strong>{row['企業名']}</strong></td>
                <td>{row['業種']}</td>
                <td><span class="{badge_class}">{row['区分']}</span></td>
                <td><strong>{row['おすすめ度(%)']}%</strong></td>
                <td class="price-buy">{row['買い指値']:,} 円</td>
                <td class="price-tp">+{row['利確目標(TP+10%)']:,} 円</td>
                <td class="price-sl">-{row['損切ライン(SL-5%)']:,} 円</td>
                <td>{row['出来高倍率']} 倍</td>
            </tr>
"""
        html_content += """
        </tbody>
    </table>
"""
    else:
        html_content += """
    <div class="no-data">
        <p>本日の厳しい基準（高値-30%超・出来高急増）を満たす銘柄はありませんでした。</p>
    </div>
"""

    html_content += """
    <footer>
        <p>© 2026 YouTubabaa / 株式投資自動スクリーニングシステム</p>
    </footer>
</body>
</html>
"""

    os.makedirs("docs", exist_ok=True)
    out_html = os.path.join("docs", "index.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  Webレポートを docs/ に生成しました: {out_html}")

def main():
    print(">>> 1. スクリーニングを実行中（daily_morning_routine）...")
    try:
        run_morning_routine()
    except Exception as e:
        print(f"スクリーニング実行エラー: {e}")
    
    print(">>> 2. WebレポートHTML生成中（generate_web_report）...")
    generate_web_report()

if __name__ == "__main__":
    main()
