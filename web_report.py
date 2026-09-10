# -*- coding: utf-8 -*-
"""
web_report.py - 最新の朝の指値プランCSVを読み込んで、Webサイト（HTML）の index.html をルートに直接生成・更新するモジュール
"""
from __future__ import annotations
import os
import glob
import pandas as pd
import config

def generate_web_report():
    print("=" * 76)
    print("Webレポート（hamaba0217-wq用）のHTML生成中...")
    print("=" * 76)

    pattern = os.path.join(config.OUTPUT_DIR, "*_morning_orders.csv")
    files = glob.glob(pattern)

    df = pd.DataFrame()
    target_date = "データなし"
    if files:
        latest_file = max(files, key=os.path.getmtime)
        target_date = os.path.basename(latest_file).split("_")[0]
        try:
            df = pd.read_csv(latest_file)
            print(f"  最新の指値CSVを読み込みました: {latest_file} (件数: {len(df)})")
        except Exception as e:
            print(f"  CSV読み込みエラー: {e}")

    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>今日の株式運用候補 ＆ 指値プラン</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #f8f9fa; color: #333; }}
        h1 {{ color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; }}
        .date-badge {{ background: #e8f0fe; color: #1a73e8; padding: 5px 12px; border-radius: 4px; font-weight: bold; display: inline-block; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 6px; overflow: hidden; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background: #1a73e8; color: white; font-weight: 600; }}
        tr:hover {{ background: #f1f3f5; }}
        .badge-main {{ background: #28a745; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.85em; }}
        .badge-sub {{ background: #6c757d; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.85em; }}
        .price {{ font-family: monospace; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>今日の株式運用候補 ＆ 指値プラン</h1>
    <div class="date-badge">基準日: {target_date}</div>
    
    <table>
        <thead>
            <tr>
                <th>銘柄コード</th>
                <th>業種</th>
                <th>区分</th>
                <th>おすすめ度(%)</th>
                <th>買い指値</th>
                <th>利確目標 (TP+10%)</th>
                <th>損切ライン (SL-5%)</th>
                <th>出来高倍率</th>
            </tr>
        </thead>
        <tbody>
"""

    if not df.empty:
        for _, row in df.iterrows():
            badge_class = "badge-main" if "本命" in str(row.get("区分", "")) else "badge-sub"
            html_content += f"""
            <tr>
                <td class="price"><b>{row.get("銘柄コード", "")}</b></td>
                <td>{row.get("業種", "")}</td>
                <td><span class="{badge_class}">{row.get("区分", "")}</span></td>
                <td>{row.get("おすすめ度(%)", "")}%</td>
                <td class="price">¥{row.get("買い指値", 0):,.1f}</td>
                <td class="price" style="color: #28a745;">¥{row.get("利確目標(TP+10%)", 0):,.1f}</td>
                <td class="price" style="color: #dc3545;">¥{row.get("損切ライン(SL-5%)", 0):,.1f}</td>
                <td>{row.get("出来高倍率", 0)}x</td>
            </tr>
            """
    else:
        html_content += """
            <tr>
                <td colspan="8" style="text-align: center; color: #6c757d; padding: 30px;">本日の条件を満たす銘柄はありません。</td>
            </tr>
        """

    html_content += f"""
        </tbody>
    </table>
    <footer style="margin-top: 30px; text-align: center; color: #6c757d; font-size: 0.9em;">
        &copy; 2026 YouTubabaa / 株式投資自動スクリーニングシステム
    </footer>
</body>
</html>
"""

    out_html = "index.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  Webレポートをルートに生成しました: {out_html}")

def main():
    generate_web_report()

if __name__ == "__main__":
    main()
