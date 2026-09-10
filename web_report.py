# -*- coding: utf-8 -*-
"""
web_report.py - 最新の朝の指値プランCSVを読み込んで、考え方・手法ごとの違いやおすすめを表示するWebレポート（index.html）生成モジュール
"""
from __future__ import annotations
import os
import glob
import pandas as pd
import config

def generate_web_report():
    print("=" * 76)
    print("Webレポート（考え方別・手法別おすすめ表示対応）のHTML生成中...")
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

    # HTMLの構築（考え方・検証方法ごとの特徴とお勧めをセクションとして追加）
    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>今日の株式運用候補 ＆ 指値プラン・検証手法別おすすめ</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #f8f9fa; color: #333; line-height: 1.6; }}
        h1 {{ color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 30px; font-size: 1.3em; border-left: 4px solid #1a73e8; padding-left: 10px; }}
        .date-badge {{ background: #e8f0fe; color: #1a73e8; padding: 5px 12px; border-radius: 4px; font-weight: bold; display: inline-block; margin-bottom: 20px; }}
        
        /* 検証手法ごとの解説カード */
        .strategy-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .strategy-card {{ background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); padding: 20px; border-top: 4px solid #1a73e8; }}
        .strategy-card.alt {{ border-top-color: #28a745; }}
        .strategy-card h3 {{ margin-top: 0; color: #1a73e8; font-size: 1.1em; }}
        .strategy-card.alt h3 {{ color: #28a745; }}
        .strategy-card p {{ margin: 8px 0; font-size: 0.95em; color: #555; }}
        .strategy-card ul {{ padding-left: 20px; margin: 8px 0; font-size: 0.92em; color: #444; }}

        /* テーブルスタイル */
        table {{ width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 6px; overflow: hidden; margin-top: 10px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background: #1a73e8; color: white; font-weight: 600; }}
        tr:hover {{ background: #f1f3f5; }}
        .badge-main {{ background: #28a745; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; }}
        .badge-sub {{ background: #6c757d; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.85em; }}
        .price {{ font-family: monospace; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>今日の株式運用候補 ＆ 指値プラン</h1>
    <div class="date-badge">基準日: {target_date}</div>

    <h2>💡 検証方法・アプローチ別の特徴とおすすめ</h2>
    <div class="strategy-container">
        <!-- 戦略1 -->
        <div class="strategy-card">
            <h3>📈 【本命セクター重視】アプローチ</h3>
            <p><b>考え方:</b> 過去のバックテストデータから「勝率55%以上」が実証されている高勝率セクター（海運、銀行、パルプ・紙など）に資金を集中させる方法。</p>
            <p><b>こんな方におすすめ:</b></p>
            <ul>
                <li>相場全体のトレンドに逆らわず、確率的に優位な業種から手堅くリターンを狙いたい方</li>
                <li>セクターローテーションの波に乗って安定感を重視したい方</li>
            </ul>
        </div>
        <!-- 戦略2 -->
        <div class="strategy-card alt">
            <h3>⚡ 【高スコア・モメンタム急増】アプローチ</h3>
            <p><b>考え方:</b> 52週高値から大幅に下落した大底圏（-30%以下）にあり、かつ本日の出来高が急増している銘柄をスコアリングで多角的に抽出する方法。</p>
            <p><b>こんな方におすすめ:</b></p>
            <ul>
                <li>底打ちからの急反発（リバウンド）による大きな値幅や短期的な利益を狙いたい方</li>
                <li>話題性や出来高の勢い（モメンタム）を重視したダイナミックなトレードが好きな方</li>
            </ul>
        </div>
    </div>

    <h2>📋 本日のスクリーニング＆指値結果一覧</h2>
    <table>
        <thead>
            <tr>
                <th>銘柄コード</th>
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

    if not df.empty:
        for _, row in df.iterrows():
            is_main = "本命" in str(row.get("区分", ""))
            badge_class = "badge-main" if is_main else "badge-sub"
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
