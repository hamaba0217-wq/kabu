# -*- coding: utf-8 -*-
"""
web_report.py - 地合い連動型＆最大3選のおすすめ銘柄を含むHTMLレポート生成モジュール
"""
from __future__ import annotations
import os
import datetime as pd_dt
import pandas as pd
import config
from sources import JQuants, JST
from badnews import _bad_flags

def generate_html_report(quotes: pd.DataFrame, fin=None, listed=None, margin=None):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    today_str = pd_dt.datetime.now(JST).date().isoformat()
    
    # セクターマップの作成
    sector_map = {}
    if listed is not None:
        code_col = "Code" if "Code" in listed.columns else ("code" if "code" in listed.columns else None)
        sec_col = None
        for c in ["sector33", "Sector33CodeName", "sector33_code", "Sector17CodeName"]:
            if c in listed.columns:
                sec_col = c
                break
        if code_col and sec_col:
            sector_map = dict(zip(listed[code_col].astype(str).str.slice(0, 4), listed[sec_col]))

    shares_map = {}
    if fin is not None and "shares_outstanding" in fin.columns:
        f_sorted = fin.sort_values("disclosed_date").dropna(subset=["shares_outstanding"])
        shares_map = dict(f_sorted.groupby("code")["shares_outstanding"].last())

    qi = quotes.sort_values(["code", "date"]).reset_index(drop=True)
    g = qi.groupby("code")

    qi["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    qi["vol_ratio"] = qi["volume"] / qi["vol_ma20"]
    qi["high_52w"] = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / qi["high_52w"] - 1.0
    qi["day_range"] = (qi["high"] - qi["low"]) / qi["open"] * 100.0

    qi["ret_1d"] = g["close"].pct_change()
    daily_market = qi.groupby("date")["ret_1d"].mean().reset_index(name="market_ret")
    daily_market["market_ma5"] = daily_market["market_ret"].rolling(5, min_periods=1).mean()
    qi = pd.merge(qi, daily_market[["date", "market_ma5"]], on="date", how="left")

    codes_str = qi["code"].astype(str)
    shares_series = codes_str.map(shares_map).fillna(0)
    qi["mcap_oku"] = qi["close"] * shares_series / 1e8
    qi["is_small_cap"] = (qi["mcap_oku"] > 0) & (qi["mcap_oku"] < 100)
    qi["is_bottom"] = qi["pct_from_high"] <= -0.30
    qi["is_vol_surge"] = qi["vol_ratio"] >= 2.0
    qi["is_range_wide"] = qi["day_range"] >= 7.0
    qi["sector"] = codes_str.str.slice(0, 4).map(sector_map).fillna("その他")

    bad_set = _bad_flags(fin) if fin is not None else set()
    qi["date_str"] = pd.to_datetime(qi["date"]).dt.strftime("%Y-%m-%d")
    qi["is_clean"] = [1 if (str(c), d) not in bad_set else 0 for c, d in zip(codes_str, qi["date_str"])]

    latest_date = qi["date"].max()
    latest_subset = qi[qi["date"] == latest_date]
    market_ma5_val = latest_subset["market_ma5"].iloc[0] if len(latest_subset) > 0 and "market_ma5" in latest_subset.columns else 0.0

    # 地合い判定
    if market_ma5_val > 0.005:
        regime = "強気（積極モード）"
        regime_class = "badge-success"
    elif market_ma5_val >= 0.0:
        regime = "通常（標準モード）"
        regime_class = "badge-primary"
    else:
        regime = "弱気（安全・見送りモード）"
        regime_class = "badge-danger"

    b6_mask = qi["is_bottom"] & qi["is_small_cap"] & qi["is_vol_surge"] & qi["is_range_wide"] & (qi["is_clean"] == 1)
    if market_ma5_val < 0:
        b6_mask = b6_mask & qi["sector"].isin(["小売業", "情報･通信業", "サービス業"])

    today_candidates = qi[(qi["date"] == latest_date) & b6_mask].copy()

    # おすすめ銘柄は最大3つ（出来高倍率が高い順、弱気相場のときは無しの判定も可だが上位最大3つ）
    recommended = []
    if len(today_candidates) > 0 and market_ma5_val >= -0.01: # 極端な暴落でなければ上位3つ
        top_picks = today_candidates.sort_values("vol_ratio", ascending=False).head(3)
        for _, r in top_picks.iterrows():
            recommended.append({
                "code": r["code"],
                "sector": r["sector"],
                "vol_ratio": f"{r['vol_ratio']:.2f}倍",
                "day_range": f"{r['day_range']:.1f}%",
                "mcap": f"{r['mcap_oku']:.1f}億"
            })

    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>株価スクリーニング & 本日の厳選おすすめレポート ({today_str})</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; color: #333; }}
        .container {{ max-width: 900px; margin: auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ font-size: 24px; border-bottom: 2px solid #007bff; padding-bottom: 10px; margin-top: 0; }}
        .card {{ background: #f1f3f5; border-radius: 6px; padding: 20px; margin-bottom: 20px; }}
        .badge {{ display: inline-block; padding: 6px 12px; font-size: 14px; font-weight: bold; border-radius: 4px; color: #fff; }}
        .badge-success {{ background-color: #28a745; }}
        .badge-primary {{ background-color: #007bff; }}
        .badge-danger {{ background-color: #dc3545; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; border: 1px solid #dee2e6; text-align: left; }}
        th {{ background-color: #e9ecef; }}
        .no-rec {{ color: #6c757d; font-style: italic; }}
    </style>
</head>
<body>
<div class="container">
    <h1>株価スクリーニング & 地合い連動レポート</h1>
    <p>レポート生成日時: {today_str} (データ基準日: {str(latest_date)[:10]})</p>
    
    <div class="card">
        <h3>市場地合いステータス</h3>
        <p>市場5日平均リターン: <strong>{market_ma5_val*100:.3f}%</strong></p>
        <p>判定モード: <span class="badge {regime_class}">{regime}</span></p>
    </div>

    <div class="card">
        <h3>本日のおすすめ銘柄（最大3選）</h3>
"""

    if len(recommended) > 0:
        html_content += """
        <table>
            <tr><th>銘柄コード</th><th>セクター</th><th>出来高倍率</th><th>日中レンジ</th><th>時価総額</th></tr>
"""
        for rec in recommended:
            html_content += f"<tr><td>{rec['code']}</td><td>{rec['sector']}</td><td>{rec['vol_ratio']}</td><td>{rec['day_range']}</td><td>{rec['mcap']}</td></tr>\n"
        html_content += "        </table>\n"
    else:
        html_content += '<p class="no-rec">本日は条件に合致するおすすめ銘柄はありません（見送り推奨）。</p>\n'

    html_content += f"""
    </div>
    
    <div class="card">
        <h3>スクリーニング条件概要</h3>
        <ul>
            <li>高値圏からの下落率: 30%以上の押し目ゾーン</li>
            <li>時価総額: 10億円〜100億円未満の小型株</li>
            <li>出来高急増: 20日平均比2倍以上</li>
            <li>日中ボラティリティ: 7%以上の幅広いレンジ</li>
            <li>地合いフィルター連動（弱気相場では小売・情報通信・サービスに限定）</li>
        </ul>
    </div>
</div>
</body>
</html>
"""

    out_path = os.path.join(config.OUTPUT_DIR, f"{today_str}_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"WEBレポートを生成しました: {out_path}")

def main():
    jq = JQuants()
    print("データ読み込み中（WEBレポート生成）...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    try:
        fin = jq.financials(config.BACKTEST_LOOKBACK_DAYS)
    except Exception:
        fin = None
    listed = jq.listed()
    try:
        margin = jq.margin(config.BACKTEST_LOOKBACK_DAYS)
    except Exception:
        margin = None
    generate_html_report(quotes, fin=fin, listed=listed, margin=margin)

if __name__ == "__main__":
    main()
