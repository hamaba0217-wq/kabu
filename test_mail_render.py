# -*- coding: utf-8 -*-
"""メール本文のHTML生成とRole 0判定を、合成データで検証する（送信はしない）。"""
import datetime as dt
import numpy as np, pandas as pd
import mailer, regime, screen, config
import test_backtest as tb

quotes, fin, listed = tb.make_data()
rg = regime.judge(quotes, listed)
label, action = regime.LEVEL_LABEL[rg["level"]]
print("=== Role 0 判定 ===")
print(f"  {label} — {action}")
for r in rg["reasons"]:
    print("   ・" + r)

prices = screen.latest_prices(quotes, config.MA_WINDOW)
yoy = screen.yoy_table(fin)
picks, log = screen.run_screen(prices, yoy, listed, pd.DataFrame())
out = screen.to_output(picks)
print(f"\n=== 抽出 {len(out)}銘柄 ===")

cols = [c for c in ["code4","company_name","market","close","market_cap_oku",
                    "turnover_oku","sales_yoy","op_yoy"] if c in out.columns]
disp = out[cols].head(5)
html = mailer.wrap("日次スクリーニング テスト",
                   f'<div class="box {rg["level"].lower()}"><b>{label}</b> — {action}</div>'
                   + mailer.df_to_html(disp, left_cols=("code4","company_name","market")))
open("preview.html","w",encoding="utf-8").write(html)
n = len(out)
today = dt.date(2026,7,23)
subject = f"[株スクリーン] {today:%-m/%-d} " + (f"候補{n}件 " if n else "候補なし ") + label.split()[0]
print(f"\n件名: {subject}")
print(f"HTML長: {len(html)}文字 → preview.html に出力")
assert "<table>" in html and "投資助言ではありません" in html
print("\nHTML生成: OK")
