# -*- coding: utf-8 -*-
"""
今日の候補：押し目 × 高勝率業種 × 決算1か月回避 × 悪材料なし

今日までの全検証で「唯一、前半・後半どちらでも対相場超過プラス」を保った条件。
その条件に、最新営業日時点で該当する銘柄を抽出する。

条件（検証で確定したもの・preentryと同一ロジックを再利用）
------------------------------------------------------------
  ・押し目：52週高値から -15〜-5%（_f_pullback）
  ・高勝率業種：保険/銀行/証券/倉庫運輸/水産農林（_f_high_win_sector）
  ・決算1か月回避：次の決算まで20営業日超（決算が近い銘柄は除外）
  ・悪材料なし：直近決算で 予想未達/減益/減収 のいずれも無い

検証結果（この条件の全期間）：対相場超過+0.95%・勝率62.1%・646件
  前半+0.70%/後半+0.56%（押し目×高勝率業種の分割検証）
  ※対相場超過は小さく、手数料・スリッページを引くと消える水準。
  ※これは分析補助であり投資助言ではありません。判断と責任は利用者にあります。

先読み防止：最新営業日までのデータだけで判定。決算・悪材料も同時点の情報のみ。
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import config
import technical
from refine import _f_pullback, _f_high_win_sector
from preearnings import _next_earnings_date
from preentry import _biz_days_until
from badnews import _latest_fin_before, _bad_flags, _prep_fin_extended


AVOID_EARNINGS_DAYS = 20    # 次の決算まで20営業日以内なら除外（決算1か月回避）
FORWARD_DAYS = 5            # 過去同条件時の「その後1週間」＝5営業日
MIN_GAP_DAYS = 5           # 同一シグナルの重複を避ける最小間隔（営業日）
HOLD_DAYS_2W = 10          # 保有2週間＝10営業日（到達確率の測定期間）
TARGET_UP = 10.0           # 上昇目標（%）：+10%到達を測る
TARGET_DOWN = -5.0         # 下落ライン（%）：-5%到達を測る


LAG_DAYS = 7    # 需給データの公表遅延（先読み防止）


def _latest_by_key(df, key_col, date_col, as_of):
    """各キーについて、as_of の7日前までの直近1行を返す辞書。"""
    if df is None or len(df) == 0:
        return {}
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=LAG_DAYS)
    sub = df[pd.to_datetime(df[date_col]) <= cutoff]
    out = {}
    for k, g in sub.groupby(key_col):
        out[str(k)] = g.sort_values(date_col).iloc[-1]
    return out


def _condition_met(r, code, as_of, fin_by_code, fin_ext, require_sector=True):
    """その時点 as_of で候補条件を全部満たすか（find と同一判定）。
    require_sector=False なら高勝率業種の絞りを外す（押し目・決算回避・悪材料のみ）。"""
    if not _f_pullback(r):
        return False
    if require_sector and not _f_high_win_sector(r):
        return False
    earnings = _next_earnings_date(fin_by_code, code, as_of)
    days_until = _biz_days_until(as_of, earnings)
    if days_until is not None and days_until <= AVOID_EARNINGS_DAYS:
        return False
    fin_row = _latest_fin_before(fin_ext, code, as_of)
    flags = _bad_flags(fin_row)
    if flags["予想未達"] or flags["減益"] or flags["減収"]:
        return False
    return True


STOP_LOSS_HIST = -5.0      # 過去履歴の損切りライン（%）。保有中に到達したら決済。


def _history_for_code(code, cqi, sec_name, fin_by_code, fin_ext):
    """1銘柄について、過去に条件を満たした時点の結果を集める。
    返り値:
      rets_nostop … 5営業日後終値リターン(%)のリスト
      rets_stop   … 損切り-5%を入れた場合の5営業日リターン(%)のリスト
      hit_up      … 保有10営業日(2週間)以内に +10% に到達した回数
      hit_down    … 保有10営業日(2週間)以内に -5% に到達した回数
      n2w         … 2週間の判定ができたサンプル数
    到達は終値ベース（保有中の各日終値で判定）。"""
    cqi = cqi.sort_values("date").reset_index(drop=True)
    closes = cqi["close"].values
    dates = cqi["date"].values
    n = len(cqi)
    rets_nostop, rets_stop = [], []
    hit_up = hit_down = n2w = 0
    last_idx = -MIN_GAP_DAYS - 1
    # 2週間(10営業日)先まで見るため、末尾は HOLD_DAYS_2W 分残す
    upper = n - max(FORWARD_DAYS, HOLD_DAYS_2W)
    for i in range(260, upper):
        if i - last_idx < MIN_GAP_DAYS:
            continue
        r = cqi.iloc[i].copy()
        r["sector33"] = sec_name
        if not _condition_met(r, code, pd.Timestamp(dates[i]), fin_by_code, fin_ext):
            continue
        entry = closes[i]
        if not entry or entry <= 0:
            continue
        # 5営業日の経路（損切り集計用）
        path5 = [(closes[i + k] / entry - 1.0) * 100 for k in range(1, FORWARD_DAYS + 1)]
        rets_nostop.append(path5[-1])
        stopped = False
        for pr in path5:
            if pr <= STOP_LOSS_HIST:
                rets_stop.append(STOP_LOSS_HIST)
                stopped = True
                break
        if not stopped:
            rets_stop.append(path5[-1])
        # 10営業日(2週間)の経路：+10%到達・-5%到達をそれぞれ判定
        path10 = [(closes[i + k] / entry - 1.0) * 100 for k in range(1, HOLD_DAYS_2W + 1)]
        n2w += 1
        if any(pr >= TARGET_UP for pr in path10):
            hit_up += 1
        if any(pr <= TARGET_DOWN for pr in path10):
            hit_down += 1
        last_idx = i
    return rets_nostop, rets_stop, hit_up, hit_down, n2w


def find(quotes, fin, listed, margin=None, short_ratio=None):
    qi = technical.add_indicators(quotes)
    # 52週高値からの下落率（refine.run と同じ計算）。押し目判定に必要。
    g = qi.sort_values(["code", "date"]).groupby("code")
    h = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / h - 1.0
    as_of = qi["date"].max()
    # 最新営業日の1日分だけを取り出す（全行コピーを避ける＝メモリ節約）。
    # 各銘柄その日に1行なので、date==as_of で最新スナップショットになる。
    snap = qi[qi["date"] == as_of].drop_duplicates("code", keep="last").set_index("code")

    # 業種を付与
    sec = {}
    sec_code = {}
    name_map = {}
    if "sector33" in listed.columns:
        for _, r in listed.iterrows():
            sec[str(r["code"])] = r["sector33"]
    if "sector33_code" in listed.columns:
        for _, r in listed.iterrows():
            sec_code[str(r["code"])] = str(r["sector33_code"])
    if "company_name" in listed.columns:
        for _, r in listed.iterrows():
            name_map[str(r["code"])] = r["company_name"]

    # 需給データの直近値（先読み防止：7日前まで）
    latest_margin = {}
    if margin is not None and len(margin):
        mm = margin.copy()
        for c in ("long_margin", "short_margin"):
            if c in mm.columns:
                mm[c] = pd.to_numeric(mm[c], errors="coerce")
        latest_margin = _latest_by_key(mm, "code", "date", as_of)
    latest_sr = {}
    if short_ratio is not None and len(short_ratio):
        latest_sr = _latest_by_key(short_ratio, "sector33", "date", as_of)

    fin_ext = _prep_fin_extended(fin)
    # _next_earnings_date は「開示日のリスト」を受け取る（preentryと同一形式）
    fin_by_code = {}
    if "code" in fin.columns and "disclosed_date" in fin.columns:
        for code_g, gg in fin.groupby("code"):
            fin_by_code[str(code_g)] = sorted(pd.to_datetime(gg["disclosed_date"]).tolist())

    # 過去同条件の値動き集計用に、指標つき全期間データを銘柄別に持つ
    qi_by_code = {str(c): gg for c, gg in qi.groupby("code")}

    rows = []
    for code, row in snap.iterrows():
        code = str(code)
        r = row.copy()
        r["sector33"] = sec.get(code)
        # 押し目
        if not _f_pullback(r):
            continue
        # 高勝率業種
        if not _f_high_win_sector(r):
            continue
        # 決算1か月回避：次の決算まで20営業日以内なら除外
        earnings = _next_earnings_date(fin_by_code, code, as_of)
        days_until = _biz_days_until(as_of, earnings)
        if days_until is not None and days_until <= AVOID_EARNINGS_DAYS:
            continue
        # 悪材料なし
        fin_row = _latest_fin_before(fin_ext, code, as_of)
        flags = _bad_flags(fin_row)
        if flags["予想未達"] or flags["減益"] or flags["減収"]:
            continue

        # 需給データを付与（先読み防止済みの直近値）
        m = latest_margin.get(code)
        long_m = float(m["long_margin"]) if m is not None and pd.notna(m.get("long_margin")) else None
        short_m = float(m["short_margin"]) if m is not None and pd.notna(m.get("short_margin")) else None
        margin_ratio = (long_m / short_m) if (long_m and short_m and short_m > 0) else None
        scode = sec_code.get(code)
        sr_row = latest_sr.get(str(scode)) if scode else None
        sr_val = float(sr_row["short_ratio"]) if sr_row is not None and pd.notna(sr_row.get("short_ratio")) else None

        # 過去に同条件だった時の、その後の値動き（銘柄ごとの癖）
        cqi = qi_by_code.get(code)
        if cqi is not None:
            hist_ns, hist_st, hit_up, hit_down, n2w = _history_for_code(
                code, cqi, sec.get(code), fin_by_code, fin_ext)
        else:
            hist_ns, hist_st, hit_up, hit_down, n2w = [], [], 0, 0, 0
        if hist_ns:
            a = np.array(hist_ns)
            hist_n = len(a)
            hist_mean = round(float(a.mean()), 2)
            hist_win = round(float((a > 0).mean() * 100), 1)
            hist_min = round(float(a.min()), 2)
            hist_max = round(float(a.max()), 2)
        else:
            hist_n, hist_mean, hist_win, hist_min, hist_max = 0, None, None, None, None
        stop_mean = round(float(np.mean(hist_st)), 2) if hist_st else None
        stop_win = round(float((np.array(hist_st) > 0).mean() * 100), 1) if hist_st else None
        # 2週間で +10%到達 / -5%到達 の確率
        up10_pct = round(hit_up / n2w * 100, 1) if n2w else None
        down5_pct = round(hit_down / n2w * 100, 1) if n2w else None

        rows.append({
            "code": code,
            "銘柄名": name_map.get(code, ""),
            "業種": sec.get(code, ""),
            "高値からの下落%": round(float(r.get("pct_from_high", np.nan)) * 100, 1)
                if pd.notna(r.get("pct_from_high")) else None,
            "次決算まで営業日": days_until if days_until is not None else "不明",
            "終値": round(float(r["close"]), 1),
            "信用買い残": int(long_m) if long_m is not None else None,
            "信用売り残": int(short_m) if short_m is not None else None,
            "信用倍率": round(margin_ratio, 2) if margin_ratio is not None else None,
            "業種空売り比率": round(sr_val, 3) if sr_val is not None else None,
            "過去該当n": hist_n,
            "平均_損切なし%": hist_mean,
            "平均_損切5%": stop_mean,
            "勝率_損切なし%": hist_win,
            "2週で+10%到達%": up10_pct,
            "2週で-5%到達%": down5_pct,
            "過去最小%": hist_min,
            "過去最大%": hist_max,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("高値からの下落%").reset_index(drop=True)
    return df, pd.Timestamp(as_of).date()


def _print_morning_summary(df):
    """今日の運用サマリー：有力候補を絞り、エントリー/損切り/利確の目安を出す。"""
    MIN_N = 30          # 過去実績の最低サンプル数
    STOP_PCT = -5.0     # 損切りライン（検証で使った値）
    TARGET_PCT = 10.0   # 利確目安（2週で+10%到達を測った値）
    d = df.copy()
    # 実績が信頼でき、リスクよりリターンが優位な銘柄を絞る
    if "過去該当n" in d.columns:
        d = d[d["過去該当n"] >= MIN_N]
    if "平均_損切5%" in d.columns:
        d = d[d["平均_損切5%"] > 0]
    if "2週で+10%到達%" in d.columns and "2週で-5%到達%" in d.columns:
        d = d[d["2週で+10%到達%"].fillna(0) >= d["2週で-5%到達%"].fillna(999)]
    # 平均_損切5%が高い順に上位を「特に有力」として出す
    if "平均_損切5%" in d.columns:
        d = d.sort_values("平均_損切5%", ascending=False)

    print("=" * 72)
    print("★ 今日の運用サマリー（特に有力な候補）")
    print("=" * 72)
    if d.empty:
        print("  本日は、実績・リスクの条件を満たす『特に有力』な候補はありません。")
        print("  （過去該当n≧30・平均プラス・+10%到達≧-5%到達 をすべて満たすもの）")
        print("  下の候補一覧は参考として見てください。無理に買う必要はありません。")
        return
    print(f"  条件を満たした {len(d)} 件（過去実績が良く、下振れより上昇が優位）:\n")
    for _, r in d.head(5).iterrows():
        close = r.get("終値")
        name = r.get("銘柄名", "")
        code = r.get("code", "")
        sec = r.get("業種", "")
        if close and close == close:  # not NaN
            entry = float(close)
            stop = entry * (1 + STOP_PCT / 100)
            target = entry * (1 + TARGET_PCT / 100)
            print(f"  ● {name}（{code}）{sec}")
            print(f"     現値 {entry:,.1f}円  →  損切り {stop:,.1f}円(-5%) / "
                  f"利確目安 {target:,.1f}円(+10%)")
            print(f"     過去: 平均(損切5%){r.get('平均_損切5%'):+.2f}% "
                  f"勝率{r.get('勝率_損切なし%','?')}% "
                  f"2週で+10%到達{r.get('2週で+10%到達%','?')}%/"
                  f"-5%到達{r.get('2週で-5%到達%','?')}% "
                  f"(n{r.get('過去該当n')})")
        else:
            print(f"  ● {name}（{code}）{sec}  ※終値データなし")
    print("\n  ※上記は過去実績にもとづく目安。将来を保証しません。手数料未考慮。")
    print("  ※損切り-5%・利確+10%は検証で用いた値。必ず自分の判断で決めてください。")


def main(argv=None):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    argv = argv or []
    full = ("full" in argv) or ("--full" in argv)
    from sources import JQuants, JST
    jq = JQuants()
    print("キャッシュからデータを読み込み中...")
    quotes = jq.quotes(config.BACKTEST_LOOKBACK_DAYS)
    fin = jq.financials(config.BACKTEST_LOOKBACK_DAYS)
    listed = jq.listed()
    print(f"  株価 {len(quotes):,}行 / 決算 {len(fin):,}行 / 銘柄 {len(listed):,}")

    # 需給データ（信用残＝銘柄別、空売り比率＝業種別）
    print("信用残・空売り比率を取得します...")
    try:
        margin = jq.margin(config.BACKTEST_LOOKBACK_DAYS)
        print(f"  信用残 {len(margin):,}行")
    except Exception as e:
        print(f"  [!] 信用残の取得に失敗: {e}")
        margin = None
    try:
        short_ratio = jq.short_ratio(config.BACKTEST_LOOKBACK_DAYS)
        print(f"  空売り比率 {len(short_ratio):,}行")
    except Exception as e:
        print(f"  [!] 空売り比率の取得に失敗: {e}")
        short_ratio = None

    df, as_of = find(quotes, fin, listed, margin=margin, short_ratio=short_ratio)
    print(f"\n{'='*72}")
    print(f"今日の候補：押し目 × 高勝率業種 × 決算1か月回避 × 悪材料なし")
    print(f"（{as_of} 時点）")
    print(f"{'='*72}\n")
    if df.empty:
        print("本日、この条件に該当する銘柄はありませんでした。")
        print("（押し目ゾーンにある高勝率業種の銘柄が無い日もあります）")
    else:
        print(f"該当 {len(df)} 件（高値からの下落が深い順）:\n")

        # ── 今日の運用サマリー（毎朝これだけ見れば動ける形）──
        # 過去実績が良くリスクが低めの銘柄を「特に有力」として絞る。
        # 条件：過去該当n≧30／平均_損切5%>0／2週で-5%到達% ≤ +10%到達%
        _print_morning_summary(df)

        # 画面は主要列のみ（信用買残/売残の実数はCSVに）
        show_cols = ["code", "銘柄名", "業種", "高値からの下落%", "信用倍率",
                     "過去該当n", "平均_損切なし%", "平均_損切5%",
                     "2週で+10%到達%", "2週で-5%到達%", "過去最大%", "過去最小%"]
        show_cols = [c for c in show_cols if c in df.columns]
        print(f"\n【参考】候補 {len(df)} 件すべての詳細:\n")
        print(df[show_cols].to_string(index=False))
        today = dt.datetime.now(JST).date().isoformat()
        path = os.path.join(config.OUTPUT_DIR, f"{today}_candidates_best.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n（信用買い残・売り残など全列はCSVに出力）")
        print(f"CSV: {path}")

    print(f"\n{'-'*72}")
    print("運用の目安（検証結果・手数料/スリッページ未考慮）:")
    print("  この条件の全期間：対相場超過+0.95%・勝率62.1%・646件")
    print("  前半+0.70%/後半+0.56%（両期間プラス＝再現性が確認された唯一の条件）")
    print("  ※対相場超過は小さく、手数料を引くと消える水準。大きく儲かる方針ではない。")
    print("  ※分析補助であり投資助言ではありません。売買判断と責任は利用者にあります。")
    print("  --- 信用残・空売り比率について ---")
    print("  ・信用残（買残/売残/倍率）は銘柄別。空売り比率は業種別（個別銘柄は取得不可）。")
    print("  ・信用倍率 = 買残÷売残。空欄は信用取引対象外（貸借銘柄でない）。")
    print("  ・信用倍率・空売り比率とも検証では対相場超過を安定改善できず＝参考情報。")
    print("  ・需給データは7日前までの直近公表値（先読み防止）。")
    print("  --- 過去同条件のその後について（銘柄ごとの癖）---")
    print("  ・過去該当n＝その銘柄で今回と同条件を過去に満たした回数。nが小さいほど偶然。")
    print("  ・平均_損切なし＝5営業日後終値の平均。平均_損切5%＝-5%到達で決済した平均。")
    print("  ・2週で+10%到達%＝保有10営業日以内に一度でも+10%に達した回の割合。")
    print("  ・2週で-5%到達%＝保有10営業日以内に一度でも-5%に達した回の割合。")
    print("    （+10%と-5%は独立判定。両方に触れる回もある＝先に-5%後に+10%等）")
    print("  ・到達判定は終値ベース。ザラ場の一時的な上下は反映しない。")
    print("  ・過去の傾向であり将来を保証しない。先読みなし・手数料未考慮。")
    print("  ・信用買い残/売り残の実数はCSVに出力（画面は倍率のみ表示）。")

    # 一気通貫（full）：業種の絞りなしで今日の押し目候補を出し、
    # 平均(損切5%)が高い上位を、業種フィルタなしの3分割で厳格再検証する。
    # 結果に業種を表示し、高勝率業種(★)かどうかが分かるようにする。
    if full:
        print(f"\n{'='*72}")
        print("一気通貫：業種の絞りなしで押し目候補を評価し、上位を3分割で厳格再検証")
        print(f"{'='*72}")
        import validate_candidates as vc
        # 業種を問わず、今日 押し目×決算回避×悪材料なし を満たす銘柄
        all_codes = vc._todays_pullback_codes(quotes, fin, listed)
        print(f"  業種の絞りなしの今日の押し目候補: {len(all_codes)}件")
        print("  各銘柄の過去実績（平均_損切5%）で上位を選び、3分割検証します")
        print("  （数百件を走査するため時間がかかります）")
        # 各銘柄の過去平均(損切5%)を出して上位を選ぶ（業種フィルタなし）
        ranked = _rank_by_stop_avg(quotes, fin, listed, all_codes)
        top = [c for c, _ in ranked[:8]]
        print(f"  上位8件を3分割検証: {top}")
        results = vc.run_strict(quotes, fin, listed, top, require_sector=False)
        print("\n" + vc.summarize_strict(results))


def _rank_by_stop_avg(quotes, fin, listed, codes, min_n=30):
    """各銘柄の過去（業種フィルタなし）の平均_損切5%を計算し、高い順に並べる。"""
    import validate_candidates as vc
    qi = technical.add_indicators(quotes)
    g = qi.sort_values(["code", "date"]).groupby("code")
    h = g["close"].transform(lambda s: s.shift(1).rolling(250, min_periods=20).max())
    qi["pct_from_high"] = qi["close"] / h - 1.0
    sec_name = {}
    if "sector33" in listed.columns:
        for _, r in listed.iterrows():
            sec_name[str(r["code"])] = r["sector33"]
    fin_ext = _prep_fin_extended(fin)
    fin_by_code = {}
    if "code" in fin.columns and "disclosed_date" in fin.columns:
        for cg, gg in fin.groupby("code"):
            fin_by_code[str(cg)] = sorted(pd.to_datetime(gg["disclosed_date"]).tolist())
    qi_by_code = {str(c): gg for c, gg in qi.groupby("code")}
    scored = []
    for code in codes:
        code = str(code)
        cqi = qi_by_code.get(code)
        if cqi is None:
            continue
        events = vc._collect_events(code, cqi, sec_name.get(code), fin_by_code, fin_ext,
                                    require_sector=False)
        if len(events) < min_n:
            continue
        stop5 = np.mean([e["stop5"] for e in events])
        scored.append((code, stop5))
    scored.sort(key=lambda x: -x[1])
    return scored


if __name__ == "__main__":
    import sys
    main(sys.argv[2:] if len(sys.argv) > 2 else [])
