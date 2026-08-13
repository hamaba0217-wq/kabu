# -*- coding: utf-8 -*-
"""
データ取得モジュール

- J-Quants API V2（JPX公式）: 株価・決算サマリー・銘柄一覧
- EDINET API v2（金融庁）  : 大量保有報告書

いずれも公式APIです。規約でスクレイピングを禁止しているサイトは使いません。
"""

from __future__ import annotations

import datetime as dt
import os
import time
from typing import Iterable

import pandas as pd
import numpy as np
import requests

JST = dt.timezone(dt.timedelta(hours=9))


# ---------------------------------------------------------------------------
# 列名の正規化
#
# J-Quants は V1 と V2 で列名が違い、今後も変わり得ます。
# ここで「正規化された名前」に寄せることで、本体のロジックを守ります。
# 想定外の列名だった場合は、下の候補リストに追加してください。
# `python main.py inspect` で実際の列名を確認できます。
# ---------------------------------------------------------------------------

QUOTE_COLUMNS = {
    "code":           ["Code", "LocalCode", "code"],
    "date":           ["Date", "date"],
    "close":          ["C", "Close", "close"],                          # 生の終値（未調整）
    "adj_factor":     ["AdjFactor", "AdjustmentFactor", "adj_factor"],  # 分割・併合の調整係数
    "volume":         ["Vo", "Volume", "volume"],
    "turnover_value": ["Va", "TurnoverValue", "Turnover", "turnover_value"],
}

# 決算サマリーの列名（2026年 V2 実測）
#   連結: Sales / OP、単体(非連結): NCSales / NCOP
#   会社予想(通期): FOP / FSales、発行済株式数: ShOutFY
# 連結が空の会社は単体を使う（financials() で処理）。
FIN_COLUMNS = {
    "code":              ["Code", "LocalCode", "code"],
    "disclosed_date":    ["DiscDate", "DisclosedDate", "disclosed_date"],
    "period_type":       ["CurPerType", "Period", "TypeOfCurrentPeriod", "period_type"],
    "period_end":        ["CurPerEn", "PeriodEnd", "CurrentPeriodEndDate", "period_end"],
    "net_sales":         ["Sales", "NetSales", "Revenue", "net_sales"],
    "operating_profit":  ["OP", "OperatingProfit", "OperatingIncome", "operating_profit"],
    "net_sales_nc":      ["NCSales"],      # 単体売上（フォールバック用）
    "operating_profit_nc": ["NCOP"],       # 単体営業利益（フォールバック用）
    "shares_outstanding": ["ShOutFY", "Shares", "SharesOutstanding", "shares_outstanding"],
    # ①資本効率（ROE・PBR）用。ROE=Profit/Equity、PBR=株価/BookValuePerShare。
    "net_profit":        ["Profit", "NetProfit", "ProfitAttributableToOwnersOfParent", "net_profit"],
    "equity":            ["Equity", "NetAssets", "equity"],
    "bps":               ["BookValuePerShare", "BPS", "bps"],
    # 会社予想（通期）。連結 FOP/FSales、単体 FNCOP/FNCSales。決算モメンタム判定に使う。
    "fc_operating_profit":    ["FOP"],
    "fc_net_sales":           ["FSales"],
    "fc_operating_profit_nc": ["FNCOP"],
    "fc_net_sales_nc":        ["FNCSales"],
}

# ③④需給データの列名マッピング（実データの列名を複数候補で吸収）
MARGIN_COLUMNS = {
    "code":            ["Code", "LocalCode", "code"],
    "date":            ["Date", "date"],
    "long_margin":     ["LongMarginTradeVolume", "long_margin"],       # 信用買い残
    "short_margin":    ["ShortMarginTradeVolume", "short_margin"],      # 信用売り残
}

SHORT_RATIO_COLUMNS = {
    "sector33":        ["Sector33Code", "sector_33_code", "sector33"],
    "date":            ["Date", "date"],
    "short_ratio":     ["ShortSellingWithRestrictionsRatio",
                        "ShortSellingWithoutRestrictionsRatio", "short_ratio"],
}

LISTED_COLUMNS = {
    # 2026年時点のV2実測列名（get_list が返すもの）を先頭に置いています。
    "code":         ["Code", "LocalCode", "code"],
    "company_name": ["CoName", "CompanyName", "Name", "company_name"],
    "market":       ["MktNm", "MarketName", "MarketCodeName", "Market", "market"],
    "sector33":     ["S33Nm", "Sector33Name", "Sector33CodeName", "Sector33", "sector33"],
    "sector33_code": ["S33", "Sector33Code", "sector33_code"],   # 業種33コード（空売り比率との結合用）
}


def code4(code: pd.Series) -> pd.Series:
    """J-Quantsの5桁コード(72030)を、一般的な4桁表記(7203)に直す。

    証券コードは通常4桁ですが、J-Quantsは末尾に0を付けた5桁で返します。
    人が見るときは4桁の方がわかりやすいので、表示用に変換します。
    """
    s = code.astype(str).str.strip()
    return s.where(s.str.len() != 5, s.str[:4])


class ColumnNotFound(Exception):
    """必須の列が見つからなかったときに出す例外"""


def normalize(df: pd.DataFrame, mapping: dict, required: Iterable[str],
              label: str) -> pd.DataFrame:
    """候補リストに従って列名を正規化する。"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=list(mapping.keys()))

    out = pd.DataFrame(index=df.index)
    found = {}
    for canonical, candidates in mapping.items():
        for cand in candidates:
            if cand in df.columns:
                out[canonical] = df[cand]
                found[canonical] = cand
                break

    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ColumnNotFound(
            f"\n[{label}] 必須の列が見つかりませんでした: {missing}\n"
            f"  実際に返ってきた列: {list(df.columns)}\n"
            f"  → sources.py の候補リストに正しい列名を追加してください。\n"
        )
    return out


# ---------------------------------------------------------------------------
# J-Quants
# ---------------------------------------------------------------------------

class JQuants:
    """J-Quants API V2 のラッパー（公式Pythonクライアントを使用）"""

    def __init__(self, api_key: str | None = None):
        try:
            import jquantsapi
        except ImportError as e:
            raise SystemExit(
                "jquants-api-client が入っていません。\n"
                "  pip install jquants-api-client\n"
            ) from e

        api_key = api_key or os.environ.get("JQUANTS_API_KEY")
        if not api_key:
            raise SystemExit(
                "環境変数 JQUANTS_API_KEY が設定されていません。\n"
                "  J-Quantsダッシュボードで発行したAPIキーを設定してください。\n"
            )
        self.cli = jquantsapi.ClientV2(api_key=api_key)

    # -- 生データ（inspect用） --------------------------------------------

    def raw_listed(self) -> pd.DataFrame:
        return self.cli.get_list()

    # -- レート制限を厳守した逐次取得 --------------------------------------
    #
    # 日付を1つ指定すると全銘柄がまとめて返る（1営業日=1リクエスト）。
    # 公式の *_range は内部で並列リクエストするため、レート上限を超えて
    # 429になりやすい。そこで並列は使わず、1営業日ずつ順に呼び、
    # 「1分あたり per_min 件」を守るトークン管理で確実に流量を抑える。

    _covered_start = None   # 契約範囲の開始日（400で判明したら記憶）
    _req_times = None       # 直近のリクエスト時刻（レート管理用）

    def _clamp_start(self, start: dt.datetime) -> dt.datetime:
        if self._covered_start is not None and start < self._covered_start:
            return self._covered_start
        return start

    def _throttle(self, per_min: int):
        """直近60秒のリクエストが per_min 件を超えないよう待機する。"""
        import time as _t
        if self._req_times is None:
            self._req_times = []
        now = _t.monotonic()
        # 60秒より古い記録は捨てる
        self._req_times = [t for t in self._req_times if now - t < 60]
        if len(self._req_times) >= per_min:
            sleep_for = 60 - (now - self._req_times[0]) + 0.5
            if sleep_for > 0:
                _t.sleep(sleep_for)
        self._req_times.append(_t.monotonic())

    def _fetch_one_day(self, single_fetch, ymd, label):
        """1営業日ぶんを取得。一時的なnetwork/429エラーは待って再試行する。

        戻り値: (DataFrame or None, "ok"/"skip"/"fail")
        """
        import time as _t, re as _re
        for attempt in range(5):
            try:
                return single_fetch(ymd), "ok"
            except Exception as e:
                msg = str(e)
                # 契約範囲外の日 → スキップ（正常。頻出するので黙って飛ばす）
                if "400" in msg and "subscription covers" in msg:
                    m = _re.search(r"dates:\s*(\d{4}-\d{2}-\d{2})", msg)
                    if m:
                        self._covered_start = dt.datetime.strptime(
                            m.group(1), "%Y-%m-%d").replace(tzinfo=JST)
                    return None, "skip"
                # レート制限 → 65秒待って再試行
                if "429" in msg or "Too Many" in msg:
                    print(f"    [429] {ymd} で制限。65秒待機して再試行します")
                    _t.sleep(65)
                    continue
                # ネットワーク断（DNS解決失敗・接続断など）→ 待って再試行
                net = any(w in msg for w in (
                    "getaddrinfo", "NameResolution", "Failed to resolve",
                    "Connection", "timed out", "Max retries", "RemoteDisconnected"))
                if net:
                    wait = 15 * (attempt + 1)
                    print(f"    [通信エラー] {ymd}。ネット接続を確認します。"
                          f"{wait}秒待って再試行（{attempt + 1}/5回目）")
                    _t.sleep(wait)
                    continue
                raise   # 想定外のエラーはそのまま投げる
        print(f"    [警告] {ymd} は5回試して取得できませんでした。スキップします")
        return None, "fail"

    def _daily_loop(self, single_fetch, start, end, label,
                    on_batch=None) -> pd.DataFrame:
        """1営業日ずつ取得する。

        on_batch(frames_so_far) が渡されると、一定件数ごとに呼ばれる。
        これを使って「取得しながら途中保存」ができる。通信が途中で切れても
        そこまでの分は保存済みになり、次回は差分から再開できる。
        """
        import config as _cfg
        per_min = max(1, int(getattr(_cfg, "RATE_LIMIT_PER_MIN", 5)))
        start = self._clamp_start(start)

        days = pd.bdate_range(start.date(), end.date())
        total = len(days)
        est = total / per_min
        print(f"  {label}: {total}営業日分を取得（レート {per_min}件/分・想定 約{est:.0f}分）")

        frames, since_save, consecutive_fail = [], 0, 0
        SAVE_EVERY = 40          # 40営業日ごとに途中保存
        MAX_CONSECUTIVE_FAIL = 3  # 3営業日連続で失敗したらネット断とみなし中断
        for k, d in enumerate(days, 1):
            self._throttle(per_min)
            ymd = d.strftime("%Y%m%d")
            df, status = self._fetch_one_day(single_fetch, ymd, label)

            if status == "ok" and df is not None and len(df):
                frames.append(df)
                since_save += 1
                consecutive_fail = 0
            elif status == "fail":
                consecutive_fail += 1
            else:  # skip（契約範囲外）は失敗ではない
                consecutive_fail = 0

            if k % 20 == 0 or k == total:
                print(f"    {k}/{total} 日 … {d.date()}")

            if on_batch is not None and since_save >= SAVE_EVERY:
                on_batch(frames)
                since_save = 0

            # 連続失敗＝ネット接続が落ちている可能性大。ここで打ち切る
            if consecutive_fail >= MAX_CONSECUTIVE_FAIL:
                if on_batch is not None and frames:
                    on_batch(frames)
                got_to = frames and len(frames) > 0
                print(f"\n  ⚠ {MAX_CONSECUTIVE_FAIL}営業日連続で取得に失敗しました。")
                print(f"     ネット接続が切れている可能性があります。")
                if got_to:
                    print(f"     ここまでの分は保存済みです。"
                          f"接続を確認して、もう一度同じコマンドを実行すると"
                          f"続きから取得します。")
                raise SystemExit("  取得を中断しました（ネット接続を確認してください）")

        if on_batch is not None and frames:
            on_batch(frames)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _call_by_date(method):
        """date引数の名前がクライアント版で異なる場合に備えたラッパー。"""
        def _f(date_yyyymmdd):
            try:
                return method(date_yyyymmdd=date_yyyymmdd)
            except TypeError:
                # 引数名が date の版
                return method(date=date_yyyymmdd)
        return _f

    def raw_quotes(self, start, end, on_batch=None) -> pd.DataFrame:
        return self._daily_loop(self._call_by_date(self.cli.get_eq_bars_daily),
                                start, end, "株価日足", on_batch=on_batch)

    def raw_fin(self, start, end, on_batch=None) -> pd.DataFrame:
        return self._daily_loop(self._call_by_date(self.cli.get_fin_summary),
                                start, end, "決算サマリー", on_batch=on_batch)

    def probe_edinet(self):
        """EDINET大株主データの契約有無を疎通確認する。
        get_edinet_major_shareholders を叩いて、取れるか・plan外かを返す。"""
        m = "get_edinet_major_shareholders"
        if not hasattr(self.cli, m):
            return {"status": "no_method",
                    "detail": f"{m} メソッドがライブラリにありません。"}
        fn = getattr(self.cli, m)
        # EDINETは code または date 指定で呼ぶ仕様。引数なしだと空配列が返る。
        # 実在する大型株コードと最近の日付で、実際にデータが返るか試す。
        attempts = [
            ("銘柄コード指定(トヨタ)", lambda: fn(code="72030")),
            ("日付指定", lambda: fn(date_yyyymmdd="20260731")),
            ("銘柄コード指定(ソニー)", lambda: fn(code="67580")),
            ("引数なし", lambda: fn()),
        ]
        last_err = None
        for how, call in attempts:
            try:
                df = call()
                n = len(df) if hasattr(df, "__len__") else "?"
                cols = list(df.columns) if hasattr(df, "columns") else []
                if n and n > 0:   # データが返ったら成功
                    return {"status": "ok", "method": m, "how": how,
                            "rows": n, "columns": cols}
                # 0件でも次の呼び方を試す（引数の組合せ次第で返る場合がある）
                last_err = f"{how}では0件"
            except TypeError:
                continue
            except Exception as e:
                last_err = str(e)
                continue
        # すべて0件・エラーだった場合
        return {"status": "empty_or_error", "method": m, "detail": last_err or "全て0件"}

    def probe_tdnet(self):
        """TDnet適時開示アドオンの契約有無を疎通確認する。
        契約があれば取得できたデータの概要を、なければエラー内容を返す。
        （2026年5月開始の別料金アドオン。標準プランには含まれない）"""
        # jquantsapi の実際のメソッド名（get_td_list = TDnet適時開示一覧）
        candidates = [
            "get_td_list", "get_td_files", "get_td_bulk",
        ]
        found = [m for m in candidates if hasattr(self.cli, m)]
        if not found:
            return {"status": "no_method",
                    "detail": "ライブラリにTDnet取得メソッドが見つかりません。"
                              "jquantsapiを最新版に更新すると追加される可能性があります。",
                    "methods_tried": candidates}
        # get_td_list を最優先で試す。引数なし→日付指定の順で試行。
        import datetime as _dt
        m = "get_td_list" if "get_td_list" in found else found[0]
        fn = getattr(self.cli, m)
        # いくつかの呼び方を試す（ライブラリの引数仕様が不明なため）
        attempts = [
            ("引数なし", lambda: fn()),
            ("日付指定", lambda: fn(date_yyyymmdd="20260807")),
            ("from-to", lambda: fn(from_yyyymmdd="20260801", to_yyyymmdd="20260807")),
        ]
        last_err = None
        for how, call in attempts:
            try:
                df = call()
                n = len(df) if hasattr(df, "__len__") else "?"
                cols = list(df.columns) if hasattr(df, "columns") else []
                return {"status": "ok", "method": m, "how": how,
                        "rows": n, "columns": cols}
            except TypeError:
                continue  # 引数の形が違う→次の呼び方
            except Exception as e:
                last_err = str(e)
                continue
        return {"status": "error", "method": m, "detail": last_err or "不明なエラー"}

    def earnings_calendar(self) -> pd.DataFrame:
        """決算発表予定日。保有期間中に決算を跨ぐかの判定に使う。"""
        try:
            df = self.cli.get_eq_earnings_cal()
        except Exception as e:
            print(f"  [警告] 決算発表日を取得できませんでした: {e}")
            return pd.DataFrame(columns=["code", "date"])
        out = pd.DataFrame()
        for cand in ("Code", "LocalCode", "code"):
            if cand in df.columns:
                out["code"] = df[cand].astype(str); break
        for cand in ("Date", "DiscDate", "AnnouncementDate", "date"):
            if cand in df.columns:
                out["date"] = pd.to_datetime(df[cand], errors="coerce"); break
        if "code" not in out or "date" not in out:
            print(f"  [警告] 決算発表日の列名が想定外です: {list(df.columns)}")
            return pd.DataFrame(columns=["code", "date"])
        return out.dropna()

    # -- 正規化済み --------------------------------------------------------

    def listed(self, use_cache: bool = True) -> pd.DataFrame:
        import cache
        try:
            df = normalize(self.raw_listed(), LISTED_COLUMNS,
                           required=["code", "company_name"], label="銘柄一覧")
            df["code"] = df["code"].astype(str)
            if use_cache:
                cache.save("listed", df)
            return df
        except Exception as e:
            cached = cache.load("listed") if use_cache else None
            if cached is not None:
                print(f"  [警告] 銘柄一覧の取得に失敗。キャッシュを使用します: {e}")
                return cached
            raise

    # Freeプランは約12週間(84日)遅延。今日を指定すると400になるため、
    # 既定で余裕を持たせて手前を終端にする。DATA_LAG_DAYS で調整可能。
    def _end_date(self) -> dt.datetime:
        import config as _cfg
        lag = int(getattr(_cfg, "DATA_LAG_DAYS", 0))
        return dt.datetime.now(JST) - dt.timedelta(days=lag)

    def _normalize_quotes(self, raw) -> pd.DataFrame:
        df = normalize(raw, QUOTE_COLUMNS,
                       required=["code", "date", "close", "volume"], label="株価日足")
        df["code"] = df["code"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        # 分割・併合の調整：AdjustmentFactor（1:2分割なら権利落ち日に0.5）を使う。
        # J-Quants公式仕様：調整済株価 = 生終値 × (その日以降の調整係数の累積積)。
        # 日付降順で累積積を取り、当日を含む係数を掛けると過去株価が現在基準に揃う。
        # 出来高は逆に、累積積で割る（株数ベースで連続させる）。
        if "adj_factor" in df.columns and df["adj_factor"].notna().any():
            df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce").fillna(1.0)
            df = df.sort_values(["code", "date"]).reset_index(drop=True)
            # 各銘柄内で「翌日以降の調整係数の累積積」を作る（groupby.applyを使わない）。
            cum = np.ones(len(df), dtype=float)
            for code, pos in df.groupby("code").groups.items():
                pos = np.array(sorted(pos))          # その銘柄の行位置（日付昇順・連番）
                fac = df["adj_factor"].values[pos]
                rev_cum = np.cumprod(fac[::-1])[::-1]  # 日付降順の累積積→昇順に戻す
                shifted = np.empty_like(rev_cum)
                shifted[:-1] = rev_cum[1:]           # 翌日以降の係数（当日は含めない）
                shifted[-1] = 1.0                    # 最新日は調整なし
                cum[pos] = shifted
            df["close"] = df["close"].values * cum
            if "volume" in df.columns:
                df["volume"] = df["volume"].values / cum
        if "turnover_value" not in df.columns or df["turnover_value"].isna().all():
            df["turnover_value"] = df["close"] * df["volume"]
        return df.dropna(subset=["close"])

    def quotes(self, days: int, use_cache: bool = True) -> pd.DataFrame:
        import cache
        end = self._end_date()
        want_start = end - dt.timedelta(days=int(days * 1.8) + 10)

        cached = cache.load("quotes") if use_cache else None
        last = cache.last_date(cached, "date")

        if cached is not None and last is not None:
            # 差分だけ取得（保存済み最終日の翌日〜今日）
            fetch_start = dt.datetime.combine(
                last + dt.timedelta(days=1), dt.time(), tzinfo=JST)
            if fetch_start.date() > end.date():
                print(f"  株価: キャッシュが最新です（最終日 {last}）。取得スキップ")
                df = cached
            else:
                print(f"  株価: キャッシュあり（最終日 {last}）。差分のみ取得します")
                new = self._normalize_quotes(self.raw_quotes(fetch_start, end))
                df = cache.merge(cached, new, keys=["code", "date"])
                cache.save("quotes", df)
        else:
            print("  株価: キャッシュなし。全期間を取得します（初回のみ時間がかかります）")
            print("         ※途中で通信が切れても、それまでの分は保存され、次回は続きから取得します")

            def _save_progress(frames):
                if not use_cache or not frames:
                    return
                partial = self._normalize_quotes(pd.concat(frames, ignore_index=True))
                merged = cache.merge(cache.load("quotes"), partial, keys=["code", "date"])
                if cache.save("quotes", merged):
                    print(f"      → 途中保存: {len(merged):,}行をキャッシュに書き込み")

            raw = self.raw_quotes(want_start, end,
                                  on_batch=_save_progress if use_cache else None)
            df = self._normalize_quotes(raw)
            if use_cache:
                cache.save("quotes", cache.merge(cache.load("quotes"), df,
                                                 keys=["code", "date"]))

        # 必要な期間だけ返す
        df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(want_start.date())]
        df.attrs["turnover_is_estimated"] = False
        return df.reset_index(drop=True)

    def _normalize_fin(self, raw) -> pd.DataFrame:
        df = normalize(raw, FIN_COLUMNS, required=["code", "disclosed_date"],
                       label="決算サマリー")
        df["code"] = df["code"].astype(str)
        df["disclosed_date"] = pd.to_datetime(df["disclosed_date"], errors="coerce")
        if "period_end" in df.columns:
            df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
        for col in ("net_sales", "operating_profit", "shares_outstanding",
                    "net_sales_nc", "operating_profit_nc",
                    "net_profit", "equity", "bps",
                    "fc_operating_profit", "fc_net_sales",
                    "fc_operating_profit_nc", "fc_net_sales_nc"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "net_sales_nc" in df.columns:
            df["net_sales"] = df["net_sales"].fillna(df["net_sales_nc"])
        if "operating_profit_nc" in df.columns:
            df["operating_profit"] = df["operating_profit"].fillna(df["operating_profit_nc"])
        # 会社予想も連結→単体でフォールバック
        if "fc_operating_profit_nc" in df.columns and "fc_operating_profit" in df.columns:
            df["fc_operating_profit"] = df["fc_operating_profit"].fillna(df["fc_operating_profit_nc"])
        if "fc_net_sales_nc" in df.columns and "fc_net_sales" in df.columns:
            df["fc_net_sales"] = df["fc_net_sales"].fillna(df["fc_net_sales_nc"])
        df = df.drop(columns=[c for c in ("net_sales_nc", "operating_profit_nc",
                                          "fc_operating_profit_nc", "fc_net_sales_nc")
                              if c in df.columns])
        return df.dropna(subset=["disclosed_date"])

    def financials(self, days: int, use_cache: bool = True) -> pd.DataFrame:
        import cache
        end = self._end_date()
        want_start = end - dt.timedelta(days=days)

        cached = cache.load("fin") if use_cache else None
        last = cache.last_date(cached, "disclosed_date")

        if cached is not None and last is not None:
            fetch_start = dt.datetime.combine(
                last + dt.timedelta(days=1), dt.time(), tzinfo=JST)
            if fetch_start.date() > end.date():
                print(f"  決算: キャッシュが最新です（最終開示 {last}）。取得スキップ")
                df = cached
            else:
                print(f"  決算: キャッシュあり（最終開示 {last}）。差分のみ取得します")
                new = self._normalize_fin(self.raw_fin(fetch_start, end))
                df = cache.merge(cached, new, keys=["code", "disclosed_date"])
                cache.save("fin", df)
        else:
            print("  決算: キャッシュなし。全期間を取得します")

            def _save_progress_fin(frames):
                if not use_cache or not frames:
                    return
                partial = self._normalize_fin(pd.concat(frames, ignore_index=True))
                merged = cache.merge(cache.load("fin"), partial,
                                     keys=["code", "disclosed_date"])
                if cache.save("fin", merged):
                    print(f"      → 途中保存: {len(merged):,}行")

            raw = self.raw_fin(want_start, end,
                               on_batch=_save_progress_fin if use_cache else None)
            df = self._normalize_fin(raw)
            if use_cache:
                cache.save("fin", cache.merge(cache.load("fin"), df,
                                              keys=["code", "disclosed_date"]))

        df = df[pd.to_datetime(df["disclosed_date"]) >= pd.Timestamp(want_start.date())]
        return df.reset_index(drop=True)

    # -- ③④ 需給データ（信用残・業種別空売り比率・投資部門別）--------------

    def raw_margin(self, start, end, on_batch=None) -> pd.DataFrame:
        """信用取引週末残高。日付指定で全銘柄がまとまって返る。"""
        return self._daily_loop(self._call_by_date(self.cli.get_mkt_margin_interest),
                                start, end, "信用取引週末残高", on_batch=on_batch)

    def raw_short_ratio(self, start, end, on_batch=None) -> pd.DataFrame:
        """業種別空売り比率。日付指定で全業種がまとまって返る。"""
        return self._daily_loop(self._call_by_date(self.cli.get_mkt_short_ratio),
                                start, end, "業種別空売り比率", on_batch=on_batch)

    @staticmethod
    def _resolve_margin_cols(df):
        """信用残の列名を、実データから頑健に解決する。

        J-Quants V2 の実データ列名（2026年時点で確認済み）:
          LongVol   = 信用買い残（合計）    ShrtVol    = 信用売り残（合計）
          LongStdVol= 制度信用買い残         ShrtStdVol = 制度信用売り残
          LongNegVol= 一般信用買い残         ShrtNegVol = 一般信用売り残
        合計残（LongVol / ShrtVol）を優先して使う。
        略称ゆれ（margin/mg/trade を含む旧V1名など）にも対応する。
        """
        cols = list(df.columns)
        low = {c: c.lower() for c in cols}
        # 信用残らしさのヒント（実データの vol も含める）
        MARGIN_HINT = ("margin", "mg", "trade", "trd", "vol", "残", "balance", "bal")

        def _pick(side_words, exclude_words, prefer_total=False):
            best = None
            for c in cols:
                name = low[c]
                if not any(w in name for w in side_words):
                    continue
                if any(w in name for w in exclude_words):
                    continue
                if not any(h in name for h in MARGIN_HINT):
                    continue
                if not pd.to_numeric(df[c], errors="coerce").notna().any():
                    continue
                # スコア: 合計残（std/negを含まない）を最優先
                is_subtype = any(k in name for k in ("std", "neg", "standard", "negoti", "制度", "一般"))
                score = 0
                if not is_subtype:
                    score += 10          # 合計残を最優先
                score += sum(k in name for k in ("margin", "mg", "vol", "残", "balance"))
                if best is None or score > best[1]:
                    best = (c, score)
            return best[0] if best else None

        long_col = _pick(("long", "buy"), ("short", "shrt", "sell"))
        short_col = _pick(("short", "shrt", "sell"), ("long", "buy"))
        return long_col, short_col

    def _normalize_margin(self, raw):
        """信用残を正規化。code/date は候補方式、買い残/売り残はパターン解決。"""
        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=["code", "date", "long_margin", "short_margin"])
        # code / date
        out = pd.DataFrame(index=raw.index)
        for canonical, cands in (("code", MARGIN_COLUMNS["code"]),
                                 ("date", MARGIN_COLUMNS["date"])):
            for cand in cands:
                if cand in raw.columns:
                    out[canonical] = raw[cand]; break
        if "code" not in out or "date" not in out:
            raise ColumnNotFound(
                f"\n[信用残] code/date が見つかりません。実際の列: {list(raw.columns)}\n")
        long_col, short_col = self._resolve_margin_cols(raw)
        if long_col is None or short_col is None:
            raise ColumnNotFound(
                f"\n[信用残] 買い残/売り残の列を特定できませんでした。\n"
                f"  実際の列: {list(raw.columns)}\n"
                f"  買い残候補={long_col} / 売り残候補={short_col}\n"
                f"  → sources.py の _resolve_margin_cols を調整してください。\n")
        print(f"  信用残の列を解決: 買い残={long_col} / 売り残={short_col}")
        out["long_margin"] = pd.to_numeric(raw[long_col], errors="coerce")
        out["short_margin"] = pd.to_numeric(raw[short_col], errors="coerce")
        out["code"] = out["code"].astype(str)
        out["date"] = pd.to_datetime(out["date"])
        return out

    def margin(self, days: int, use_cache: bool = True) -> pd.DataFrame:
        """信用取引週末残高（銘柄別）。信用買い残・売り残。

        生データ（全列）をキャッシュし、読み出し時に正規化する。
        これにより、実データの列名がV1/V2で違っても取り直し不要で対応できる。
        """
        import cache
        end = self._end_date()
        want_start = end - dt.timedelta(days=days)
        raw_cached = cache.load("margin_raw") if use_cache else None
        date_col_raw = None
        if raw_cached is not None:
            date_col_raw = next((c for c in ("Date", "date") if c in raw_cached.columns), None)
        last = cache.last_date(raw_cached, date_col_raw) if date_col_raw else None
        # 旧キャッシュ（正規化済みで壊れている可能性）があれば無視して raw を使う
        if raw_cached is not None and last is not None:
            fetch_start = dt.datetime.combine(last + dt.timedelta(days=1), dt.time(), tzinfo=JST)
            if fetch_start.date() > end.date():
                print(f"  信用残: キャッシュが最新です（最終 {last}）。取得スキップ")
                raw = raw_cached
            else:
                print(f"  信用残: キャッシュあり（最終 {last}）。差分のみ取得します")
                new = self.raw_margin(fetch_start, end)
                # 生データの重複除去キー（実列名 Code/Date 系を検出）
                code_k = next((c for c in ("Code", "LocalCode", "code") if c in raw_cached.columns), None)
                date_k = next((c for c in ("Date", "date") if c in raw_cached.columns), None)
                dedup = [k for k in (code_k, date_k) if k]
                raw = cache.merge(raw_cached, new, keys=dedup if dedup else list(raw_cached.columns))
                cache.save("margin_raw", raw)
        else:
            print("  信用残: 生データを取得します（全列を保持）")
            raw = self.raw_margin(want_start, end)
            if use_cache and raw is not None and len(raw):
                cache.save("margin_raw", raw)
        df = self._normalize_margin(raw)
        df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(want_start.date())]
        return df.reset_index(drop=True)

    @staticmethod
    def _resolve_short_ratio_cols(df):
        """業種別空売り比率の列を実データから解決する。

        J-Quants V2 の実データ列名（2026年時点で確認済み）:
          S33           = 業種33コード          Date = 日付
          SellExShortVa = 空売りを除く売買代金
          ShrtWithResVa = 価格規制ありの空売り代金
          ShrtNoResVa   = 価格規制なしの空売り代金
        空売り比率 = (規制あり + 規制なし) / (規制あり + 規制なし + 除く分)
        略称ゆれ（Sector33Code / ...Ratio 等）にも対応する。
        """
        cols = list(df.columns)
        low = {c: c.lower() for c in cols}
        def _find_any(*words):
            for c in cols:
                if any(w in low[c] for w in words):
                    return c
            return None
        sector_col = _find_any("s33", "sector33", "sector_33", "sector")
        date_col = _find_any("date")
        # 既に比率列があればそれを使う
        ratio_col = next((c for c in cols if "ratio" in low[c] or "比率" in low[c]), None)
        # 空売り金額（規制あり・なしの2列を合算）。
        # 注意: SellExShortVa は「空売りを"除く"売買代金」なので短売り列ではない。
        #       sell や ex を含む列は分子から除外する。
        short_vals = [c for c in cols
                      if ("short" in low[c] or "shrt" in low[c])
                      and ("value" in low[c] or low[c].endswith("va") or "turnover" in low[c])
                      and "sell" not in low[c] and "ex" not in low[c]]
        # 分母の「空売りを除く売買代金」: sell + ex + short を含む列
        excl_val = next((c for c in cols
                         if ("sell" in low[c]) and ("ex" in low[c])
                         and ("short" in low[c] or "shrt" in low[c])), None)
        if excl_val is None:  # 予備：exclを含む金額列
            excl_val = next((c for c in cols if "excl" in low[c]
                             and ("value" in low[c] or low[c].endswith("va"))), None)
        return sector_col, date_col, ratio_col, short_vals, excl_val

    def _normalize_short_ratio(self, raw):
        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=["sector33", "date", "short_ratio"])
        sector_col, date_col, ratio_col, short_vals, excl_val = \
            self._resolve_short_ratio_cols(raw)
        if sector_col is None or date_col is None:
            raise ColumnNotFound(
                f"\n[空売り比率] sector33/date が見つかりません。実際の列: {list(raw.columns)}\n")
        out = pd.DataFrame(index=raw.index)
        out["sector33"] = raw[sector_col].astype(str)
        out["date"] = pd.to_datetime(raw[date_col])
        if ratio_col is not None:
            print(f"  空売り比率の列を解決: 比率={ratio_col}")
            out["short_ratio"] = pd.to_numeric(raw[ratio_col], errors="coerce")
        elif short_vals and excl_val is not None:
            print(f"  空売り比率を金額から計算: 空売り={short_vals} / 除く分={excl_val}")
            short_sum = sum(pd.to_numeric(raw[c], errors="coerce").fillna(0) for c in short_vals)
            ev = pd.to_numeric(raw[excl_val], errors="coerce").fillna(0)
            denom = short_sum + ev
            ratio = short_sum / denom
            ratio[~(denom > 0)] = pd.NA
            out["short_ratio"] = ratio
        else:
            raise ColumnNotFound(
                f"\n[空売り比率] 比率にできる列がありません。実際の列: {list(raw.columns)}\n"
                f"  空売り金額列={short_vals} / 除く分={excl_val}\n")
        return out

    def short_ratio(self, days: int, use_cache: bool = True) -> pd.DataFrame:
        """業種別空売り比率。生データ（全列）をキャッシュし読み出し時に正規化。"""
        import cache
        end = self._end_date()
        want_start = end - dt.timedelta(days=days)
        raw_cached = cache.load("short_ratio_raw") if use_cache else None
        date_col_raw = None
        if raw_cached is not None:
            date_col_raw = next((c for c in ("Date", "date") if c in raw_cached.columns), None)
        last = cache.last_date(raw_cached, date_col_raw) if date_col_raw else None
        if raw_cached is not None and last is not None:
            fetch_start = dt.datetime.combine(last + dt.timedelta(days=1), dt.time(), tzinfo=JST)
            if fetch_start.date() > end.date():
                print(f"  空売り比率: キャッシュが最新です（最終 {last}）。取得スキップ")
                raw = raw_cached
            else:
                print(f"  空売り比率: キャッシュあり（最終 {last}）。差分のみ取得します")
                new = self.raw_short_ratio(fetch_start, end)
                code_k = next((c for c in ("Sector33Code", "sector_33_code", "sector33") if c in raw_cached.columns), None)
                date_k = next((c for c in ("Date", "date") if c in raw_cached.columns), None)
                dedup = [k for k in (code_k, date_k) if k]
                raw = cache.merge(raw_cached, new, keys=dedup if dedup else list(raw_cached.columns))
                cache.save("short_ratio_raw", raw)
        else:
            print("  空売り比率: 生データを取得します（全列を保持）")
            raw = self.raw_short_ratio(want_start, end)
            if use_cache and raw is not None and len(raw):
                cache.save("short_ratio_raw", raw)
        df = self._normalize_short_ratio(raw)
        df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(want_start.date())]
        return df.reset_index(drop=True)

    def investor_types(self, use_cache: bool = True) -> pd.DataFrame:
        """投資部門別売買（海外勢の動向）。市場全体の週次データ。"""
        import cache
        try:
            raw = self.cli.get_eq_investor_types()
        except Exception as e:
            print(f"  [警告] 投資部門別データを取得できませんでした: {e}")
            cached = cache.load("investor") if use_cache else None
            return cached if cached is not None else pd.DataFrame()
        if use_cache:
            cache.save("investor", raw)
        return raw

EDINET_URL = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
DOCTYPE_LARGE_HOLDING = {"350", "360"}   # 350=大量保有報告書 / 360=変更報告書


def fetch_large_holdings(days: int, api_key: str | None = None,
                         sleep_sec: float = 1.0) -> pd.DataFrame:
    """直近 days 日分の大量保有報告書の提出一覧を取得する。

    機関投資家・ファンドの新規参入シグナル（Role 5）に使います。
    """
    api_key = api_key or os.environ.get("EDINET_API_KEY")
    if not api_key:
        print("  [警告] EDINET_API_KEY が未設定のため、大量保有報告書はスキップします。")
        return pd.DataFrame(columns=["code", "filer_name", "doc_description", "submit_date"])

    rows = []
    today = dt.datetime.now(JST).date()
    for i in range(days):
        day = today - dt.timedelta(days=i)
        if day.weekday() >= 5:          # 土日はスキップ
            continue
        params = {"date": day.isoformat(), "type": 2, "Subscription-Key": api_key}
        try:
            res = requests.get(EDINET_URL, params=params, timeout=30)
            res.raise_for_status()
            results = res.json().get("results") or []
        except Exception as e:
            print(f"  [警告] EDINET {day} の取得に失敗: {e}")
            time.sleep(sleep_sec)
            continue

        for r in results:
            if str(r.get("docTypeCode")) in DOCTYPE_LARGE_HOLDING:
                sec = r.get("secCode")
                if not sec:
                    continue
                rows.append({
                    "code": str(sec)[:4],           # 5桁コードを4桁に寄せる
                    "filer_name": r.get("filerName"),
                    "doc_description": r.get("docDescription"),
                    "submit_date": r.get("submitDateTime"),
                })
        time.sleep(sleep_sec)            # レート制限に配慮

    return pd.DataFrame(rows)


# 主要な書類種別コード（EDINET）
DOC_TYPES = {
    "120": "有価証券報告書", "130": "訂正有価証券報告書",
    "140": "四半期報告書", "160": "半期報告書",
    "180": "臨時報告書", "190": "訂正臨時報告書",
    "200": "自己株券買付状況報告書",
    "220": "親会社等状況報告書",
    "230": "有価証券届出書（組込方式）",
    "030": "有価証券届出書", "040": "訂正有価証券届出書",
    "350": "大量保有報告書", "360": "変更報告書",
}


def fetch_disclosures(days: int, api_key: str | None = None,
                      sleep_sec: float = 1.0) -> pd.DataFrame:
    """直近 days 日分の提出書類一覧を取得する。

    書類名（docDescription）は **EDINETが返した原文のまま** 保持します。
    要約も解釈もしません。利用者が原文を読んで判断するための材料です。
    """
    api_key = api_key or os.environ.get("EDINET_API_KEY")
    if not api_key:
        print("  [警告] EDINET_API_KEY が未設定のため、開示書類はスキップします。")
        return pd.DataFrame(columns=["code", "doc_type", "doc_description",
                                     "filer_name", "submit_date", "doc_id"])

    rows = []
    today = dt.datetime.now(JST).date()
    for i in range(days):
        day = today - dt.timedelta(days=i)
        if day.weekday() >= 5:
            continue
        try:
            res = requests.get(EDINET_URL, timeout=30, params={
                "date": day.isoformat(), "type": 2, "Subscription-Key": api_key})
            res.raise_for_status()
            results = res.json().get("results") or []
        except Exception as e:
            print(f"  [警告] EDINET {day} の取得に失敗: {e}")
            time.sleep(sleep_sec)
            continue

        for r in results:
            sec = r.get("secCode")
            if not sec:
                continue
            dtc = str(r.get("docTypeCode") or "")
            rows.append({
                "code": str(sec)[:4],
                "doc_type": DOC_TYPES.get(dtc, f"その他({dtc})"),
                "doc_description": r.get("docDescription"),   # 原文のまま
                "filer_name": r.get("filerName"),
                "submit_date": (r.get("submitDateTime") or "")[:10],
                "doc_id": r.get("docID"),
            })
        time.sleep(sleep_sec)

    return pd.DataFrame(rows)
