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


QUOTE_COLUMNS = {
    "code":           ["Code", "LocalCode", "code"],
    "date":           ["Date", "date"],
    "open":           ["O", "Open", "open"],
    "high":           ["H", "High", "high"],
    "low":            ["L", "Low", "low"],
    "close":          ["C", "Close", "close"],
    "adj_factor":     ["AdjFactor", "AdjustmentFactor", "adj_factor"],
    "volume":         ["Vo", "Volume", "volume"],
    "turnover_value": ["Va", "TurnoverValue", "Turnover", "turnover_value"],
}

FIN_COLUMNS = {
    "code":              ["Code", "LocalCode", "code"],
    "disclosed_date":    ["DiscDate", "DisclosedDate", "disclosed_date"],
    "period_type":       ["CurPerType", "Period", "TypeOfCurrentPeriod", "period_type"],
    "period_end":        ["CurPerEn", "PeriodEnd", "CurrentPeriodEndDate", "period_end"],
    "net_sales":         ["Sales", "NetSales", "Revenue", "net_sales"],
    "operating_profit":  ["OP", "OperatingProfit", "OperatingIncome", "operating_profit"],
    "net_sales_nc":      ["NCSales"],
    "operating_profit_nc": ["NCOP"],
    "shares_outstanding": ["ShOutFY", "Shares", "SharesOutstanding", "shares_outstanding"],
    "net_profit":        ["Profit", "NetProfit", "ProfitAttributableToOwnersOfParent", "net_profit"],
    "equity":            ["Equity", "NetAssets", "equity"],
    "bps":               ["BookValuePerShare", "BPS", "bps"],
    "fc_operating_profit":    ["FOP"],
    "fc_net_sales":           ["FSales"],
    "fc_operating_profit_nc": ["FNCOP"],
    "fc_net_sales_nc":        ["FNCSales"],
}

MARGIN_COLUMNS = {
    "code":            ["Code", "LocalCode", "code"],
    "date":            ["Date", "date"],
    "long_margin":     ["LongMarginTradeVolume", "long_margin"],
    "short_margin":    ["ShortMarginTradeVolume", "short_margin"],
}

SHORT_RATIO_COLUMNS = {
    "sector33":        ["Sector33Code", "sector_33_code", "sector33"],
    "date":            ["Date", "date"],
    "short_ratio":     ["ShortSellingWithRestrictionsRatio",
                        "ShortSellingWithoutRestrictionsRatio", "short_ratio"],
}

LISTED_COLUMNS = {
    "code":         ["Code", "LocalCode", "code"],
    "company_name": ["CoName", "CompanyName", "Name", "company_name"],
    "market":       ["MktNm", "MarketName", "MarketCodeName", "Market", "market"],
    "sector33":     ["S33Nm", "Sector33Name", "Sector33CodeName", "Sector33", "sector33"],
    "sector33_code": ["S33", "Sector33Code", "sector33_code"],
}


def code4(code: pd.Series) -> pd.Series:
    s = code.astype(str).str.strip()
    return s.where(s.str.len() != 5, s.str[:4])


class ColumnNotFound(Exception):
    pass


def normalize(df: pd.DataFrame, mapping: dict, required: Iterable[str],
              label: str) -> pd.DataFrame:
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
        )
    return out


class JQuants:
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
            )
        self.cli = jquantsapi.ClientV2(api_key=api_key)

    def raw_listed(self) -> pd.DataFrame:
        return self.cli.get_list()

    _covered_start = None
    _req_times = None

    def _clamp_start(self, start: dt.datetime) -> dt.datetime:
        if self._covered_start is not None and start < self._covered_start:
            return self._covered_start
        return start

    def _throttle(self, per_min: int):
        import time as _t
        if self._req_times is None:
            self._req_times = []
        now = _t.monotonic()
        self._req_times = [t for t in self._req_times if now - t < 60]
        if len(self._req_times) >= per_min:
            sleep_for = 60 - (now - self._req_times[0]) + 0.5
            if sleep_for > 0:
                _t.sleep(sleep_for)
        self._req_times.append(_t.monotonic())

    def _fetch_one_day(self, single_fetch, ymd, label):
        import time as _t, re as _re
        for attempt in range(5):
            try:
                return single_fetch(ymd), "ok"
            except Exception as e:
                msg = str(e)
                if "400" in msg and "subscription covers" in msg:
                    m = _re.search(r"dates:\s*(\d{4}-\d{2}-\d{2})", msg)
                    if m:
                        self._covered_start = dt.datetime.strptime(
                            m.group(1), "%Y-%m-%d").replace(tzinfo=JST)
                    return None, "skip"
                if "429" in msg or "Too Many" in msg:
                    _t.sleep(65)
                    continue
                net = any(w in msg for w in (
                    "getaddrinfo", "NameResolution", "Failed to resolve",
                    "Connection", "timed out", "Max retries", "RemoteDisconnected"))
                if net:
                    _t.sleep(15 * (attempt + 1))
                    continue
                raise
        return None, "fail"

    def _daily_loop(self, single_fetch, start, end, label,
                    on_batch=None) -> pd.DataFrame:
        import config as _cfg
        per_min = max(1, int(getattr(_cfg, "RATE_LIMIT_PER_MIN", 5)))
        start = self._clamp_start(start)

        days = pd.bdate_range(start.date(), end.date())
        total = len(days)
        
        # ★ 0営業日の場合はAPI呼び出しやスリープを一切行わず即座に空DataFrameを返す
        if total == 0:
            print(f"  {label}: 0営業日分のためスキップします")
            return pd.DataFrame()

        est = total / per_min
        print(f"  {label}: {total}営業日分を取得（レート {per_min}件/分・想定 約{est:.0f}分）")

        frames, since_save, consecutive_fail = [], 0, 0
        SAVE_EVERY = 40
        MAX_CONSECUTIVE_FAIL = 3
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
            else:
                consecutive_fail = 0

            if k % 20 == 0 or k == total:
                print(f"    {k}/{total} 日 … {d.date()}")

            if on_batch is not None and since_save >= SAVE_EVERY:
                on_batch(frames)
                since_save = 0

            if consecutive_fail >= MAX_CONSECUTIVE_FAIL:
                if on_batch is not None and frames:
                    on_batch(frames)
                raise SystemExit("  取得を中断しました（ネット接続を確認してください）")

        if on_batch is not None and frames:
            on_batch(frames)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _call_by_date(method):
        def _f(date_yyyymmdd):
            try:
                return method(date_yyyymmdd=date_yyyymmdd)
            except TypeError:
                return method(date=date_yyyymmdd)
        return _f

    def raw_quotes(self, start, end, on_batch=None) -> pd.DataFrame:
        return self._daily_loop(self._call_by_date(self.cli.get_eq_bars_daily),
                                start, end, "株価日足", on_batch=on_batch)

    def raw_fin(self, start, end, on_batch=None) -> pd.DataFrame:
        return self._daily_loop(self._call_by_date(self.cli.get_fin_summary),
                                start, end, "決算サマリー", on_batch=on_batch)

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
                return cached
            raise

    def _end_date(self) -> dt.datetime:
        import config as _cfg
        lag = int(getattr(_cfg, "DATA_LAG_DAYS", 0))
        return dt.datetime.now(JST) - dt.timedelta(days=lag)

    def _normalize_quotes(self, raw) -> pd.DataFrame:
        df = normalize(raw, QUOTE_COLUMNS,
                       required=["code", "date", "close", "volume"], label="株価日足")
        df["code"] = df["code"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        if "adj_factor" in df.columns and df["adj_factor"].notna().any():
            df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce").fillna(1.0)
            df = df.sort_values(["code", "date"]).reset_index(drop=True)
            cum = np.ones(len(df), dtype=float)
            for code, pos in df.groupby("code").groups.items():
                pos = np.array(sorted(pos))
                fac = df["adj_factor"].values[pos]
                rev_cum = np.cumprod(fac[::-1])[::-1]
                shifted = np.empty_like(rev_cum)
                shifted[:-1] = rev_cum[1:]
                shifted[-1] = 1.0
                cum[pos] = shifted
            for pcol in ("open", "high", "low", "close"):
                if pcol in df.columns:
                    df[pcol] = df[pcol].values * cum
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
            raw = self.raw_quotes(want_start, end)
            df = self._normalize_quotes(raw)
            if use_cache:
                cache.save("quotes", cache.merge(cache.load("quotes"), df,
                                                 keys=["code", "date"]))

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
            raw = self.raw_fin(want_start, end)
            df = self._normalize_fin(raw)
            if use_cache:
                cache.save("fin", cache.merge(cache.load("fin"), df,
                                              keys=["code", "disclosed_date"]))

        df = df[pd.to_datetime(df["disclosed_date"]) >= pd.Timestamp(want_start.date())]
        return df.reset_index(drop=True)

    def raw_margin(self, start, end, on_batch=None) -> pd.DataFrame:
        return self._daily_loop(self._call_by_date(self.cli.get_mkt_margin_interest),
                                start, end, "信用取引週末残高", on_batch=on_batch)

    def raw_short_ratio(self, start, end, on_batch=None) -> pd.DataFrame:
        return self._daily_loop(self._call_by_date(self.cli.get_mkt_short_ratio),
                                start, end, "業種別空売り比率", on_batch=on_batch)

    @staticmethod
    def _resolve_margin_cols(df):
        cols = list(df.columns)
        low = {c: c.lower() for c in cols}
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
                is_subtype = any(k in name for k in ("std", "neg", "standard", "negoti", "制度", "一般"))
                score = 0
                if not is_subtype:
                    score += 10
                score += sum(k in name for k in ("margin", "mg", "vol", "残", "balance"))
                if best is None or score > best[1]:
                    best = (c, score)
            return best[0] if best else None

        long_col = _pick(("long", "buy"), ("short", "shrt", "sell"))
        short_col = _pick(("short", "shrt", "sell"), ("long", "buy"))
        return long_col, short_col

    def _normalize_margin(self, raw):
        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=["code", "date", "long_margin", "short_margin"])
        out = pd.DataFrame(index=raw.index)
        for canonical, cands in (("code", MARGIN_COLUMNS["code"]),
                                 ("date", MARGIN_COLUMNS["date"])):
            for cand in cands:
                if cand in raw.columns:
                    out[canonical] = raw[cand]; break
        if "code" not in out or "date" not in out:
            raise ColumnNotFound("code/date が見つかりません")
        long_col, short_col = self._resolve_margin_cols(raw)
        if long_col is None or short_col is None:
            raise ColumnNotFound("信用残の列を特定できませんでした")
        out["long_margin"] = pd.to_numeric(raw[long_col], errors="coerce")
        out["short_margin"] = pd.to_numeric(raw[short_col], errors="coerce")
        out["code"] = out["code"].astype(str)
        out["date"] = pd.to_datetime(out["date"])
        return out

    def margin(self, days: int, use_cache: bool = True) -> pd.DataFrame:
        import cache
        end = self._end_date()
        want_start = end - dt.timedelta(days=days)
        raw_cached = cache.load("margin_raw") if use_cache else None
        date_col_raw = next((c for c in ("Date", "date") if c in raw_cached.columns), None) if raw_cached is not None else None
        last = cache.last_date(raw_cached, date_col_raw) if date_col_raw else None
        if raw_cached is not None and last is not None:
            fetch_start = dt.datetime.combine(last + dt.timedelta(days=1), dt.time(), tzinfo=JST)
            if fetch_start.date() > end.date():
                raw = raw_cached
            else:
                new = self.raw_margin(fetch_start, end)
                code_k = next((c for c in ("Code", "LocalCode", "code") if c in raw_cached.columns), None)
                date_k = next((c for c in ("Date", "date") if c in raw_cached.columns), None)
                dedup = [k for k in (code_k, date_k) if k]
                raw = cache.merge(raw_cached, new, keys=dedup if dedup else list(raw_cached.columns))
                cache.save("margin_raw", raw)
        else:
            raw = self.raw_margin(want_start, end)
            if use_cache and raw is not None and len(raw):
                cache.save("margin_raw", raw)
        df = self._normalize_margin(raw)
        df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(want_start.date())]
        return df.reset_index(drop=True)

    @staticmethod
    def _resolve_short_ratio_cols(df):
        cols = list(df.columns)
        low = {c: c.lower() for c in cols}
        def _find_any(*words):
            for c in cols:
                if any(w in low[c] for w in words):
                    return c
            return None
        sector_col = _find_any("s33", "sector33", "sector_33", "sector")
        date_col = _find_any("date")
        ratio_col = next((c for c in cols if "ratio" in low[c] or "比率" in low[c]), None)
        short_vals = [c for c in cols
                      if ("short" in low[c] or "shrt" in low[c])
                      and ("value" in low[c] or low[c].endswith("va") or "turnover" in low[c])
                      and "sell" not in low[c] and "ex" not in low[c]]
        excl_val = next((c for c in cols
                         if ("sell" in low[c]) and ("ex" in low[c])
                         and ("short" in low[c] or "shrt" in low[c])), None)
        if excl_val is None:
            excl_val = next((c for c in cols if "excl" in low[c]
                             and ("value" in low[c] or c.endswith("va"))), None)
        return sector_col, date_col, ratio_col, short_vals, excl_val

    def _normalize_short_ratio(self, raw):
        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=["sector33", "date", "short_ratio"])
        sector_col, date_col, ratio_col, short_vals, excl_val = \
            self._resolve_short_ratio_cols(raw)
        if sector_col is None or date_col is None:
            raise ColumnNotFound("sector33/date が見つかりません")
        out = pd.DataFrame(index=raw.index)
        out["sector33"] = raw[sector_col].astype(str)
        out["date"] = pd.to_datetime(raw[date_col])
        if ratio_col is not None:
            out["short_ratio"] = pd.to_numeric(raw[ratio_col], errors="coerce")
        elif short_vals and excl_val is not None:
            short_sum = sum(pd.to_numeric(raw[c], errors="coerce").fillna(0) for c in short_vals)
            ev = pd.to_numeric(raw[excl_val], errors="coerce").fillna(0)
            denom = short_sum + ev
            ratio = short_sum / denom
            ratio[~(denom > 0)] = pd.NA
            out["short_ratio"] = ratio
        else:
            raise ColumnNotFound("空売り比率の計算列がありません")
        return out

    def short_ratio(self, days: int, use_cache: bool = True) -> pd.DataFrame:
        import cache
        end = self._end_date()
        want_start = end - dt.timedelta(days=days)
        raw_cached = cache.load("short_ratio_raw") if use_cache else None
        date_col_raw = next((c for c in ("Date", "date") if c in raw_cached.columns), None) if raw_cached is not None else None
        last = cache.last_date(raw_cached, date_col_raw) if date_col_raw else None
        if raw_cached is not None and last is not None:
            fetch_start = dt.datetime.combine(last + dt.timedelta(days=1), dt.time(), tzinfo=JST)
            if fetch_start.date() > end.date():
                raw = raw_cached
            else:
                new = self.raw_short_ratio(fetch_start, end)
                code_k = next((c for c in ("Sector33Code", "sector_33_code", "sector33") if c in raw_cached.columns), None)
                date_k = next((c for c in ("Date", "date") if c in raw_cached.columns), None)
                dedup = [k for k in (code_k, date_k) if k]
                raw = cache.merge(raw_cached, new, keys=dedup if dedup else list(raw_cached.columns))
                cache.save("short_ratio_raw", raw)
        else:
            raw = self.raw_short_ratio(want_start, end)
            if use_cache and raw is not None and len(raw):
                cache.save("short_ratio_raw", raw)
        df = self._normalize_short_ratio(raw)
        df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(want_start.date())]
        return df.reset_index(drop=True)
