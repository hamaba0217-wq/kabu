
import limitup, limitup_analyze, cache
import pandas as pd
import numpy as np

quotes = cache.load("quotes")
fin = cache.load("fin")
listed = cache.load("listed")
if quotes is not None and len(quotes):
    st = limitup.detect_limit_up(quotes)
    reason_df = limitup_analyze.classify_reason(quotes, st, fin=fin, listed=listed)
    reason_df["event_id"] = reason_df["code"].astype(str) + "_" + pd.to_datetime(reason_df["date"]).dt.strftime('%Y%m%d')
    reason_map = dict(zip(reason_df["event_id"], reason_df["reason_type"]))

    quotes_sorted = quotes.sort_values(["code", "date"]).copy()
    grouped = {code: group.reset_index(drop=True) for code, group in quotes_sorted.groupby("code")}

    records = []
    for _, row in st.iterrows():
        code = row["code"]
        st_date = pd.to_datetime(row["date"])
        ev_id = f"{code}_{st_date.strftime('%Y%m%d')}"
        reason = reason_map.get(ev_id, "説明不能")
        if reason not in ["仕手的特徴", "説明不能"]:
            continue
        if code not in grouped:
            continue
        cqi = grouped[code]
        m = cqi.index[cqi["date"] == row["date"]].tolist()
        if not m or m[0] + 1 >= len(cqi):
            continue
        b = cqi.iloc[m[0] + 1]
        o, l, c = b["open"], b["low"], b["close"]
        if o <= 0:
            continue
        for dip in [3, 5, 7]:
            buy_px = o * (1.0 - dip / 100.0)
            if l <= buy_px:
                ret_close = (c / buy_px - 1.0) * 100.0
                records.append({
                    "date": st_date,
                    "dip": dip,
                    "ret_close": ret_close,
                    "is_win": 1 if ret_close > 0 else 0
                })

    df_res = pd.DataFrame(records)
    print(df_res.groupby("dip").agg(
        件数=("ret_close", "count"),
        平均終値R=("ret_close", "mean"),
        中央値終値R=("ret_close", "median"),
        終値勝率=("is_win", "mean")
    ))
else:
    print("No cached data in sandbox")
