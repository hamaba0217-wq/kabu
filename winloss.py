# -*- coding: utf-8 -*-
"""
過去の推奨（history/picks.csv）を勝ち・負けに分け、
『推奨した時点のチャートの形』の特徴から、勝ちやすい/負けやすい条件を洗い出す。
  py main.py winloss

使う特徴量（すべて推奨日までの情報だけ＝先読みなし）
  ・25日線/5日線からの乖離（トレンドの上か下か）
  ・直近5日/20日リターン（勢い）
  ・52週高値からの位置（大底圏か）
  ・出来高比（当日出来高 ÷ 直近20日平均）
  ・当日の値幅、陽線/陰線、上ヒゲ/下ヒゲ
  ・直前の連続上昇/下落日数、ギャップ、20日ボラティリティ

出力
  (A) 勝ち群 vs 負け群の特徴量 平均比較
  (B) 条件別の勝率（例：25日線の上/下、出来高急増あり/なし 等）
  (C) 勝率の差が大きい条件 ＝ フィルタ候補
事実にもとづく注意：手数料未考慮。勝敗は track_picks（+10%/-5%・終値ベース）準拠。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import track_picks


def _rec_index(g, rec_date):
    for i, t in enumerate(g["date"]):
        if pd.Timestamp(t) >= pd.Timestamp(rec_date):
            return i
    return None


def _features(g, ri):
    """推奨日index ri 時点の特徴量（ri までの情報のみ使用）。"""
    c = g["close"].values; o = g["open"].values if "open" in g else c
    h = g["high"].values if "high" in g else c
    lo = g["low"].values if "low" in g else c
    v = g["volume"].values if "volume" in g else np.full(len(c), np.nan)
    if ri < 25:
        return None
    f = {}
    ma5 = np.mean(c[ri-5:ri+1]); ma25 = np.mean(c[ri-25:ri+1])
    f["対5日線%"] = (c[ri]/ma5 - 1)*100 if ma5 else np.nan
    f["対25日線%"] = (c[ri]/ma25 - 1)*100 if ma25 else np.nan
    f["直近5日%"] = (c[ri]/c[ri-5]-1)*100 if c[ri-5] else np.nan
    f["直近20日%"] = (c[ri]/c[ri-20]-1)*100 if c[ri-20] else np.nan
    hi120 = np.max(h[max(0,ri-120):ri+1])
    f["52週高値から%"] = (c[ri]/hi120 - 1)*100 if hi120 else np.nan
    vma = np.nanmean(v[max(0,ri-20):ri]) if ri>0 else np.nan
    f["出来高比"] = (v[ri]/vma) if (vma and vma==vma and vma>0) else np.nan
    rng = (h[ri]-lo[ri])
    f["当日値幅%"] = (rng/o[ri]*100) if o[ri] else np.nan
    f["陽線"] = 1.0 if c[ri] >= o[ri] else 0.0
    f["上ヒゲ比"] = ((h[ri]-max(o[ri],c[ri]))/rng) if rng>0 else np.nan
    f["下ヒゲ比"] = ((min(o[ri],c[ri])-lo[ri])/rng) if rng>0 else np.nan
    # 連続上昇/下落日数（ri直前まで）
    up=0; dn=0
    for k in range(ri, 0, -1):
        if c[k] > c[k-1]:
            if dn: break
            up+=1
        elif c[k] < c[k-1]:
            if up: break
            dn+=1
        else: break
    f["連続上昇日"]=float(up); f["連続下落日"]=float(dn)
    f["ギャップ%"] = (o[ri]/c[ri-1]-1)*100 if c[ri-1] else np.nan
    rets = np.diff(c[max(0,ri-20):ri+1]) / c[max(0,ri-20):ri]
    f["20日ボラ%"] = (np.nanstd(rets)*100) if len(rets) else np.nan
    return f


def run(quotes, picks=None):
    q = quotes.copy(); q["code"]=q["code"].astype(str); q["date"]=pd.to_datetime(q["date"])
    tracked = track_picks.track(q)
    if tracked is None or tracked.empty:
        print("追跡データがありません。"); return
    tracked["win"] = tracked["結果"].astype(str).str.contains("利確")
    tracked["lose"] = tracked["結果"].astype(str).str.contains("損切り")
    d = tracked[tracked["win"] | tracked["lose"]].copy()
    bycode = {c: g.sort_values("date").reset_index(drop=True) for c, g in q.groupby("code")}

    rows=[]
    for _, r in d.iterrows():
        g = bycode.get(str(r["code"]))
        if g is None:
            g = bycode.get(str(r["code"])[:4])
        if g is None: continue
        ri = _rec_index(g, r["推奨日"])
        if ri is None: continue
        f = _features(g, ri)
        if f is None: continue
        f["win"]=bool(r["win"]); f["業種"]=r.get("業種",""); f["推奨日"]=str(r["推奨日"])
        rows.append(f)
    df = pd.DataFrame(rows)
    n=len(df)
    print(f"分析対象（勝敗確定・特徴量算出できた推奨）: {n}件　勝ち{int(df['win'].sum())}／負け{int((~df['win']).sum())}（勝率{df['win'].mean()*100:.0f}%）\n")
    if n < 20:
        print("件数が少なく傾向は参考程度です。")

    feats = ["対5日線%","対25日線%","直近5日%","直近20日%","52週高値から%","出来高比",
             "当日値幅%","陽線","上ヒゲ比","下ヒゲ比","連続上昇日","連続下落日","ギャップ%","20日ボラ%"]
    print("="*72)
    print("(A) 勝ち群 vs 負け群：推奨時点の特徴 平均")
    print("="*72)
    print(f"  {'特徴量':<12}{'勝ち平均':>10}{'負け平均':>10}{'差':>9}")
    w=df[df.win]; l=df[~df.win]
    diffs=[]
    for f in feats:
        wm=w[f].mean(); lm=l[f].mean()
        diffs.append((f, wm, lm, wm-lm))
        print(f"  {f:<12}{wm:>10.2f}{lm:>10.2f}{wm-lm:>+9.2f}")

    print("\n"+"="*72)
    print("(B) 条件別の勝率（その条件に当てはまる推奨の勝率）")
    print("="*72)
    conds = [
        ("25日線の上(上昇トレンド)", df["対25日線%"]>0),
        ("25日線の下(下落トレンド)", df["対25日線%"]<=0),
        ("5日線の上", df["対5日線%"]>0),
        ("直近20日がプラス(上昇中)", df["直近20日%"]>0),
        ("直近20日がマイナス(下落中)", df["直近20日%"]<=0),
        ("出来高2倍以上", df["出来高比"]>=2),
        ("出来高1.5倍未満", df["出来高比"]<1.5),
        ("当日 陽線", df["陽線"]==1),
        ("当日 陰線", df["陽線"]==0),
        ("大底圏(52週高値-25%以下)", df["52週高値から%"]<=-25),
        ("高値近辺(52週高値-10%以内)", df["52週高値から%"]>=-10),
        ("当日値幅7%以上(荒い)", df["当日値幅%"]>=7),
        ("直前に連続上昇(2日以上)", df["連続上昇日"]>=2),
        ("直前に連続下落(2日以上)", df["連続下落日"]>=2),
        ("ギャップアップ(+2%以上)", df["ギャップ%"]>=2),
    ]
    base=df['win'].mean()*100
    print(f"  （全体勝率 {base:.0f}%）")
    print(f"  {'条件':<28}{'該当数':>6}{'勝率':>7}{'全体比':>8}")
    rankB=[]
    for name,mask in conds:
        sub=df[mask];
        if len(sub)<8:
            print(f"  {name:<28}{len(sub):>6}{'—(少)':>7}"); continue
        wr=sub['win'].mean()*100
        rankB.append((name,len(sub),wr,wr-base))
        print(f"  {name:<28}{len(sub):>6}{wr:>6.0f}%{wr-base:>+7.0f}")

    print("\n"+"="*72)
    print("(C) 勝率を上げそうな条件 / 避けるべき条件（全体比の大きい順）")
    print("="*72)
    rankB.sort(key=lambda x:x[3], reverse=True)
    print("  ◎ 勝率が高い条件:")
    for name,c,wr,d_ in rankB[:4]:
        print(f"     {name}（{c}件・勝率{wr:.0f}%・全体比{d_:+.0f}）")
    print("  × 避けたい条件:")
    for name,c,wr,d_ in rankB[-4:]:
        print(f"     {name}（{c}件・勝率{wr:.0f}%・全体比{d_:+.0f}）")
    print("\n※この傾向を『絞り込みフィルタ』に入れれば勝率改善の可能性。要・対相場超過＆分割での再確認。")

    # (D) 複合条件（勝ち条件を組合せ）の勝率＋前半後半の再現性
    print("\n"+"="*72)
    print("(D) 勝ち条件の組合せ：勝率・平均R(+10/-5換算)・前半後半")
    print("="*72)
    df2 = df.copy()
    df2["R"] = np.where(df2["win"], 10.0, -5.0)
    # 推奨日順で前半後半（_features には推奨日が無いので rows 構築時の順＝picks順≒日付順で近似）
    half = len(df2)//2
    combos = [
        ("出来高<1.5", df2["出来高比"]<1.5),
        ("25日線上 & 出来高<1.5", (df2["対25日線%"]>0)&(df2["出来高比"]<1.5)),
        ("25日線上 & 深い底でない(>-25%) & 出来高<1.5",
             (df2["対25日線%"]>0)&(df2["52週高値から%"]>-25)&(df2["出来高比"]<1.5)),
        ("25日線上 & 値幅<7% & 出来高<1.5",
             (df2["対25日線%"]>0)&(df2["当日値幅%"]<7)&(df2["出来高比"]<1.5)),
        ("深い底でない & 荒くない(値幅<7%) & 出来高<1.5",
             (df2["52週高値から%"]>-25)&(df2["当日値幅%"]<7)&(df2["出来高比"]<1.5)),
    ]
    print(f"  {'組合せ':<40}{'件数':>5}{'勝率':>7}{'平均R':>7}{'前半勝':>7}{'後半勝':>7}")
    for name, mask in combos:
        sub=df2[mask]
        if len(sub)<8:
            print(f"  {name:<40}{len(sub):>5}{'—(少)':>7}"); continue
        wr=sub["win"].mean()*100; r=sub["R"].mean()
        idx=sub.index
        fh=sub.iloc[:max(1,len(sub)//2)]["win"].mean()*100
        sh=sub.iloc[len(sub)//2:]["win"].mean()*100
        print(f"  {name:<40}{len(sub):>5}{wr:>6.0f}%{r:>+7.1f}{fh:>6.0f}%{sh:>6.0f}%")
    print("\n  ※平均R＝+10%/-5%換算の1件あたり期待値（プラスなら有望）。前半後半の勝率が揃えば再現性あり。")

    # (E) マクロ（市場動向・ドル円・米長期金利）別の勝率
    print("\n"+"="*72)
    print("(E) マクロ環境別の勝率（推奨日のマクロが追い風か逆風か）")
    print("="*72)
    dmac = df.copy()
    dmac["d"] = pd.to_datetime(dmac["推奨日"])

    # 市場トレンド：全銘柄の日次平均リターンの5日移動平均（quotesから算出・外部不要）
    qq = q.sort_values(["code","date"]).copy()
    qq["ret"] = qq.groupby("code")["close"].pct_change()
    mret = qq.groupby("date")["ret"].mean().sort_index()
    mma5 = mret.rolling(5, min_periods=2).mean()
    dmac["市場5日"] = dmac["d"].map(mma5.to_dict())

    # ドル円・米10年（yfinance）。取れなければスキップ
    usdjpy5 = us10y5 = None
    try:
        import yfinance as yf
        lo = (dmac["d"].min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        hi = (dmac["d"].max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        fx = yf.download("JPY=X", start=lo, end=hi, progress=False)["Close"]
        fx = fx.iloc[:,0] if hasattr(fx,"columns") else fx
        fx5 = fx.pct_change(5)
        usdjpy5 = {pd.Timestamp(k).normalize(): v for k, v in fx5.items()}
        tnx = yf.download("^TNX", start=lo, end=hi, progress=False)["Close"]
        tnx = tnx.iloc[:,0] if hasattr(tnx,"columns") else tnx
        tnx5 = tnx.diff(5)
        us10y5 = {pd.Timestamp(k).normalize(): v for k, v in tnx5.items()}
        print("  （ドル円=JPY=X、米10年=^TNX をyfinanceから取得）")
    except Exception as e:
        print(f"  ※ドル円/米金利はyfinanceから取得できずスキップ（{type(e).__name__}）。市場トレンドのみ表示。")

    def _asof(mp, d):
        if not mp:
            return np.nan
        ks = [k for k in mp if k <= d]
        return mp[max(ks)] if ks else np.nan

    dmac["ドル円5日"] = dmac["d"].map(lambda x: _asof(usdjpy5, x)) if usdjpy5 else np.nan
    dmac["米10年5日"] = dmac["d"].map(lambda x: _asof(us10y5, x)) if us10y5 else np.nan

    base = dmac["win"].mean()*100
    conds = [("市場が上向き(5日平均≥0)", dmac["市場5日"]>=0),
             ("市場が下向き(5日平均<0)", dmac["市場5日"]<0)]
    if usdjpy5:
        conds += [("円安方向(ドル円5日↑)", dmac["ドル円5日"]>0),
                  ("円高方向(ドル円5日↓)", dmac["ドル円5日"]<=0)]
    if us10y5:
        conds += [("米金利 上昇(5日↑)", dmac["米10年5日"]>0),
                  ("米金利 低下(5日↓)", dmac["米10年5日"]<=0)]
    print(f"  （全体勝率 {base:.0f}%）")
    print(f"  {'マクロ条件':<24}{'該当数':>6}{'勝率':>7}{'全体比':>8}")
    for name, mask in conds:
        sub = dmac[mask.fillna(False)]
        if len(sub) < 8:
            print(f"  {name:<24}{len(sub):>6}{'—(少)':>7}"); continue
        wr = sub["win"].mean()*100
        print(f"  {name:<24}{len(sub):>6}{wr:>6.0f}%{wr-base:>+7.0f}")
    print("\n  ※サンプルは約2か月分。マクロで細かく切ると件数が減り、統計的な確度は低め（傾向の参考）。")


