# ===== alerts.py v3 — sentinelle H4 (Dow + cassure) via ntfy.sh =====
import requests
import yfinance as yf
import pandas as pd

NTFY_TOPIC = "roma-moula-k7p2x9"   # ton canal privé — ne pas partager

def contrats_actifs():
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    COMEX_MONTHS = [("G",2),("J",4),("M",6),("Q",8),("V",10),("Z",12)]
    CME_MONTHS = [("H",3),("M",6),("U",9),("Z",12)]
    def roll_cutoff(year, month):
        first_day = pd.Timestamp(year=year, month=month, day=1)
        prev_month_end = first_day - pd.Timedelta(days=1)
        last_bday = pd.bdate_range(end=prev_month_end, periods=1)[0]
        return last_bday - pd.offsets.BDay(5)
    def candidates(root, months, suffix):
        out = []
        for year in [today.year, today.year+1]:
            for code, month in months:
                if today < roll_cutoff(year, month):
                    out.append(f"{root}{code}{str(year)[-2:]}{suffix}")
        return out[:6]
    def choose(root, months, suffix, fallback):
        best = None
        for tk in candidates(root, months, suffix):
            try:
                hist = yf.Ticker(tk).history(interval="1d", period="10d").dropna()
                if hist.empty: continue
                last_date = hist.index[-1]
                if getattr(last_date, "tzinfo", None) is not None:
                    last_date = last_date.tz_localize(None)
                if (today - last_date.normalize()).days > 5: continue
                vol = float(hist["Volume"].tail(5).mean()) if "Volume" in hist.columns and hist["Volume"].dropna().shape[0] else 0
                if best is None or vol > best[0]:
                    best = (vol, tk)
            except Exception:
                continue
        return best[1] if best else fallback
    gc = choose("GC", COMEX_MONTHS, ".CMX", "GC=F")
    si = choose("SI", COMEX_MONTHS, ".CMX", "SI=F")
    es = choose("ES", CME_MONTHS, ".CME", "ES=F")
    nq = choose("NQ", CME_MONTHS, ".CME", "NQ=F")
    return gc, si, es, nq

GC_SYM, SI_SYM, ES_SYM, NQ_SYM = contrats_actifs()
ACTIFS = [("BTC-USD", "Bitcoin"), ("ETH-USD", "Ethereum"), ("SOL-USD", "Solana"),
          (GC_SYM, "Or"), (SI_SYM, "Argent"), (ES_SYM, "S&P 500"), (NQ_SYM, "Nasdaq 100")]

def resample_4h(h1):
    return h1.resample("4h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()

def tendance_dow(df, k=5):
    h, l = df["High"].values, df["Low"].values
    n = len(df); ev = []
    for i in range(k, n-k):
        if h[i] >= h[i-k:i].max() and h[i] >= h[i+1:i+k+1].max(): ev.append((i+k, "H", h[i]))
        if l[i] <= l[i-k:i].min() and l[i] <= l[i+1:i+k+1].min(): ev.append((i+k, "L", l[i]))
    ev.sort(key=lambda x: x[0])
    trend = {}; sh = []; sl = []; e = 0
    idx = df.index
    for j in range(n):
        while e < len(ev) and ev[e][0] <= j:
            (sh if ev[e][1] == "H" else sl).append(ev[e][2]); e += 1
        t = 0
        if len(sh) >= 2 and len(sl) >= 2:
            hh, hl = sh[-1] > sh[-2], sl[-1] > sl[-2]
            lh, ll = sh[-1] < sh[-2], sl[-1] < sl[-2]
            if hh and hl: t = 1
            elif lh and ll: t = -1
        trend[idx[j]] = t
    return trend

def send(msg):
    try:
        r = requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=msg.encode("utf-8"),
                          headers={"Title": "Alerte trading", "Tags": "chart"}, timeout=10)
        print("Envoyé :", msg, "->", r.status_code)
    except Exception as e:
        print("Erreur envoi :", e)

for sym, nom in ACTIFS:
    try:
        t = yf.Ticker(sym)
        h1 = t.history(interval="1h", period="60d")
        h4 = resample_4h(h1)
        d1 = t.history(interval="1d", period="5y")
        if len(h4) < 210 or len(d1) < 210:
            continue
        H4 = h4.copy()
        H4["HH20"] = H4["High"].rolling(20).max()
        H4["LL20"] = H4["Low"].rolling(20).min()
        tr = pd.concat([H4["High"]-H4["Low"], (H4["High"]-H4["Close"].shift(1)).abs(),
                        (H4["Low"]-H4["Close"].shift(1)).abs()], axis=1).max(axis=1)
        H4["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

        td = tendance_dow(d1, k=5).get(d1.index[-1], 0)
        i = len(H4) - 1
        last = H4.iloc[-1]
        prev = H4.iloc[-2]
        msg = None
        if td == 1:
            lvl = H4["HH20"].shift(1).iloc[i]
            if last["Close"] > lvl and prev["Close"] <= lvl:
                stop = lvl - 2*last["ATR"]
                msg = f"{nom} : LONG confirmé (cassure H4 au-dessus de {lvl:.2f}). Stop : {stop:.2f}. Vérifie le rapport avant décision."
        elif td == -1:
            lvl = H4["LL20"].shift(1).iloc[i]
            if last["Close"] < lvl and prev["Close"] >= lvl:
                stop = lvl + 2*last["ATR"]
                msg = f"{nom} : SHORT confirmé (cassure H4 en dessous de {lvl:.2f}). Stop : {stop:.2f}. Vérifie le rapport avant décision."

        ts = last.name
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        age_h = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600
        if msg and age_h <= 5:
            send(msg)
        else:
            print(f"{nom} : pas de nouvelle cassure (clôture {last['Close']:.2f})")
    except Exception as e:
        print(f"{nom} : erreur -> {e}")
