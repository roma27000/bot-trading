# ===== alerts.py — sentinelle H4 via ntfy.sh (sans compte ni bot) =====
import requests
import yfinance as yf
import pandas as pd

NTFY_TOPIC = "roma-moula-k7p2x9"   # ton canal privé — ne le partage pas

ACTIFS = [("BTC-USD", "Bitcoin"), ("ETH-USD", "Ethereum"), ("SOL-USD", "Solana"),
          ("GC=F", "Or"), ("SI=F", "Argent"), ("^GSPC", "S&P 500"), ("^NDX", "Nasdaq 100")]

def indicateurs(df):
    df = df.copy()
    c, h, l = df["Close"], df["High"], df["Low"]
    df["EMA200"] = c.ewm(span=200, adjust=False).mean()
    hh9, ll9 = h.rolling(9).max(), l.rolling(9).min()
    hh26, ll26 = h.rolling(26).max(), l.rolling(26).min()
    hh52, ll52 = h.rolling(52).max(), l.rolling(52).min()
    df["Tenkan"] = (hh9 + ll9) / 2
    df["Kijun"] = (hh26 + ll26) / 2
    sa = ((df["Tenkan"] + df["Kijun"]) / 2).shift(26)
    sb = ((hh52 + ll52) / 2).shift(26)
    df["CloudTop"] = pd.concat([sa, sb], axis=1).max(axis=1)
    df["CloudBot"] = pd.concat([sa, sb], axis=1).min(axis=1)
    df["HH20"] = h.rolling(20).max()
    df["LL20"] = l.rolling(20).min()
    df["ATR"] = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
                          axis=1).max(axis=1).ewm(alpha=1/14, adjust=False).mean()
    return df

def bc(r):
    if r["Close"] > r["CloudTop"]: return 1
    if r["Close"] < r["CloudBot"]: return -1
    return 0

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
        h4 = indicateurs(h1.resample("4h").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna())
        d1 = indicateurs(t.history(interval="1d", period="2y"))
        w1 = indicateurs(t.history(interval="1wk", period="5y"))
        score = 2 * bc(w1.iloc[-1]) + 2 * bc(d1.iloc[-1]) + bc(h4.iloc[-1])
        if score >= 3: direction = "LONG"
        elif score >= 1: direction = "LONG SWING"
        elif score <= -3: direction = "SHORT"
        elif score <= -1: direction = "SHORT SWING"
        else: continue

        last = h4.iloc[-1]
        if "LONG" in direction:
            entree = h4["HH20"].shift(1).iloc[-1]
            casse = last["Close"] > entree
            txt = f"clôture H4 AU-DESSUS de {entree:.2f}"
        else:
            entree = h4["LL20"].shift(1).iloc[-1]
            casse = last["Close"] < entree
            txt = f"clôture H4 EN DESSOUS de {entree:.2f}"

        ts = last.name
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        age_h = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600

        if casse and age_h <= 5:
            stop = entree - 2 * last["ATR"] if "LONG" in direction else entree + 2 * last["ATR"]
            send(f"{nom} : {direction} confirmé ({txt}). Stop : {stop:.2f}. Vérifie le rapport avant décision.")
        else:
            print(f"{nom} : pas de nouvelle cassure (clôture {last['Close']:.2f} vs niveau {entree:.2f})")
    except Exception as e:
        print(f"{nom} : erreur -> {e}")
