# Options Deribit : IV, skew, max pain, volume + GEX estimé
# v5 : transport "navigateur" (curl_cffi) puis requests + prix via underlying_price
import math
from datetime import datetime
import streamlit as st

try:
    from curl_cffi import requests as _cr
    _CURL = True
except Exception:
    _CURL = False
import requests as _rq

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "*/*"}
MAPPING = {"BTC-USD": "BTC", "ETH-USD": "ETH"}

def _get(url, params=None, timeout=20):
    if _CURL:
        try:
            return _cr.get(url, params=params, impersonate="chrome", timeout=timeout)
        except Exception:
            pass
    return _rq.get(url, params=params, headers=UA, timeout=timeout)

def _pdf(x): return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)

def _parse(name):
    p = name.split("-")
    if len(p) != 4: return None
    try: return float(p[2]), datetime.strptime(p[1], "%d%b%y"), p[3]
    except Exception: return None

@st.cache_data(ttl=1800)
def _donnees(currency):
    r = _get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
             params={"currency": currency, "kind": "option"})
    r.raise_for_status()
    d = r.json()
    if "result" not in d:
        raise Exception(str(d.get("error", {}))[:120])
    items = d["result"]
    if not items: raise Exception("livre vide")
    S = float(items[0].get("underlying_price") or 0)
    if S <= 0: raise Exception("prix sous-jacent absent")
    opts = []
    for it in items:
        pa = _parse(it.get("instrument_name", ""))
        if not pa: continue
        K, exp, cp = pa
        T = (exp - datetime.now()).days/365.0
        iv = it.get("mark_iv") or 0
        if T <= 0 or T > 0.35 or iv <= 0: continue
        opts.append(dict(K=K, T=T, iv=iv/100, oi=it.get("open_interest") or 0, vol=it.get("volume") or 0, cp=cp))
    return S, opts

@st.cache_data(ttl=1800)
def get_options_deribit(symbol):
    cur = MAPPING.get(symbol)
    if not cur: return None
    try:
        S, opts = _donnees(cur)
    except Exception as e:
        print("DERIBIT erreur:", e); return None
    if not opts: return None
    w = [o for o in opts if o["vol"] > 0]
    iv_moy = (sum(o["iv"]*o["vol"] for o in w)/sum(o["vol"] for o in w)*100) if w else sum(o["iv"] for o in opts)/len(opts)*100
    fen = [o for o in opts if 7/365 <= o["T"] <= 45/365]
    skew = None
    if fen:
        atm = min(fen, key=lambda o: abs(o["K"]-S))["K"]
        pu = [o for o in fen if o["cp"] == "P" and o["K"] == atm]
        ca = [o for o in fen if o["cp"] == "C" and o["K"] == atm]
        if pu and ca: skew = (pu[0]["iv"]-ca[0]["iv"])*100
    strikes = sorted({o["K"] for o in opts if o["oi"] > 0})
    max_pain = None
    if strikes:
        best, bp = None, None
        for P in strikes:
            pain = sum(o["oi"]*max(0, P-o["K"]) for o in opts if o["cp"] == "C" and o["oi"] > 0) + \
                   sum(o["oi"]*max(0, o["K"]-P) for o in opts if o["cp"] == "P" and o["oi"] > 0)
            if bp is None or pain < bp: best, bp = P, pain
        max_pain = best
    return {"iv_moyen_24h": iv_moy, "skew_put_call": skew, "max_pain": max_pain,
            "volume_24h": sum(o["vol"] for o in opts), "options_source": "DERIBIT"}

@st.cache_data(ttl=1800)
def estimer_gex_deribit(currency):
    """GEX estimé (heuristique de recherche, convention calls + / puts -)."""
    try:
        S, opts = _donnees(currency)
    except Exception as e:
        print("DERIBIT erreur (gex):", e); return None
    if not opts: return None
    net = 0
    for o in opts:
        d1 = (math.log(S/o["K"]) + 0.5*o["iv"]**2*o["T"])/(o["iv"]*math.sqrt(o["T"]))
        net += (o["oi"]*(_pdf(d1)/(S*o["iv"]*math.sqrt(o["T"])))*S*S) * (1 if o["cp"] == "C" else -1)
    vc = sum(o["vol"] for o in opts if o["cp"] == "C")
    vp = sum(o["vol"] for o in opts if o["cp"] == "P")
    pcr = vp/max(1e-9, vc)
    return {"gex_regime": "POS" if net > 0 else ("NEG" if net < 0 else "NEUTRE"),
            "gex_flip": None,
            "dex_biais": "BAISSIER" if pcr > 1.2 else ("HAUSSIER" if pcr < 0.8 else "NEUTRE"),
            "gex_strength": 1 if net > 0 else (-1 if net < 0 else 0),
            "gex_source": "DERIBIT-EST"}
