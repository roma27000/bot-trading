# Options Deribit (API publique) : IV, skew, max pain, volume + GEX estimé
# v3 : navigation "navigateur" (curl_cffi) + journaux d'erreurs
import math
from datetime import datetime
import streamlit as st

try:
    from curl_cffi import requests as _req
    _CURL = True
except Exception:
    import requests as _req
    _CURL = False

MAPPING = {"BTC-USD": "BTC", "ETH-USD": "ETH"}

def _get(url, params=None, timeout=15):
    if _CURL:
        return _req.get(url, params=params, impersonate="chrome", timeout=timeout)
    return _req.get(url, params=params, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})

def _pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def _parse(name):
    p = name.split("-")
    if len(p) != 4:
        return None
    try:
        return float(p[2]), datetime.strptime(p[1], "%d%b%y"), p[3]
    except Exception:
        return None

@st.cache_data(ttl=1800)
def _donnees(currency):
    r = _get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
             params={"currency": currency, "kind": "option"})
    r.raise_for_status()
    items = r.json().get("result", [])
    ri = _get("https://www.deribit.com/api/v2/public/get_index_price",
              params={"index_name": f"{currency}_USD"}, timeout=10)
    S = float(ri.json()["result"]["index_price"])
    opts = []
    for it in items:
        pa = _parse(it.get("instrument_name", ""))
        if not pa:
            continue
        K, exp, cp = pa
        T = (exp - datetime.now()).days / 365.0
        iv = it.get("mark_iv") or 0
        if T <= 0 or T > 0.35 or iv <= 0:
            continue
        opts.append({"K": K, "T": T, "iv": iv / 100.0,
                     "oi": it.get("open_interest") or 0, "vol": it.get("volume") or 0, "cp": cp})
    return S, opts

@st.cache_data(ttl=1800)
def get_options_deribit(symbol):
    cur = MAPPING.get(symbol)
    if not cur:
        return None
    try:
        S, opts = _donnees(cur)
    except Exception as e:
        print("DERIBIT erreur:", repr(e))
        return None
    if not opts:
        print("DERIBIT : aucune option valide pour", cur)
        return None
    w = [o for o in opts if o["vol"] > 0]
    iv_moy = (sum(o["iv"] * o["vol"] for o in w) / sum(o["vol"] for o in w) * 100) if w \
        else sum(o["iv"] for o in opts) / len(opts) * 100
    fen = [o for o in opts if 7 / 365 <= o["T"] <= 45 / 365]
    skew = None
    if fen:
        atm = min(fen, key=lambda o: abs(o["K"] - S))["K"]
        put = [o for o in fen if o["cp"] == "P" and o["K"] == atm]
        call = [o for o in fen if o["cp"] == "C" and o["K"] == atm]
        if put and call:
            skew = (put[0]["iv"] - call[0]["iv"]) * 100
        else:
            ps = [o for o in fen if o["cp"] == "P"]; cs = [o for o in fen if o["cp"] == "C"]
            if ps and cs:
                skew = (sum(o["iv"] for o in ps) / len(ps) - sum(o["iv"] for o in cs) / len(cs)) * 100
    strikes = sorted({o["K"] for o in opts if o["oi"] > 0})
    max_pain = None
    if strikes:
        best, bp = None, None
        for P in strikes:
            pain = sum(o["oi"] * max(0.0, P - o["K"]) for o in opts if o["cp"] == "C" and o["oi"] > 0) + \
                   sum(o["oi"] * max(0.0, o["K"] - P) for o in opts if o["cp"] == "P" and o["oi"] > 0)
            if bp is None or pain < bp:
                best, bp = P, pain
        max_pain = best
    return {"iv_moyen_24h": iv_moy, "skew_put_call": skew, "max_pain": max_pain,
            "volume_24h": sum(o["vol"] for o in opts), "options_source": "DERIBIT"}

@st.cache_data(ttl=1800)
def estimer_gex_deribit(currency):
    """GEX estimé (heuristique de recherche, convention calls + / puts -)."""
    try:
        S, opts = _donnees(currency)
    except Exception as e:
        print("DERIBIT erreur (gex):", repr(e))
        return None
    if not opts:
        return None
    net = 0.0
    for o in opts:
        d1 = (math.log(S / o["K"]) + 0.5 * o["iv"] ** 2 * o["T"]) / (o["iv"] * math.sqrt(o["T"]))
        dg = o["oi"] * (_pdf(d1) / (S * o["iv"] * math.sqrt(o["T"]))) * S * S
        net += dg if o["cp"] == "C" else -dg
    vc = sum(o["vol"] for o in opts if o["cp"] == "C")
    vp = sum(o["vol"] for o in opts if o["cp"] == "P")
    pcr = vp / max(1e-9, vc)
    return {"gex_regime": "POS" if net > 0 else ("NEG" if net < 0 else "NEUTRE"),
            "gex_flip": None,
            "dex_biais": "BAISSIER" if pcr > 1.2 else ("HAUSSIER" if pcr < 0.8 else "NEUTRE"),
            "gex_strength": 1 if net > 0 else (-1 if net < 0 else 0),
            "gex_source": "DERIBIT-EST"}
