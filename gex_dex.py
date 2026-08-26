# Couche GEX/DEX
# v2 : + estimateur GEX/DEX/IV depuis les chaînes d'options Yahoo (SPY, QQQ, GLD, SLV, IBIT, ETHA)
# Ordre : FlashAlpha (clé) -> CryptoGamma -> Deribit-EST (BTC/ETH) -> Yahoo-OI (tous proxys)
import os, time, math
from datetime import datetime
import requests
import yfinance as yf
import streamlit as st

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "*/*"}
MAPPING = {"BTC-USD": "BTC", "ETH-USD": "ETH"}
YAHOO_PROXY = {"BTC-USD": "IBIT", "ETH-USD": "ETHA", "^GSPC": "SPY",
               "^NDX": "QQQ", "GC=F": "GLD", "SI=F": "SLV"}

def _cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _construit(net_gex, flip, net_dex, strength, source, iv_atm=None):
    regime = "POS" if net_gex > 0 else ("NEG" if net_gex < 0 else "NEUTRE")
    biais = "HAUSSIER" if (net_dex or 0) > 0 else ("BAISSIER" if (net_dex or 0) < 0 else "NEUTRE")
    return {"gex_regime": regime, "gex_flip": flip, "dex_biais": biais,
            "gex_strength": strength, "gex_source": source, "iv_atm": iv_atm}

@st.cache_data(ttl=3600)
def get_gex_yahoo(proxy):
    """GEX/DEX/IV estimés depuis l'open interest des chaînes d'options (Black-Scholes)."""
    try:
        t = yf.Ticker(proxy)
        exps = list(t.options)
        if not exps:
            return None
        spot = float(t.history(period="5d", interval="1d")["Close"].iloc[-1])
        now = datetime.now()
        gex = 0.0; dex = 0.0; ivs = []
        for exp in exps[:2]:
            dte = (datetime.strptime(exp, "%Y-%m-%d") - now).days
            if dte <= 0 or dte > 90:
                continue
            T = dte / 365.0
            ch = t.option_chain(exp)
            for df, sgn in ((ch.calls, 1), (ch.puts, -1)):
                for _, r in df.iterrows():
                    oi = r.get("openInterest") or 0
                    iv = r.get("impliedVolatility") or 0
                    K = float(r.get("strike") or 0)
                    if oi <= 0 or iv <= 0 or K <= 0:
                        continue
                    sig = iv / 100 if iv > 3 else iv
                    sq = math.sqrt(T)
                    d1 = (math.log(spot / K) + 0.5 * sig * sig * T) / (sig * sq)
                    gamma = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (spot * sig * sq)
                    delta = _cdf(d1) if sgn == 1 else _cdf(d1) - 1
                    gex += sgn * oi * gamma * spot * spot
                    dex += oi * delta * spot
                    if abs(math.log(spot / K)) < 0.03:
                        ivs.append(sig * 100)
        if gex == 0 and dex == 0:
            return None
        iv_atm = sum(ivs) / len(ivs) if ivs else None
        strength = 1 if gex > 0 else (-1 if gex < 0 else 0)
        return _construit(gex, None, dex, strength, "YAHOO-OI", iv_atm)
    except Exception as e:
        print(f"YAHOO GEX {proxy} : {e}")
        return None

@st.cache_data(ttl=1800)
def get_gex_dex(symbol):
    sym = MAPPING.get(symbol)
    key = os.environ.get("FLASHALPHA_API_KEY", "")
    if key and sym:  # 1) FlashAlpha (si clé)
        for _ in range(2):
            try:
                r = requests.get(f"https://api.flashalpha.com/v1/exposure/summary/{sym}",
                                 headers={"X-API-KEY": key}, timeout=10)
                if r.status_code == 200:
                    d = r.json()
                    net = d.get("net_gex", d.get("gex"))
                    if net is not None:
                        return _construit(float(net), d.get("gamma_flip"), d.get("net_dex"),
                                          int(d.get("gex_strength", 1 if net > 0 else -1)), "FLASHALPHA")
            except Exception:
                time.sleep(2)
    if sym in ("BTC", "ETH"):  # 2) CryptoGamma puis Deribit-EST
        try:
            r = requests.get(f"https://cryptogamma.io/api/v1/gex/{sym.lower()}", headers=UA, timeout=10)
            if r.status_code == 200:
                d = r.json()
                net = d.get("net_gex", d.get("gex"))
                if net is not None:
                    return _construit(float(net), d.get("gamma_flip"), d.get("net_dex"),
                                      1 if net > 0 else -1, "CRYPTOGAMMA")
        except Exception:
            pass
        try:
            import deribit_options
            est = deribit_options.estimer_gex_deribit(sym)
            if est:
                est["iv_atm"] = None
                return est
        except Exception:
            pass
    proxy = YAHOO_PROXY.get(symbol)  # 3) Yahoo-OI (tous proxys)
    if proxy:
        return get_gex_yahoo(proxy)
    return None
