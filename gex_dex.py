# GEX/DEX : FlashAlpha (clé) -> CryptoGamma -> estimation Deribit (BTC/ETH)
import os, time
import requests
import streamlit as st

MAPPING = {"BTC-USD": "BTC", "ETH-USD": "ETH", "SOL-USD": None,
           "^GSPC": "SPY", "^NDX": "QQQ", "GC=F": None, "SI=F": None}

def _construit(net_gex, flip, net_dex, strength, source):
    regime = "POS" if net_gex > 0 else ("NEG" if net_gex < 0 else "NEUTRE")
    biais = "HAUSSIER" if (net_dex or 0) > 0 else ("BAISSIER" if (net_dex or 0) < 0 else "NEUTRE")
    return {"gex_regime": regime, "gex_flip": flip, "dex_biais": biais,
            "gex_strength": strength, "gex_source": source}

@st.cache_data(ttl=1800)
def get_gex_dex(symbol):
    sym = MAPPING.get(symbol)
    if not sym:
        return None
    key = os.environ.get("FLASHALPHA_API_KEY", "")
    if key:  # 1) FlashAlpha
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
    if sym in ("BTC", "ETH"):  # 2) CryptoGamma
        try:
            r = requests.get(f"https://cryptogamma.io/api/v1/gex/{sym.lower()}", timeout=10)
            if r.status_code == 200:
                d = r.json()
                net = d.get("net_gex", d.get("gex"))
                if net is not None:
                    return _construit(float(net), d.get("gamma_flip"), d.get("net_dex"),
                                      1 if net > 0 else -1, "CRYPTOGAMMA")
        except Exception:
            pass
    if sym in ("BTC", "ETH"):  # 3) estimation Deribit
        try:
            import deribit_options
            return deribit_options.estimer_gex_deribit(sym)
        except Exception:
            return None
    return None
