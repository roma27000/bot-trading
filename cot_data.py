# COT hebdomadaire via fichiers historiques CFTC (ZIP CSV), sans clé
# v3 : navigation "navigateur" (curl_cffi) + journaux d'erreurs
import io, zipfile
import pandas as pd
import streamlit as st

try:
    from curl_cffi import requests as _req
    _CURL = True
except Exception:
    import requests as _req
    _CURL = False

CIBLES = {
    "GC=F": ("https://www.cftc.gov/files/dea/history/com_dis_txt.zip", "GOLD", "dis"),
    "SI=F": ("https://www.cftc.gov/files/dea/history/com_dis_txt.zip", "SILVER", "dis"),
    "^GSPC": ("https://www.cftc.gov/files/dea/history/fut_fin_txt.zip", "S&P 500", "fin"),
}

def _get(url, timeout=120):
    if _CURL:
        return _req.get(url, impersonate="chrome", timeout=timeout)
    return _req.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})

@st.cache_data(ttl=604800)  # 7 jours
def _charger(url):
    r = _get(url)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open(z.namelist()[0]), on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    return df

def _col(df, motifs, exclus=None):
    for c in df.columns:
        u = c.upper()
        if all(m in u for m in motifs) and (exclus is None or not any(e in u for e in exclus)):
            return c
    return None

@st.cache_data(ttl=86400)  # 1 jour
def get_cot(symbol):
    if symbol not in CIBLES:
        return None
    url, mot, typ = CIBLES[symbol]
    try:
        df = _charger(url)
        mcol = _col(df, ["MARKET"])
        dcol = _col(df, ["REPORT", "DATE"]) or _col(df, ["DATE"])
        sub = df[df[mcol].astype(str).str.contains(mot, na=False)].copy()
        if sub.empty or not dcol:
            print("COT : marché introuvable pour", symbol)
            return None
        sub[dcol] = pd.to_datetime(sub[dcol], errors="coerce")
        sub = sub.dropna(subset=[dcol]).sort_values(dcol)
        if typ == "dis":
            lg, sh = _col(sub, ["MANAGED", "LONG"], ["SHORT"]), _col(sub, ["MANAGED", "SHORT"])
            clg, csh = _col(sub, ["PRODUCER", "LONG"], ["SHORT"]), _col(sub, ["PRODUCER", "SHORT"])
        else:
            lg = _col(sub, ["NONCOMMERCIAL", "LONG"], ["SHORT"]) or _col(sub, ["NON-COMMERCIAL", "LONG"], ["SHORT"])
            sh = _col(sub, ["NONCOMMERCIAL", "SHORT"]) or _col(sub, ["NON-COMMERCIAL", "SHORT"])
            clg, csh = _col(sub, ["COMMERCIAL", "LONG"], ["SHORT", "NON"]), _col(sub, ["COMMERCIAL", "SHORT"], ["NON"])
        if not lg or not sh:
            print("COT : colonnes positions introuvables pour", symbol)
            return None
        an = sub.tail(53)
        net = (pd.to_numeric(an[lg], errors="coerce") - pd.to_numeric(an[sh], errors="coerce")).dropna()
        if len(net) < 10:
            print("COT : historique trop court pour", symbol)
            return None
        cur, prev = float(net.iloc[-1]), float(net.iloc[-2])
        com = None
        if clg and csh:
            try:
                com = float(pd.to_numeric(an[clg], errors="coerce").iloc[-1] -
                            pd.to_numeric(an[csh], errors="coerce").iloc[-1])
            except Exception:
                com = None
        return {"net_position_commercials": com, "net_position_noncommercials": cur,
                "changement_semaine": cur - prev,
                "percentile_1an": float((net <= cur).mean() * 100), "cot_source": "CFTC"}
    except Exception as e:
        print("COT erreur:", repr(e))
        return None
