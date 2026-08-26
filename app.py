import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
import requests
import os, time
import gex_dex, deribit_options, cot_data

st.set_page_config(page_title="Bot Trading Pro", layout="wide")

USE_GEX_DEX = True
USE_COT = True
USE_OPTIONS = True
USE_DEBUG = False
RR_MIN = 1.5
METAUX = ["GC=F", "SI=F"]
RISQUE_PAR_REGIME = {"RISK-ON": 1.0, "NEUTRE": 0.5, "RISK-OFF": 0.25}
PLAFOND_TOTAL = 2.0          # % max de risque si tous les trades se déclenchent
SEUIL_ZONE = 2.0             # % de distance = "zone alerte"

STATS = {
    "BTC-USD": "NEUTRE : 50 % (IC 26-74) n=16 +0.37R — tous régimes : 58 % n=62 +0.66R",
    "ETH-USD": "NEUTRE : 36 % (IC 11-61) n=14 +0.06R — tous régimes : 48 % n=27 +0.44R",
    "SOL-USD": "NEUTRE : 38 % (IC 4-71) n=8 +0.12R — tous régimes : 53 % n=19 +0.57R",
    "GC=F": "tous régimes : 51.7 % (IC 43-61) n=120 +0.46R (robuste dans les 3 régimes)",
    "SI=F": "tous régimes : 43.5 % (IC 33-54) n=92 +0.26R",
}

# ================= OUTILS TECHNIQUES (inchangés) =================
def ajouter_indicateurs(df):
    df = df.copy()
    close, high, low = df["Close"], df["High"], df["Low"]
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()
    delta = close.diff()
    ag = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + ag / al.replace(0, 1e-10)))
    pc = close.shift(1)
    tr = pd.concat([high-low, (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = e12 - e26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["HH20"] = high.rolling(20).max()
    df["LL20"] = low.rolling(20).min()
    hh9, ll9 = high.rolling(9).max(), low.rolling(9).min()
    hh26, ll26 = high.rolling(26).max(), low.rolling(26).min()
    hh52, ll52 = high.rolling(52).max(), low.rolling(52).min()
    df["Tenkan"] = (hh9 + ll9) / 2
    df["Kijun"] = (hh26 + ll26) / 2
    sa = ((df["Tenkan"] + df["Kijun"]) / 2).shift(26)
    sb = ((hh52 + ll52) / 2).shift(26)
    df["CloudTop"] = pd.concat([sa, sb], axis=1).max(axis=1)
    df["CloudBot"] = pd.concat([sa, sb], axis=1).min(axis=1)
    return df

def resample_4h(h1):
    return h1.resample("4h").agg({"Open":"first","High":"max","Low":"min",
                                  "Close":"last","Volume":"sum"}).dropna()

def etat_cloud(row):
    if row["Close"] > row["CloudTop"]: return "au-dessus du nuage"
    if row["Close"] < row["CloudBot"]: return "sous le nuage"
    return "dans le nuage"

def _analyser(symbole):
    try:
        t = yf.Ticker(symbole)
        h1 = t.history(interval="1h", period="1y")
        if h1.empty: return None
        h4 = resample_4h(h1)
        d1 = t.history(interval="1d", period="2y")
        w1 = t.history(interval="1wk", period="5y")
        if min(len(h4), len(d1), len(w1)) < 210: return None
        H4 = ajouter_indicateurs(h4).iloc[-1]
        D1 = ajouter_indicateurs(d1).iloc[-1]
        W1 = ajouter_indicateurs(w1).iloc[-1]
        def bc(r):
            if r["Close"] > r["CloudTop"]: return 1
            if r["Close"] < r["CloudBot"]: return -1
            return 0
        score = 2*bc(W1) + 2*bc(D1) + bc(H4)
        if score >= 3: direction = "LONG"
        elif score >= 1: direction = "LONG SWING"
        elif score <= -3: direction = "SHORT"
        elif score <= -1: direction = "SHORT SWING"
        else: direction = "ATTENTE"
        alerte = ""
        if "LONG" in direction and D1["RSI"] > 75: alerte = "RSI D1 sur-étiré"
        if "SHORT" in direction and D1["RSI"] < 25: alerte = "RSI D1 sur-étiré"
        res = []
        if "LONG" in direction:
            scen = [("PULLBACK", float(H4["Kijun"]),
                     min(float(H4["CloudBot"]), float(H4["LL20"])) - float(H4["ATR"]),
                     [float(H4["HH20"]), float(D1["HH20"]), float(W1["CloudBot"])]),
                    ("CASSE", float(H4["HH20"]), float(H4["HH20"]) - 2*float(H4["ATR"]),
                     [float(D1["HH20"]), float(W1["CloudBot"]), float(W1["CloudTop"])])]
            for ns, e, s, tg in scen:
                rk = e - s
                if rk <= 0: continue
                tps = [(tp, (tp-e)/rk) for tp in tg if tp > e]
                if tps: res.append((ns, e, s, max(x[1] for x in tps), tps))
        elif "SHORT" in direction:
            scen = [("PULLBACK", float(H4["Kijun"]),
                     max(float(H4["CloudTop"]), float(H4["HH20"])) + float(H4["ATR"]),
                     [float(H4["LL20"]), float(D1["LL20"]), float(W1["CloudTop"])]),
                    ("CASSE", float(H4["LL20"]), float(H4["LL20"]) + 2*float(H4["ATR"]),
                     [float(D1["LL20"]), float(W1["CloudTop"]), float(W1["CloudBot"])])]
            for ns, e, s, tg in scen:
                rk = s - e
                if rk <= 0: continue
                tps = [(tp, (e-tp)/rk) for tp in tg if tp < e]
                if tps: res.append((ns, e, s, max(x[1] for x in tps), tps))
        base = {"prix": float(H4["Close"]), "atr": float(H4["ATR"]), "score": score,
                "direction": direction, "alerte": alerte, "rsi_d1": float(D1["RSI"]),
                "etat_w": etat_cloud(W1), "etat_d": etat_cloud(D1), "etat_h4": etat_cloud(H4),
                "date_h4": h4.index[-1]}
        if direction == "ATTENTE" or not res:
            base.update({"scenario": "-", "entree": None, "stop": None, "rr": None, "tps": [],
                         "statut": "ATTENTE" if direction == "ATTENTE" else "REJETÉ"})
            return base
        ns, e, s, rr, tps = max(res, key=lambda x: x[3])
        stt = "PRÉFÉRÉ" if rr >= 2 else ("ACCEPTABLE" if rr >= RR_MIN else "REJETÉ")
        base.update({"scenario": ns, "entree": e, "stop": s, "rr": rr, "tps": tps, "statut": stt})
        return base
    except Exception:
        return None

@st.cache_data(ttl=900)
def analyser_actif(symbole):
    return _analyser(symbole)

# ================= DONNÉES EXTERNES =================
@st.cache_data(ttl=43200)
def get_regime():
    key = os.environ.get("FRED_API_KEY", "")
    try:
        def fred_hist(sid):
            r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                             params={"series_id": sid, "api_key": key, "file_type": "json",
                                     "observation_start": "2018-01-01", "sort_order": "asc"}, timeout=20)
            r.raise_for_status()
            s = pd.Series({o["date"]: float(o["value"]) for o in r.json()["observations"] if o["value"] != "."})
            s.index = pd.to_datetime(s.index)
            return s
        vix = yf.Ticker("^VIX").history(interval="1d", period="7y")["Close"]
        dxy = yf.Ticker("DX-Y.NYB").history(interval="1d", period="7y")["Close"]
        if vix.index.tz is not None: vix.index = vix.index.tz_localize(None)
        if dxy.index.tz is not None: dxy.index = dxy.index.tz_localize(None)
        reg = pd.DataFrame({"r10": fred_hist("DGS10").resample("ME").last(),
                            "vix": vix.resample("ME").mean(),
                            "dxy": dxy.resample("ME").last()}).dropna()
        d10 = reg["r10"] - reg["r10"].shift(3)
        ddxy = (reg["dxy"] / reg["dxy"].shift(3) - 1) * 100
        sc = (d10.apply(lambda x: 1 if x <= -0.25 else (-1 if x >= 0.25 else 0))
              + ddxy.apply(lambda x: 1 if x <= -2 else (-1 if x >= 2 else 0))
              + reg["vix"].apply(lambda x: 1 if x < 20 else (0 if x <= 30 else -1)))
        return sc.apply(lambda s: "RISK-ON" if s >= 2 else ("RISK-OFF" if s <= -1 else "NEUTRE"))
    except Exception:
        return pd.Series(dtype=object)

@st.cache_data(ttl=14400)
def get_positionnement():
    out = {}
    try:
        items = None
        for essai in range(3):
            try:
                r = requests.get("https://api.coingecko.com/api/v3/derivatives", timeout=20)
                r.raise_for_status()
                data = r.json()
                items = data.get("data", data) if isinstance(data, dict) else data
                break
            except Exception:
                time.sleep(5)
        if items:
            for coin in ["BTC", "ETH", "SOL"]:
                rows = []
                for it in items:
                    if not isinstance(it, dict): continue
                    if str(it.get("contract_type", "")).lower() != "perpetual": continue
                    if str(it.get("index_id", "")).upper() != coin: continue
                    sym = str(it.get("symbol", "")).upper()
                    if not (sym.endswith("USDT") or sym.endswith("USDC") or sym.endswith("USD")): continue
                    fr, oi = it.get("funding_rate"), it.get("open_interest")
                    rows.append({"funding": float(fr) if fr is not None else None,
                                 "oi": float(oi) if oi is not None else None})
                if rows:
                    df = pd.DataFrame(rows).dropna(subset=["oi"]).sort_values("oi", ascending=False).head(10)
                    f = df["funding"].dropna()
                    out[coin] = {"oi": float(df["oi"].sum()),
                                 "funding": float(f.mean()) if len(f) else None}
    except Exception:
        pass
    return out

@st.cache_data(ttl=900)
def get_metaux():
    try:
        g = yf.Ticker("GC=F").history(interval="1d", period="1y")["Close"]
        s = yf.Ticker("SI=F").history(interval="1d", period="1y")["Close"]
        if g.index.tz is not None: g.index = g.index.tz_localize(None)
        if s.index.tz is not None: s.index = s.index.tz_localize(None)
        g_now, s_now = float(g.iloc[-1]), float(s.iloc[-1])
        k = min(63, len(g) - 1)
        return {"g": g_now, "s": s_now, "ema": float(g.ewm(span=200, adjust=False).mean().iloc[-1]),
                "g3m": (g_now / float(g.iloc[k]) - 1) * 100, "ratio": g_now / s_now}
    except Exception:
        return {}

# ================= PÉDAGOGIE & LECTURE SIMPLE =================
CONCEPTS = """
<div class='card'><div class='top'><h2>📚 Comprendre les couches avancées (langage simple)</h2></div>
<p><b>IV (volatilité implicite)</b> = l'agitation que le marché <i>attend</i>. Comme la météo : IV élevée = tempête attendue ; IV basse = temps calme.</p>
<p><b>Skew put/call</b> = le prix de la peur. Si les “puts” (protections contre la baisse) sont plus chers que les “calls” (paris hausse), le marché se protège → peur présente.</p>
<p><b>Max pain</b> = le prix où les acheteurs d'options perdent le plus à l'expiration ; souvent un “aimant” en fin de semaine.</p>
<p><b>GEX</b> = le sens des amortisseurs. GEX positif = les gros acteurs calmant le jeu (marché stable, cassures moins fiables) ; GEX négatif = ils amplifient (marché nerveux).</p>
<p><b>DEX</b> = le biais directionnel du positionnement options (haussier / baissier / neutre).</p>
<p><b>COT</b> = le positionnement des “gros” (spéculateurs vs professionnels). Un percentile extrême = foule déjà pleine d'un côté → risque de retournement.</p>
<p class='histo'>Règle d'or : ces couches <b>informent et modulent la taille</b> ; elles ne déclenchent jamais un trade. La technique (tendance + cassure + R:R) décide.</p></div>
"""

def lecture_options_simple(opt, gex):
    if not opt and not gex:
        return None
    ph = []
    if opt:
        iv = opt.get("iv_moyen_24h")
        if iv is not None:
            if iv > 80: ph.append(f"Volatilité très élevée ({iv:.0f} %) : prudence et tailles réduites.")
            elif iv > 55: ph.append(f"Volatilité élevée ({iv:.0f} %) : attends des confirmations nettes.")
            elif iv > 35: ph.append(f"Volatilité modérée ({iv:.0f} %) : marché plutôt calme, signaux techniques plus fiables.")
            else: ph.append(f"Volatilité basse ({iv:.0f} %) : méfie-toi des faux calmes.")
        sk = opt.get("skew_put_call")
        if sk is not None:
            if sk > 8: ph.append("Protections contre la baisse chères : peur présente, replis potentiellement violents.")
            elif sk < -5: ph.append("Paris hausse chers : appétit haussier, ne pas courir après le prix.")
            else: ph.append("Peur et appétit équilibrés : pas de stress particulier dans les options.")
        mp = opt.get("max_pain")
        if mp: ph.append(f"Le “max pain” ({mp:,.0f}) agit comme un aimant de fin de semaine.")
    if gex:
        if gex.get("gex_regime") == "POS": ph.append("GEX positif : market-makers amortisseurs → marché stabilisé, casse à confirmer deux fois.")
        elif gex.get("gex_regime") == "NEG": ph.append("GEX négatif : market-makers amplificateurs → marché nerveux, stops stricts.")
    return " ".join(ph)

def lecture_metaux(m):
    if not m: return "Données or/argent indisponibles."
    n = []
    n.append("Or au-dessus de sa moyenne 200j : demande monétaire/refuge soutenue" if m["g"] > m["ema"]
             else "Or sous sa moyenne 200j : demande monétaire affaiblie")
    if m["ratio"] >= 85: n.append(f"ratio or/argent {m['ratio']:.0f} (élevé) : environnement défensif")
    elif m["ratio"] <= 70: n.append(f"ratio or/argent {m['ratio']:.0f} (bas) : appétit industriel")
    else: n.append(f"ratio or/argent {m['ratio']:.0f} : neutre")
    if m["g3m"] >= 5: n.append(f"or +{m['g3m']:.1f} % sur 3 mois : forte demande de protection")
    elif m["g3m"] <= -5: n.append(f"or {m['g3m']:.1f} % sur 3 mois : détente du stress")
    return " | ".join(n)

def conseil_systeme(r, reg, metal, gex=None, cot=None, opt=None):
    d = r["direction"]
    if d == "ATTENTE": return "Aucune action. Pas de biais clair : le système attend."
    if r["statut"] == "REJETÉ": return "Aucune action. R:R insuffisant : le système protège ton capital."
    if reg == "RISK-OFF" and "LONG" in d and not metal:
        return "VETO MACRO : pas de position longue risquée dans ce régime (métaux exceptés)."
    if reg == "RISK-ON" and "SHORT" in d: return "VETO MACRO : pas de position courte dans ce régime."
    t = []
    if r["scenario"] == "CASSE":
        sens = "au-dessus" if "LONG" in d else "en dessous"
        t.append(f"Attendre une clôture H4 {sens} de <b>{r['entree']:.2f}</b> avant d'entrer.")
    else:
        t.append(f"Attendre un repli vers <b>{r['entree']:.2f}</b> pour envisager une entrée.")
    t.append(f"Stop loss obligatoire à <b>{r['stop']:.2f}</b>, jamais élargi.")
    if r["alerte"] or "SWING" in d:
        t.append("Contexte fragile : <b>taille réduite</b>, ne pas courir après le prix.")
    t.append("Sorties par tiers (TP1/TP2/TP3) ; après TP1, remonter le stop au prix d'entrée.")
    bonus = []
    if gex:
        if gex["gex_regime"] == "POS" and "LONG" in d:
            bonus.append("GEX positif : marché stabilisé, favorable à un long confirmé.")
        elif gex["gex_regime"] == "NEG" and "LONG" in d:
            bonus.append("GEX négatif : marché nerveux, prudence sur le long.")
        elif gex["gex_regime"] == "NEG" and "SHORT" in d:
            bonus.append("GEX négatif : environnement instable, cohérent avec un short prudent.")
    if cot and cot.get("percentile_1an") is not None:
        if cot["percentile_1an"] >= 85 and "LONG" in d:
            bonus.append(f"COT spéculatifs très longs (percentile {cot['percentile_1an']:.0f} %) : positionnement chargé.")
        elif cot["percentile_1an"] <= 15 and "SHORT" in d:
            bonus.append(f"COT spéculatifs très shorts (percentile {cot['percentile_1an']:.0f} %) : risque de short squeeze.")
    if opt and opt.get("iv_moyen_24h"):
        if opt["iv_moyen_24h"] > 70:
            bonus.append("IV élevée : marché tendu, privilégier taille réduite.")
        elif (opt.get("skew_put_call") or 0) > 8:
            bonus.append("Skew put marqué : peur présente, les replis peuvent être violents.")
    if bonus:
        t.append(" ".join(bonus[:2]))
    return " ".join(t)

def badge_classe(r):
    return {"PRÉFÉRÉ": "ok", "ACCEPTABLE": "warn", "REJETÉ": "no"}.get(r["statut"], "wait")

def fmt(v):
    return f"{v:.2f}" if v is not None else "—"

# ================= PACK 1 : FEU, QUALITÉ =================
def feu_txt(dist, actionnable):
    if not actionnable:
        return ""
    if dist <= SEUIL_ZONE:
        return "🟠 ZONE ALERTE : déclencheur proche → surveille la prochaine clôture H4"
    return f"⚪ Surveillance : déclencheur à {dist:.1f} % (🟢 = confirmé par toi sur le graphique)"

def qualite(r, gex):
    d = r["direction"]
    long = "LONG" in d
    pts = [("R:R", (r.get("rr") or 0) >= 2),
           ("RSI", r["rsi_d1"] < 75 if long else r["rsi_d1"] > 25),
           ("GEX", bool(gex) and (gex["gex_regime"] == "POS" if long else gex["gex_regime"] == "NEG")),
           ("TendW", r["etat_w"] == "au-dessus du nuage" if long else r["etat_w"] == "sous le nuage")]
    s = sum(1 for _, ok in pts if ok)
    det = " · ".join(f"{n} {'✔' if ok else '✘'}" for n, ok in pts)
    return s, det

CSS = """
<style>
body{font-family:-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:16px}
h1{font-size:22px} .meta{color:#94a3b8;font-size:13px;margin-bottom:16px}
.card{background:#1e293b;border-radius:16px;padding:16px;margin-bottom:16px}
.top{display:flex;justify-content:space-between;align-items:center}
h2{font-size:18px;margin:0} .sym{color:#94a3b8;font-size:13px}
.badge{padding:4px 12px;border-radius:999px;font-weight:700;font-size:12px;color:#fff}
.ok{background:#16a34a} .warn{background:#d97706} .no{background:#dc2626} .wait{background:#64748b}
.verdict{color:#38bdf8;font-weight:600;margin:8px 0}
.prix{color:#e2e8f0;font-size:13px}
table{width:100%;border-collapse:collapse;margin:8px 0}
th,td{padding:8px 6px;border-bottom:1px solid #334155;text-align:left;font-size:14px}
.niveaux td{font-size:16px;font-weight:700}
.taille{color:#4ade80} .histo{color:#94a3b8;font-size:12px}
.layers{background:#0f172a;padding:8px;border-radius:6px;margin:8px 0;font-size:12px;color:#cbd5e1}
.conseil{background:#0b1220;border-left:4px solid #38bdf8;padding:10px;border-radius:8px;margin-top:10px;font-size:14px}
.alert{background:#7f1d1d;border-left:4px solid #f87171;padding:10px;border-radius:8px;margin-top:10px;font-size:14px;color:#fecaca}
</style>
"""

# ================= INTERFACE =================
st.title("📊 Rapport de trading Pro")
st.caption("Document d'analyse — pas un conseil personnalisé. Prix Yahoo (délai possible).")

capital = st.sidebar.number_input("Capital simulé (€)", min_value=10.0, value=100.0, step=10.0)
r_semaine = st.sidebar.number_input("R cumulé cette semaine (paper)", value=0.0, step=0.5)
st.sidebar.markdown("---")
st.sidebar.write("**Couches actives**")
st.sidebar.write(f"GEX/DEX : {'ON' if USE_GEX_DEX else 'OFF'} | COT : {'ON' if USE_COT else 'OFF'} | Options : {'ON' if USE_OPTIONS else 'OFF'}")
st.sidebar.markdown("---")
if st.sidebar.button("🧹 Forcer des prix frais"):
    st.cache_data.clear()
    st.sidebar.success("Cache vidé. Clique sur Générer.")

if st.button("🔄 Générer l'analyse du jour", type="primary"):
    with st.spinner("Analyse en cours (1-2 minutes)..."):
        regime_hist = get_regime()
        regime_actuel = str(regime_hist.iloc[-1]) if len(regime_hist) else "INCONNU"
        risque_pct = RISQUE_PAR_REGIME.get(regime_actuel, 0.5)
        positionnement = get_positionnement()
        metaux_indic = get_metaux()
        now = pd.Timestamp.now(tz="UTC")
        next_h4 = now.floor("4h") + pd.Timedelta("4h")

        # ---------- PASSE 1 : analyse ----------
        resultats = []
        ACTIFS = [("BTC-USD","Bitcoin","BTC"), ("ETH-USD","Ethereum","ETH"), ("SOL-USD","Solana","SOL"),
                  ("GC=F","Or",None), ("SI=F","Argent",None),
                  ("^GSPC","S&P 500",None), ("^NDX","Nasdaq 100",None)]
        for sym, nom, coin in ACTIFS:
            r = analyser_actif(sym)
            if r is None:
                continue
            gex = gex_dex.get_gex_dex(sym) if USE_GEX_DEX else None
            cot = cot_data.get_cot(sym) if USE_COT else None
            opt = deribit_options.get_options_deribit(sym) if USE_OPTIONS else None
            metal = sym in METAUX
            veto = (regime_actuel == "RISK-OFF" and "LONG" in r["direction"] and not metal)
            actionnable = r["entree"] is not None and r["statut"] in ("PRÉFÉRÉ", "ACCEPTABLE") and not veto
            dist = abs(r["entree"] - r["prix"]) / r["prix"] * 100 if r["entree"] else None
            sens = "au-dessus du prix" if (r["entree"] and r["entree"] > r["prix"]) else "en dessous du prix"
            q, qdet = (qualite(r, gex) if r["entree"] else (0, ""))
            resultats.append(dict(sym=sym, nom=nom, coin=coin, r=r, gex=gex, cot=cot, opt=opt,
                                  metal=metal, veto=veto, actionnable=actionnable,
                                  dist=dist, sens=sens, q=q, qdet=qdet))

        # ---------- PASSE 2 : garde-fous portefeuille ----------
        act = [x for x in resultats if x["actionnable"]]
        n_rej = sum(1 for x in resultats if x["r"]["statut"] == "REJETÉ")
        crypto_longs = [x for x in act if x["coin"] in ("BTC", "ETH", "SOL") and "LONG" in x["r"]["direction"]]
        crypto_factor = 0.5 if len(crypto_longs) >= 3 else 1.0
        total_brut = risque_pct * len(act)
        global_factor = min(1.0, PLAFOND_TOTAL / total_brut) if total_brut > PLAFOND_TOTAL else 1.0
        total_adj = sum(risque_pct * (crypto_factor if (x["coin"] in ("BTC","ETH","SOL") and "LONG" in x["r"]["direction"]) else 1.0)
                        * global_factor for x in act)
        plus_proche = min([x for x in act if x["dist"] is not None], key=lambda x: x["dist"], default=None)

        adj_html = ""
        if crypto_factor < 1:
            adj_html += "<p>⚠️ <b>Corrélation</b> : 3 cryptos long = 1 seul panier → tailles crypto ÷2.</p>"
        if global_factor < 1:
            adj_html += f"<p>⚠️ <b>Plafond 2 %</b> : risque total brut {total_brut:.1f} % → toutes tailles ×{global_factor:.2f}.</p>"
        cb_html = ""
        if r_semaine <= -2.0:
            cb_html = "<div class='alert'>🛑 <b>CIRCUIT BREAKER</b> : −2 % atteint cette semaine → pause jusqu'à lundi. Le capital se protège d'abord.</div>"

        nearest_html = ""
        if plus_proche:
            nearest_html = (f"<p>🔎 Déclencheur le plus proche : <b>{plus_proche['nom']}</b> à "
                            f"<b>{plus_proche['dist']:.1f} %</b> ({plus_proche['sens']}) — "
                            f"{feu_txt(plus_proche['dist'], True)}</p>")

        resume_html = f"""
        <div class='card'><div class='top'><h2>🎯 Aujourd'hui — {now.strftime('%d/%m, %H:%M UTC')}</h2></div>
        <p><b>{len(act)}</b> signal(s) actionnable(s) | <b>{n_rej}</b> rejeté(s) | risque par trade : <b>{risque_pct} %</b> (régime {regime_actuel})</p>
        <p>⏱ Prochaine clôture H4 : <b>{next_h4.strftime('%H:%M UTC')}</b> — les confirmations se jugent à cette clôture.</p>
        {nearest_html}
        <p>💼 Risque total si tout se déclenche : <b>{total_adj:.2f} %</b> (plafond {PLAFOND_TOTAL:.0f} %)</p>
        {adj_html}{cb_html}
        <p class='histo'>Feux : ⚪ loin · 🟠 zone (&le;{SEUIL_ZONE:.0f} %) · 🟢 confirmé par toi sur le graphique après clôture H4.</p></div>
        """

        # ---------- PASSE 3 : cartes ----------
        metaux_html = ""
        if metaux_indic:
            m = metaux_indic
            metaux_html = f"""
            <div class='card'><div class='top'><h2>🥇 Contexte macro — Or & Argent</h2></div>
            <p>Or : <b>{m['g']:.2f} $</b> | Argent : <b>{m['s']:.2f} $</b> | Ratio or/argent : <b>{m['ratio']:.0f}</b></p>
            <p>Or vs moyenne 200j : {m['ema']:.2f} | Or sur 3 mois : {m['g3m']:+.1f} %</p>
            <div class='conseil'>💡 {lecture_metaux(m)}</div></div>"""

        cartes = ""
        lignes_journal = []
        for x in resultats:
            r, gex, cot, opt = x["r"], x["gex"], x["cot"], x["opt"]
            p = positionnement.get(x["coin"])
            histo = STATS.get(x["sym"], "n/a (pas de backtest pour cet actif)")
            h4_txt = pd.Timestamp(r["date_h4"]).strftime("%d/%m %Hh UTC") if r.get("date_h4") is not None else "?"

            facteur = 1.0
            if x["actionnable"]:
                if x["coin"] in ("BTC", "ETH", "SOL") and "LONG" in r["direction"]:
                    facteur *= crypto_factor
                facteur *= global_factor
            taille_html = ""
            risque_e = taille_e = unites = None
            if x["actionnable"]:
                risque_e = capital * risque_pct / 100 * facteur
                dist_stop = abs(r["entree"] - r["stop"]) / r["entree"] * 100
                taille_e = risque_e / (dist_stop / 100)
                unites = taille_e / r["entree"]
                note = f" (ajustée ×{facteur:.2f})" if facteur < 1 else ""
                taille_html = (f"<p class='taille'>Taille : <b>{taille_e:.2f} €</b> ({unites:.6f} unité) "
                               f"— risque {risque_e:.2f} €{note}</p>")

            tps = r["tps"]
            tp1 = fmt(tps[0][0]) if len(tps) > 0 else "—"
            tp2 = fmt(tps[1][0]) if len(tps) > 1 else "—"
            tp3 = fmt(tps[2][0]) if len(tps) > 2 else "—"

            pos_html = ""
            if p is not None and p.get("funding") is not None:
                pos_html = f"<p>Positionnement : funding {p['funding']:.4f} %/8h | OI {p['oi']/1e9:.1f} Md$</p>"

            lignes = []
            if USE_GEX_DEX:
                if gex:
                    iv_txt = f" | IV ATM : {gex['iv_atm']:.0f} %" if gex.get("iv_atm") else ""
                    lignes.append(f"GEX : {gex['gex_regime']} | DEX : {gex['dex_biais']} | Force : {gex['gex_strength']:+d}{iv_txt} ({gex['gex_source']})")
                else:
                    lignes.append("<span style='color:#64748b'>GEX/DEX : non disponible aujourd'hui</span>")
            if USE_COT:
                if cot:
                    lignes.append(f"COT : Non-commercials net {cot['net_position_noncommercials']:,.0f} | Δ semaine : {cot['changement_semaine']:+,.0f} | Percentile 1 an : {cot['percentile_1an']:.0f} %")
                else:
                    lignes.append("<span style='color:#64748b'>COT : non disponible (CFTC anti-robots) — couche reportée</span>")
            if USE_OPTIONS:
                if opt:
                    lignes.append(f"Options Deribit : IV 24h : {opt['iv_moyen_24h']:.0f} % | Skew : {fmt(opt['skew_put_call'])} | Max pain : {fmt(opt['max_pain'])} | Vol : {opt['volume_24h']:,.0f}")
                elif x["sym"] in ("BTC-USD", "ETH-USD"):
                    lignes.append("<span style='color:#64748b'>Options Deribit : non disponible aujourd'hui (serveur protégé) — chiffres dans le LAB Colab hebdo</span>")
            lect = lecture_options_simple(opt, gex)
            layers_html = f"<div class='layers'>{'<br>'.join(lignes)}</div>" if lignes else ""
            if lect:
                layers_html += f"<div class='conseil'>📖 Lecture simple : {lect}</div>"

            dist_html = ""
            if x["dist"] is not None:
                dist_html = (f"<p class='prix'>Prix actuel : <b>{r['prix']:.2f}</b> | dernière bougie H4 : {h4_txt} | "
                             f"🚦 {feu_txt(x['dist'], x['actionnable']) or f'déclencheur à {x[\"dist\"]:.1f} %'}</p>"
                             f"<p class='histo'>Qualité : {x['q']}/4 — {x['qdet']}</p>")
            else:
                dist_html = f"<p class='prix'>Prix actuel : <b>{r['prix']:.2f}</b> | dernière bougie H4 : {h4_txt}</p>"

            cartes += f"""
            <div class='card'>
              <div class='top'><h2>{x['nom']} <span class='sym'>{x['sym']}</span></h2>
              <span class='badge {badge_classe(r)}'>{r['statut']}</span></div>
              <p class='verdict'>{r['direction']} — scénario {r['scenario']}</p>
              {dist_html}
              <table><tr><th>Entrée</th><th>Stop</th><th>TP1</th><th>TP2</th><th>TP3</th></tr>
              <tr class='niveaux'><td>{fmt(r['entree'])}</td><td>{fmt(r['stop'])}</td>
              <td>{tp1}</td><td>{tp2}</td><td>{tp3}</td></tr></table>
              {taille_html}
              <p>Macro : régime <b>{regime_actuel}</b> | RSI D1 : {r['rsi_d1']:.1f} | Ichimoku W : {r['etat_w']}</p>
              {pos_html}
              {layers_html}
              <p class='histo'>Historique mesuré (25/08/2026) : {histo}</p>
              <div class='conseil'>💡 {conseil_systeme(r, regime_actuel, x['metal'], gex, cot, opt)}</div>
            </div>"""

            lignes_journal.append({
                "date": now.date(), "actif": x["sym"], "direction": r["direction"],
                "scenario": r["scenario"], "statut": r["statut"],
                "prix_actuel": round(r["prix"], 2), "entree": r["entree"], "stop": r["stop"],
                "distance_pct": round(x["dist"], 2) if x["dist"] is not None else None,
                "qualite": x["q"], "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "risque_e": risque_e, "taille_e": taille_e, "unites": unites,
                "gex_regime": gex["gex_regime"] if gex else None,
                "gex_source": gex["gex_source"] if gex else None,
                "cot_percentile": round(cot["percentile_1an"], 1) if cot else None,
                "iv_24h": round(opt["iv_moyen_24h"], 1) if opt else None,
                "skew": round(opt["skew_put_call"], 2) if opt and opt["skew_put_call"] is not None else None,
                "max_pain": opt["max_pain"] if opt else None,
                "historique_regime": histo
            })

        if USE_DEBUG:
            st.caption(f"Debug — régime {regime_actuel} | facteurs : crypto={crypto_factor}, global={global_factor:.2f}")

        html = f"""<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'>{CSS}</head><body>
        <h1>📊 Rapport de trading Pro — {now.strftime('%d/%m/%Y')}</h1>
        <div class='meta'>Régime macro : <b>{regime_actuel}</b> | Capital : {capital:.0f} € |
        Risque modulé : <b>{risque_pct} %</b> par trade | Couches : GEX/DEX, COT, Options Deribit</div>
        {resume_html}
        {metaux_html}
        {CONCEPTS}
        {cartes}
        <div class='conseil'>Rappel : aucun trade réel sans confirmation du prix, sans respect du stop,
        et sans paper trading validé. Les couches avancées informent et modulent la taille ; la technique décide.</div>
        </body></html>"""

        components.html(html, height=6000, scrolling=True)

        csv = pd.DataFrame(lignes_journal).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Télécharger le journal CSV", csv,
                           file_name="journal_signaux.csv", mime="text/csv")
else:
    st.info("Clique sur « Générer l'analyse du jour » pour produire le rapport complet.")
