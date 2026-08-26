import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
import requests
import os, time

st.set_page_config(page_title="Bot Trading", layout="wide")

# ================= RÉGLAGES =================
RR_MIN = 1.5
METAUX = ["GC=F", "SI=F"]
RISQUE_PAR_REGIME = {"RISK-ON": 1.0, "NEUTRE": 0.5, "RISK-OFF": 0.25}

STATS = {
    "BTC-USD": "NEUTRE : 50 % (IC 26-74) n=16 +0.37R — tous régimes : 58 % n=62 +0.66R",
    "ETH-USD": "NEUTRE : 36 % (IC 11-61) n=14 +0.06R — tous régimes : 48 % n=27 +0.44R",
    "SOL-USD": "NEUTRE : 38 % (IC 4-71) n=8 +0.12R — tous régimes : 53 % n=19 +0.57R",
    "GC=F": "tous régimes : 51.7 % (IC 43-61) n=120 +0.46R (robuste dans les 3 régimes)",
    "SI=F": "tous régimes : 43.5 % (IC 33-54) n=92 +0.26R",
}

# ================= OUTILS TECHNIQUES =================
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
                "etat_w": etat_cloud(W1), "etat_d": etat_cloud(D1), "etat_h4": etat_cloud(H4)}
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

@st.cache_data(ttl=3600)
def analyser_actif(symbole):
    return _analyser(symbole)

# ================= DONNÉES EXTERNES (cache anti-429) =================
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

@st.cache_data(ttl=3600)
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

# ================= LECTURE & CONSEILS =================
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

def conseil_systeme(r, reg, metal):
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
    return " ".join(t)

def badge_classe(r):
    return {"PRÉFÉRÉ": "ok", "ACCEPTABLE": "warn", "REJETÉ": "no"}.get(r["statut"], "wait")

def fmt(v):
    return f"{v:.2f}" if v is not None else "—"

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
table{width:100%;border-collapse:collapse;margin:8px 0}
th,td{padding:8px 6px;border-bottom:1px solid #334155;text-align:left;font-size:14px}
.niveaux td{font-size:16px;font-weight:700}
.taille{color:#4ade80} .histo{color:#94a3b8;font-size:12px}
.conseil{background:#0b1220;border-left:4px solid #38bdf8;padding:10px;border-radius:8px;margin-top:10px;font-size:14px}
</style>
"""

# ================= INTERFACE =================
st.title("📊 Rapport de trading")
st.caption("Document d'analyse — pas un conseil personnalisé.")

capital = st.sidebar.number_input("Capital simulé (€)", min_value=10.0, value=100.0, step=10.0)
st.sidebar.markdown("---")
st.sidebar.write("**Routine quotidienne**")
st.sidebar.write("1. Générer l'analyse\n2. Lire les cartes\n3. Télécharger le journal")

if st.button("🔄 Générer l'analyse du jour", type="primary"):
    with st.spinner("Analyse en cours (environ 1 minute)..."):
        regime_hist = get_regime()
        regime_actuel = str(regime_hist.iloc[-1]) if len(regime_hist) else "INCONNU"
        risque_pct = RISQUE_PAR_REGIME.get(regime_actuel, 0.5)
        positionnement = get_positionnement()
        metaux_indic = get_metaux()

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
        ACTIFS = [("BTC-USD","Bitcoin","BTC"), ("ETH-USD","Ethereum","ETH"), ("SOL-USD","Solana","SOL"),
                  ("GC=F","Or",None), ("SI=F","Argent",None),
                  ("^GSPC","S&P 500",None), ("^NDX","Nasdaq 100",None)]

        for sym, nom, coin in ACTIFS:
            r = analyser_actif(sym)
            if r is None:
                continue
            metal = sym in METAUX
            p = positionnement.get(coin)
            histo = STATS.get(sym, "n/a (pas de backtest pour cet actif)")

            taille_html = ""
            risque_e = taille_e = unites = None
            veto = (regime_actuel == "RISK-OFF" and "LONG" in r["direction"] and not metal)
            if r["entree"] is not None and r["statut"] in ("PRÉFÉRÉ", "ACCEPTABLE") and not veto:
                risque_e = capital * risque_pct / 100
                dist = abs(r["entree"] - r["stop"]) / r["entree"] * 100
                taille_e = risque_e / (dist / 100)
                unites = taille_e / r["entree"]
                taille_html = (f"<p class='taille'>Taille : <b>{taille_e:.2f} €</b> ({unites:.6f} unité) "
                               f"— risque {risque_e:.2f} € ({risque_pct} %)</p>")

            tps = r["tps"]
            tp1 = fmt(tps[0][0]) if len(tps) > 0 else "—"
            tp2 = fmt(tps[1][0]) if len(tps) > 1 else "—"
            tp3 = fmt(tps[2][0]) if len(tps) > 2 else "—"

            pos_html = ""
            if p is not None and p.get("funding") is not None:
                pos_html = f"<p>Positionnement : funding {p['funding']:.4f} %/8h | OI {p['oi']/1e9:.1f} Md$</p>"

            cartes += f"""
            <div class='card'>
              <div class='top'><h2>{nom} <span class='sym'>{sym}</span></h2>
              <span class='badge {badge_classe(r)}'>{r['statut']}</span></div>
              <p class='verdict'>{r['direction']} — scénario {r['scenario']}</p>
              <table><tr><th>Entrée</th><th>Stop</th><th>TP1</th><th>TP2</th><th>TP3</th></tr>
              <tr class='niveaux'><td>{fmt(r['entree'])}</td><td>{fmt(r['stop'])}</td>
              <td>{tp1}</td><td>{tp2}</td><td>{tp3}</td></tr></table>
              {taille_html}
              <p>Macro : régime <b>{regime_actuel}</b> | RSI D1 : {r['rsi_d1']:.1f} | Ichimoku W : {r['etat_w']}</p>
              {pos_html}
              <p class='histo'>Historique mesuré (25/08/2026) : {histo}</p>
              <div class='conseil'>💡 {conseil_systeme(r, regime_actuel, metal)}</div>
            </div>"""

            lignes_journal.append({"date": pd.Timestamp.now().date(), "actif": sym,
                                   "direction": r["direction"], "scenario": r["scenario"],
                                   "statut": r["statut"], "entree": r["entree"], "stop": r["stop"],
                                   "tp1": tp1, "tp2": tp2, "tp3": tp3, "risque_e": risque_e,
                                   "taille_e": taille_e, "unites": unites, "historique_regime": histo})

        html = f"""<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'>{CSS}</head><body>
        <h1>📊 Rapport de trading — {pd.Timestamp.now().strftime('%d/%m/%Y')}</h1>
        <div class='meta'>Régime macro : <b>{regime_actuel}</b> | Capital : {capital:.0f} € |
        Risque modulé : <b>{risque_pct} %</b> par trade</div>
        {metaux_html}
        {cartes}
        <div class='conseil'>Rappel : aucun trade réel sans confirmation du prix, sans respect du stop,
        et sans paper trading validé.</div>
        </body></html>"""

        # Affichage garanti du rapport (page complète dans un cadre à faire défiler)
        components.html(html, height=4000, scrolling=True)

        csv = pd.DataFrame(lignes_journal).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Télécharger le journal CSV", csv,
                           file_name="journal_signaux.csv", mime="text/csv")
else:
    st.info("Clique sur « Générer l'analyse du jour » pour produire le rapport complet.")
