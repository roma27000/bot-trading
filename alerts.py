        st.download_button("📥 Télécharger le journal CSV", csv,
                           file_name="journal_signaux.csv", mime="text/csv")
else:
    st.info("Clique sur « Générer l'analyse du jour » pour produire le rapport complet.")

# ================= GRAPHIQUES (TradingView) =================
st.markdown("## 📈 Graphiques (H4 — unité de temps de la stratégie)")
TV_SYMBOL = {"BTC-USD": "BINANCE:BTCUSDT", "ETH-USD": "BINANCE:ETHUSDT", "SOL-USD": "BINANCE:SOLUSDT",
             "GC=F": "TVC:GOLD", "SI=F": "TVC:SILVER", "^GSPC": "OANDA:SPX500", "^NDX": "OANDA:NAS100"}
actif_chart = st.selectbox("Actif à afficher", list(TV_SYMBOL.keys()))
tv_id = "tv_" + actif_chart.replace("^", "").replace("=", "")
components.html(f"""
<div class="tradingview-widget-container" style="height:480px;">
  <div id="{tv_id}" style="height:480px;"></div>
  <script src="https://s3.tradingview.com/tv.js"></script>
  <script>
  new TradingView.widget({{
    "autosize": false, "width": "100%", "height": 480,
    "symbol": "{TV_SYMBOL[actif_chart]}", "interval": "240", "timezone": "Etc/UTC",
    "theme": "dark", "style": "1", "locale": "fr",
    "container_id": "{tv_id}"
  }});
  </script>
</div>""", height=500)
st.caption("Rappel : le graphique sert à visualiser ; la décision vient du rapport (niveaux, feux, qualité).")
