import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import datetime

from data.loader import (
    kpis_globaux, retard_journalier, retard_par_zone,
    congestion_heure_zone, load_accidents,
)


def render():
    st.markdown('<p class="section-title">🏠 Tableau de Bord · Vue Globale du Réseau</p>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 📅 Période d'analyse")
        debut = st.date_input("Début", value=datetime.date(2019, 1, 1), key="sb_debut")
        fin   = st.date_input("Fin",   value=datetime.date(2022, 12, 31), key="sb_fin")
    debut_str, fin_str = str(debut), str(fin)

    try:
        kpis   = kpis_globaux(debut_str, fin_str)
        daily  = retard_journalier(debut_str, fin_str)
        zones  = retard_par_zone(debut_str, fin_str)
        heatdf = congestion_heure_zone(debut_str, fin_str)
    except Exception as e:
        st.error(f"❌ Connexion DW impossible : **{e}**")
        st.stop()

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, icon, label, val, sub in [
        (c1,"⏱️","Retard moy.",
         f"{kpis['retard_moyen']}s",
         f"médiane {kpis['retard_median']}s"),
        (c2,"👥","Charge moy.",
         f"{kpis['charge_moyenne']} passagers", ""),
        (c3,"🚦","Congestion",
         f"{kpis['congestion_moy']:.3f}", ""),
        (c4,"🚗","Vitesse moy.",
         f"{kpis['vitesse_moy_kmh']} km/h", ""),
        (c5,"❌","Taux annulation",
         f"{kpis['taux_annulation']:.1f}%", ""),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div style="font-size:1.4rem">{icon}</div>
          <div class="metric-value">{val}</div>
          <div class="metric-label">{label}</div>
          <div style="font-size:.73rem;color:#475569;margin-top:4px">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        f"<br><small style='color:#475569'>📦 <b>dwurbanmobility</b> · "
        f"{kpis['nb_voyages']:,} voyages · {debut_str} → {fin_str}</small>",
        unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### Évolution du Retard Journalier Moyen")
    if not daily.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["jour"], y=daily["retard_moy"],
            fill="tozeroy", line=dict(color="#818cf8", width=2),
            fillcolor="rgba(129,140,248,0.1)", name="Retard moyen (s)",
        ))
        fig.add_hline(y=120, line_dash="dash", line_color="#ef4444",
                      annotation_text="Seuil 120s")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=260,
            margin=dict(l=0,r=0,t=10,b=0),
            xaxis_title="Date", yaxis_title="Retard (s)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucune donnée de retard sur cette période.")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Retard moyen par Zone")
        if not zones.empty:
            fig2 = go.Figure(go.Bar(
                x=zones["zone_nom"], y=zones["retard_moy"],
                marker_color="#818cf8",
                text=zones["retard_moy"].round(1), textposition="outside",
            ))
            fig2.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=280,
                margin=dict(l=0,r=0,t=10,b=0), yaxis_title="Retard (s)",
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.markdown("#### Heatmap Congestion · Heure × Zone")
        if not heatdf.empty:
            fig3 = px.imshow(heatdf, color_continuous_scale="Blues",
                             labels=dict(x="Heure",y="Zone",color="Congestion"),
                             aspect="auto")
            fig3.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=280,
                margin=dict(l=0,r=0,t=10,b=0),
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Pas assez de données pour la heatmap.")

    st.markdown("#### Top Incidents (factaccidents)")
    try:
        df_inc = load_accidents()
        if not df_inc.empty:
            top = (df_inc.nlargest(10, "severity")
                   [["zone_nom","type_incident","severity",
                     "usager_vulnerable","nb_crimes"]]
                   .rename(columns={
                       "zone_nom":          "Zone",
                       "type_incident":     "Type Incident",
                       "severity":          "Sévérité",
                       "usager_vulnerable": "Vulnérable",
                       "nb_crimes":         "Nb Crimes",
                   }))
            st.dataframe(
                top.style.background_gradient(subset=["Sévérité"], cmap="Reds"),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Aucun incident trouvé.")
    except Exception as e:
        st.warning(f"factaccidents : {e}")