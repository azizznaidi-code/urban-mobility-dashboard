"""
pages/report.py — Rapport XAI · données réelles DW.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

from data.loader import (
    kpis_globaux, retard_par_zone, retard_par_ligne,
    load_accidents,
)


def render():
    st.markdown('<p class="section-title">📊 Rapport Explicable (XAI) · DW réel</p>',
                unsafe_allow_html=True)
    st.caption(
        '<span class="module-tag tag-ml">SHAP Summary</span> &nbsp;'
        '<span class="module-tag tag-ml">Feature Importance</span> &nbsp;'
        '<span class="module-tag tag-ml">Incidents</span>',
        unsafe_allow_html=True)
    st.markdown("---")

    c1, c2 = st.columns(2)
    debut = str(c1.date_input("Du",  datetime.date(2019, 1, 1)))
    fin   = str(c2.date_input("Au",  datetime.date(2022, 12, 31)))

    if st.button("📄 Générer le rapport", use_container_width=False):
        # ── Charger toutes les données DW ────────────────────────────────
        try:
            kpis    = kpis_globaux(debut, fin)
            by_zone = retard_par_zone(debut, fin)
            by_mode = retard_par_ligne(debut, fin)
            df_inc  = load_accidents()
        except Exception as e:
            st.error(f"Erreur chargement DW : {e}")
            st.stop()

        st.success("Rapport généré depuis **dwurbanmobility** ✅")
        st.markdown("---")

        # ── Résumé exécutif ───────────────────────────────────────────────
        mode_pire = by_mode.nlargest(1, "retard_moy")["ligne_id"].iloc[0] if not by_mode.empty else "—"
        zone_pire = by_zone.nlargest(1, "retard_moy")["zone_nom"].iloc[0] if not by_zone.empty else "—"
        charge_moy = kpis.get("charge_moyenne", "N/A")
        charge_str = f"{charge_moy:.2f}/5" if isinstance(charge_moy, (int, float)) else "N/A"

        st.markdown(f"""
        <div style="background:rgba(30,41,59,0.9);border:1px solid rgba(129,140,248,0.3);
                    border-radius:12px;padding:24px;">
          <h3 style="color:#f1f5f9">📋 Rapport UrbanAI · {debut} → {fin}</h3>
          <hr style="border-color:rgba(99,102,241,0.2)"/>
          <b style="color:#818cf8">Résumé exécutif (données DW réelles)</b>
          <ul style="color:#94a3b8;margin-top:8px">
            <li>Retard moyen : <b style='color:#e2e8f0'>{kpis['retard_moyen']}s</b>
                (médiane {kpis['retard_median']}s)</li>
            <li>Taux d'annulation : <b style='color:#e2e8f0'>{kpis['taux_annulation']:.1f}%</b></li>
            <li>Satisfaction usager : <b style='color:#e2e8f0'>{charge_str}</b></li>
            <li>Mode le plus retardé : <b style='color:#ef4444'>{mode_pire}</b></li>
            <li>Zone la plus impactée : <b style='color:#ef4444'>{zone_pire}</b></li>
            <li>Congestion moyenne : <b style='color:#e2e8f0'>{kpis['congestion_moy']:.3f}</b>
                · Vitesse moy : {kpis['vitesse_moy_kmh']} km/h</li>
          </ul>
        </div>""", unsafe_allow_html=True)

        # ── Retard par zone ────────────────────────────────────────────────
        st.markdown("#### Retard moyen par Zone (Dim_Zone × facttransport)")
        if not by_zone.empty:
            fig1 = px.bar(by_zone.sort_values("retard_moy", ascending=True),
                          x="retard_moy", y="zone_nom", orientation="h",
                          color="retard_moy", color_continuous_scale="Reds",
                          labels={"retard_moy": "Retard (s)", "zone_nom": "Zone"})
            fig1.update_layout(template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)", height=280,
                                margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        # ── Incidents ──────────────────────────────────────────────────────
        if not df_inc.empty:
            st.markdown("#### Répartition des Incidents (factaccidents)")
            col_inc = "type_incident" if "type_incident" in df_inc.columns else df_inc.columns[0]
            inc_cat = df_inc[col_inc].value_counts().reset_index()
            inc_cat.columns = ["categorie", "count"]
            fig2 = px.pie(inc_cat, values="count", names="categorie",
                          hole=0.4,
                          color_discrete_sequence=["#818cf8", "#34d399", "#f59e0b",
                                                    "#ef4444", "#a78bfa", "#60a5fa"])
            fig2.update_layout(template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)", height=300,
                                margin=dict(l=0, r=0, t=10, b=0),
                                legend=dict(font=dict(color="#94a3b8")))
            st.plotly_chart(fig2, use_container_width=True)

        # ── Export ─────────────────────────────────────────────────────────
        report_dict = {
            "kpis":            kpis,
            "retard_par_zone": by_zone.to_dict(),
            "retard_par_mode": by_mode.to_dict(),
        }
        import json
        st.download_button(
            "⬇️ Télécharger le rapport (JSON)",
            data=json.dumps(report_dict, default=str, ensure_ascii=False),
            file_name=f"rapport_urbanai_{debut}_{fin}.json",
            mime="application/json",
        )