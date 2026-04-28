"""
pages/whatif.py — Simulation What-If + Pareto · baseline depuis DW réel.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data.loader import kpis_globaux


def _build_baseline(kpis: dict) -> dict:
    return {
        "retard_moyen":  kpis["retard_moyen"],
        "co2":           72.0,   # g/pkm moyen estimé (pas dans DW)
        "cout":          55.0,   # unités arbitraires
        "satisfaction":  kpis.get("satisfaction", 3.5),
        "demande":       kpis.get("nb_voyages", 8000),
    }


def render():
    st.markdown('<p class="section-title">🎲 Simulation What-If & Optimisation Pareto</p>',
                unsafe_allow_html=True)
    st.caption(
        '<span class="module-tag tag-optim">pymoo NSGA-II</span> &nbsp;'
        '<span class="module-tag tag-optim">Front de Pareto</span> &nbsp;'
        '<span class="module-tag tag-optim">Baseline DW réelle</span>',
        unsafe_allow_html=True)
    st.markdown("---")

    import datetime
    c1,c2 = st.columns(2)
    debut = str(c1.date_input("Début", datetime.date(2019, 1, 1), key="wi_d"))
    fin   = str(c2.date_input("Fin",   datetime.date(2022, 12, 31), key="wi_f"))

    # ── Charger la baseline depuis le DW réel ────────────────────────────
    try:
        kpis     = kpis_globaux(debut, fin)
        baseline = _build_baseline(kpis)
        st.markdown(f"""
        <div style="background:rgba(30,41,59,0.8);border:1px solid rgba(129,140,248,0.2);
                    border-radius:10px;padding:14px;margin-bottom:16px;">
          📡 <b style='color:#818cf8'>Baseline DW réelle</b> ·
          Retard moyen : <b>{baseline['retard_moyen']}s</b> ·
          Satisfaction : <b>{baseline['satisfaction']:.2f}/5</b> ·
          {kpis['nb_voyages']:,} voyages analysés
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Impossible de charger la baseline DW : {e}"); st.stop()

    tab1, tab2 = st.tabs(["🎬 Simulation de Scénario", "🎯 Optimisation Pareto"])

    with tab1:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### Paramètres du scénario")
            greve    = st.checkbox("🚨 Grève partielle (−50% bus)")
            pluie    = st.checkbox("🌧️ Épisode pluvieux")
            sport    = st.checkbox("⚽ Événement sportif")
            voie_bus = st.checkbox("🛣️ Voie bus dédiée (+efficacité)")
            st.markdown("---")
            d_bus  = st.slider("Variation fréquence bus (%)", -50, 50, 0) / 100
            d_tram = st.slider("Variation fréquence tram (%)", -30, 30, 0) / 100

            if st.button("▶️ Simuler", use_container_width=True):
                from models.whatif_optimizer import WhatIfOptimizer
                opt    = WhatIfOptimizer()
                params = {"greve":greve,"pluie":pluie,"evenement_sportif":sport,
                          "voie_bus_dediee":voie_bus,"freq_bus":d_bus,"freq_tram":d_tram}
                res    = opt.simulate_scenario(baseline, params)
                st.session_state["whatif_res"] = res

        with col_r:
            if "whatif_res" in st.session_state:
                res = st.session_state["whatif_res"]
                st.markdown("#### Résultats vs Baseline DW")
                for label, before, after, unit, lib in [
                    ("⏱ Retard moyen",   baseline["retard_moyen"], res["retard_moyen"],   "s",    True),
                    ("🌿 CO₂ estimé",     baseline["co2"],           res["co2"],            " g/pkm",True),
                    ("💰 Coût opérat.",   baseline["cout"],          res["cout"],           " u.a.",True),
                    ("⭐ Satisfaction",   baseline["satisfaction"],  res["satisfaction"],   "/5",  False),
                ]:
                    delta = after - before
                    ok    = (delta < 0) == lib
                    color = "#34d399" if ok else "#ef4444"
                    st.markdown(
                        f"**{label}** &nbsp; {before:.1f}{unit} → "
                        f"<b style='color:{color}'>{after:.1f}{unit}</b> "
                        f"<small style='color:{color}'>({'▼' if delta<0 else '▲'}{abs(delta):.1f}{unit})</small>",
                        unsafe_allow_html=True)

                # Gauge retard
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=res["retard_moyen"],
                    delta={"reference": baseline["retard_moyen"]},
                    gauge={"axis":{"range":[0,max(900,res["retard_moyen"]*1.5)]},
                           "bar":{"color":"#818cf8"},
                           "steps":[{"range":[0,120],"color":"rgba(52,211,153,0.2)"},
                                    {"range":[120,300],"color":"rgba(245,158,11,0.2)"},
                                    {"range":[300,900],"color":"rgba(239,68,68,0.2)"}]},
                    title={"text":"Retard moyen (s)","font":{"color":"#94a3b8"}},
                    number={"font":{"color":"#818cf8"}},
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=240,
                                   margin=dict(l=20,r=20,t=30,b=10),
                                   font={"color":"#94a3b8"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Configurez un scénario et cliquez **Simuler**.")

    with tab2:
        st.markdown("#### Front de Pareto · Baseline issue du DW réel")
        st.caption(
            f"Baseline : retard={baseline['retard_moyen']}s · "
            f"satisfaction={baseline['satisfaction']:.2f}/5"
        )
        if st.button("🔬 Calculer le Front de Pareto (NSGA-II)", use_container_width=False):
            from models.whatif_optimizer import WhatIfOptimizer
            opt = WhatIfOptimizer()
            with st.spinner("Optimisation NSGA-II en cours…"):
                pareto_df = opt.run_pareto(baseline, n_gen=80)
                st.session_state["pareto_df"]  = pareto_df
                st.session_state["pareto_opt"] = opt

        if "pareto_df" in st.session_state:
            df = st.session_state["pareto_df"]
            opt = st.session_state["pareto_opt"]

            F_norm = (df[["CO2 (g/pkm)","Coût (u.a.)","Retard moy. (s)"]].values
                      if all(c in df.columns for c in ["CO2 (g/pkm)","Coût (u.a.)","Retard moy. (s)"])
                      else df.values)
            F_norm = (F_norm - F_norm.min(0)) / (F_norm.max(0) - F_norm.min(0) + 1e-9)
            best   = int(np.argmin(np.linalg.norm(F_norm, axis=1)))

            fig = px.scatter_3d(
                df, x="CO2 (g/pkm)", y="Coût (u.a.)", z="Retard moy. (s)",
                color="CO2 (g/pkm)", color_continuous_scale="Viridis",
                opacity=0.7, template="plotly_dark",
            )
            bp = df.iloc[best]
            fig.add_trace(go.Scatter3d(
                x=[bp["CO2 (g/pkm)"]], y=[bp["Coût (u.a.)"]],
                z=[bp["Retard moy. (s)"]],
                mode="markers",
                marker=dict(size=12,color="#ef4444",symbol="diamond"),
                name="★ Meilleur compromis",
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=480,
                               margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)

            comp = opt.get_best_compromise()
            if comp:
                st.markdown("#### ★ Solution optimale recommandée")
                c1,c2,c3 = st.columns(3)
                c1.metric("CO₂", f"{comp['CO2']:.1f} g/pkm",
                           delta=f"{comp['CO2']-baseline['co2']:+.1f}")
                c2.metric("Coût", f"{comp['cout']:.1f} u.a.",
                           delta=f"{comp['cout']-baseline['cout']:+.1f}")
                c3.metric("Retard", f"{comp['retard']:.0f}s",
                           delta=f"{comp['retard']-baseline['retard_moyen']:+.0f}s",
                           delta_color="inverse")
