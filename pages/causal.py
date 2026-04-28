"""
pages/causal.py — IA Causale · données réelles DW dwurbanmobility.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data.loader import load_transport, load_trafic, load_accidents


@st.cache_resource(show_spinner="Analyse causale en cours…")
def get_causal_analyzer(debut: str, fin: str):
    from models.causal_ai import CausalAnalyzer
    # Fusionner transport + trafic sur zone_id
    df_t  = load_transport(debut, fin)
    df_tr = load_trafic(debut, fin)

    # facttransport n'a pas zone_id ni mode_id — on joint via time_id/heure
    df_t_agg  = df_t[["retard_s","annule","charge_estimee"]].copy()
    df_tr_agg = df_tr[["zone_id","congestion_index","vitesse_kmh",
                        "meteo_id","event_id","zone_nom"]].copy()

    # Réinitialiser les index pour concat simple
    df_t_agg  = df_t_agg.reset_index(drop=True)
    df_tr_agg = df_tr_agg.reset_index(drop=True)

    # Prendre le minimum des deux longueurs
    n = min(len(df_t_agg), len(df_tr_agg))
    common = pd.concat([df_t_agg.iloc[:n], df_tr_agg.iloc[:n]], axis=1)

    common = common.rename(columns={
        "meteo_id": "fkmeteo",
        "event_id": "fkevent",
    })
    common["FK_mode"] = 0  # pas de mode dans facttransport
    common["categorie_incident"] = 0

    if len(common) < 50:
        raise ValueError(f"Seulement {len(common)} lignes — insuffisant.")

    # Incidents : ajouter categorie
    try:
        df_inc = load_accidents()
        if not df_inc.empty:
            inc_agg = df_inc.groupby("zone_id")["categorie_incident"].first().reset_index()
            common  = pd.merge(common, inc_agg, on="zone_id", how="left")
    except Exception:
        common["categorie_incident"] = 0

    analyzer = CausalAnalyzer()
    analyzer.fit(common)
    return analyzer, common


def render():
    st.markdown('<p class="section-title">🧠 IA Causale · Comprendre le Pourquoi</p>',
                unsafe_allow_html=True)
    st.caption(
        '<span class="module-tag tag-causal">DoWhy</span> &nbsp;'
        '<span class="module-tag tag-causal">DAG Causal</span> &nbsp;'
        '<span class="module-tag tag-causal">Effets Contrefactuels</span>',
        unsafe_allow_html=True)
    st.markdown("---")

    import datetime
    c1,c2 = st.columns(2)
    debut = str(c1.date_input("Début", datetime.date(2019, 1, 1), key="ca_d"))
    fin   = str(c2.date_input("Fin",   datetime.date(2022, 12, 31), key="ca_f"))

    if st.button("🔬 Lancer l'analyse causale", use_container_width=False):
        try:
            analyzer, df_merged = get_causal_analyzer(debut, fin)
            st.session_state["causal_analyzer"] = analyzer
            st.session_state["df_causal"]        = df_merged
            st.success(f"✅ Analyse causale sur {len(df_merged):,} observations réelles du DW.")
        except Exception as e:
            st.error(f"Erreur analyse causale : {e}"); st.stop()

    if "causal_analyzer" not in st.session_state:
        st.info("Cliquez sur **Lancer l'analyse causale** pour charger les données DW.")
        return

    analyzer  = st.session_state["causal_analyzer"]
    df_merged = st.session_state["df_causal"]

    tab1, tab2, tab3 = st.tabs([
        "📊 Attribution Causale",
        "🔬 Analyser un Trajet",
        "🗺️ DAG Causal",
    ])

    # ── Tab 1 : Attribution globale ──────────────────────────────────────
    with tab1:
        st.markdown("#### Décomposition causale des retards (données DW réelles)")
        attrib = analyzer.attribution_breakdown(df_merged)

        col_l, col_r = st.columns([3,2])
        with col_l:
            labels = list(attrib.keys()); values = list(attrib.values())
            COLORS = {"Météo":"#60a5fa","Événement urbain":"#f59e0b",
                      "Incident / Accident":"#f87171","Congestion routière":"#818cf8",
                      "Zone géographique":"#34d399","Annulation":"#a78bfa",
                      "Mode de transport":"#94a3b8","Cause inconnue":"#64748b"}
            fig = go.Figure(go.Bar(
                x=values, y=labels, orientation="h",
                marker_color=[COLORS.get(l,"#64748b") for l in labels],
                text=[f"{v:.1f}%" for v in values], textposition="outside",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=max(200,len(labels)*40+40),
                margin=dict(l=0,r=60,t=10,b=0), xaxis_title="Attribution (%)",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            fig2 = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.5,
                marker_colors=[COLORS.get(l,"#64748b") for l in labels],
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", template="plotly_dark",
                height=max(200,len(labels)*40+40),
                margin=dict(l=0,r=0,t=10,b=0),
                legend=dict(font=dict(color="#94a3b8",size=11)),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Effet DoWhy si disponible
        val = getattr(analyzer.result, "value", None) if getattr(analyzer, "result", None) else None
        if val is not None:
            st.markdown("#### Effet causal estimé (DoWhy · données DW)")
            st.markdown(f"""
            <div style="background:rgba(30,41,59,0.9);border:1px solid rgba(129,140,248,0.3);
                        border-radius:12px;padding:20px;">
              <table style="width:100%;color:#94a3b8;font-size:.9rem">
                <tr><td><b style='color:#e2e8f0'>Traitement</b></td><td>congestion_index</td></tr>
                <tr><td><b style='color:#e2e8f0'>Résultat</b></td><td>retard_s</td></tr>
                <tr><td><b style='color:#e2e8f0'>Effet (ATE)</b></td>
                    <td><b style='color:#818cf8;font-size:1.1rem'>
                    {val:.2f} s / unité</b></td></tr>
              </table>
            </div>""", unsafe_allow_html=True)
            ref = analyzer.refutation_test()
            color = "#34d399" if ref.get("valid") else "#ef4444"
            st.markdown(f"**Test de réfutation** : "
                        f"<span style='color:{color}'>{ref['status']}</span>",
                        unsafe_allow_html=True)

    # ── Tab 2 : Trajet individuel ─────────────────────────────────────────
    with tab2:
        st.markdown("#### Analyser un trajet (sélectionner depuis les données DW)")
        idx = st.slider("Index observation", 0, len(df_merged)-1, 0)
        row = df_merged.iloc[idx]

        st.dataframe(row.to_frame().T, use_container_width=True)

        expl = analyzer.explain_single_trip(row)
        color = "#ef4444" if expl["severite"]=="CRITIQUE" else (
                "#f59e0b" if expl["severite"]=="ÉLEVÉ" else "#fbbf24")

        st.markdown(f"""
        <div style="background:rgba(30,41,59,0.9);border:1px solid {color}44;
                    border-radius:12px;padding:16px;margin:12px 0">
          <b style="color:{color}">⚠️ {expl['severite']}</b> ·
          Retard = <b>{expl['retard_s']:.0f}s</b><br>
          Cause principale : <b style="color:#818cf8">{expl['cause_principale']}</b>
          ({expl['cause_weights'].get(expl['cause_principale'],0):.1f}%)
        </div>""", unsafe_allow_html=True)

        cause_rows = [{"Cause":k,"Attribution (%)":v,
                       "Impact estimé (s)":round(expl["retard_s"]*v/100)}
                      for k,v in sorted(expl["cause_weights"].items(),key=lambda x:-x[1])]
        st.dataframe(pd.DataFrame(cause_rows), use_container_width=True, hide_index=True)

    # ── Tab 3 : DAG ───────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### DAG Causal — Relations extraites du DW dwurbanmobility")
        from models.causal_ai import CAUSAL_GRAPH, CAUSE_LABELS

        pos = {"fkmeteo":(0.1,0.9),"fkevent":(0.35,0.95),
               "categorie_incident":(0.6,0.88),"fkzone_id":(0.85,0.8),
               "FK_mode":(0.95,0.5),"congestion_index":(0.5,0.6),
               "retard_s":(0.5,0.35),"charge_estimee":(0.25,0.1),
               "retard_s":(0.75,0.1)}
        NCOLORS = {"fkmeteo":"#60a5fa","fkevent":"#f59e0b",
                   "categorie_incident":"#f87171","fkzone_id":"#34d399",
                   "FK_mode":"#a78bfa","congestion_index":"#818cf8",
                   "retard_s":"#ef4444","charge_estimee":"#34d399",
                   "retard_s":"#f87171"}
        EDGES = [("fkmeteo","retard_s"),("fkevent","retard_s"),
                 ("categorie_incident","retard_s"),("congestion_index","retard_s"),
                 ("fkzone_id","retard_s"),("fkzone_id","congestion_index"),
                 ("FK_mode","retard_s"),("FK_mode","congestion_index"),
                 ("retard_s","charge_estimee"),("retard_s","retard_s"),
                 ("congestion_index","charge_estimee")]

        fig3 = go.Figure()
        for src,dst in EDGES:
            if src in pos and dst in pos:
                x0,y0=pos[src]; x1,y1=pos[dst]
                fig3.add_annotation(ax=x0,ay=y0,x=x1,y=y1,
                    xref="x",yref="y",axref="x",ayref="y",
                    arrowhead=3,arrowsize=1.5,arrowwidth=1.5,arrowcolor="#475569")
        for node,(x,y) in pos.items():
            label = CAUSE_LABELS.get(node, node)
            fig3.add_trace(go.Scatter(
                x=[x],y=[y],mode="markers+text",
                marker=dict(size=28,color=NCOLORS.get(node,"#64748b")),
                text=[label.split("/")[0][:12]],
                textposition="middle center",
                textfont=dict(size=9,color="white"),
                showlegend=False,
                hovertext=[f"{label}"],hoverinfo="text",
            ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            height=420,margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(showgrid=False,zeroline=False,visible=False,range=[-0.05,1.05]),
            yaxis=dict(showgrid=False,zeroline=False,visible=False,range=[-0.05,1.05]),
        )
        st.plotly_chart(fig3, use_container_width=True)
