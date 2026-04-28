"""
pages/graph.py — Graphe Spatio-Temporel · topologie reelle DW.
Correction : sub -> filt + robustesse zone_nom
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data.loader import load_dim_arrets, load_trafic


@st.cache_resource(show_spinner="Construction du graphe reseau…")
def get_graph(debut: str, fin: str):
    from models.graph_model import TransportGraph
    topo   = load_dim_arrets()
    trafic = load_trafic(debut, fin)
    G = TransportGraph()
    G.build_from_dw(topo, trafic)
    return G, topo


ZONE_PALETTE = [
    "#818cf8", "#34d399", "#f59e0b", "#ef4444", "#ec4899",
    "#06b6d4", "#a78bfa", "#fbbf24", "#4ade80", "#fb923c",
]


def render():
    st.markdown('<p class="section-title">🕸️ Graphe Spatio-Temporel du Reseau</p>',
                unsafe_allow_html=True)
    st.caption(
        '<span class="module-tag tag-graph">NetworkX</span> &nbsp;'
        '<span class="module-tag tag-graph">Propagation Retard</span> &nbsp;'
        '<span class="module-tag tag-graph">Centralite Betweenness</span>',
        unsafe_allow_html=True)
    st.markdown("---")

    import datetime
    c1, c2 = st.columns(2)
    debut = str(c1.date_input("Debut", datetime.date(2019, 1, 1),  key="gr_d"))
    fin   = str(c2.date_input("Fin",   datetime.date(2022, 12, 31), key="gr_f"))

    if st.button("🔧 Construire / Recharger le Graphe", use_container_width=False):
        try:
            G, topo = get_graph(debut, fin)
            st.session_state["graph"] = G
            st.session_state["topo"]  = topo
        except Exception as e:
            st.error(f"Erreur construction graphe : {e}"); st.stop()

    if "graph" not in st.session_state:
        st.info("Cliquez sur **Construire le Graphe** pour charger les donnees DW.")
        return

    G    = st.session_state["graph"]
    topo = st.session_state["topo"]

    # Metriques reseau
    c1, c2, c3 = st.columns(3)
    c1.metric("Noeuds (arrets)",    G.G.number_of_nodes())
    c2.metric("Aretes (liaisons)",  G.G.number_of_edges())
    c3.metric("Zones",
              topo["zone_nom"].nunique()
              if "zone_nom" in topo.columns else "—")

    st.markdown("---")

    # ── Filtres ───────────────────────────────────────────────────────────────
    # FIX : verifier que zone_nom existe avant d'utiliser
    if "zone_nom" in topo.columns:
        zones_dispo = topo["zone_nom"].dropna().unique().tolist()
    else:
        zones_dispo = []

    col_ctrl, col_g = st.columns([1, 3])

    with col_ctrl:
        st.markdown("**Filtres**")
        if zones_dispo:
            zones_sel = st.multiselect("Zone", zones_dispo,
                                        default=zones_dispo[:4])
        else:
            zones_sel = []
            st.info("Aucune zone disponible.")

        show_prop = st.checkbox("Simuler propagation retard")
        if show_prop:
            nodes_list = list(G.G.nodes())
            src_node   = st.selectbox("Arret source", nodes_list[:50])
            delay_inj  = st.slider("Retard injecte (s)", 60, 1800, 300)

    with col_g:
        # FIX : utiliser 'filt' (pas 'sub' qui n'existe pas)
        filt = topo.copy()
        if zones_sel and "zone_nom" in filt.columns:
            filt = filt[filt["zone_nom"].isin(zones_sel)]

        # FIX : verifier que filt n'est pas vide avant de continuer
        if filt.empty:
            st.warning("Aucun arret disponible pour les zones selectionnees.")
            return

        valid_nodes = set(filt["arret_id"].tolist())

        import numpy as np
        rng = np.random.default_rng(42)
        pos = {n: (rng.uniform(0, 10), rng.uniform(0, 10))
               for n in G.G.nodes()}

        fig = go.Figure()

        # Aretes
        for u, v, data in G.G.edges(data=True):
            if u in valid_nodes and v in valid_nodes:
                x0, y0 = pos[u]; x1, y1 = pos[v]
                cong = data.get("congestion", 0)
                fig.add_trace(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode="lines",
                    line=dict(color=f"rgba(239,68,68,{0.2+cong*0.6})",
                              width=0.8 + cong * 2.5),
                    showlegend=False, hoverinfo="skip",
                ))

        # Propagation retard
        prop_dict = {}
        if show_prop and src_node in G.G.nodes():
            prop_dict = G.propagate_delay(src_node, delay_inj)

        # FIX : zone_list depuis 'filt' (pas 'sub')
        if "zone_nom" in filt.columns and not filt["zone_nom"].dropna().empty:
            zone_list = sorted(filt["zone_nom"].dropna().unique().tolist())
        else:
            zone_list = []

        zone_color_map = {z: ZONE_PALETTE[i % len(ZONE_PALETTE)]
                          for i, z in enumerate(zone_list)}

        # FIX : utiliser 'filt' (pas 'sub') pour le groupby
        if not filt.empty and "zone_nom" in filt.columns:
            for zone_nom, grp in filt.groupby("zone_nom"):
                xs, ys, hover, sizes = [], [], [], []
                for _, row in grp.iterrows():
                    nid = row["arret_id"]
                    if nid not in pos:
                        continue
                    xs.append(pos[nid][0])
                    ys.append(pos[nid][1])
                    sizes.append(12)
                    hover.append(
                        f"Arret {nid} | Zone: {row.get('zone_nom','?')} "
                        f"| {row.get('stop_nom','')}"
                    )
                color = zone_color_map.get(zone_nom, "#818cf8")
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers",
                    name=str(zone_nom),
                    marker=dict(size=sizes, color=color, opacity=0.85,
                                line=dict(width=1, color="rgba(255,255,255,0.3)")),
                    hovertext=hover, hoverinfo="text",
                ))
        elif not filt.empty:
            # Pas de zone_nom — afficher tous les noeuds sans groupe
            xs, ys, hover = [], [], []
            for _, row in filt.iterrows():
                nid = row["arret_id"]
                if nid not in pos:
                    continue
                xs.append(pos[nid][0])
                ys.append(pos[nid][1])
                hover.append(f"Arret {nid}")
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                name="Arrets",
                marker=dict(size=10, color="#818cf8", opacity=0.85),
                hovertext=hover, hoverinfo="text",
            ))

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=480,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            legend=dict(font=dict(color="#94a3b8")),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Points critiques ──────────────────────────────────────────────────────
    st.markdown("#### Top 10 Arrets Critiques (Centralite Betweenness)")
    try:
        crit = G.critical_nodes(10)
        st.dataframe(
            crit.style.background_gradient(subset=["betweenness"], cmap="Purples"),
            use_container_width=True, hide_index=True,
        )
    except Exception as e:
        st.warning(f"Calcul centralite non disponible : {e}")