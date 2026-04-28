"""
pages/recommandation.py — Système de Recommandation d'Itinéraires Urbains
Corrigé :
  Fix 1 — Seuils congestion adaptés au KPI IC réel (0-4+) : 1.5 Modéré · 2.5 Critique
  Fix 2 — Poids Dijkstra recalculés par heure (profils horaires réels)
  Fix 3 — Temps total calculé depuis profils horaires réels (fallback TTM KPI 17.8 min)
  Fix 4 — Suppression @st.cache_data sur _build_zone_graph pour forcer recalcul
  Fix 5 — Message explicatif si chemin direct (2 zones seulement)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime

from data.loader import load_trafic, load_dim_zones


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Calcul des profils de zones…")
def _build_zone_profiles(debut: str, fin: str) -> pd.DataFrame:
    """Profil moyen par zone × heure depuis facttrfic."""
    df = load_trafic(debut, fin)

    # Heure numérique — robuste aux formats string et datetime
    try:
        df["heure_n"] = pd.to_datetime(
            df["heure"].astype(str).str[:8], format="%H:%M:%S", errors="coerce"
        ).dt.hour
    except Exception:
        df["heure_n"] = None

    # Fallback si parsing échoue
    if df["heure_n"].isna().all():
        try:
            df["heure_n"] = df["heure"].astype(str).str[:2].astype(int)
        except Exception:
            df["heure_n"] = (df.index % 24)

    df["heure_n"] = df["heure_n"].fillna(0).astype(int)

    # S'assurer que zone_id existe
    zone_col = "zone_id" if "zone_id" in df.columns else "fkzone_id"

    agg = (df.groupby([zone_col, "heure_n"])
             .agg(
                 congestion_moy=("congestion_index", "mean"),
                 vitesse_moy=("vitesse_kmh", "mean"),
                 temps_moy=("temps_trajet_min", "mean"),
             )
             .reset_index()
             .fillna(0))

    # Normaliser le nom de colonne zone
    if zone_col != "zone_id":
        agg = agg.rename(columns={zone_col: "zone_id"})

    return agg


# FIX 4 — Pas de @st.cache_data : les poids varient selon l'heure
def _build_zone_graph(debut: str, fin: str):
    """Graphe complet entre zones — poids = congestion × temps moyen."""
    import networkx as nx
    df = load_trafic(debut, fin)
    dz = load_dim_zones()

    # Colonnes disponibles dans dz
    dz_cols = ["zone_id", "zone_nom"]
    for c in ["lat_centre", "lon_centre"]:
        if c in dz.columns:
            dz_cols.append(c)
    dz = dz[dz_cols].copy()

    # Colonne zone dans facttrfic
    zone_col = "zone_id" if "zone_id" in df.columns else "fkzone_id"

    zone_stats = (df.groupby(zone_col)
                  .agg(cong=("congestion_index", "mean"),
                       tps=("temps_trajet_min", "mean"))
                  .reset_index()
                  .fillna(0))
    zone_stats = zone_stats.rename(columns={zone_col: "zone_id"})
    zone_stats = zone_stats.reset_index(drop=True)

    G = nx.DiGraph()
    zones = zone_stats["zone_id"].tolist()

    for z in zones:
        row  = dz[dz["zone_id"] == z]
        name = str(row["zone_nom"].values[0])   if not row.empty else str(z)
        lat  = float(row["lat_centre"].values[0]) if (not row.empty and "lat_centre" in dz.columns) else 0.0
        lon  = float(row["lon_centre"].values[0]) if (not row.empty and "lon_centre" in dz.columns) else 0.0
        G.add_node(int(z), label=name, lat=lat, lon=lon)

    # Arêtes avec poids de base
    for zi in zones:
        for zj in zones:
            if zi == zj:
                continue
            ci_row = zone_stats[zone_stats["zone_id"] == zi]["cong"]
            tj_row = zone_stats[zone_stats["zone_id"] == zj]["tps"]
            ci = float(ci_row.values[0]) if len(ci_row) > 0 else 0.0
            tj = float(tj_row.values[0]) if len(tj_row) > 0 else 0.0
            weight = max(0.01, ci * tj + 1)
            G.add_edge(int(zi), int(zj), weight=weight, congestion=ci, temps=tj)

    return G, dz


# FIX 2 — Poids Dijkstra recalculés par heure
def _find_best_path(G, source_id, target_id, heure: int, profiles: pd.DataFrame):
    """Dijkstra avec poids recalculés à partir des profils horaires réels."""
    import networkx as nx

    G_h = G.copy()

    for (u, v) in G_h.edges():
        prof_u = profiles[(profiles["zone_id"] == u) & (profiles["heure_n"] == heure)]
        prof_v = profiles[(profiles["zone_id"] == v) & (profiles["heure_n"] == heure)]

        cong_u = float(prof_u["congestion_moy"].values[0]) if not prof_u.empty else G[u][v]["congestion"]
        tps_v  = float(prof_v["temps_moy"].values[0])      if not prof_v.empty else G[u][v]["temps"]

        # Poids horaire : plus congestionné → plus évité
        G_h[u][v]["weight_h"] = max(0.01, cong_u * tps_v + 1)

    try:
        path   = nx.dijkstra_path(G_h, source_id, target_id, weight="weight_h")
        length = nx.dijkstra_path_length(G_h, source_id, target_id, weight="weight_h")
        return path, length
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────────────────────

def render():
    st.markdown('<p class="section-title">🗺️ Recommandation d\'Itinéraire Optimal</p>',
                unsafe_allow_html=True)
    st.caption(
        '<span class="module-tag tag-graph">Content-Based</span> &nbsp;'
        '<span class="module-tag tag-graph">Dijkstra Pondéré</span> &nbsp;'
        '<span class="module-tag tag-graph">Profils de Zones</span>',
        unsafe_allow_html=True)
    st.markdown("---")

    c1, c2 = st.columns(2)
    debut = str(c1.date_input("Début", datetime.date(2019, 1, 1), key="rec_d"))
    fin   = str(c2.date_input("Fin",   datetime.date(2022, 12, 31), key="rec_f"))

    # Charger zones
    try:
        dz = load_dim_zones()
    except Exception as e:
        st.error(f"Erreur chargement zones : {e}"); return

    zone_map  = dict(zip(dz["zone_nom"], dz["zone_id"]))
    zone_noms = sorted(dz["zone_nom"].dropna().tolist())

    with st.expander("⚙️ Paramètres du trajet", expanded=True):
        rc1, rc2, rc3 = st.columns(3)
        zone_dep  = rc1.selectbox("Zone de départ",  zone_noms, key="rec_dep",
                                   index=0)
        zone_arr  = rc2.selectbox("Zone d'arrivée", zone_noms, key="rec_arr",
                                   index=min(1, len(zone_noms) - 1))
        heure_sel = rc3.slider("Heure du voyage", 0, 23, 8, key="rec_h")

    rec_btn = st.button("🔍 Trouver le meilleur itinéraire", use_container_width=True)

    if not rec_btn and "rec_results" not in st.session_state:
        st.info("Sélectionnez une zone de départ, d'arrivée et cliquez sur **Trouver**.")
        st.markdown("---")
        st.markdown("#### Profil de congestion par Zone × Heure")
        try:
            profiles = _build_zone_profiles(debut, fin)
            pivot = (profiles.pivot_table(
                index="zone_id", columns="heure_n",
                values="congestion_moy", aggfunc="mean"
            ).round(3))
            pivot.index = pivot.index.map(
                lambda zid: dz.loc[dz["zone_id"] == zid, "zone_nom"].values[0]
                if zid in dz["zone_id"].values else str(zid)
            )
            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[f"{h}h" for h in pivot.columns],
                y=pivot.index.tolist(),
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(tickfont=dict(color="#94a3b8")),
            ))
            fig_heat.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=max(200, len(pivot) * 35),
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Heure", yaxis_title="Zone",
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception as e:
            st.info(f"Profils non disponibles : {e}")
        return

    if rec_btn:
        if zone_dep == zone_arr:
            st.warning("Zone de départ et d'arrivée identiques."); return
        try:
            with st.spinner("Construction du graphe de zones..."):
                profiles = _build_zone_profiles(debut, fin)
                G, dz2   = _build_zone_graph(debut, fin)
        except Exception as e:
            st.error(f"Erreur construction graphe : {e}")
            import traceback; st.code(traceback.format_exc()); return

        src_id = zone_map.get(zone_dep)
        dst_id = zone_map.get(zone_arr)
        if src_id is None or dst_id is None:
            st.error("Zone introuvable dans le DW."); return

        src_id = int(src_id)
        dst_id = int(dst_id)

        if src_id not in G.nodes:
            st.error(f"Zone départ (id={src_id}) absente du graphe trafic. Vérifier facttrfic.")
            st.info(f"Zones disponibles dans le graphe : {sorted(G.nodes())[:20]}")
            return
        if dst_id not in G.nodes:
            st.error(f"Zone arrivée (id={dst_id}) absente du graphe trafic.")
            return

        path, length = _find_best_path(G, src_id, dst_id, heure_sel, profiles)

        if path is None:
            st.error("Aucun itinéraire trouvé entre ces zones.")
            return

        path_labels = [G.nodes[n].get("label", str(n)) for n in path]

        path_details = []
        for i, z in enumerate(path):
            prof_h = profiles[(profiles["zone_id"] == z) & (profiles["heure_n"] == heure_sel)]
            cong = float(prof_h["congestion_moy"].mean()) if not prof_h.empty else 0.0
            tps  = float(prof_h["temps_moy"].mean())      if not prof_h.empty else 0.0
            # FIX 1 — Seuils KPI IC : <1.5 Fluide · 1.5-2.5 Modéré · >2.5 Critique
            path_details.append({
                "Étape":           i + 1,
                "Zone":            G.nodes[z].get("label", str(z)),
                "Congestion Moy":  round(cong, 3),
                "Temps moy (min)": round(tps, 1),
                "Statut":          "🔴" if cong > 2.5 else "🟡" if cong > 1.5 else "🟢",
            })

        # FIX 3 — Temps total depuis profils réels, fallback TTM KPI 17.8 min/zone
        total_tps = sum(d["Temps moy (min)"] for d in path_details if d["Temps moy (min)"] > 0)
        if total_tps == 0:
            total_tps = len(path) * 17.8  # TTM KPI actuel = 17.8 min/zone

        max_cong = max(d["Congestion Moy"] for d in path_details)

        st.session_state["rec_results"] = {
            "path":         path,
            "path_labels":  path_labels,
            "length":       length,
            "path_details": path_details,
            "total_tps":    total_tps,
            "max_cong":     max_cong,
            "G":            G,
            "zone_dep":     zone_dep,
            "zone_arr":     zone_arr,
            "heure_sel":    heure_sel,
        }

    res = st.session_state.get("rec_results")
    if not res:
        return

    # FIX 5 — Message explicatif si chemin direct (2 zones seulement)
    if len(res["path"]) <= 2:
        statut_label = (
            "Critique"  if res["max_cong"] > 2.5 else
            "Modéré"    if res["max_cong"] > 1.5 else
            "Fluide"
        )
        st.info(
            f"💡 Chemin direct **{res['zone_dep']} → {res['zone_arr']}** — "
            f"Aucune zone intermédiaire moins congestionnée à **{res['heure_sel']}h**. "
            f"Congestion KPI IC : **{res['max_cong']:.2f}** ({statut_label})"
        )

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Zones traversées",   len(res["path"]))
    k2.metric("Temps total estimé", f"{res['total_tps']:.0f} min")
    k3.metric("Congestion max (IC)", f"{res['max_cong']:.2f}", delta_color="inverse")
    st.markdown("---")

    # Tableau
    st.markdown("#### 📋 Étapes de l'itinéraire optimal")
    st.dataframe(pd.DataFrame(res["path_details"]), use_container_width=True, hide_index=True)

    # Visualisation graphe
    st.markdown("#### 🕸️ Réseau de zones")
    G = res["G"]
    nodes = list(G.nodes())
    lats  = [G.nodes[n].get("lat", 0) for n in nodes]
    lons  = [G.nodes[n].get("lon", 0) for n in nodes]

    if max(abs(l) for l in lats + lons) < 0.01:
        angle = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
        lats  = list(np.sin(angle))
        lons  = list(np.cos(angle))

    pos_map  = {n: (lons[i], lats[i]) for i, n in enumerate(nodes)}
    path_set = set(zip(res["path"][:-1], res["path"][1:]))

    fig = go.Figure()
    for (u, v) in G.edges():
        x0, y0 = pos_map[u]; x1, y1 = pos_map[v]
        on_path = (u, v) in path_set
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(
                color="#818cf8" if on_path else "rgba(71,85,105,0.3)",
                width=3 if on_path else 0.6,
            ),
            showlegend=False, hoverinfo="skip",
        ))

    on_path_nodes = set(res["path"])
    node_colors   = ["#34d399" if n in on_path_nodes else "#475569" for n in nodes]
    node_sizes    = [16        if n in on_path_nodes else 9         for n in nodes]
    labels        = [G.nodes[n].get("label", str(n)) for n in nodes]

    fig.add_trace(go.Scatter(
        x=[pos_map[n][0] for n in nodes],
        y=[pos_map[n][1] for n in nodes],
        mode="markers+text",
        marker=dict(size=node_sizes, color=node_colors,
                    line=dict(width=1, color="rgba(255,255,255,0.3)")),
        text=labels, textposition="top center",
        textfont=dict(size=10, color="#94a3b8"),
        hovertext=labels, hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=480,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟢 Zones itinéraire · 🔵 Liaisons empruntées · Gris = autres zones")

    # Profil congestion le long de l'itinéraire
    st.markdown("#### Congestion le long de l'itinéraire")
    det_df = pd.DataFrame(res["path_details"])
    # FIX 1 — Couleurs et seuils adaptés au KPI IC (0-4+)
    bar_colors = [
        "#ef4444" if c > 2.5 else "#f59e0b" if c > 1.5 else "#34d399"
        for c in det_df["Congestion Moy"]
    ]
    fig_bar = go.Figure(go.Bar(
        x=det_df["Zone"],
        y=det_df["Congestion Moy"],
        marker_color=bar_colors,
    ))
    # FIX 1 — Deux lignes de seuil adaptées au KPI IC
    fig_bar.add_hline(
        y=1.5, line_dash="dash", line_color="#f59e0b",
        annotation_text="Seuil Modéré KPI IC 1.5",
        annotation_font_color="#f59e0b",
    )
    fig_bar.add_hline(
        y=2.5, line_dash="dash", line_color="#ef4444",
        annotation_text="Seuil Critique KPI IC 2.5",
        annotation_font_color="#ef4444",
    )
    fig_bar.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Zone", yaxis_title="Congestion Index (IC)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)