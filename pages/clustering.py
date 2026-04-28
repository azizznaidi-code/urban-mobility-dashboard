"""
pages/clustering.py — Clustering Spatio-Temporel des Zones Urbaines
K-Means + DBSCAN automatiques — parametres caches a l'utilisateur.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime

from data.loader import load_trafic, load_dim_zones


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_zone_features(df_trafic: pd.DataFrame) -> pd.DataFrame:
    """Agregation facttrfic par zone pour features de clustering."""
    # Detecter nom colonne zone
    zone_col = "zone_id" if "zone_id" in df_trafic.columns else "fkzone_id"

    agg = (
        df_trafic.groupby(zone_col)
        .agg(
            vitesse_moy      = ("vitesse_kmh",      "mean"),
            vitesse_std      = ("vitesse_kmh",      "std"),
            congestion_moy   = ("congestion_index",  "mean"),
            congestion_max   = ("congestion_index",  "max"),
            temps_trajet_moy = ("temps_trajet_min",  "mean"),
            nb_obs           = ("vitesse_kmh",       "count"),
        )
        .reset_index()
        .fillna(0)
    )
    # Normaliser le nom de colonne
    if zone_col != "zone_id":
        agg = agg.rename(columns={zone_col: "zone_id"})
    return agg


@st.cache_data(ttl=300, show_spinner="Chargement donnees clustering…")
def _load_cluster_data(debut: str, fin: str) -> pd.DataFrame:
    df_tr = load_trafic(debut, fin)
    dz    = load_dim_zones()
    feats = _build_zone_features(df_tr)
    # Jointure zone_nom
    if "zone_nom" in dz.columns and "zone_id" in dz.columns:
        merged = feats.merge(dz[["zone_id", "zone_nom"]], on="zone_id", how="left")
    else:
        merged = feats
        merged["zone_nom"] = merged["zone_id"].astype(str)
    return merged


def _scale(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(df[feature_cols])


def _auto_k(X: np.ndarray, k_min: int = 2, k_max: int = 8) -> int:
    """Detecte automatiquement le K optimal via Silhouette Score."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best_k, best_sil = k_min, -1
    for k in range(k_min, min(k_max + 1, len(X))):
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X)
        if len(set(lbl)) > 1:
            sil = silhouette_score(X, lbl)
            if sil > best_sil:
                best_sil, best_k = sil, k
    return best_k


# ─────────────────────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────────────────────

def render():
    st.markdown(
        '<p class="section-title">🗂️ Clustering des Zones Urbaines</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        '<span class="module-tag tag-ml">K-Means</span> &nbsp;'
        '<span class="module-tag tag-ml">DBSCAN</span> &nbsp;'
        '<span class="module-tag tag-graph">PCA 2D</span> &nbsp;'
        '<span class="module-tag tag-graph">Silhouette · Elbow · Davies-Bouldin</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Seuils KPI IC (Fiche KPI officielle) ─────────────────────────────────
    # <1.5 Fluide · 1.5-2.0 Modere · 2.0-2.5 Critique · >2.5 Tres critique
    KPI_IC_FLUIDE   = 1.5
    KPI_IC_CRITIQUE = 2.5

    # ── Date range uniquement (parametres algo caches) ────────────────────────
    c1, c2 = st.columns(2)
    debut = str(c1.date_input("Début", datetime.date(2019, 1, 1),  key="cl_d"))
    fin   = str(c2.date_input("Fin",   datetime.date(2022, 12, 31), key="cl_f"))

    # Parametres algo FIXES (invisibles pour l'utilisateur)
    K_MAX      = 8
    EPS_DBSCAN = 0.8
    MIN_SAMP   = 3

    run_btn = st.button("🚀 Lancer le Clustering", use_container_width=True)

    if not run_btn and "cluster_results" not in st.session_state:
        st.info("Cliquez sur **Lancer le Clustering** pour analyser les zones depuis le DW.")
        return

    if run_btn:
        try:
            df = _load_cluster_data(debut, fin)
        except Exception as e:
            st.error(f"Erreur chargement donnees : {e}")
            return

        if df.empty or len(df) < 3:
            st.warning("Pas assez de zones dans cette periode.")
            return

        feature_cols = [
            "vitesse_moy", "vitesse_std", "congestion_moy",
            "congestion_max", "temps_trajet_moy",
        ]
        feature_cols = [c for c in feature_cols if c in df.columns]
        X = _scale(df, feature_cols)

        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        from sklearn.decomposition import PCA

        # K optimal detecte automatiquement
        with st.spinner("Optimisation automatique du nombre de clusters..."):
            k_sel = _auto_k(X, k_min=2, k_max=min(K_MAX, len(df) - 1))

        # Elbow data
        inertias, sil_scores, db_scores = [], [], []
        ks = list(range(2, min(K_MAX + 1, len(df))))
        for k in ks:
            km  = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km.fit_predict(X)
            inertias.append(km.inertia_)
            if len(set(lbl)) > 1:
                sil_scores.append(silhouette_score(X, lbl))
                db_scores.append(davies_bouldin_score(X, lbl))
            else:
                sil_scores.append(0); db_scores.append(99)

        # K-Means final avec K optimal
        km_final = KMeans(n_clusters=k_sel, random_state=42, n_init=10)
        df["cluster_kmeans"] = km_final.fit_predict(X).astype(str)

        # DBSCAN
        db = DBSCAN(eps=EPS_DBSCAN, min_samples=MIN_SAMP)
        df["cluster_dbscan"]  = db.fit_predict(X).astype(str)
        n_dbscan_clusters     = len(set(df["cluster_dbscan"].unique()) - {"-1"})
        n_noise               = int((df["cluster_dbscan"] == "-1").sum())

        # PCA 2D
        pca   = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        df["PCA1"] = X_pca[:, 0]
        df["PCA2"] = X_pca[:, 1]

        try:
            sil_final = silhouette_score(X, km_final.labels_)
            db_final  = davies_bouldin_score(X, km_final.labels_)
        except Exception:
            sil_final, db_final = 0, 0

        st.session_state["cluster_results"] = {
            "df": df, "X": X, "feature_cols": feature_cols,
            "ks": ks, "inertias": inertias,
            "sil_scores": sil_scores, "db_scores": db_scores,
            "sil_final": sil_final, "db_final": db_final,
            "k_sel": k_sel, "n_dbscan": n_dbscan_clusters,
            "n_noise": n_noise, "pca_var": pca.explained_variance_ratio_,
            "kpi_ic_fluide": KPI_IC_FLUIDE, "kpi_ic_critique": KPI_IC_CRITIQUE,
        }

    res = st.session_state.get("cluster_results")
    if res is None:
        return

    df           = res["df"]
    feature_cols = res["feature_cols"]
    KPI_IC_FLUIDE   = res.get("kpi_ic_fluide", 1.5)
    KPI_IC_CRITIQUE = res.get("kpi_ic_critique", 2.5)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zones analysees",   len(df))
    c2.metric("Clusters K-Means",  res["k_sel"],
              help=f"K optimal detecte automatiquement (Silhouette max)")
    c3.metric("Clusters DBSCAN",   res["n_dbscan"])
    c4.metric("Bruit DBSCAN",      res["n_noise"])

    col1, col2 = st.columns(2)
    col1.metric("Silhouette Score",     f"{res['sil_final']:.3f}",
                help="Plus proche de 1 = meilleur clustering")
    col2.metric("Davies-Bouldin Index", f"{res['db_final']:.3f}",
                help="Plus bas = meilleur clustering")

    st.info(
        f"**K optimal detecte automatiquement : {res['k_sel']} clusters** "
        f"(Silhouette Score = {max(res['sil_scores']):.3f})"
    )
    st.markdown("---")

    # ── Onglets ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Elbow & Silhouette", "🔵 PCA K-Means",
        "🟠 PCA DBSCAN", "🔥 Profil des Clusters", "📋 Donnees",
    ])

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Methode Elbow — Inertie par K")
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=res["ks"], y=res["inertias"],
            mode="lines+markers",
            line=dict(color="#818cf8", width=2.5),
            marker=dict(size=9, color="#818cf8"),
            name="Inertie",
        ))
        fig_elbow.add_vline(
            x=res["k_sel"], line_dash="dash", line_color="#34d399",
            annotation_text=f"K={res['k_sel']} optimal (auto)"
        )
        fig_elbow.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Nombre de clusters K", yaxis_title="Inertie (SSE)",
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

        col_s, col_d = st.columns(2)
        with col_s:
            st.markdown("#### Silhouette Score par K")
            fig_sil = go.Figure(go.Bar(
                x=res["ks"], y=res["sil_scores"],
                marker_color=["#34d399" if s == max(res["sil_scores"]) else "#818cf8"
                              for s in res["sil_scores"]],
            ))
            fig_sil.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=250,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="K", yaxis_title="Silhouette Score",
            )
            st.plotly_chart(fig_sil, use_container_width=True)

        with col_d:
            st.markdown("#### Davies-Bouldin Index par K")
            fig_db = go.Figure(go.Bar(
                x=res["ks"], y=res["db_scores"],
                marker_color=["#34d399" if s == min(res["db_scores"]) else "#f59e0b"
                              for s in res["db_scores"]],
            ))
            fig_db.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=250,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="K", yaxis_title="Davies-Bouldin Index",
            )
            st.plotly_chart(fig_db, use_container_width=True)

        pv = res["pca_var"]
        st.info(
            f"**PCA** : composante 1 = **{pv[0]*100:.1f}%** de la variance, "
            f"composante 2 = **{pv[1]*100:.1f}%** — total : **{sum(pv)*100:.1f}%**"
        )

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Visualisation PCA 2D — Clusters K-Means")
        fig_pca = px.scatter(
            df, x="PCA1", y="PCA2", color="cluster_kmeans",
            hover_data=["zone_nom", "congestion_moy", "vitesse_moy"]
                       if "zone_nom" in df.columns
                       else ["zone_id", "congestion_moy", "vitesse_moy"],
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={"cluster_kmeans": "Cluster K-Means"},
        )
        fig_pca.update_traces(marker=dict(size=14, opacity=0.85,
                                          line=dict(width=1, color="rgba(255,255,255,0.3)")))
        for _, row in df.iterrows():
            label = str(row.get("zone_nom", row.get("zone_id", "")))[:15]
            fig_pca.add_annotation(
                x=row["PCA1"], y=row["PCA2"], text=label,
                showarrow=False, font=dict(size=9, color="#94a3b8"),
                xshift=8, yshift=8,
            )
        fig_pca.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=420,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    # ── Tab 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### Visualisation PCA 2D — Clusters DBSCAN (−1 = bruit)")
        color_map = {}
        palette   = px.colors.qualitative.Vivid
        for i, c in enumerate(sorted(df["cluster_dbscan"].unique())):
            color_map[c] = "#64748b" if c == "-1" else palette[i % len(palette)]

        fig_db2 = px.scatter(
            df, x="PCA1", y="PCA2", color="cluster_dbscan",
            hover_data=["zone_nom", "congestion_moy", "vitesse_moy"]
                       if "zone_nom" in df.columns
                       else ["zone_id", "congestion_moy", "vitesse_moy"],
            color_discrete_map=color_map,
            labels={"cluster_dbscan": "Cluster DBSCAN"},
        )
        fig_db2.update_traces(marker=dict(size=14, opacity=0.85,
                                          line=dict(width=1, color="rgba(255,255,255,0.3)")))
        fig_db2.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=420,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_db2, use_container_width=True)

        noise_df = df[df["cluster_dbscan"] == "-1"]
        if not noise_df.empty:
            disp_cols = (["zone_nom"] if "zone_nom" in noise_df.columns else ["zone_id"]) + feature_cols
            st.markdown("**Zones anormales (bruit DBSCAN)**")
            st.dataframe(noise_df[disp_cols].round(2), use_container_width=True, hide_index=True)

    # ── Tab 4 ─────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("#### Profil moyen des clusters K-Means (heatmap normalisee)")
        profile = df.groupby("cluster_kmeans")[feature_cols].mean()
        profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

        fig_heat = go.Figure(go.Heatmap(
            z=profile_norm.values,
            x=feature_cols,
            y=[f"Cluster {c}" for c in profile_norm.index],
            colorscale="RdBu_r",
            text=profile.values.round(2),
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(tickfont=dict(color="#94a3b8")),
        ))
        fig_heat.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=max(300, len(profile) * 60),
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(tickangle=-30),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("**Interpretation des clusters (seuils Fiche KPI IC) :**")
        for clust in sorted(df["cluster_kmeans"].unique()):
            sub = df[df["cluster_kmeans"] == clust]
            # FIX zone_nom : verifier que la colonne existe et n'est pas vide
            if "zone_nom" in sub.columns and not sub["zone_nom"].dropna().empty:
                zones_noms = ", ".join(sub["zone_nom"].dropna().tolist()[:5])
            else:
                zones_noms = ", ".join(sub["zone_id"].astype(str).tolist()[:5])

            cong  = sub["congestion_moy"].mean()
            vit   = sub["vitesse_moy"].mean()
            # Seuils Fiche KPI IC : <1.5 Fluide · 1.5-2.5 Critique · >2.5 Tres critique
            label = (
                "🔴 Tres Critique (IC>2.5)"  if cong > KPI_IC_CRITIQUE
                else "🟡 Critique (IC 1.5-2.5)" if cong > KPI_IC_FLUIDE
                else "🟢 Fluide (IC<1.5)"
            )
            st.markdown(
                f"- **Cluster {clust}** ({label}) — "
                f"Congestion moy: `{cong:.2f}` | Vitesse moy: `{vit:.1f} km/h` — "
                f"Zones: {zones_noms or 'N/A'}"
            )

    # ── Tab 5 ─────────────────────────────────────────────────────────────────
    with tab5:
        st.markdown("#### Resultats clustering par zone")
        display_cols = (
            (["zone_nom"] if "zone_nom" in df.columns else []) +
            ["zone_id"] + feature_cols +
            ["cluster_kmeans", "cluster_dbscan", "nb_obs"]
        )
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[display_cols].round(3).style.background_gradient(
                subset=["congestion_moy"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True,
        )