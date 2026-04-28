"""
pages/mlops_page.py — MLOps · suivi des modèles entraînés sur le DW réel.
"""
# Correctif pkg_resources (setuptools < 67 / Python 3.12)
try:
    import pkg_resources  # noqa: F401
except ImportError:
    import sys, types
    pkg_resources = types.ModuleType("pkg_resources")
    sys.modules["pkg_resources"] = pkg_resources

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def _load_mlflow_runs() -> pd.DataFrame:
    """Charge les runs MLflow réels (sqlite:///mlruns.db)."""
    try:
        import mlflow
        from config import settings
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exps   = client.search_experiments()
        rows   = []
        for exp in exps:
            runs = client.search_runs(exp.experiment_id,
                                       max_results=50,
                                       order_by=["start_time DESC"])
            for r in runs:
                rows.append({
                    "run_name":  r.info.run_name or r.info.run_id[:8],
                    "expérience": exp.name,
                    "status":    r.info.status,
                    "date":      pd.to_datetime(r.info.start_time, unit="ms")
                                   .strftime("%Y-%m-%d %H:%M"),
                    "durée_s":   round((r.info.end_time - r.info.start_time) / 1000)
                                   if r.info.end_time else 0,
                    **{f"metric_{k}": v for k,v in r.data.metrics.items()},
                    **{f"param_{k}":  v for k,v in r.data.params.items()},
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:
        return pd.DataFrame({"Erreur": [str(e)]})


def _drift_from_dw(debut: str, fin: str) -> dict:
    """Calcule la dérive réelle sur les colonnes clés du DW."""
    from data.loader import load_trafic, load_transport
    import numpy as np
    results = {}
    try:
        df_tr = load_trafic(debut, fin)
        for col in ["vitesse_kmh","congestion_index","temps_trajet_min"]:
            if col in df_tr.columns:
                vals = df_tr[col].dropna()
                mid  = len(vals)//2
                ref  = vals.iloc[:mid]; cur = vals.iloc[mid:]
                from scipy.stats import ks_2samp
                stat, _ = ks_2samp(ref, cur)
                results[col] = round(stat, 4)
    except Exception: pass
    try:
        df_tp = load_transport(debut, fin)
        for col in ["retard_s","charge_estimee","charge_estimee"]:
            if col in df_tp.columns:
                vals = df_tp[col].dropna()
                mid  = len(vals)//2
                ref  = vals.iloc[:mid]; cur = vals.iloc[mid:]
                from scipy.stats import ks_2samp
                stat, _ = ks_2samp(ref, cur)
                results[col] = round(stat, 4)
    except Exception: pass
    return results


def render():
    st.markdown('<p class="section-title">⚙️ MLOps · Gestion des Modèles</p>',
                unsafe_allow_html=True)
    st.caption(
        '<span class="module-tag tag-ml">MLflow</span> &nbsp;'
        '<span class="module-tag tag-ml">Model Registry</span> &nbsp;'
        '<span class="module-tag tag-ml">Drift (KS-test) · DW réel</span>',
        unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2 = st.tabs(["📋 Registre MLflow", "⚠️ Drift · DW réel"])

    with tab1:
        st.markdown("#### Runs MLflow (entraînements réels sur dwurbanmobility)")
        if st.button("🔄 Recharger les runs"):
            st.cache_data.clear()

        df_runs = _load_mlflow_runs()
        if df_runs.empty:
            st.info("Aucun run MLflow trouvé. Entraînez d'abord un modèle.")
        elif "Erreur" in df_runs.columns:
            st.warning(f"MLflow non disponible : {df_runs['Erreur'].iloc[0]}")
        else:
            # Filtres
            exp_list = df_runs["expérience"].unique().tolist() if "expérience" in df_runs else []
            exp_sel  = st.multiselect("Expérience", exp_list, default=exp_list)
            df_view  = df_runs[df_runs["expérience"].isin(exp_sel)] if exp_sel else df_runs

            st.dataframe(df_view, use_container_width=True, hide_index=True)

            # Métriques clés
            metric_cols = [c for c in df_view.columns if c.startswith("metric_")]
            if metric_cols:
                st.markdown("#### Évolution des métriques")
                metric_sel = st.selectbox("Métrique", metric_cols)
                fig = go.Figure(go.Scatter(
                    x=df_view["date"], y=df_view[metric_sel],
                    mode="lines+markers",
                    line=dict(color="#818cf8",width=2),
                    marker=dict(size=8,color="#818cf8"),
                ))
                fig.update_layout(
                    template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",height=280,
                    margin=dict(l=0,r=0,t=10,b=0),
                    yaxis_title=metric_sel.replace("metric_",""),
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Dérive des données (KS-test · colonnes DW réelles)")
        import datetime
        c1,c2 = st.columns(2)
        debut = str(c1.date_input("Début", datetime.date(2019, 1, 1), key="ml_d"))
        fin   = str(c2.date_input("Fin",   datetime.date(2022, 12, 31), key="ml_f"))

        if st.button("📊 Calculer le drift (KS-test sur DW)"):
            with st.spinner("Calcul du drift sur les données réelles…"):
                drift = _drift_from_dw(debut, fin)
            st.session_state["drift"] = drift

        if "drift" in st.session_state:
            drift = st.session_state["drift"]
            if not drift:
                st.warning("Aucune donnée disponible pour calculer le drift.")
            else:
                drift_df = pd.DataFrame([
                    {"Feature": k, "Score KS": v,
                     "Statut": "🔴 Dérive" if v>0.25 else "🟡 Attention" if v>0.15 else "🟢 Stable"}
                    for k,v in drift.items()
                ])
                fig2 = go.Figure(go.Bar(
                    x=drift_df["Score KS"], y=drift_df["Feature"],
                    orientation="h",
                    marker_color=["#ef4444" if v>0.25 else "#f59e0b" if v>0.15 else "#34d399"
                                  for v in drift_df["Score KS"]],
                    text=drift_df["Score KS"].round(4), textposition="outside",
                ))
                fig2.add_vline(x=0.25, line_dash="dash", line_color="#ef4444",
                               annotation_text="Seuil réentraînement (0.25)")
                fig2.update_layout(
                    template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",height=max(200,len(drift)*50),
                    margin=dict(l=0,r=60,t=10,b=0),
                    xaxis_title="Score KS (0=stable, 1=dérive totale)",
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(drift_df, use_container_width=True, hide_index=True)

                drifted = [k for k,v in drift.items() if v > 0.25]
                if drifted:
                    st.markdown(
                        f'<div class="alert-avertissement">🟡 <b>Dérive détectée</b> sur : '
                        f'<b>{", ".join(drifted)}</b> — Réentraînement recommandé.</div>',
                        unsafe_allow_html=True)
