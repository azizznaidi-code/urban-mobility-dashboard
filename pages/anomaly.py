"""
pages/anomaly.py — Detection d'anomalies · donnees reelles DW facttrfic.
Fix : cause_distribution NaN + shap_cause NaN + top_anomalies robuste
"""
try:
    import pkg_resources
except ImportError:
    import sys, types
    pkg_resources = types.ModuleType("pkg_resources")
    sys.modules["pkg_resources"] = pkg_resources

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data.loader import anomalies_from_dw


@st.cache_resource(show_spinner="Entrainement du detecteur d'anomalies…")
def get_detector(debut: str, fin: str):
    from models.anomaly_detector import AnomalyDetector
    df = anomalies_from_dw(debut, fin)
    if len(df) < 50:
        raise ValueError(f"Seulement {len(df)} lignes — insuffisant.")
    det = AnomalyDetector()
    det.fit(df)
    return det, df


# ── Fonctions de correction NaN ───────────────────────────────────────────────

def _fix_shap_cause(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace shap_cause NaN par la feature la plus deviante (z-score)."""
    df = df.copy()
    feat_cols = [c for c in ["retard_s", "congestion_index",
                              "vitesse_kmh", "temps_trajet_min"]
                 if c in df.columns]
    if not feat_cols:
        df["shap_cause"] = "Inconnu"
        return df

    # Verifier si shap_cause est absent ou majoritairement NaN/nan-string
    need_fix = False
    if "shap_cause" not in df.columns:
        need_fix = True
    else:
        nan_pct = (df["shap_cause"].isna() |
                   df["shap_cause"].astype(str).str.lower().isin(["nan","none","null",""])).mean()
        if nan_pct > 0.3:
            need_fix = True

    if need_fix:
        # Calculer la feature la plus deviante pour chaque ligne
        X_df  = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        means = X_df.mean()
        stds  = X_df.std().replace(0, 1)
        zscores = ((X_df - means) / stds).abs()
        df["shap_cause"] = zscores.idxmax(axis=1)

    # Nettoyer les NaN restants
    df["shap_cause"] = df["shap_cause"].fillna("Inconnu")
    df["shap_cause"] = df["shap_cause"].astype(str).replace(
        {"nan":"Inconnu","none":"Inconnu","null":"Inconnu","":"Inconnu"})
    return df


def _cause_distribution_clean(df: pd.DataFrame, det) -> dict:
    """
    Recupere cause_distribution depuis det et filtre les NaN.
    Si tout est NaN → recalcule depuis shap_cause fixe.
    """
    LABEL_MAP = {
        "retard_s":         "Retard (s)",
        "congestion_index": "Congestion IC",
        "vitesse_kmh":      "Vitesse VM",
        "temps_trajet_min": "Temps trajet TTM",
        "anomaly_score":    "Score anomalie",
    }

    # Essayer la methode du modele
    try:
        raw = det.cause_distribution(df)
    except Exception:
        raw = {}

    # Filtrer les cles nan
    clean = {}
    for k, v in raw.items():
        k_str = str(k).lower().strip()
        if k_str not in ("nan", "none", "null", "") and v > 0:
            label = LABEL_MAP.get(k, k)
            clean[label] = v
    if clean:
        return clean

    # Fallback : calculer depuis shap_cause (deja fixe)
    df_anom = df[df["anomaly"]] if "anomaly" in df.columns else df
    if "shap_cause" in df_anom.columns:
        series = df_anom["shap_cause"].astype(str)
        series = series[~series.str.lower().isin(["nan","none","null","","inconnu"])]
        if not series.empty:
            counts = series.value_counts()
            total  = counts.sum()
            return {LABEL_MAP.get(k, k): round(v/total*100, 1)
                    for k, v in counts.items()}

    # Fallback final : z-score sur anomalies
    feat_cols = [c for c in ["retard_s","congestion_index","vitesse_kmh","temps_trajet_min"]
                 if c in df.columns]
    df_anom2  = df[df["anomaly"]] if "anomaly" in df.columns else df
    if feat_cols and not df_anom2.empty:
        X_df   = df_anom2[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        means  = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).mean()
        stds   = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).std().replace(0,1)
        top_f  = ((X_df - means) / stds).abs().idxmax(axis=1).value_counts()
        total  = top_f.sum()
        return {LABEL_MAP.get(k, k): round(v/total*100, 1) for k, v in top_f.items()}

    return {}


# ── Render ────────────────────────────────────────────────────────────────────

def render():
    st.markdown("<p class='section-title'>🔍 Detection d'Anomalies Proactive</p>",
                unsafe_allow_html=True)
    st.caption(
        '<span class="module-tag tag-ml">Isolation Forest</span> &nbsp;'
        '<span class="module-tag tag-ml">Autoencoder PyTorch</span> &nbsp;'
        '<span class="module-tag tag-ml">SHAP Cause</span>',
        unsafe_allow_html=True)
    st.markdown("---")

    st.info(
        "**Fiche KPI SAN** : < 0.4 Normal · 0.4–0.6 Information · "
        "0.6–0.8 Avertissement · > 0.8 Critique"
    )

    import datetime
    c1, c2 = st.columns(2)
    debut = str(c1.date_input("Debut", datetime.date(2019, 1, 1),  key="an_d"))
    fin   = str(c2.date_input("Fin",   datetime.date(2022, 12, 31), key="an_f"))

    col_a, col_b = st.columns(2)
    train_btn  = col_a.button("Entrainer le detecteur",  use_container_width=True)
    detect_btn = col_b.button("Detecter les anomalies",  use_container_width=True)

    # Entrainement
    if train_btn:
        try:
            det, df_raw = get_detector(debut, fin)
            st.session_state["detector"]       = det
            st.session_state["df_anomaly_raw"] = df_raw
            st.success(f"Detecteur entraine sur {len(df_raw):,} observations reelles.")
        except Exception as e:
            st.error(f"Erreur : {e}"); st.stop()

    # Detection
    if detect_btn:
        if "detector" not in st.session_state:
            st.warning("Cliquez d'abord sur **Entrainer le detecteur**."); st.stop()
        det    = st.session_state["detector"]
        df_raw = st.session_state["df_anomaly_raw"]
        try:
            df_det = det.detect(df_raw)
            # FIX : corriger shap_cause NaN immediatement apres detect()
            df_det = _fix_shap_cause(df_det)
            st.session_state["df_detected"] = df_det
        except Exception as e:
            st.error(f"Erreur detection : {e}"); st.stop()

    if "df_detected" not in st.session_state:
        st.info("Entrainer d'abord le detecteur puis cliquer sur **Detecter**.")
        return

    df  = st.session_state["df_detected"]
    det = st.session_state["detector"]

    seuil = st.slider("Seuil score anomalie (Fiche KPI SAN)",
                      0.3, 0.95, 0.5, 0.05,
                      help="< 0.4 Normal · 0.4-0.6 Info · 0.6-0.8 Avert. · > 0.8 Critique")

    # KPIs
    n_tot  = len(df)
    n_anom = int(df["anomaly"].sum()) if "anomaly" in df.columns else 0
    n_crit = int((df["severity"] == "CRITIQUE").sum()) if "severity" in df.columns else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations",         f"{n_tot:,}")
    c2.metric("Anomalies detectees",   f"{n_anom:,}",
              delta=f"{n_anom/max(n_tot,1)*100:.1f}%", delta_color="inverse")
    c3.metric("Critiques (>0.8)",      f"{n_crit:,}", delta_color="inverse")
    c4.metric("Score moyen anomalies",
              f"{df[df['anomaly']]['anomaly_score'].mean():.3f}"
              if n_anom > 0 and "anomaly_score" in df.columns else "—")

    st.markdown("---")

    col_l, col_r = st.columns([3, 1])

    with col_l:
        st.markdown("#### Score d'anomalie dans le temps (DW reel)")
        fig = go.Figure()
        if "anomaly_score" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["anomaly_score"],
                mode="lines", line=dict(color="#475569", width=1),
                name="Score", opacity=0.5,
            ))
            colors = {"CRITIQUE":"#ef4444","AVERTISSEMENT":"#f59e0b","INFO":"#3b82f6"}
            for sev, col in colors.items():
                if "severity" not in df.columns: break
                sub = df[df["severity"] == sev]
                if sub.empty: continue
                fig.add_trace(go.Scatter(
                    x=sub.index, y=sub["anomaly_score"],
                    mode="markers",
                    marker=dict(size=8, color=col), name=sev,
                ))
            # Seuils KPI SAN
            for s_val, s_col, s_lbl in [
                (0.4, "#34d399", "Normal/Info 0.4"),
                (0.6, "#f59e0b", "Avert. 0.6"),
                (0.8, "#ef4444", "Critique 0.8"),
            ]:
                fig.add_hline(y=s_val, line_dash="dot", line_color=s_col,
                              opacity=0.6, annotation_text=s_lbl,
                              annotation_font_color=s_col)
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=300,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Causes (SHAP)")
        # FIX : utiliser notre fonction qui filtre les NaN
        cause_dist = _cause_distribution_clean(df, det)
        if cause_dist:
            fig2 = px.pie(
                values=list(cause_dist.values()),
                names=list(cause_dist.keys()),
                color_discrete_sequence=["#818cf8","#34d399","#f59e0b","#ef4444","#ec4899"],
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", template="plotly_dark",
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(font=dict(color="#94a3b8", size=11)),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Causes non disponibles.")

    # Top anomalies
    st.markdown("#### Top anomalies detectees (donnees DW reelles)")
    try:
        top = det.top_anomalies(df, 20)
    except Exception:
        top = df.nlargest(20, "anomaly_score") if "anomaly_score" in df.columns else df.head(20)

    # FIX : nettoyer shap_cause dans top aussi
    if "shap_cause" in top.columns:
        top = top.copy()
        LABEL_MAP = {
            "retard_s":"Retard (s)","congestion_index":"Congestion IC",
            "vitesse_kmh":"Vitesse VM","temps_trajet_min":"Temps trajet TTM",
        }
        top["shap_cause"] = (top["shap_cause"].astype(str)
                             .replace({"nan":"Inconnu","none":"Inconnu",
                                       "null":"Inconnu","":"Inconnu"})
                             .map(lambda x: LABEL_MAP.get(x, x)))

    display_cols = [c for c in ["zone_nom","retard_s","congestion_index",
                                 "vitesse_kmh","anomaly_score","severity","shap_cause"]
                    if c in top.columns]
    if display_cols:
        try:
            st.dataframe(
                top[display_cols].style.background_gradient(
                    subset=["anomaly_score"], cmap="Reds"),
                use_container_width=True, hide_index=True,
            )
        except Exception:
            st.dataframe(top[display_cols], use_container_width=True, hide_index=True)
    else:
        st.dataframe(top, use_container_width=True, hide_index=True)

    # Zones les plus touchees
    if "zone_nom" in df.columns and "anomaly" in df.columns:
        st.markdown("#### Anomalies par Zone (DW Dim_Zone)")
        zone_anom = (
            df[df["anomaly"]].groupby("zone_nom")["anomaly_score"]
            .agg(["count","mean"])
            .rename(columns={"count":"nb","mean":"score_moy"})
            .sort_values("nb", ascending=False)
            .reset_index()
        )
        if not zone_anom.empty:
            fig3 = px.bar(
                zone_anom, x="zone_nom", y="nb",
                color="score_moy", color_continuous_scale="Reds",
                labels={"zone_nom":"Zone","nb":"Nb anomalies","score_moy":"Score moyen"},
            )
            fig3.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=280,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig3, use_container_width=True)