"""
pages/nlp.py — NLP Analyse de Sentiment & Classification des Incidents Urbains
Données : factaccidents + dimzones (DW réel 2019-2022).
Lexique basé sur les catégories d'incidents (pas de données texte réelles dans le DW).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data.loader import load_accidents, load_dim_zones


# ─────────────────────────────────────────────────────────────────────────────
# Mapping incidents → contexte NLP
# ─────────────────────────────────────────────────────────────────────────────

# Libellés des types d'incidents (fktypeincident 1-10)
INCIDENT_LABELS = {
    1:  "Collision frontale",
    2:  "Collision latérale",
    3:  "Renversement piéton",
    4:  "Accident vélo",
    5:  "Chute motocycliste",
    6:  "Dérapage véhicule",
    7:  "Obstacle sur voie",
    8:  "Embouteillage critique",
    9:  "Incident météorologique",
    10: "Vandalisme / Crime",
}

# Libellés de sévérité (fK_severity 1-5)
SEVERITY_LABELS = {1: "Mineur", 2: "Modéré", 3: "Grave", 4: "Très Grave", 5: "Fatal"}

# Sentiment lexique par type d'incident
SENTIMENT_MAP = {
    1: "Négatif",   # Collision frontale
    2: "Négatif",   # Collision latérale
    3: "Très Négatif",
    4: "Négatif",
    5: "Négatif",
    6: "Modéré",
    7: "Modéré",
    8: "Négatif",
    9: "Modéré",
    10: "Très Négatif",
}

SENTIMENT_SCORE = {
    "Très Négatif": -1.0,
    "Négatif":      -0.5,
    "Modéré":        0.0,
    "Positif":       0.5,
    "Très Positif":  1.0,
}

SENTIMENT_COLORS = {
    "Très Négatif": "#ef4444",
    "Négatif":      "#f97316",
    "Modéré":       "#f59e0b",
    "Positif":      "#34d399",
    "Très Positif": "#22c55e",
}

# Catégories thématiques
CATEGORIES = {
    "Collision":    [1, 2],
    "Vulnérable":   [3, 4, 5],
    "Infrastructure": [6, 7, 8],
    "Environnement":  [9],
    "Sécurité":       [10],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Analyse NLP des incidents…")
def _enrich_accidents(debut: str = "2019-01-01", fin: str = "2022-12-31") -> pd.DataFrame:
    df = load_accidents()
    dz = load_dim_zones()[["zone_id", "zone_nom"]]

    # Enrichissement
    df["type_label"]     = df["type_incident"].map(INCIDENT_LABELS).fillna("Inconnu")
    df["severity_label"] = df["severity"].map(SEVERITY_LABELS).fillna("Inconnu")
    df["sentiment"]      = df["type_incident"].map(SENTIMENT_MAP).fillna("Modéré")
    df["score_sent"]     = df["sentiment"].map(SENTIMENT_SCORE).fillna(0.0)

    # Catégorie
    def _cat(t):
        for cat, types in CATEGORIES.items():
            if t in types:
                return cat
        return "Autre"
    df["categorie"] = df["type_incident"].apply(_cat)

    # Importance composite : sévérité × usager_vulnerable × taux_pour_1000
    df["importance"] = (
        df["severity"].fillna(1) * 0.5 +
        df["usager_vulnerable"].fillna(0) * 2.0 +
        df["taux_pour_1000"].fillna(0) * 0.1
    )

    # Pseudo-texte pour wordcloud
    df["description"] = (
        df["type_label"] + " " +
        df["severity_label"] + " " +
        df["categorie"] + " " +
        np.where(df["usager_vulnerable"] > 0, "usager_vulnerable piéton cycliste", "vehicule")
    )

    df = df.merge(dz, on="zone_id", how="left")
    return df


def _make_wordcloud(df: pd.DataFrame) -> str | None:
    """Génère une wordcloud sous forme d'image base64."""
    try:
        from wordcloud import WordCloud
        import base64, io
        text = " ".join(df["description"].dropna().tolist())
        wc   = WordCloud(width=700, height=350, background_color="#0f172a",
                         colormap="Blues", max_words=80,
                         font_path=None).generate(text)
        buf  = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return b64
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────────────────────

def render():
    st.markdown('<p class="section-title">💬 NLP — Analyse des Incidents Urbains</p>',
                unsafe_allow_html=True)
    st.caption(
        '<span class="module-tag tag-causal">Analyse de Sentiment</span> &nbsp;'
        '<span class="module-tag tag-causal">Classification Incidents</span> &nbsp;'
        '<span class="module-tag tag-causal">WordCloud</span>',
        unsafe_allow_html=True)
    st.markdown("---")

    st.info(
        "**Source** : `factaccidents` + `dimzones` (DW dwurbanmobility 2019-2022). "
        "Le sentiment est attribué automatiquement par type d'incident via un lexique expert."
    )

    nlp_btn = st.button("🔍 Lancer l'analyse NLP", use_container_width=True)

    if not nlp_btn and "nlp_results" not in st.session_state:
        st.info("Cliquez sur **Lancer l'analyse NLP** pour analyser les incidents.")
        return

    if nlp_btn:
        try:
            df = _enrich_accidents()
            st.session_state["nlp_results"] = df
        except Exception as e:
            st.error(f"Erreur chargement : {e}"); return

    df = st.session_state.get("nlp_results")
    if df is None or df.empty:
        st.warning("Aucune donnée d'incidents disponible."); return

    # ── KPIs ─────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total incidents",       f"{len(df):,}")
    c2.metric("Score sentiment moyen", f"{df['score_sent'].mean():.3f}")
    c3.metric("Usagers vulnérables",   f"{int(df['usager_vulnerable'].sum()):,}")
    c4.metric("Types d'incidents",     f"{df['type_label'].nunique()}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distribution Sentiment",
        "🗂️ Classification par Catégorie",
        "☁️ WordCloud",
        "📋 Données Enrichies",
    ])

    # ── Tab 1 : Distribution sentiment ───────────────────────────────────────
    with tab1:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### Distribution des Sentiments")
            sent_counts = df["sentiment"].value_counts().reset_index()
            sent_counts.columns = ["sentiment", "count"]
            sent_counts["color"] = sent_counts["sentiment"].map(SENTIMENT_COLORS)
            fig_sent = px.pie(
                sent_counts, values="count", names="sentiment",
                color="sentiment",
                color_discrete_map=SENTIMENT_COLORS,
            )
            fig_sent.update_traces(textposition="inside", textinfo="percent+label")
            fig_sent.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(font=dict(color="#94a3b8")),
            )
            st.plotly_chart(fig_sent, use_container_width=True)

        with col_r:
            st.markdown("#### Score Sentiment par Zone")
            if "zone_nom" in df.columns:
                zone_sent = (df.groupby("zone_nom")["score_sent"]
                             .mean().reset_index().sort_values("score_sent"))
                zone_sent.columns = ["zone_nom", "score_moy"]
                fig_z = px.bar(
                    zone_sent, x="score_moy", y="zone_nom", orientation="h",
                    color="score_moy", color_continuous_scale="RdYlGn",
                    range_color=[-1, 0],
                    labels={"score_moy": "Score sentiment", "zone_nom": "Zone"},
                )
                fig_z.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", height=320,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_z, use_container_width=True)

        # Score par type d'incident
        st.markdown("#### Score de Sentiment par Type d'Incident")
        type_sent = (df.groupby(["type_label", "sentiment"])["accident_id"]
                     .count().reset_index())
        type_sent.columns = ["type_label", "sentiment", "count"]
        fig_ts2 = px.bar(
            type_sent, x="type_label", y="count", color="sentiment",
            color_discrete_map=SENTIMENT_COLORS, barmode="stack",
            labels={"type_label": "Type incident", "count": "Nombre"},
        )
        fig_ts2.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_ts2, use_container_width=True)

    # ── Tab 2 : Classification par catégorie ──────────────────────────────────
    with tab2:
        st.markdown("#### Classification automatique des incidents")

        col_l, col_r = st.columns(2)

        with col_l:
            cat_counts = df["categorie"].value_counts().reset_index()
            cat_counts.columns = ["categorie", "count"]
            fig_cat = px.pie(
                cat_counts, values="count", names="categorie",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_cat.update_traces(textposition="inside", textinfo="percent+label")
            fig_cat.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(font=dict(color="#94a3b8")),
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        with col_r:
            st.markdown("#### Sévérité × Catégorie")
            sev_cat = (df.groupby(["categorie", "severity_label"])["accident_id"]
                       .count().unstack(fill_value=0))
            fig_sc = px.imshow(
                sev_cat,
                color_continuous_scale="Reds",
                text_auto=True,
                labels=dict(x="Sévérité", y="Catégorie", color="Nb"),
            )
            fig_sc.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                height=320, margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # Top incidents par importance
        st.markdown("#### Top 10 incidents les plus impactants (score composite)")
        top_imp = (df.groupby("type_label")["importance"]
                   .mean().reset_index()
                   .sort_values("importance", ascending=False).head(10))
        fig_imp = px.bar(
            top_imp, x="importance", y="type_label", orientation="h",
            color="importance", color_continuous_scale="Reds",
            labels={"importance": "Score importance", "type_label": "Type incident"},
        )
        fig_imp.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=300,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Tab 3 : WordCloud ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### WordCloud des descriptions d'incidents")
        b64 = _make_wordcloud(df)
        if b64:
            st.markdown(
                f'<img src="data:image/png;base64,{b64}" style="width:100%; '
                f'border-radius:12px; border:1px solid rgba(99,102,241,0.2)">',
                unsafe_allow_html=True
            )
        else:
            st.warning(
                "La librairie `wordcloud` n'est pas installée. "
                "Exécutez `pip install wordcloud` pour activer le WordCloud."
            )
            # Fallback : fréquences en barres
            st.markdown("#### Fréquences des termes (alternative)")
            word_freq = {}
            for desc in df["description"].dropna():
                for w in desc.split():
                    word_freq[w] = word_freq.get(w, 0) + 1
            wf_df = (pd.DataFrame.from_dict(word_freq, orient="index", columns=["freq"])
                     .sort_values("freq", ascending=False).head(20).reset_index())
            wf_df.columns = ["terme", "freq"]
            fig_wf = px.bar(wf_df, x="freq", y="terme", orientation="h",
                            color="freq", color_continuous_scale="Blues",
                            labels={"freq": "Fréquence", "terme": "Terme"})
            fig_wf.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=400,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_wf, use_container_width=True)

    # ── Tab 4 : Données ───────────────────────────────────────────────────────
    with tab4:
        st.markdown("#### Données enrichies par incident")
        display_cols = [c for c in
                        ["accident_id", "zone_nom", "type_label", "severity_label",
                         "categorie", "sentiment", "score_sent",
                         "usager_vulnerable", "nb_crimes", "taux_pour_1000", "importance"]
                        if c in df.columns]
        st.dataframe(
            df[display_cols].sort_values("importance", ascending=False)
              .head(500)
              .round(3)
              .style.background_gradient(subset=["score_sent"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True,
        )
