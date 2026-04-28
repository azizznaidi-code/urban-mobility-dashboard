"""
app.py — UrbanAI avec Login + Roles selon Fiches KPI
"""
import streamlit as st

st.set_page_config(
    page_title="UrbanAI — Mobilité Intelligente",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }
.module-tag {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
}
.tag-ml     { background: rgba(129,140,248,0.15); color: #818cf8; border: 1px solid rgba(129,140,248,0.3); }
.tag-graph  { background: rgba(52,211,153,0.15);  color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.tag-causal { background: rgba(251,191,36,0.15);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.tag-optim  { background: rgba(236,72,153,0.15);  color: #ec4899; border: 1px solid rgba(236,72,153,0.3); }
.section-title {
    font-size: 1.4rem; font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 16px;
}
.stButton>button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 8px 20px; transition: opacity 0.2s;
}
.stButton>button:hover { opacity: 0.85; }
div[data-testid="stMetric"] {
    background: #1e293b; border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px; padding: 16px;
}
.alert-critique      { border-left:4px solid #ef4444; background:rgba(239,68,68,0.08); padding:12px 16px; border-radius:0 8px 8px 0; margin:8px 0; }
.alert-avertissement { border-left:4px solid #f59e0b; background:rgba(245,158,11,0.08); padding:12px 16px; border-radius:0 8px 8px 0; margin:8px 0; }
.alert-info          { border-left:4px solid #3b82f6; background:rgba(59,130,246,0.08); padding:12px 16px; border-radius:0 8px 8px 0; margin:8px 0; }
</style>
""", unsafe_allow_html=True)

# ── Toutes les pages disponibles ──────────────────────────────────────────────
ALL_PAGES = {
    "🔮 Prédiction Multi-Horizon": "prediction",
    "🔍 Détection d'Anomalies":    "anomaly",
    "🧠 IA Causale":               "causal",
    "🗺️ Recommandation":           "recommandation",
    "💬 NLP Incidents":            "nlp",
    "🕸️ Graphe du Réseau":         "graph",
    "🎲 Simulation What-If":       "whatif",
    "📊 Rapport & XAI":            "report",
    "⚙️ Configuration MLOps":      "mlops",
}

# ── Verifier authentification ─────────────────────────────────────────────────
authenticated = st.session_state.get("authenticated", False)
user          = st.session_state.get("user", {})

if not authenticated:
    from pages.login import render as login_render
    login_render()
    st.stop()

# ── Utilisateur connecte ──────────────────────────────────────────────────────
allowed_pages = user.get("pages", [])
PAGES = {label: key for label, key in ALL_PAGES.items() if key in allowed_pages}

# ── Ajouter mlops pour tous les roles si pas present ─────────────────────────
mlops_label = "⚙️ Configuration MLOps"
if mlops_label not in PAGES:
    PAGES[mlops_label] = "mlops"

if not PAGES:
    st.error("Aucune page disponible pour votre role. Contactez l'administrateur.")
    st.stop()

pages_list = list(PAGES.keys())

# ── Initialiser la page courante dans session_state ──────────────────────────
if "current_page" not in st.session_state:
    st.session_state["current_page"] = pages_list[0]

# Verifier que la page courante existe toujours
if st.session_state["current_page"] not in pages_list:
    st.session_state["current_page"] = pages_list[0]

# ── Sidebar avec profil ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏙️ **UrbanAI**")
    st.caption("Mobilité Intelligente · DW Urban Mobility")
    st.divider()

    from pages.profile import render_profile
    render_profile()

    st.divider()

    # Navigation avec index stable
    current_index = pages_list.index(st.session_state["current_page"])

    page_label = st.radio(
        "Navigation",
        pages_list,
        index=current_index,
        key="nav_radio",
        label_visibility="collapsed",
    )

    # Sauvegarder la page selectionnee
    st.session_state["current_page"] = page_label

    st.divider()

    couleur = user.get("couleur", "#818cf8")
    st.markdown(f"""
    <div style='font-size:0.75rem; color:#475569;'>
    📦 <b style='color:#818cf8'>DW</b> dwurbanmobility<br>
    🗄️ <b style='color:#34d399'>SQL Server</b> 2022<br>
    🎭 <b style='color:{couleur}'>{user.get("role","")}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Recharger les données DW"):
        st.cache_data.clear()
        st.success("Cache vidé !")

# ── Routage selon page selectionnee ──────────────────────────────────────────
page_key = PAGES[page_label]

if   page_key == "prediction":
    from pages import prediction;     prediction.render()
elif page_key == "anomaly":
    from pages import anomaly;        anomaly.render()
elif page_key == "causal":
    from pages import causal;         causal.render()
elif page_key == "recommandation":
    from pages import recommandation; recommandation.render()
elif page_key == "nlp":
    from pages import nlp;            nlp.render()
elif page_key == "graph":
    from pages import graph;          graph.render()
elif page_key == "whatif":
    from pages import whatif;         whatif.render()
elif page_key == "report":
    from pages import report;         report.render()
elif page_key == "mlops":
    from pages import mlops_page;     mlops_page.render()
