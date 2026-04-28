"""
auth.py — Gestion authentification UrbanAI
Utilisateurs, roles et acces aux pages selon Fiches KPI officielles
"""
import hashlib

# ── Hash mot de passe ─────────────────────────────────────────────────────────
def hash_pwd(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()

# ── Base utilisateurs (Decideurs Fiches KPI) ─────────────────────────────────
# Structure : {email: {nom, prenom, role, pages, kpis, pwd_hash}}
USERS = {
    "president.aom@urbanmobility.fr": {
        "nom": "Dupont", "prenom": "Marie",
        "role": "Président AOM",
        "role_full": "Président(e) de l'Autorité Organisatrice de la Mobilité",
        "kpis": ["RM", "TA", "CM", "SU"],
        "pages": ["prediction", "anomaly", "recommandation", "report"],
        "pwd_hash": hash_pwd("aom2024!"),
        "couleur": "#818cf8",
    },
    "dir.planification@urbanmobility.fr": {
        "nom": "Martin", "prenom": "Pierre",
        "role": "Directeur Planification",
        "role_full": "Directeur de la Planification de l'Offre (AOM)",
        "kpis": ["TA", "CM", "VM", "TTM"],
        "pages": ["prediction", "anomaly", "whatif", "report"],
        "pwd_hash": hash_pwd("plan2024!"),
        "couleur": "#34d399",
    },
    "vp.mobilite@urbanmobility.fr": {
        "nom": "Bernard", "prenom": "Sophie",
        "role": "VP Mobilité",
        "role_full": "Vice-Président(e) chargé(e) de la Mobilité et des Déplacements",
        "kpis": ["IC", "VM", "TTM"],
        "pages": ["prediction", "graph", "recommandation", "report"],
        "pwd_hash": hash_pwd("mob2024!"),
        "couleur": "#f59e0b",
    },
    "dir.voirie@urbanmobility.fr": {
        "nom": "Leroy", "prenom": "Jean",
        "role": "Directeur Voirie",
        "role_full": "Directeur de la Voirie et des Déplacements",
        "kpis": ["IC", "VM", "TTM"],
        "pages": ["prediction", "graph", "whatif", "report"],
        "pwd_hash": hash_pwd("voirie2024!"),
        "couleur": "#f59e0b",
    },
    "dir.clients@urbanmobility.fr": {
        "nom": "Petit", "prenom": "Lucie",
        "role": "Directeur Relation Clients",
        "role_full": "Directeur de la Relation Clients et de la Qualité de Service",
        "kpis": ["SU", "RM", "TA"],
        "pages": ["prediction", "nlp", "recommandation", "report"],
        "pwd_hash": hash_pwd("clients2024!"),
        "couleur": "#ec4899",
    },
    "dir.dreal@urbanmobility.fr": {
        "nom": "Moreau", "prenom": "Eric",
        "role": "Directeur DREAL",
        "role_full": "Directeur Régional de l'Environnement, de l'Aménagement et du Logement",
        "kpis": ["QA", "CO2"],
        "pages": ["prediction", "anomaly", "causal", "report"],
        "pwd_hash": hash_pwd("dreal2024!"),
        "couleur": "#22c55e",
    },
    "dir.securite@urbanmobility.fr": {
        "nom": "Simon", "prenom": "Marc",
        "role": "Préfet / Directeur Sécurité",
        "role_full": "Préfet de département + Directeur Départemental de la Police Nationale",
        "kpis": ["TD", "SAN"],
        "pages": ["prediction", "nlp", "anomaly", "report"],
        "pwd_hash": hash_pwd("secu2024!"),
        "couleur": "#ef4444",
    },
    "cdo@urbanmobility.fr": {
        "nom": "Lambert", "prenom": "Alice",
        "role": "Chief Data Officer",
        "role_full": "Directeur des Données et de l'Intelligence Artificielle (CDO)",
        "kpis": ["SAN", "RM", "TA", "CM", "IC", "VM", "SU", "QA", "TD"],
        "pages": ["prediction", "anomaly", "causal", "recommandation",
                  "nlp", "graph", "whatif", "report", "mlops"],
        "pwd_hash": hash_pwd("cdo2024!"),
        "couleur": "#4f46e5",
    },
    "admin@urbanmobility.fr": {
        "nom": "Admin", "prenom": "UrbanAI",
        "role": "Administrateur",
        "role_full": "Administrateur Système UrbanAI",
        "kpis": ["RM", "TA", "CM", "IC", "VM", "TTM", "SU", "QA", "TD", "SAN"],
        "pages": ["prediction", "anomaly", "causal", "recommandation",
                  "nlp", "graph", "whatif", "report", "mlops"],
        "pwd_hash": hash_pwd("admin2024!"),
        "couleur": "#fbbf24",
    },
}

def login(email: str, pwd: str):
    """Retourne l'utilisateur si credentials valides, None sinon."""
    user = USERS.get(email.lower().strip())
    if user and user["pwd_hash"] == hash_pwd(pwd):
        return {**user, "email": email}
    return None

def get_user_pages(user: dict) -> dict:
    """Retourne le dict PAGES filtre selon le role de l'utilisateur."""
    ALL_PAGES = {
        "prediction":    "🔮 Prédiction Multi-Horizon",
        "anomaly":       "🔍 Détection d'Anomalies",
        "causal":        "🧠 IA Causale",
        "recommandation":"🗺️ Recommandation",
        "nlp":           "💬 NLP Incidents",
        "graph":         "🕸️ Graphe du Réseau",
        "whatif":        "🎲 Simulation What-If",
        "report":        "📊 Rapport & XAI",
        "mlops":         "⚙️ Configuration MLOps",
    }
    allowed = user.get("pages", [])
    return {v: k for k, v in ALL_PAGES.items() if k in allowed}

