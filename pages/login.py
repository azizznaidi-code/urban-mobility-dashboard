"""
pages/login.py — Page de connexion + Création de compte UrbanAI
"""
import streamlit as st
from auth import login, USERS, hash_pwd

# Roles disponibles selon Fiches KPI
ROLES_DISPONIBLES = {
    "Président AOM":           {"kpis": ["RM","TA","CM","SU"],        "pages": ["prediction","anomaly","recommandation","report","mlops"],                          "couleur": "#818cf8"},
    "Directeur Planification": {"kpis": ["TA","CM","VM","TTM"],       "pages": ["prediction","anomaly","whatif","report","mlops"],                                  "couleur": "#34d399"},
    "VP Mobilité":             {"kpis": ["IC","VM","TTM"],            "pages": ["prediction","graph","recommandation","report","mlops"],                             "couleur": "#f59e0b"},
    "Directeur Voirie":        {"kpis": ["IC","VM","TTM"],            "pages": ["prediction","graph","whatif","report","mlops"],                                     "couleur": "#f59e0b"},
    "Directeur Relation Clients": {"kpis": ["SU","RM","TA"],          "pages": ["prediction","nlp","recommandation","report","mlops"],                               "couleur": "#ec4899"},
    "Directeur DREAL":         {"kpis": ["QA","CO2"],                 "pages": ["prediction","anomaly","causal","report","mlops"],                                   "couleur": "#22c55e"},
    "Préfet / Directeur Sécurité": {"kpis": ["TD","SAN"],            "pages": ["prediction","nlp","anomaly","report","mlops"],                                      "couleur": "#ef4444"},
    "Chief Data Officer":      {"kpis": ["RM","TA","CM","IC","VM","SU","QA","TD","SAN"], "pages": ["prediction","anomaly","causal","recommandation","nlp","graph","whatif","report","mlops"], "couleur": "#4f46e5"},
}

def render():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Header
        st.markdown("""
        <div style='text-align:center; padding: 30px 0 16px 0;'>
            <div style='font-size:3.5rem;'>🏙️</div>
            <h1 style='background: linear-gradient(90deg, #818cf8, #34d399);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 2rem; font-weight: 800; margin: 0;'>UrbanAI</h1>
            <p style='color:#64748b; font-size:0.9rem; margin-top:4px;'>
                Mobilité Intelligente · DW Urban Mobility
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Onglets Connexion / Créer un compte
        tab_login, tab_register = st.tabs(["🔐 Connexion", "✨ Créer un compte"])

        # ── TAB 1 : CONNEXION ─────────────────────────────────────────────────
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            email = st.text_input("📧 Email professionnel",
                                   placeholder="prenom.nom@urbanmobility.fr",
                                   key="login_email")
            pwd   = st.text_input("🔑 Mot de passe", type="password",
                                   placeholder="••••••••", key="login_pwd")

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            login_btn = st.button("Se connecter", use_container_width=True,
                                   type="primary", key="btn_login")

            if login_btn:
                if not email or not pwd:
                    st.error("Veuillez saisir votre email et mot de passe.")
                else:
                    user = login(email, pwd)
                    if user:
                        st.session_state["user"]          = user
                        st.session_state["authenticated"] = True
                        st.success(f"Bienvenue {user['prenom']} {user['nom']} !")
                        st.rerun()
                    else:
                        st.error("Email ou mot de passe incorrect.")

            st.markdown("---")
            st.markdown("""
            <p style='text-align:center; color:#475569; font-size:0.75rem;'>
            Accès sécurisé · Données confidentielles AOM<br>
            Vous n'avez pas de compte ? Utilisez l'onglet <b>Créer un compte</b>
            </p>
            """, unsafe_allow_html=True)

        # ── TAB 2 : CRÉER UN COMPTE ───────────────────────────────────────────
        with tab_register:
            st.markdown("<br>", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            prenom = col_a.text_input("👤 Prénom", placeholder="Marie", key="reg_prenom")
            nom    = col_b.text_input("👤 Nom",    placeholder="Dupont",  key="reg_nom")

            email_reg = st.text_input("📧 Email professionnel",
                                       placeholder="marie.dupont@urbanmobility.fr",
                                       key="reg_email")

            role_sel = st.selectbox(
                "🎭 Rôle (selon Fiche KPI officielle)",
                list(ROLES_DISPONIBLES.keys()),
                key="reg_role",
            )

            # Afficher les KPIs et pages du role selectionne
            role_info = ROLES_DISPONIBLES[role_sel]
            couleur   = role_info["couleur"]

            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03); border:1px solid {couleur}30;
                        border-radius:8px; padding:10px 14px; margin:4px 0 10px 0;'>
                <div style='font-size:0.75rem; color:#94a3b8; margin-bottom:4px;'>
                    KPIs surveillés pour ce rôle :
                </div>
                <div>
                    {''.join([f"<span style='background:{couleur}20; color:{couleur}; border:1px solid {couleur}40; border-radius:4px; padding:2px 7px; font-size:0.7rem; margin:2px; display:inline-block;'>{k}</span>" for k in role_info["kpis"]])}
                </div>
                <div style='font-size:0.72rem; color:#64748b; margin-top:6px;'>
                    Pages accessibles : {len(role_info["pages"])} modules
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_p1, col_p2 = st.columns(2)
            pwd_reg  = col_p1.text_input("🔑 Mot de passe", type="password",
                                          placeholder="Min. 8 caractères", key="reg_pwd")
            pwd_reg2 = col_p2.text_input("🔑 Confirmer", type="password",
                                          placeholder="Répéter le mot de passe", key="reg_pwd2")

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            register_btn = st.button("Créer mon compte", use_container_width=True,
                                      type="primary", key="btn_register")

            if register_btn:
                # Validations
                errors = []
                if not prenom.strip(): errors.append("Le prénom est requis.")
                if not nom.strip():    errors.append("Le nom est requis.")
                if not email_reg.strip(): errors.append("L'email est requis.")
                if "@" not in email_reg: errors.append("Email invalide.")
                if email_reg.lower() in USERS: errors.append("Cet email est déjà utilisé.")
                if len(pwd_reg) < 8:   errors.append("Le mot de passe doit faire au moins 8 caractères.")
                if pwd_reg != pwd_reg2: errors.append("Les mots de passe ne correspondent pas.")

                if errors:
                    for err in errors:
                        st.error(err)
                else:
                    # Créer le compte
                    new_user = {
                        "nom":       nom.strip(),
                        "prenom":    prenom.strip(),
                        "role":      role_sel,
                        "role_full": role_sel,
                        "kpis":      role_info["kpis"],
                        "pages":     role_info["pages"],
                        "pwd_hash":  hash_pwd(pwd_reg),
                        "couleur":   couleur,
                        "email":     email_reg.lower().strip(),
                    }
                    # Ajouter en session (persistance en memoire)
                    USERS[email_reg.lower().strip()] = new_user

                    st.success(f"✅ Compte créé pour {prenom} {nom} !")
                    st.info(f"🎭 Rôle : **{role_sel}** · KPIs : {', '.join(role_info['kpis'])}")

                    # Connexion automatique
                    st.session_state["user"]          = new_user
                    st.session_state["authenticated"] = True
                    st.balloons()
                    st.rerun()

            st.markdown("---")
            st.markdown("""
            <p style='text-align:center; color:#475569; font-size:0.75rem;'>
            Le compte est créé pour cette session.<br>
            Pour un accès permanent, contactez l'administrateur :<br>
            <b style='color:#818cf8'>admin@urbanmobility.fr</b>
            </p>
            """, unsafe_allow_html=True)