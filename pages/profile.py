"""
Composant profil utilisateur (sidebar)
"""
import streamlit as st

def render_profile():
    """Affiche le profil utilisateur dans la sidebar."""
    user = st.session_state.get("user", {})
    if not user:
        return

    couleur = user.get("couleur", "#818cf8")

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e293b, #0f172a);
                border: 1px solid {couleur}40; border-radius:12px; padding:16px; margin-bottom:12px;'>
        <div style='display:flex; align-items:center; gap:10px;'>
            <div style='width:42px; height:42px; border-radius:50%;
                        background: linear-gradient(135deg, {couleur}, {couleur}80);
                        display:flex; align-items:center; justify-content:center;
                        font-size:1.2rem; font-weight:700; color:white;'>
                {user.get("prenom","?")[0]}{user.get("nom","?")[0]}
            </div>
            <div>
                <div style='font-weight:700; color:#f1f5f9; font-size:0.9rem;'>
                    {user.get("prenom","")} {user.get("nom","")}
                </div>
                <div style='color:{couleur}; font-size:0.72rem; font-weight:600;'>
                    {user.get("role","")}
                </div>
            </div>
        </div>
        <div style='margin-top:10px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.06);'>
            <div style='color:#64748b; font-size:0.7rem;'>{user.get("email","")}</div>
            <div style='margin-top:6px;'>
                <span style='font-size:0.68rem; color:#94a3b8;'>KPIs surveillés : </span>
                {''.join([f"<span style='background:{couleur}20; color:{couleur}; border:1px solid {couleur}40; border-radius:4px; padding:1px 6px; font-size:0.65rem; margin:1px; display:inline-block;'>{k}</span>" for k in user.get("kpis",[])])}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Se déconnecter", use_container_width=True):
        for key in ["user", "authenticated", "current_page"]:
            st.session_state.pop(key, None)
        st.rerun()
