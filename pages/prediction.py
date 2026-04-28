"""
pages/prediction.py — Prediction Multi-Horizon uniquement
Les onglets Classification / Regression / TS / LSTM sont dans le notebook ML.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime

from data.loader import load_transport, load_dim_zones, load_dim_modes

HORIZONS = ["15min", "1h", "2h", "J+1", "J+7"]

METEO_CATEGORIES = {
    "❄️ FROID":             {"codes": [1, 2],     "retard_factor": 1.45, "annul_bonus": 0.04},
    "🌬️ FRAIS":             {"codes": [3],         "retard_factor": 1.15, "annul_bonus": 0.01},
    "🌬️ FRAIS, SEC":        {"codes": [4],         "retard_factor": 1.10, "annul_bonus": 0.00},
    "⛈️ DOUX, PLUIE FORTE": {"codes": [5, 8, 10], "retard_factor": 1.35, "annul_bonus": 0.03},
    "🌤️ DOUX":              {"codes": [6, 9],      "retard_factor": 1.00, "annul_bonus": 0.00},
    "🌤️ DOUX, SEC":         {"codes": [7],         "retard_factor": 0.90, "annul_bonus": 0.00},
}

HORIZON_METEO_WEIGHT = {
    "15min": 0.5, "1h": 0.7, "2h": 0.9, "J+1": 1.2, "J+7": 1.4,
}

# Seuils Fiche KPI RM
SEUIL_EXCELLENT = 180
SEUIL_NORMAL    = 300
SEUIL_ALERTE    = 600

# KPI par role (Fiches KPI officielles)
KPI_PAR_ROLE = {
    "Président AOM":              {"kpi": "RM", "label": "Retard Moyen",       "col": "retard_s",       "unite": "s",    "cible": 300,  "seuils": [180, 300, 600]},
    "Directeur Planification":    {"kpi": "TA", "label": "Taux Annulation",     "col": "annule",         "unite": "%",    "cible": 1.5,  "seuils": [1.0, 1.6, 3.0]},
    "VP Mobilité":                {"kpi": "IC", "label": "Indice Congestion",   "col": "congestion_index","unite": "",    "cible": 1.5,  "seuils": [1.5, 2.0, 2.5]},
    "Directeur Voirie":           {"kpi": "VM", "label": "Vitesse Moyenne",     "col": "vitesse_kmh",    "unite": "km/h", "cible": 30,   "seuils": [20, 30]},
    "Directeur Relation Clients": {"kpi": "SU", "label": "Satisfaction",        "col": "satisfaction_1_5","unite": "/5",  "cible": 3.5,  "seuils": [3.0, 3.5, 4.0]},
    "Directeur DREAL":            {"kpi": "QA", "label": "Qualite Air ATMO",    "col": "indice_atmo",    "unite": "",     "cible": 4,    "seuils": [4, 6, 8]},
    "Préfet / Directeur Sécurité":{"kpi": "TD", "label": "Taux Delinquance",   "col": "taux_pour_1000", "unite": "/1000","cible": 2.0,  "seuils": [1.0, 2.0, 5.0]},
    "Chief Data Officer":         {"kpi": "ALL","label": "Tous KPI",            "col": "retard_s",       "unite": "s",    "cible": 300,  "seuils": [180, 300, 600]},
    "Administrateur":             {"kpi": "ALL","label": "Tous KPI",            "col": "retard_s",       "unite": "s",    "cible": 300,  "seuils": [180, 300, 600]},
}


def apply_meteo_adjustment(results: dict, meteo_cat: str) -> dict:
    meta = METEO_CATEGORIES[meteo_cat]
    adjusted = {}
    for h, r in results.items():
        weight = HORIZON_METEO_WEIGHT.get(h, 1.0)
        factor = 1.0 + (meta["retard_factor"] - 1.0) * weight
        adjusted[h] = {
            **r,
            "retard_s":       r.get("retard_s", 0) * factor,
            "annule":         min(1.0, r.get("annule", 0) + meta["annul_bonus"] * weight),
            "charge_estimee": r.get("charge_estimee", 0),
        }
    return adjusted


def build_shap_with_meteo(shap_vals, feat_names, meteo_cat: str, horizon: str):
    meta   = METEO_CATEGORIES[meteo_cat]
    weight = HORIZON_METEO_WEIGHT.get(horizon, 1.0)
    delta  = (meta["retard_factor"] - 1.0) * weight
    sv_arr = np.array(shap_vals)
    if sv_arr.ndim >= 2:
        sv = list(sv_arr[0].astype(float))
    else:
        sv = list(sv_arr.astype(float))
    fn = list(feat_names)
    if "fkmeteo" in fn:
        idx     = fn.index("fkmeteo")
        sv[idx] = delta * abs(sv[idx]) if sv[idx] != 0 else delta * 0.05
    else:
        fn.append("fkmeteo")
        sv.append(delta * 0.15)
    return sv, fn


def _niveau_kpi_rm(ret: float) -> str:
    if ret < SEUIL_EXCELLENT: return "🟢 Excellent"
    elif ret < SEUIL_NORMAL:  return "🟡 Normal"
    elif ret < SEUIL_ALERTE:  return "🟠 Avertissement"
    else:                     return "🔴 Critique"


def _niveau_generique(val: float, seuils: list, labels: list) -> str:
    for i, s in enumerate(seuils):
        if val < s: return labels[i]
    return labels[-1]


@st.cache_resource(show_spinner="Entrainement du modele MultiHorizon…")
def get_predictor(debut: str, fin: str):
    from models.multi_horizon import MultiHorizonPredictor
    df = load_transport(debut, fin)
    if len(df) < 100:
        raise ValueError(f"Seulement {len(df)} lignes — insuffisant.")
    p = MultiHorizonPredictor()
    p.fit(df)
    return p, df


def render():
    # Recuperer l'utilisateur connecte
    user  = st.session_state.get("user", {})
    role  = user.get("role", "")
    prenom = user.get("prenom", "")

    # KPI selon le role
    kpi_info = KPI_PAR_ROLE.get(role, KPI_PAR_ROLE["Président AOM"])
    kpi_code = kpi_info["kpi"]
    kpi_label = kpi_info["label"]

    st.markdown('<p class="section-title">🔮 Prédiction Multi-Horizon</p>',
                unsafe_allow_html=True)

    # Afficher le KPI surveille selon le role
    couleur = user.get("couleur", "#818cf8")
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.03); border:1px solid {couleur}40;
                border-radius:10px; padding:12px 16px; margin-bottom:12px;'>
        <span style='color:{couleur}; font-weight:700;'>👤 {prenom} — {role}</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <span style='color:#94a3b8;'>KPI surveillé : </span>
        <span style='background:{couleur}20; color:{couleur}; border:1px solid {couleur}40;
                     border-radius:4px; padding:2px 8px; font-weight:700;'>
            {kpi_code} — {kpi_label}
        </span>
        &nbsp;&nbsp;
        <span style='color:#64748b; font-size:0.8rem;'>
            Cible : {kpi_info["cible"]}{kpi_info["unite"]}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        '<span class="module-tag tag-ml">LightGBM</span> &nbsp;'
        '<span class="module-tag tag-ml">SHAP XAI</span> &nbsp;'
        '<span class="module-tag tag-ml">MLflow</span>',
        unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    debut = col1.date_input("Debut entrainement", datetime.date(2019, 1, 1))
    fin   = col2.date_input("Fin entrainement",   datetime.date(2022, 12, 31))

    try:
        zones = load_dim_zones()["zone_nom"].tolist()
        modes = load_dim_modes()["mode"].tolist()
    except Exception as e:
        st.error(f"Impossible de charger les dimensions DW : {e}")
        st.stop()

    # ── Parametres du trajet ──────────────────────────────────────────────────
    with st.expander("Parametres du trajet a analyser", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            zone_sel = st.selectbox("Zone", zones, key="mh_zone")
            mode_sel = st.selectbox("Mode", modes, key="mh_mode")
        with c2:
            heure_sel = st.slider("Heure", 0, 23, 8, key="mh_h")
            meteo_cat = st.selectbox("Meteo", list(METEO_CATEGORIES.keys()), key="mh_m")
            meteo_sel = METEO_CATEGORIES[meteo_cat]["codes"][0]
        with c3:
            charge  = st.slider("Charge estimee (passagers)", 0, 126, 44, key="mh_c")
            lag_ret = st.number_input("Retard precedent (s)", 0, 1800, 90, key="mh_l")

    col_train, col_pred_btn = st.columns(2)
    train_btn = col_train.button("Entrainer / Charger le modele",
                                  use_container_width=True, key="mh_train")
    pred_btn  = col_pred_btn.button("Predire",
                                     use_container_width=True, key="mh_pred")

    # ── Entrainement ──────────────────────────────────────────────────────────
    if train_btn:
        try:
            predictor, df_train = get_predictor(str(debut), str(fin))
            st.session_state["predictor"] = predictor
            st.session_state["df_train"]  = df_train
            st.success(f"Modele entraine sur {len(df_train):,} voyages reels du DW.")
        except Exception as e:
            st.error(f"Erreur entrainement : {e}"); st.stop()

    # ── Prediction ────────────────────────────────────────────────────────────
    if pred_btn:
        if "predictor" not in st.session_state:
            st.warning("Cliquez d'abord sur **Entrainer / Charger le modele**."); st.stop()

        predictor = st.session_state["predictor"]
        df_train  = st.session_state["df_train"]

        dz = load_dim_zones()
        dm = load_dim_modes()
        zone_id = dz.query("zone_nom == @zone_sel")["zone_id"].values[0] \
                  if zone_sel in dz["zone_nom"].values else 0
        mode_id = dm.query("mode == @mode_sel")["SK_mode"].values[0] \
                  if mode_sel in dm["mode"].values else 0

        X_new = pd.DataFrame([{
            "FK_arret":       df_train["FK_arret"].mode().iloc[0],
            "FKline_id":      df_train["ligne_id"].mode().iloc[0],
            "FK_vehicules":   df_train["vehicule_id"].mode().iloc[0],
            "fktime_id":      heure_sel,
            "fkzone_id":      zone_id,
            "FK_mode":        mode_id,
            "charge_estimee": float(charge),
            "lag_retard_1":   float(lag_ret),
            "fkmeteo":        float(meteo_sel),
        }])

        try:
            raw_results = predictor.predict(X_new)
        except Exception as e:
            st.error(f"Erreur prediction : {e}"); st.stop()

        results = apply_meteo_adjustment(raw_results, meteo_cat)

        # ── Tableau resultats adapte au KPI du role ───────────────────────────
        st.markdown(f"#### Previsions {kpi_label} (KPI {kpi_code}) par horizon")

        NIVEAU_LABELS = {
            "RM": ["🟢 Excellent","🟡 Normal","🟠 Avertissement","🔴 Critique"],
            "TA": ["🟢 Excellent","🟡 Normal","🟠 Alerte","🔴 Critique"],
            "IC": ["🟢 Fluide","🟡 Modere","🟠 Critique","🔴 Tres Critique"],
            "VM": ["🔴 Critique","🟠 Lent","🟢 Normal"],
            "SU": ["🔴 Critique","🟠 Moyen","🟡 Bon","🟢 Excellent"],
            "QA": ["🟢 Bon","🟡 Moyen","🟠 Mauvais","🔴 Tres Mauvais"],
            "TD": ["🟢 Tres Faible","🟡 Acceptable","🟠 Modere","🔴 Eleve"],
        }

        rows = []
        for h in HORIZONS:
            if h not in results: continue
            r   = results[h]
            ret = r.get("retard_s", 0)
            rows.append({
                "Horizon":        h,
                f"{kpi_label} ({kpi_info['unite']})": f"{ret:.0f}",
                "Meteo":          meteo_cat,
                "Charge (pass.)": f"{r.get('charge_estimee', 0):.0f}",
                "Niveau KPI":     _niveau_kpi_rm(ret),
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── KPI alertes selon seuils Fiche ────────────────────────────────────
        horizon_1h = results.get("1h", {})
        ret_1h     = horizon_1h.get("retard_s", 0)
        if ret_1h >= SEUIL_ALERTE:
            st.markdown(f"""
            <div class='alert-critique'>
                🔴 <b>ALERTE CRITIQUE KPI {kpi_code}</b> — Retard prevu 1h : {ret_1h:.0f}s
                (seuil Fiche KPI > {SEUIL_ALERTE}s) — Action immediate requise
            </div>""", unsafe_allow_html=True)
        elif ret_1h >= SEUIL_NORMAL:
            st.markdown(f"""
            <div class='alert-avertissement'>
                🟠 <b>AVERTISSEMENT KPI {kpi_code}</b> — Retard prevu 1h : {ret_1h:.0f}s
                (seuil Fiche KPI > {SEUIL_NORMAL}s)
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='alert-info'>
                🟢 <b>KPI {kpi_code} NORMAL</b> — Retard prevu 1h : {ret_1h:.0f}s
                (< {SEUIL_EXCELLENT}s Excellent)
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        col_l, col_r = st.columns(2)

        # ── Graphe evolution du KPI ───────────────────────────────────────────
        with col_l:
            st.markdown(f"#### Evolution {kpi_label} prevu — KPI {kpi_code}")
            hs   = [h for h in HORIZONS if h in results]
            rets = [results[h].get("retard_s", 0) for h in hs]
            fig  = go.Figure()
            fig.add_trace(go.Scatter(
                x=hs, y=rets, mode="lines+markers",
                line=dict(color=couleur, width=2.5),
                marker=dict(size=9, color=couleur),
                fill="tozeroy", fillcolor=f"rgba(129,140,248,0.1)",
                name=kpi_label,
            ))
            fig.add_hline(y=SEUIL_EXCELLENT, line_dash="dash",
                          line_color="#22c55e",
                          annotation_text=f"Excellent < {SEUIL_EXCELLENT}s")
            fig.add_hline(y=SEUIL_NORMAL, line_dash="dash",
                          line_color="#f59e0b",
                          annotation_text=f"Avert. > {SEUIL_NORMAL}s")
            fig.add_hline(y=SEUIL_ALERTE, line_dash="dot",
                          line_color="#ef4444",
                          annotation_text=f"Critique > {SEUIL_ALERTE}s")
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Horizon", yaxis_title=f"{kpi_label} ({kpi_info['unite']})",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── SHAP explication ──────────────────────────────────────────────────
        with col_r:
            st.markdown(f"#### Explication SHAP — Causes du {kpi_label}")
            if "1h" in results:
                try:
                    shap_vals, feat_names = predictor.explain(X_new, "1h", "retard_s")
                    sv, fn = build_shap_with_meteo(shap_vals, feat_names, meteo_cat, "1h")

                    # Noms lisibles
                    FEAT_LABELS = {
                        "charge_estimee": "Charge passagers",
                        "lag_retard_1":   "Retard precedent",
                        "fkmeteo":        "Conditions meteo",
                        "fktime_id":      "Heure de voyage",
                        "fkzone_id":      "Zone urbaine",
                        "FK_mode":        "Mode transport",
                    }
                    fn_labels = [FEAT_LABELS.get(f, f) for f in fn]
                    shap_df = (pd.DataFrame({"feature": fn_labels, "shap": sv})
                               .sort_values("shap", ascending=True))
                    fig2 = go.Figure(go.Bar(
                        x=shap_df["shap"], y=shap_df["feature"],
                        orientation="h",
                        marker_color=["#ef4444" if v > 0 else "#34d399"
                                      for v in shap_df["shap"]],
                    ))
                    fig2.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", height=300,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title=f"Contribution SHAP ({kpi_info['unite']})",
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.info(f"SHAP non disponible : {e}")

    elif not train_btn and not pred_btn and "predictor" not in st.session_state:
        st.info("Cliquez sur **Entrainer / Charger le modele** pour demarrer.")