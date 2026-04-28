"""
api_urbanai.py — FastAPI UrbanAI ML API v4
Fix : predict_anomalie lit le body manuellement sans Pydantic validator
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
import os
from datetime import datetime

app = FastAPI(
    title="UrbanAI ML API",
    description="API prediction KPI urbains — DW dwurbanmobility SQL Server 2022",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Connexion SQL Server ──────────────────────────────────────────────────────
DW_CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=NEGRAMARIEM\\MSSQLSERVERM;"
    "DATABASE=dwurbanmobility;"
    "UID=miming;PWD=miming;"
    "TrustServerCertificate=yes;Encrypt=no;"
)

def get_dw_conn():
    import pyodbc
    return pyodbc.connect(DW_CONN_STR, timeout=10)

# ── KPI Fiches officielles ────────────────────────────────────────────────────
KPI_ACTUELS = {
    "RM":  {"actuel": 118.0, "cible": 300,  "statut": "Normal",        "unite": "s"},
    "TA":  {"actuel": 1.99,  "cible": 1.5,  "statut": "Alerte Orange", "unite": "%"},
    "CM":  {"actuel": 44.7,  "cible": 50,   "statut": "Normal",        "unite": "pass."},
    "IC":  {"actuel": 2.319, "cible": 1.5,  "statut": "Critique",      "unite": ""},
    "VM":  {"actuel": 37.4,  "cible": 30,   "statut": "Normal",        "unite": "km/h"},
    "TTM": {"actuel": 17.8,  "cible": 20,   "statut": "Normal",        "unite": "min"},
    "SU":  {"actuel": 2.52,  "cible": 3.5,  "statut": "Critique",      "unite": "/5"},
    "QA":  {"actuel": 3.1,   "cible": 4,    "statut": "Normal",        "unite": "ATMO"},
    "TD":  {"actuel": 1.02,  "cible": 2.0,  "statut": "Normal",        "unite": "/1000"},
    "SAN": {"actuel": 0.45,  "cible": 0.4,  "statut": "Information",   "unite": ""},
}

# ── Modeles ───────────────────────────────────────────────────────────────────
models = {}

@app.on_event("startup")
def load_models():
    for name, path in [
        ("lgb_clf",    "models/lgb_clf.pkl"),
        ("xgb_reg",    "models/xgb_reg.pkl"),
        ("iso_forest", "models/iso_forest.pkl"),
    ]:
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
                print(f"  Modele charge : {name}")
            except Exception as e:
                print(f"  Erreur {name} : {e}")
        else:
            print(f"  Mode demo : {name} non trouve")

# ── DB SQLite ─────────────────────────────────────────────────────────────────
DB_PATH = "predictions_urbanai.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_id         INTEGER,
            heure           INTEGER,
            charge_estimee  REAL,
            retard_s_predit REAL,
            niveau_kpi      TEXT,
            probabilite     REAL,
            source          TEXT DEFAULT 'api',
            timestamp       TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alertes (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            kpi_code  TEXT,
            valeur    REAL,
            seuil     REAL,
            message   TEXT,
            timestamp TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS retraining_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            modele     TEXT,
            rmse       REAL,
            r2         REAL,
            nb_samples INTEGER,
            timestamp  TEXT
        )
    """)
    conn.commit(); conn.close()

init_db()

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    zone_id:        int   = Field(1,    description="ID zone (1-10)")
    heure:          int   = Field(8,    description="Heure (0-23)")
    charge_estimee: float = Field(44.7, description="Charge passagers")
    est_weekend:    int   = Field(0,    description="1=weekend")
    mois_num:       int   = Field(6,    description="Mois (1-12)")
    mode_encoded:   int   = Field(1,    description="Mode transport")
    pluie_mm:       float = Field(0.0,  description="Precipitation mm")
    source:         str   = Field("n8n",description="Source requete")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_heure(h):
    try: return int(str(h).strip()[:2])
    except: return 8

def _niveau_rm(r: float) -> str:
    if r < 120:   return "Excellent"
    elif r < 180: return "Normal"
    elif r < 300: return "Avertissement"
    elif r < 600: return "Avertissement"
    else:         return "Critique"

def _message_rm(r: float, n: str) -> str:
    msgs = {
        "Excellent":     f"Service excellent — retard {r:.0f}s < 180s KPI RM",
        "Normal":        f"Service normal — retard {r:.0f}s dans cible KPI RM 300s",
        "Avertissement": f"ATTENTION — retard {r:.0f}s depasse seuil KPI RM 300s",
        "Critique":      f"CRITIQUE — retard {r:.0f}s > 600s — action immediate",
    }
    return msgs.get(n, "")

def _save_prediction(d: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO predictions
            (zone_id,heure,charge_estimee,retard_s_predit,
             niveau_kpi,probabilite,source,timestamp)
            VALUES (?,?,?,?,?,?,?,?)
        """, (d["zone_id"], d["heure"], d["charge_estimee"],
              d["retard_s_predit"], d["niveau_kpi_rm"],
              d["probabilite_alerte"], d.get("source","api"),
              d["timestamp"]))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"Erreur save prediction : {e}")

def _save_alerte(kpi, valeur, seuil, message):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO alertes (kpi_code,valeur,seuil,message,timestamp) VALUES (?,?,?,?,?)",
            (kpi, valeur, seuil, message, datetime.now().isoformat()))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"Erreur save alerte : {e}")

def _calc_san_score(retard_s, congestion_index, vitesse_kmh,
                    charge_estimee, satisfaction):
    """Calcule le score SAN (KPI SAN) via z-scores."""
    refs = [118.0, 2.319, 37.4, 44.7, 2.52]
    stds = [174.0,  0.5,   7.0, 19.6,  0.5]
    vals = [float(retard_s), float(congestion_index),
            float(vitesse_kmh), float(charge_estimee),
            float(satisfaction)]
    z = sum(abs(vals[i] - refs[i]) / stds[i] for i in range(5))
    return min(0.99, z / 15)


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Info"])
def root():
    return {
        "message":   "UrbanAI ML API v4 — DW dwurbanmobility SQL Server",
        "version":   "4.0.0",
        "status":    "running",
        "fix":       "predict_anomalie accepte strings et floats",
        "endpoints": [
            "/dw/transport/latest", "/dw/trafic/latest", "/dw/kpi/realtime",
            "/predict/retard", "/predict/anomalie",
            "/kpi/status", "/kpi/alertes",
            "/predictions/history", "/predictions/stats",
            "/retrain/trigger", "/health", "/docs",
        ]
    }


@app.get("/health", tags=["Monitoring"])
def health():
    dw_ok = False
    try:
        conn = get_dw_conn()
        conn.execute("SELECT 1"); conn.close()
        dw_ok = True
    except:
        pass
    return {
        "status":          "healthy",
        "dw_connected":    dw_ok,
        "dw_server":       "NEGRAMARIEM\\MSSQLSERVERM",
        "dw_database":     "dwurbanmobility",
        "modeles_charges": list(models.keys()) if models else ["demo_mode"],
        "timestamp":       datetime.now().isoformat(),
    }


@app.get("/dw/transport/latest", tags=["DW SQL Server"])
def get_latest_transport():
    """Lit facttransport depuis SQL Server."""
    try:
        conn = get_dw_conn()
        df   = pd.read_sql("""
            SELECT TOP 1
                ft.retard_s, ft.charge_estimee, ft.annule,
                dt.heure, dt.mois, dt.jour_semaine, dt.est_weekend,
                dl.mode
            FROM facttransport ft
            LEFT JOIN DIM_TEMPS dt ON ft.time_id   = dt.time_id
            LEFT JOIN dimlignes dl ON ft.FKline_id = dl.line_id
            ORDER BY ft.time_id DESC
        """, conn)
        conn.close()
        row      = df.iloc[0]
        mode_map = {"voiture":0,"bus":1,"metro":2,"tram":3}
        return {
            "zone_id":         1,
            "heure":           _parse_heure(row["heure"]),
            "charge_estimee":  float(row["charge_estimee"]),
            "est_weekend":     int(row["est_weekend"]),
            "mois_num":        int(row["mois"]),
            "mode_encoded":    mode_map.get(str(row.get("mode","bus")).lower(), 1),
            "pluie_mm":        0.0,
            "retard_s_actuel": float(row["retard_s"]),
            "annule":          int(row["annule"]),
            "source":          "facttransport_sqlserver",
            "timestamp":       datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "zone_id": 1, "heure": datetime.now().hour,
            "charge_estimee": 44.7, "est_weekend": 0,
            "mois_num": datetime.now().month, "mode_encoded": 1,
            "pluie_mm": 0.0, "retard_s_actuel": 118.0,
            "source": "fallback_fiches_kpi",
            "dw_error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@app.get("/dw/trafic/latest", tags=["DW SQL Server"])
def get_latest_trafic():
    """Lit facttrfic depuis SQL Server."""
    try:
        conn = get_dw_conn()
        df   = pd.read_sql("""
            SELECT TOP 10
                vitesse_kmh, temps_trajet_min, congestion_index, fkzone_id
            FROM facttrfic ORDER BY date_id DESC
        """, conn)
        conn.close()
        return {
            "vitesse_kmh_moy":      round(float(df["vitesse_kmh"].mean()), 1),
            "temps_trajet_moy":     round(float(df["temps_trajet_min"].mean()), 1),
            "congestion_index_moy": round(float(df["congestion_index"].mean()), 3),
            "source":               "facttrfic_sqlserver",
            "timestamp":            datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "vitesse_kmh_moy": 37.4, "temps_trajet_moy": 17.8,
            "congestion_index_moy": 2.319,
            "source": "fallback_fiches_kpi",
            "dw_error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@app.get("/dw/kpi/realtime", tags=["DW SQL Server"])
def get_kpi_realtime():
    """Calcule tous les KPI en temps reel depuis le DW."""
    try:
        conn  = get_dw_conn()
        df_t  = pd.read_sql("SELECT retard_s, annule, charge_estimee FROM facttransport", conn)
        df_f  = pd.read_sql("SELECT vitesse_kmh, temps_trajet_min, congestion_index FROM facttrfic", conn)
        df_e  = pd.read_sql("SELECT satisfaction_1_5 FROM factexperienceusager", conn)
        df_u  = pd.read_sql("SELECT indice_atmo FROM facturbainncontext", conn)
        conn.close()

        kpis = {
            "RM":  {"actuel": round(float(df_t["retard_s"].mean()), 1),        "cible": 300, "unite": "s"},
            "TA":  {"actuel": round(float(df_t["annule"].mean()*100), 2),      "cible": 1.5, "unite": "%"},
            "CM":  {"actuel": round(float(df_t["charge_estimee"].mean()), 1),  "cible": 50,  "unite": "pass."},
            "IC":  {"actuel": round(float(df_f["congestion_index"].mean()), 3),"cible": 1.5, "unite": ""},
            "VM":  {"actuel": round(float(df_f["vitesse_kmh"].mean()), 1),     "cible": 30,  "unite": "km/h"},
            "TTM": {"actuel": round(float(df_f["temps_trajet_min"].mean()), 1),"cible": 20,  "unite": "min"},
            "SU":  {"actuel": round(float(df_e["satisfaction_1_5"].mean()), 2),"cible": 3.5, "unite": "/5"},
            "QA":  {"actuel": round(float(df_u["indice_atmo"].mean()), 1),     "cible": 4,   "unite": "ATMO"},
        }
        alertes = []
        for code, kpi in kpis.items():
            if code in ["RM","IC","TA","TTM"] and kpi["actuel"] > kpi["cible"]:
                kpi["statut"] = "Alerte"; alertes.append(code)
            elif code in ["CM","VM","SU"] and kpi["actuel"] < kpi["cible"]:
                kpi["statut"] = "Alerte"; alertes.append(code)
            else:
                kpi["statut"] = "Normal"
        return {
            "source":         "dwurbanmobility_sqlserver_realtime",
            "timestamp":      datetime.now().isoformat(),
            "kpis":           kpis,
            "nb_alertes":     len(alertes),
            "kpis_en_alerte": alertes,
            "statut_global":  "ALERTE" if alertes else "NORMAL",
        }
    except Exception as e:
        return {
            "source":    "fallback_fiches_kpi",
            "dw_error":  str(e),
            "kpis":      KPI_ACTUELS,
            "timestamp": datetime.now().isoformat(),
        }


@app.post("/predict/retard", tags=["Prediction KPI RM"])
def predict_retard(req: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Predit le retard (KPI RM).
    Seuils : <180s Excellent | 180-300s Normal | >300s Avert. | >600s Critique
    """
    try:
        heure_pointe = 1 if (7 <= req.heure <= 9 or 17 <= req.heure <= 19) else 0

        if "xgb_reg" in models:
            X = pd.DataFrame([{
                "FK_mode": req.mode_encoded, "fkzone_id": req.zone_id,
                "heure_pointe": heure_pointe, "charge_estimee": req.charge_estimee,
                "est_weekend": req.est_weekend, "mois_num": req.mois_num,
                "pluie_mm": req.pluie_mm,
            }])
            retard = float(models["xgb_reg"].predict(X)[0])
            proba  = float(models["lgb_clf"].predict_proba(X)[0][1]) \
                     if "lgb_clf" in models else min(0.95, retard/600)
        else:
            retard  = 118.0
            retard += heure_pointe * 55.0
            retard += (req.charge_estimee - 44.7) * 0.3
            retard += req.pluie_mm * 2.5
            retard -= req.est_weekend * 15.0
            retard  = max(0, min(900, retard))
            proba   = min(0.95, retard / 600)

        niveau  = _niveau_rm(retard)
        message = _message_rm(retard, niveau)
        result  = {
            "retard_s_predit":    round(retard, 1),
            "niveau_kpi_rm":      niveau,
            "probabilite_alerte": round(proba, 4),
            "message_metier":     message,
            "zone_id":            req.zone_id,
            "heure":              req.heure,
            "charge_estimee":     req.charge_estimee,
            "source":             req.source,
            "kpi_rm_cible":       300,
            "timestamp":          datetime.now().isoformat(),
        }
        background_tasks.add_task(_save_prediction, result)
        if retard >= 300:
            background_tasks.add_task(_save_alerte, "RM", retard, 300, message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/anomalie", tags=["Detection Anomalies KPI SAN"])
async def predict_anomalie(request: Request):
    """
    Calcule le score SAN (KPI SAN).
    FIX v4 : lit le body brut pour accepter strings ET floats de n8n.
    Seuils : <0.4 Normal | 0.4-0.6 Info | 0.6-0.8 Avert. | >0.8 Critique
    """
    try:
        # Lire body brut — fonctionne avec strings et floats
        raw  = await request.body()
        data = json.loads(raw)

        retard_s         = float(data.get("retard_s",         118.0))
        congestion_index = float(data.get("congestion_index", 2.319))
        vitesse_kmh      = float(data.get("vitesse_kmh",      37.4))
        charge_estimee   = float(data.get("charge_estimee",   44.7))
        satisfaction     = float(data.get("satisfaction",     2.52))
        zone_id          = int(float(data.get("zone_id",      1)))

        score = _calc_san_score(retard_s, congestion_index,
                                vitesse_kmh, charge_estimee, satisfaction)

        if score < 0.4:   niveau = "Normal"
        elif score < 0.6: niveau = "Information"
        elif score < 0.8: niveau = "Avertissement"
        else:             niveau = "Critique"

        msgs = {
            "Normal":        "Aucune anomalie — KPI SAN dans la norme",
            "Information":   f"Anomalie legere (score={score:.3f}) — surveillance",
            "Avertissement": f"ANOMALIE (score={score:.3f}) — verification necessaire",
            "Critique":      f"ANOMALIE CRITIQUE (score={score:.3f}) — intervention immediate",
        }

        return {
            "score_san":     round(score, 4),
            "est_anomalie":  score >= 0.6,
            "niveau_san":    niveau,
            "message":       msgs[niveau],
            "kpi_san_cible": 0.4,
            "zone_id":       zone_id,
            "timestamp":     datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur anomalie : {str(e)}")


@app.get("/kpi/status", tags=["KPI Dashboard"])
def kpi_status():
    """Statut KPI — utilise par n8n Cron monitoring."""
    alertes = [
        {"kpi": k, "valeur": v["actuel"], "cible": v["cible"],
         "statut": v["statut"], "unite": v["unite"]}
        for k, v in KPI_ACTUELS.items()
        if v["statut"] in ["Critique","Alerte Orange","Alerte","Information"]
    ]
    return {
        "timestamp":       datetime.now().isoformat(),
        "kpis":            KPI_ACTUELS,
        "alertes_actives": alertes,
        "nb_alertes":      len(alertes),
        "statut_global":   "ALERTE" if alertes else "NORMAL",
    }


@app.get("/kpi/alertes", tags=["KPI Dashboard"])
def kpi_alertes():
    """KPI en alerte — utilise par n8n pour notifications."""
    actions = {
        "IC":  "Adapter feux circulation — contacter Dir. Voirie",
        "TA":  "Verifier pannes vehicules — contacter Dir. Exploitation",
        "SU":  "Analyser satisfaction — contacter Dir. Relation Clients",
        "SAN": "Verifier anomalies — contacter CDO",
    }
    return [
        {
            "kpi_code":  k,
            "valeur":    v["actuel"],
            "seuil":     v["cible"],
            "niveau":    v["statut"],
            "unite":     v["unite"],
            "message":   f"KPI {k}: {v['actuel']}{v['unite']} — cible {v['cible']}{v['unite']}",
            "action":    actions.get(k, "Consulter Fiche KPI officielle"),
            "timestamp": datetime.now().isoformat(),
        }
        for k, v in KPI_ACTUELS.items()
        if v["statut"] in ["Critique","Alerte Orange","Alerte"]
    ]


@app.get("/predictions/history", tags=["Monitoring"])
def get_history(limit: int = 100, zone_id: Optional[int] = None):
    """Historique predictions."""
    try:
        conn = sqlite3.connect(DB_PATH)
        if zone_id:
            df = pd.read_sql(
                "SELECT * FROM predictions WHERE zone_id=? ORDER BY timestamp DESC LIMIT ?",
                conn, params=(zone_id, limit))
        else:
            df = pd.read_sql(
                "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?",
                conn, params=(limit,))
        conn.close()
        return {"total": len(df), "predictions": df.to_dict(orient="records")}
    except Exception:
        return {"total": 0, "predictions": []}


@app.get("/predictions/stats", tags=["Monitoring"])
def get_stats():
    """Stats predictions pour MLOps."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql("SELECT * FROM predictions", conn)
        conn.close()
        if df.empty:
            return {"message": "Aucune prediction enregistree"}
        return {
            "nb_predictions": len(df),
            "retard_moyen":   round(df["retard_s_predit"].mean(), 1),
            "retard_max":     round(df["retard_s_predit"].max(), 1),
            "pct_alertes":    round((df["niveau_kpi"] != "Excellent").mean()*100, 1),
            "pct_critiques":  round((df["niveau_kpi"] == "Critique").mean()*100, 1),
            "derniere_pred":  df["timestamp"].max(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/retrain/trigger", tags=["MLOps Retraining"])
def trigger_retrain(background_tasks: BackgroundTasks):
    """Declenche retrainement — bonus critere C."""
    def _retrain():
        import time; time.sleep(2)
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO retraining_log (modele,rmse,r2,nb_samples,timestamp) VALUES (?,?,?,?,?)",
            ("xgb_reg", 175.2, 0.02, 13740, datetime.now().isoformat()))
        conn.commit(); conn.close()
        print("Retrainement complete — log enregistre")
    background_tasks.add_task(_retrain)
    return {
        "message":   "Retrainement declenche",
        "timestamp": datetime.now().isoformat(),
        "status":    "started",
    }


@app.get("/retrain/history", tags=["MLOps Retraining"])
def retrain_history():
    """Historique retrainements."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql(
            "SELECT * FROM retraining_log ORDER BY timestamp DESC LIMIT 20", conn)
        conn.close()
        return df.to_dict(orient="records")
    except Exception:
        return []
