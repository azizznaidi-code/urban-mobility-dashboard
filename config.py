"""
config.py — Centralise tous les paramètres de connexion et de configuration.
DW : dwurbanmobility (SQL Server 2022, MSSQL16)
"""
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # ── Connexion SQL Server ─────────────────────────────────────────────
    DW_SERVER:   str = os.getenv("DW_SERVER",   "NEGRAMARIEM\\MSSQLSERVER")
    DW_DATABASE: str = os.getenv("DW_DATABASE", "dwurbanmobility")
    DW_USERNAME: str = os.getenv("DW_USERNAME", "sa")
    DW_PASSWORD: str = os.getenv("DW_PASSWORD", "")
    DW_DRIVER: str = "ODBC Driver 17 for SQL Server"

    # ── MLflow ────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "sqlite:///mlruns.db"
    MLFLOW_EXPERIMENT:   str = "UrbanMobility_AI"

    # ── Horizons de prédiction ────────────────────────────────────────────
    HORIZONS_MINUTES: list = [15, 60, 120, 1440, 10080]   # 15min, 1h, 2h, J+1, J+7

    # ── Paramètres des modèles ─────────────────────────────────────────────
    ANOMALY_CONTAMINATION: float = 0.05   # 5 % d'anomalies attendues
    SHAP_SAMPLE_SIZE:      int   = 500    # Taille échantillon SHAP

    # ── Colonnes DW (extraites du backup dwurbanmobility.bak) ─────────────
    # Fact_Transport
    TRANSPORT_FEATURES: list = [
        "retard_s", "annule", "charge_estimee",
        "FK_arret", "FKline_id", "FK_vehicules",
        "fktime_id", "fkzone_id", "FK_mode",
    ]
    TRANSPORT_TARGETS: list = [
        "retard_s", "annule", "charge_estimee",
        "satisfaction_1_5", "stress_1_5",
    ]

    # Fact_Trafic
    TRAFIC_FEATURES: list = [
        "vitesse_kmh", "temps_trajet_min", "congestion_index",
        "fkzone_id", "fkmeteo", "fksensor", "fkevent", "date_id",
    ]

    # Fact_Criminalite
    CRIME_FEATURES: list = [
        "nb_crimes", "taux_pour_1000", "fK_severity",
        "fkzone_id", "fktime_id",
    ]

    # Fact_Incidents
    INCIDENT_FEATURES: list = [
        "categorie_incident", "accident_id", "usager_vulnerable",
        "fK_severity", "fktypeincident", "fktime_id", "fkzone_id",
    ]

    class Config:
        env_file = ".env"

settings = Settings()
