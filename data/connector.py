"""
data/connector.py — Connexion et chargement depuis dwurbanmobility (SQL Server).
"""
import urllib.parse
import pyodbc
import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger
from functools import lru_cache

from config import settings


def get_connection_string() -> str:
    params = urllib.parse.quote_plus(
        f"DRIVER={{{settings.DW_DRIVER}}};"
        f"SERVER={settings.DW_SERVER};"
        f"DATABASE={settings.DW_DATABASE};"
        f"UID={settings.DW_USERNAME};"
        f"PWD={settings.DW_PASSWORD};"
        f"TrustServerCertificate=yes;"
    )
    return f"mssql+pyodbc:///?odbc_connect={params}"


@lru_cache(maxsize=1)
def get_engine():
    """Retourne un moteur SQLAlchemy mis en cache."""
    engine = create_engine(get_connection_string(), fast_executemany=True)
    logger.info(f"Connexion DW établie → {settings.DW_DATABASE}")
    return engine


def load_fact_transport(date_debut: str = "2020-01-01",
                         date_fin:   str = "2099-12-31") -> pd.DataFrame:
    """Charge facttransport avec dimensions jointes."""
    sql = """
    SELECT
        ft.retard_s,
        ft.annule,
        ft.charge_estimee,
        ft.FK_arret,
        ft.FKline_id,
        ft.FK_vehicules,
        ft.time_id,
        dt.BK_date   AS date_voyage,
        dt.heure,
        dt.jour,
        dt.mois      AS mois_num,
        dt.annee
    FROM dbo.facttransport ft
    LEFT JOIN dbo.DIM_TEMPS dt ON ft.time_id = dt.time_id
    WHERE dt.BK_date BETWEEN :debut AND :fin
    """
    with get_engine().connect() as conn:
        df = pd.read_sql(text(sql), conn,
                         params={"debut": date_debut, "fin": date_fin})
    logger.info(f"facttransport : {len(df):,} lignes chargées")
    return df


def load_fact_trafic(date_debut: str = "2020-01-01",
                      date_fin:   str = "2099-12-31") -> pd.DataFrame:
    sql = """
    SELECT
        ft.vitesse_kmh,
        ft.temps_trajet_min,
        ft.congestion_index,
        ft.fkzone_id   AS zone_id,
        ft.fkmeteo     AS meteo_id,
        ft.fksensor    AS sensor_id,
        ft.fkevent     AS event_id,
        ft.date_id,
        dz.zone_nom,
        dz.lat_centre,
        dz.lon_centre
    FROM dbo.facttrfic ft
    LEFT JOIN dbo.dimzones dz ON ft.fkzone_id = dz.zone_id
    WHERE ft.date_id BETWEEN :debut AND :fin
    """
    with get_engine().connect() as conn:
        df = pd.read_sql(text(sql), conn,
                         params={"debut": date_debut, "fin": date_fin})
    return df


def load_fact_accidents() -> pd.DataFrame:
    sql = """
    SELECT
        fa.accident_id,
        fa.usager_vulnerable,
        fa.fK_severity      AS severity,
        fa.fktypeincident   AS type_incident,
        fa.fktime_id        AS time_id,
        fa.fkzone_id        AS zone_id,
        fa.nb_crimes,
        fa.taux_pour_1000,
        dz.zone_nom
    FROM dbo.factaccidents fa
    LEFT JOIN dbo.dimzones dz ON fa.fkzone_id = dz.zone_id
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn)


def load_network_topology() -> pd.DataFrame:
    """Charge la topologie réseau : arrêts pour le graphe."""
    sql = """
    SELECT
        da.SK_arret     AS arret_id,
        da.stop_nom,
        da.lat,
        da.lon,
        da.zone_id,
        dz.zone_nom
    FROM dbo.dimarret da
    LEFT JOIN dbo.dimzones dz ON da.zone_id = dz.zone_id
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn)
