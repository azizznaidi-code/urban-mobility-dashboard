import pandas as pd
import streamlit as st
from sqlalchemy import text
from loguru import logger
from data.connector import get_engine


def _sql(query: str, params: dict | None = None) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn, params=params or {})


@st.cache_data(ttl=600, show_spinner="Chargement facttransport…")
def load_transport(date_debut: str = "2019-01-01",
                   date_fin:   str = "2022-12-31") -> pd.DataFrame:
    df = _sql("""
        SELECT
            ft.retard_s,
            ft.annule,
            ft.charge_estimee,
            ft.FK_arret,
            ft.FKline_id    AS ligne_id,
            ft.FK_vehicules AS vehicule_id,
            ft.time_id      AS fktime_id,
            dt.BK_date      AS date_voyage,
            dt.heure,
            dt.jour,
            dt.mois         AS mois_num,
            dt.annee,
            dt.periode
        FROM  dbo.facttransport ft
        LEFT JOIN dbo.DIM_TEMPS dt ON ft.time_id = dt.time_id
        WHERE dt.BK_date BETWEEN :debut AND :fin
    """, {"debut": date_debut, "fin": date_fin})
    df["date_voyage"] = pd.to_datetime(df["date_voyage"], errors="coerce")
    logger.info(f"facttransport : {len(df):,} lignes")
    return df


@st.cache_data(ttl=600, show_spinner="Chargement facttrfic…")
def load_trafic(date_debut: str = "2019-01-01",
                date_fin:   str = "2022-12-31") -> pd.DataFrame:
    df = _sql("""
        SELECT
            ft.vitesse_kmh,
            ft.temps_trajet_min,
            ft.congestion_index,
            ft.fkzone_id    AS zone_id,
            ft.fkmeteo      AS meteo_id,
            ft.fksensor     AS sensor_id,
            ft.fkevent      AS event_id,
            ft.date_id,
            dt.BK_date      AS date_trafic,
            dt.heure,
            dz.zone_nom,
            dz.lat_centre,
            dz.lon_centre
        FROM  dbo.facttrfic ft
        LEFT JOIN dbo.DIM_TEMPS dt ON ft.date_id = dt.date_id
        LEFT JOIN dbo.dimzones  dz ON ft.fkzone_id = dz.zone_id
        WHERE dt.BK_date BETWEEN :debut AND :fin
    """, {"debut": date_debut, "fin": date_fin})
    logger.info(f"facttrfic : {len(df):,} lignes")
    return df


@st.cache_data(ttl=600, show_spinner="Chargement factaccidents…")
def load_accidents() -> pd.DataFrame:
    df = _sql("""
        SELECT
            fa.accident_id,
            fa.usager_vulnerable,
            fa.fK_severity    AS severity,
            fa.fktypeincident AS type_incident,
            fa.fktime_id      AS time_id,
            fa.fkzone_id      AS zone_id,
            fa.nb_crimes,
            fa.taux_pour_1000,
            dz.zone_nom
        FROM  dbo.factaccidents fa
        LEFT JOIN dbo.dimzones dz ON fa.fkzone_id = dz.zone_id
    """)
    logger.info(f"factaccidents : {len(df):,} lignes")
    return df


@st.cache_data(ttl=3600)
def load_dim_zones() -> pd.DataFrame:
    return _sql("SELECT zone_id, zone_nom, ville, region, pays, lat_centre, lon_centre FROM dbo.dimzones")

@st.cache_data(ttl=3600)
def load_dim_arrets() -> pd.DataFrame:
    return _sql("""
        SELECT da.SK_arret AS arret_id, da.stop_nom, da.lat, da.lon,
               da.zone_id, COALESCE(dz.zone_nom, da.ville) AS zone_nom
        FROM   dbo.dimarret da
        LEFT JOIN dbo.dimzones dz ON da.zone_id = dz.zone_id
        WHERE  da.SK_arret IS NOT NULL
          AND  da.lat IS NOT NULL
          AND  da.lon IS NOT NULL
    """)

@st.cache_data(ttl=3600)
def load_dim_temps() -> pd.DataFrame:
    return _sql("SELECT time_id, date_id, BK_date, heure, jour, mois, annee, periode FROM dbo.DIM_TEMPS")

@st.cache_data(ttl=3600)
def load_dim_modes() -> pd.DataFrame:
    return _sql("SELECT * FROM dbo.dimmodes")

@st.cache_data(ttl=3600)
def load_dim_lignes() -> pd.DataFrame:
    return _sql("SELECT * FROM dbo.dimlignes")

@st.cache_data(ttl=3600)
def load_dim_sensor() -> pd.DataFrame:
    return _sql("SELECT * FROM dbo.dimsensor")


@st.cache_data(ttl=300, show_spinner="Calcul des KPIs…")
def kpis_globaux(date_debut: str, date_fin: str) -> dict:
    df_t  = load_transport(date_debut, date_fin)
    df_tr = load_trafic(date_debut, date_fin)
    return {
        "retard_moyen":    round(float(df_t["retard_s"].mean()),          1),
        "retard_median":   round(float(df_t["retard_s"].median()),        1),
        "taux_annulation": round(float(df_t["annule"].mean()) * 100,      2),
        "charge_moyenne":  round(float(df_t["charge_estimee"].mean()),    1),
        "congestion_moy":  round(float(df_tr["congestion_index"].mean()), 3),
        "vitesse_moy_kmh": round(float(df_tr["vitesse_kmh"].mean()),      1),
        "nb_voyages":      len(df_t),
    }

@st.cache_data(ttl=300)
def retard_journalier(date_debut: str, date_fin: str) -> pd.DataFrame:
    df = load_transport(date_debut, date_fin)
    df["jour_date"] = df["date_voyage"].dt.date
    return (df.groupby("jour_date")["retard_s"]
              .mean().reset_index()
              .rename(columns={"jour_date": "jour", "retard_s": "retard_moy"}))

@st.cache_data(ttl=300)
def retard_par_zone(date_debut: str, date_fin: str) -> pd.DataFrame:
    df_tr = load_trafic(date_debut, date_fin)
    df_t  = load_transport(date_debut, date_fin)
    retard_global = float(df_t["retard_s"].mean())
    zone_stats = (df_tr.groupby("zone_nom")[["congestion_index","vitesse_kmh"]]
                  .mean().reset_index())
    zone_stats["retard_moy"] = retard_global * (1 + zone_stats["congestion_index"] * 0.5)
    return zone_stats.sort_values("retard_moy", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=300)
def retard_par_ligne(date_debut: str, date_fin: str) -> pd.DataFrame:
    df = load_transport(date_debut, date_fin)
    return (df.groupby("ligne_id")["retard_s"]
              .agg(["mean","count"])
              .rename(columns={"mean":"retard_moy","count":"nb"})
              .reset_index()
              .sort_values("retard_moy", ascending=False))

@st.cache_data(ttl=300)
def congestion_heure_zone(date_debut: str, date_fin: str) -> pd.DataFrame:
    df = load_trafic(date_debut, date_fin)
    if "heure" not in df.columns or df["heure"].nunique() <= 1:
        df["heure"] = df.index % 24
    else:
        try:
            df["heure"] = pd.to_datetime(df["heure"], format="%H:%M:%S",
                                          errors="coerce").dt.hour
        except Exception:
            df["heure"] = df.index % 24
    return df.pivot_table(index="zone_nom", columns="heure",
                           values="congestion_index", aggfunc="mean")

@st.cache_data(ttl=300)
def anomalies_from_dw(date_debut: str, date_fin: str) -> pd.DataFrame:
    df_tr = load_trafic(date_debut, date_fin)
    df_t  = load_transport(date_debut, date_fin)
    df_tr["retard_s"]       = float(df_t["retard_s"].mean())
    df_tr["charge_estimee"] = float(df_t["charge_estimee"].mean())
    df_tr["annule"]         = float(df_t["annule"].mean())
    if len(df_t) == len(df_tr):
        df_tr["retard_s"]       = df_t["retard_s"].values
        df_tr["charge_estimee"] = df_t["charge_estimee"].values
        df_tr["annule"]         = df_t["annule"].values
    return df_tr