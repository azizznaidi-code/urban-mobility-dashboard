"""
models/causal_ai.py — IA Causale avec DoWhy.

Répond à : "Pourquoi ce retard ?"
  → Météo à 70 % | Grève à 15 % | Événement à 10 % | Autre 5 %

DAG causal basé sur le DW dwurbanmobility :
  Meteo     → Retard
  Evenement → Retard
  Incident  → Retard
  Zone      → Retard, Congestion
  Congestion → Retard, Satisfaction
  Retard    → Satisfaction, Stress
"""
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict
from loguru import logger

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    logger.warning("DoWhy non installé — mode dégradé activé.")


# ── DAG Causal du transport urbain ────────────────────────────────────────
CAUSAL_GRAPH = """
    digraph {
        fkmeteo     -> retard_s;
        fkevent     -> retard_s;
        categorie_incident -> retard_s;
        congestion_index -> retard_s;
        fkzone_id   -> retard_s;
        fkzone_id   -> congestion_index;
        retard_s    -> satisfaction_1_5;
        retard_s    -> stress_1_5;
        congestion_index -> satisfaction_1_5;
        annule      -> satisfaction_1_5;
        FK_mode     -> retard_s;
        FK_mode     -> congestion_index;
    }
"""

CAUSE_LABELS = {
    "fkmeteo":            "Météo",
    "fkevent":            "Événement urbain",
    "categorie_incident": "Incident / Accident",
    "congestion_index":   "Congestion routière",
    "fkzone_id":          "Zone géographique",
    "annule":             "Annulation",
    "FK_mode":            "Mode de transport",
}


class CausalAnalyzer:
    """Analyse causale : identifie les facteurs qui expliquent retard_s."""

    def __init__(self):
        self.model  = None
        self.result = None

    def fit(self, df: pd.DataFrame):
        """
        Construit le modèle causal et estime l'effet de chaque cause.
        Nécessite les colonnes : retard_s, fkmeteo, fkevent,
        categorie_incident, congestion_index, fkzone_id.
        """
        if not DOWHY_AVAILABLE:
            logger.warning("DoWhy absent — estimation par corrélation partielle.")
            self._fit_fallback(df)
            return

        # Encoder les variables catégorielles
        df_work = df.copy()
        for col in ["fkzone_id", "FK_mode", "categorie_incident"]:
            if col in df_work.columns:
                df_work[col] = pd.Categorical(df_work[col]).codes

        self.model = CausalModel(
            data=df_work,
            treatment="congestion_index",   # cause principale à étudier
            outcome="retard_s",
            graph=CAUSAL_GRAPH,
        )
        identified = self.model.identify_effect(
            proceed_when_unidentifiable=True
        )
        self.result = self.model.estimate_effect(
            identified,
            method_name="backdoor.linear_regression",
        )
        if getattr(self.result, "value", None) is not None:
            logger.info(
                f"Effet causal de congestion → retard : "
                f"{self.result.value:.4f} s / unité"
            )
        else:
            logger.info("Effet causal de congestion → retard : Non estimé (None).")

    def _fit_fallback(self, df: pd.DataFrame):
        """Corrélations partielles comme proxy causal (sans DoWhy)."""
        causes   = [c for c in CAUSE_LABELS if c in df.columns]
        self._partial_corr = {}
        for c in causes:
            try:
                self._partial_corr[c] = abs(
                    df[c].fillna(0).corr(df["retard_s"].fillna(0))
                )
            except Exception:
                self._partial_corr[c] = 0.0

    def attribution_breakdown(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Retourne la part d'attribution causale de chaque facteur
        pour expliquer le retard observé.
        Format : {"Météo": 0.30, "Congestion": 0.45, ...}
        """
        causes = [c for c in CAUSE_LABELS if c in df.columns]
        raw    = {}

        if hasattr(self, "_partial_corr"):
            raw = {CAUSE_LABELS[c]: self._partial_corr.get(c, 0.0)
                   for c in causes}
        else:
            # Moyenne des valeurs absolues des colonnes × effet estimé
            for c in causes:
                mean_val = abs(df[c].fillna(0).mean())
                raw[CAUSE_LABELS[c]] = mean_val

        # Normaliser en % (somme = 1)
        total = sum(raw.values()) or 1.0
        return {k: round(v / total * 100, 1) for k, v in raw.items()}

    def explain_single_trip(
        self, row: pd.Series, threshold_s: float = 120
    ) -> Dict:
        """
        Explique un trajet individuel.
        Retourne un dict avec la cause principale et les poids.
        """
        retard = row.get("retard_s", 0)
        severy = "CRITIQUE" if retard > 600 else (
                 "ÉLEVÉ"    if retard > 300 else (
                 "MODÉRÉ"   if retard > threshold_s else "FAIBLE"))

        weights = {}
        if row.get("fkmeteo", 0) > 2:     # météo défavorable
            weights["Météo"] = 0.35
        if row.get("fkevent", 0) > 0:
            weights["Événement urbain"] = 0.25
        if row.get("categorie_incident", 0):
            weights["Incident / Accident"] = 0.20
        if row.get("congestion_index", 0) > 0.6:
            weights["Congestion routière"] = 0.40
        if row.get("annule", 0):
            weights["Annulation"] = 0.50

        if not weights:
            weights["Cause inconnue"] = 1.0
        total = sum(weights.values())
        weights = {k: round(v / total * 100, 1) for k, v in weights.items()}

        return {
            "retard_s":      retard,
            "severite":      severy,
            "cause_weights": weights,
            "cause_principale": max(weights, key=weights.get),
        }

    def refutation_test(self) -> Dict:
        """Test de réfutation DoWhy (si disponible)."""
        if self.model is None or self.result is None:
            return {"status": "Non disponible", "valid": None}
        try:
            ref = self.model.refute_estimate(
                self.result,
                method_name="random_common_cause",
            )
            valid = abs(ref.new_effect - self.result.value) < 0.1 * abs(self.result.value)
            return {
                "status":     "Validé" if valid else "Sensible",
                "original":   self.result.value,
                "new_effect": ref.new_effect,
                "valid":      valid,
            }
        except Exception as e:
            return {"status": f"Erreur: {e}", "valid": None}
