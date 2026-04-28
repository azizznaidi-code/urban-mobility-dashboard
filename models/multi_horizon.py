import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from typing import Dict, List, Tuple
from loguru import logger


class MultiHorizonPredictor:
    """
    Modèle de prédiction multi-horizon pour UrbanAI.
    Prédit retard_s, annule, et charge_estimee pour différents horizons :
    15min, 1h, 2h, J+1, J+7.
    """

    HORIZONS = ["15min", "1h", "2h", "J+1", "J+7"]
    TARGETS  = ["retard_s", "annule", "charge_estimee"]

    FEATURE_COLS = [
        "FK_arret", "FKline_id", "FK_vehicules",
        "fktime_id", "fkzone_id", "FK_mode",
        "charge_estimee", "lag_retard_1", "fkmeteo",
        "heure", "mois_num", "annee", "jour_semaine",
    ]

    def __init__(self):
        self.models     = {}   # { (horizon, target): model }
        self.explainers = {}   # { (horizon, target): explainer }
        self.is_fitted  = False

    # ── Conversion heure ────────────────────────────────────────────────
    def _convert_heure(self, series: pd.Series) -> pd.Series:
        """char 'HH:MM:SS' → entier 0-23."""
        try:
            return (series.astype(str)
                          .str.strip()
                          .str[:2]
                          .replace('', '0')
                          .astype(int))
        except Exception:
            return pd.Series([0] * len(series))

    # ── Préparation des features ─────────────────────────────────────────
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Conversion heure char → int
        if "heure" in df.columns:
            df["heure"] = self._convert_heure(df["heure"])

        # Jour de la semaine depuis date_voyage
        if "date_voyage" in df.columns:
            df["date_voyage"]  = pd.to_datetime(df["date_voyage"], errors="coerce")
            df["jour_semaine"] = df["date_voyage"].dt.dayofweek.fillna(0)
        else:
            df["jour_semaine"] = 0

        # lag_retard_1 si absent
        if "lag_retard_1" not in df.columns:
            if "retard_s" in df.columns:
                df = df.sort_values(["FK_arret", "date_voyage"] if "date_voyage" in df.columns else ["FK_arret"])
                df["lag_retard_1"] = df.groupby("FK_arret")["retard_s"].shift(1).fillna(0)
            else:
                df["lag_retard_1"] = 0

        # Colonnes manquantes → 0
        for col in self.FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df[self.FEATURE_COLS]

    # ── Entraînement ─────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame):
        logger.info(f"Début entraînement MultiHorizon sur {len(df)} lignes…")

        X = self._prepare_data(df)

        for target in self.TARGETS:
            y = pd.to_numeric(df[target], errors="coerce").fillna(0)

            if target == "annule":
                y_bin = (y > 0).astype(int)
                for horizon in self.HORIZONS:
                    model = lgb.LGBMClassifier(
                        n_estimators=100, random_state=42,
                        verbose=-1, n_jobs=-1,
                    )
                    model.fit(X, y_bin)
                    self.models[(horizon, target)] = model
            else:
                for horizon in self.HORIZONS:
                    model = lgb.LGBMRegressor(
                        n_estimators=100, random_state=42,
                        verbose=-1, n_jobs=-1,
                    )
                    model.fit(X, y)
                    self.models[(horizon, target)] = model

        self.is_fitted = True
        logger.success("Entraînement MultiHorizon terminé ✅")

    # ── Prédiction ───────────────────────────────────────────────────────
    def predict(self, X_new: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant la prédiction.")

        X       = self._prepare_data(X_new)
        results = {}

        for horizon in self.HORIZONS:
            results[horizon] = {}
            for target in self.TARGETS:
                model = self.models.get((horizon, target))
                if model is None:
                    continue
                pred = model.predict(X)[0]
                if target == "annule":
                    pred = float(min(1.0, max(0.0, pred)))
                else:
                    pred = float(max(0.0, pred))
                results[horizon][target] = pred

        return results

    # ── Explication SHAP ─────────────────────────────────────────────────
    def explain(
        self, X_new: pd.DataFrame, horizon: str, target: str
    ) -> Tuple[np.ndarray, List[str]]:

        model = self.models.get((horizon, target))
        if model is None:
            raise ValueError(f"Modèle non trouvé pour {horizon}/{target}")

        X = self._prepare_data(X_new)

        if (horizon, target) not in self.explainers:
            self.explainers[(horizon, target)] = shap.TreeExplainer(model)

        explainer  = self.explainers[(horizon, target)]
        shap_vals  = explainer.shap_values(X)

        # LGBM Classification → liste [class0, class1] → prendre class1
        if isinstance(shap_vals, list):
            sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        else:
            sv = shap_vals

        sv = np.array(sv)

        # Forcer en 2D
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)

        return sv, self.FEATURE_COLS