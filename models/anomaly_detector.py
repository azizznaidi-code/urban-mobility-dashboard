"""
models/anomaly_detector.py — Détection d'Anomalies Proactive.

Deux niveaux de détection :
  1. Isolation Forest    — signaux faibles multi-variables (rapide)
  2. Autoencoder PyTorch — anomalies subtiles dans les séries temporelles

Pour chaque anomalie :
  • Score d'anomalie (0-1)
  • Cause probable via SHAP
  • Niveau de sévérité : INFO / AVERTISSEMENT / CRITIQUE
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from loguru import logger
from typing import Dict, List, Tuple
import mlflow

from config import settings


# ── Autoencoder pour la détection d'anomalies temporelles ────────────────
class AnomalyAutoencoder(nn.Module):
    """
    Encode → Bouteille → Décode.
    L'erreur de reconstruction est le score d'anomalie.
    """

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon = self(x)
            return ((x - recon) ** 2).mean(dim=1)


# ── Détecteur principal ───────────────────────────────────────────────────
class AnomalyDetector:

    FEATURE_COLS = [
        "retard_s", "vitesse_kmh", "congestion_index",
        "charge_estimee", "temps_trajet_min",
        "fkmeteo", "fksensor",
    ]
    SEVERITY_THRESHOLDS = {
        "CRITIQUE":      0.80,
        "AVERTISSEMENT": 0.60,
        "INFO":          0.40,
    }

    def __init__(self):
        self.iso_forest:  IsolationForest | None     = None
        self.autoencoder: AnomalyAutoencoder | None  = None
        self.scaler       = StandardScaler()
        self.threshold_ae = 0.0   # seuil percentile 95
        self.explainer    = None
        self.device       = torch.device("cpu")

    # ── Entraînement ────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, ae_epochs: int = 40):
        feat_cols = [c for c in self.FEATURE_COLS if c in df.columns]
        X = self.scaler.fit_transform(df[feat_cols].fillna(0).values).astype(np.float32)

        # — Isolation Forest ———————————————————————————————————————————
        self.iso_forest = IsolationForest(
            contamination=settings.ANOMALY_CONTAMINATION,
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        self.iso_forest.fit(X)
        # Explainer SHAP pour l'Isolation Forest
        sample = shap.sample(X, min(settings.SHAP_SAMPLE_SIZE, len(X)))
        self.explainer = shap.Explainer(
            self.iso_forest.predict, sample,
            feature_names=feat_cols,
        )
        logger.info("Isolation Forest entraîné.")

        # — Autoencoder ————————————————————————————————————————————————
        X_tr, X_val = train_test_split(X, test_size=0.1, random_state=42)
        self.autoencoder = AnomalyAutoencoder(X.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        with mlflow.start_run(run_name="AnomalyAutoencoder"):
            mlflow.log_param("ae_epochs", ae_epochs)
            for epoch in range(ae_epochs):
                self.autoencoder.train()
                bx   = torch.tensor(X_tr).to(self.device)
                recon = self.autoencoder(bx)
                loss  = criterion(recon, bx)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                if epoch % 10 == 0:
                    mlflow.log_metric("ae_loss", loss.item(), step=epoch)

        # Seuil : percentile 95 sur les données d'entraînement normales
        X_t = torch.tensor(X_tr).to(self.device)
        errs = self.autoencoder.reconstruction_error(X_t).numpy()
        self.threshold_ae = float(np.percentile(errs, 95))
        logger.success(f"Autoencoder entraîné. Seuil AE = {self.threshold_ae:.4f}")

    # ── Détection ────────────────────────────────────────────────────────
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retourne le DataFrame enrichi de colonnes :
          anomaly_iso, anomaly_ae, anomaly_score, severity, shap_cause
        """
        feat_cols = [c for c in self.FEATURE_COLS if c in df.columns]
        X = self.scaler.transform(df[feat_cols].fillna(0).values).astype(np.float32)

        # — Isolation Forest ———————————————————————————————————————────
        iso_scores = -self.iso_forest.score_samples(X)   # plus haut = + anormal
        iso_norm   = (iso_scores - iso_scores.min()) / (
            iso_scores.max() - iso_scores.min() + 1e-9
        )
        iso_flag   = self.iso_forest.predict(X) == -1   # -1 = anomalie

        # — Autoencoder ────────────────────────────────────────────────
        X_t    = torch.tensor(X)
        ae_err = self.autoencoder.reconstruction_error(X_t).numpy()
        ae_norm = (ae_err - ae_err.min()) / (ae_err.max() - ae_err.min() + 1e-9)
        ae_flag = ae_err > self.threshold_ae

        # — Score combiné (50/50) ──────────────────────────────────────
        combined = 0.5 * iso_norm + 0.5 * ae_norm

        # — Sévérité ───────────────────────────────────────────────────
        def severity(score: float) -> str:
            for level, thresh in self.SEVERITY_THRESHOLDS.items():
                if score >= thresh:
                    return level
            return "NORMAL"

        result = df.copy()
        result["anomaly_iso"]   = iso_flag
        result["anomaly_ae"]    = ae_flag
        result["anomaly_score"] = np.round(combined, 3)
        result["anomaly"]       = iso_flag | ae_flag
        result["severity"]      = [severity(s) for s in combined]

        # — Cause principale via SHAP ─────────────────────────────────
        anomalous_idx = np.where(iso_flag)[0]
        if len(anomalous_idx) > 0 and self.explainer is not None:
            sample = X[anomalous_idx[:min(50, len(anomalous_idx))]]
            try:
                shap_vals = self.explainer(sample)
                top_cause = [
                    feat_cols[int(np.argmax(np.abs(shap_vals.values[i])))]
                    for i in range(len(sample))
                ]
                result.loc[result.index[anomalous_idx[:len(top_cause)]],
                           "shap_cause"] = top_cause
            except Exception as e:
                logger.warning(f"SHAP cause error: {e}")

        n_anomalies = result["anomaly"].sum()
        logger.info(f"Anomalies détectées : {n_anomalies} / {len(df)}")
        return result

    # ── Top anomalies ────────────────────────────────────────────────────
    def top_anomalies(self, df_detected: pd.DataFrame,
                      n: int = 20) -> pd.DataFrame:
        return (
            df_detected[df_detected["anomaly"]]
            .nlargest(n, "anomaly_score")
            .reset_index(drop=True)
        )

    # ── Distribution des causes ───────────────────────────────────────────
    def cause_distribution(self, df_detected: pd.DataFrame) -> Dict[str, int]:
        if "shap_cause" not in df_detected.columns:
            return {}
        return (
            df_detected["shap_cause"].dropna().value_counts().to_dict()
        )
