"""
models/multi_task.py — Apprentissage Multi-Tâches (PyTorch).

Un seul réseau apprend simultanément :
  • retard_s          (régression)
  • satisfaction_1_5  (régression)
  • congestion_index  (régression)
  • annule            (classification binaire)

Avantage : le modèle capture les corrélations entre ces variables
(ex : fort congestion → retard élevé → satisfaction basse).
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from loguru import logger
import mlflow
from typing import Dict, Tuple

from config import settings


# ── Architecture du réseau partagé ──────────────────────────────────────
class MultiTaskNet(nn.Module):
    """
    Tronc partagé → 4 têtes spécialisées.
    Tronc : 3 couches FC + BatchNorm + Dropout
    Têtes  : 1 couche linéaire chacune
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Tronc partagé
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        dim = hidden_dim // 2
        # Têtes spécialisées
        self.head_retard        = nn.Linear(dim, 1)   # régression
        self.head_satisfaction  = nn.Linear(dim, 1)   # régression
        self.head_congestion    = nn.Linear(dim, 1)   # régression
        self.head_annule        = nn.Sequential(      # classification
            nn.Linear(dim, 1), nn.Sigmoid()
        )

    def forward(self, x):
        shared = self.trunk(x)
        return {
            "retard_s":         self.head_retard(shared).squeeze(-1),
            "satisfaction_1_5": self.head_satisfaction(shared).squeeze(-1),
            "congestion_index": self.head_congestion(shared).squeeze(-1),
            "annule":           self.head_annule(shared).squeeze(-1),
        }


# ── Perte multi-tâches pondérée ─────────────────────────────────────────
class MultiTaskLoss(nn.Module):
    """Poids appris automatiquement (incertitude homoscédastique – Kendall)."""

    def __init__(self):
        super().__init__()
        # log(sigma²) par tâche – initialisé à 0
        self.log_vars = nn.Parameter(torch.zeros(4))

    def forward(self, preds: dict, targets: dict) -> torch.Tensor:
        mse = nn.MSELoss()
        bce = nn.BCELoss()

        losses = [
            mse(preds["retard_s"],         targets["retard_s"]),
            mse(preds["satisfaction_1_5"], targets["satisfaction_1_5"]),
            mse(preds["congestion_index"], targets["congestion_index"]),
            bce(preds["annule"],           targets["annule"]),
        ]
        total = sum(
            torch.exp(-lv) * l + lv
            for lv, l in zip(self.log_vars, losses)
        )
        return total


# ── Wrapper de haut niveau ──────────────────────────────────────────────
class MultiTaskPredictor:

    FEATURE_COLS = [
        "FK_arret", "FKline_id", "FK_vehicules",
        "fktime_id", "fkzone_id", "FK_mode",
        "charge_estimee", "heure_sin", "heure_cos",
        "jour_semaine", "mois", "lag_retard_1",
    ]
    TARGET_COLS = [
        "retard_s", "satisfaction_1_5", "congestion_index", "annule",
    ]

    def __init__(self, hidden_dim: int = 128):
        self.hidden_dim = hidden_dim
        self.scaler     = StandardScaler()
        self.model: MultiTaskNet | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        feat_cols = [c for c in self.FEATURE_COLS if c in df.columns]
        X = self.scaler.fit_transform(df[feat_cols].fillna(0).values)
        y = {col: df[col].fillna(0).values.astype(np.float32)
             for col in self.TARGET_COLS if col in df.columns}
        return X.astype(np.float32), y

    def fit(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 512):
        X, y = self._prepare(df)
        X_tr, X_te, *splits = train_test_split(
            X, *[y[c] for c in y], test_size=0.2, random_state=42
        )
        y_tr = dict(zip(y.keys(), splits[:len(y)]))
        y_te = dict(zip(y.keys(), splits[len(y):]))

        self.model = MultiTaskNet(X_tr.shape[1], self.hidden_dim).to(self.device)
        criterion  = MultiTaskLoss().to(self.device)
        optimizer  = torch.optim.AdamW(
            list(self.model.parameters()) + list(criterion.parameters()),
            lr=1e-3, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        # DataLoader
        tensors = [torch.tensor(X_tr)] + [
            torch.tensor(y_tr[c]) for c in y_tr
        ]
        loader = DataLoader(
            TensorDataset(*tensors), batch_size=batch_size, shuffle=True
        )

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name="MultiTask_PyTorch"):
            mlflow.log_params({"epochs": epochs, "hidden_dim": self.hidden_dim})
            for epoch in range(epochs):
                self.model.train()
                epoch_loss = 0.0
                for batch in loader:
                    bx = batch[0].to(self.device)
                    by = dict(zip(y_tr.keys(),
                                  [b.to(self.device) for b in batch[1:]]))
                    optimizer.zero_grad()
                    out  = self.model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                scheduler.step()

                if epoch % 10 == 0:
                    logger.info(
                        f"Époque {epoch:3d}/{epochs} | loss={epoch_loss:.4f}"
                    )
                    mlflow.log_metric("train_loss", epoch_loss, step=epoch)

        logger.success("MultiTaskNet entraîné avec succès.")

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Modèle non entraîné – appelez fit() d'abord.")
        feat_cols = [c for c in self.FEATURE_COLS if c in df.columns]
        X = torch.tensor(
            self.scaler.transform(df[feat_cols].fillna(0).values).astype(np.float32)
        ).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(X)
        return {k: v.cpu().numpy() for k, v in out.items()}
