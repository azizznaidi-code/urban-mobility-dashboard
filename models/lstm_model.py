"""
models/lstm_model.py — LSTM PyTorch pour prédiction de retard_s (série temporelle).
Données : facttransport + DIM_TEMPS (2019-2022).
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict


class LSTMPredictor:
    """
    LSTM en PyTorch pour prédire retard_s en série temporelle.
    Séquences glissantes de longueur `seq_len`.
    """

    def __init__(self, seq_len: int = 24, hidden_size: int = 64,
                 num_layers: int = 2, epochs: int = 30, lr: float = 1e-3):
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.epochs      = epochs
        self.lr          = lr
        self.model       = None
        self.scaler      = None
        self.feature_col = "retard_s"
        self.trained     = False
        self.train_losses: list = []

    # ── Données ───────────────────────────────────────────────────────────────

    def _prepare_series(self, df: pd.DataFrame) -> np.ndarray:
        """
        Regroupe facttransport par heure (mois+jour+heure) et retourne
        la série retard_s normalisée.
        """
        df = df.copy()
        if "heure" in df.columns:
            # char 'HH:MM:SS' → entier 0-23
            df["heure"] = (df["heure"].astype(str)
                           .str.strip().str[:2]
                           .replace('', '0').astype(int))

        ts = (df.groupby(["annee", "mois_num", "jour", "heure"])["retard_s"]
              .mean().reset_index().sort_values(["annee", "mois_num", "jour", "heure"]))
        return ts["retard_s"].values.astype(np.float32)

    def _make_sequences(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(series) - self.seq_len):
            X.append(series[i: i + self.seq_len])
            y.append(series[i + self.seq_len])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ── Modèle ────────────────────────────────────────────────────────────────

    def _build_model(self, input_size: int = 1):
        import torch.nn as nn
        import torch

        class _LSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True, dropout=0.2)
                self.fc   = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)      # (batch, seq, hidden)
                return self.fc(out[:, -1]) # dernière étape temporelle

        return _LSTM(input_size, self.hidden_size, self.num_layers)

    # ── Entraînement ─────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, progress_callback=None) -> "LSTMPredictor":
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import MinMaxScaler

        raw   = self._prepare_series(df)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = self.scaler.fit_transform(raw.reshape(-1, 1)).flatten()

        X, y = self._make_sequences(scaled)
        # (samples, seq_len, features)
        X_t = torch.tensor(X).unsqueeze(-1)   # (N, seq, 1)
        y_t = torch.tensor(y).unsqueeze(-1)   # (N, 1)

        split = int(0.8 * len(X_t))
        X_train, X_val = X_t[:split], X_t[split:]
        y_train, y_val = y_t[:split], y_t[split:]

        self.model = self._build_model()
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion  = nn.MSELoss()

        self.train_losses = []
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out   = self.model(X_train)
            loss  = criterion(out, y_train)
            loss.backward()
            optimizer.step()
            self.train_losses.append(float(loss.item()))
            if progress_callback:
                progress_callback(epoch + 1, self.epochs, float(loss.item()))

        # Validation predictions stored
        self.model.eval()
        with torch.no_grad():
            y_pred_val = self.model(X_val).numpy().flatten()
            y_true_val = y_val.numpy().flatten()

        # Inverse transform
        self.y_true_val = self.scaler.inverse_transform(
            y_true_val.reshape(-1, 1)).flatten()
        self.y_pred_val = self.scaler.inverse_transform(
            y_pred_val.reshape(-1, 1)).flatten()

        self.trained = True
        return self

    # ── Métriques ─────────────────────────────────────────────────────────────

    def metrics(self) -> Dict[str, float]:
        if not self.trained:
            raise RuntimeError("Modèle non entraîné.")
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        y_t = self.y_true_val
        y_p = self.y_pred_val
        mse  = float(mean_squared_error(y_t, y_p))
        return {
            "MSE":  round(mse, 2),
            "RMSE": round(float(np.sqrt(mse)), 2),
            "MAE":  round(float(mean_absolute_error(y_t, y_p)), 2),
            "R2":   round(float(r2_score(y_t, y_p)), 4),
        }

    # ── Prédiction rolling ────────────────────────────────────────────────────

    def predict_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne (y_true, y_pred) sur le jeu de validation."""
        if not self.trained:
            raise RuntimeError("Modèle non entraîné.")
        return self.y_true_val, self.y_pred_val
