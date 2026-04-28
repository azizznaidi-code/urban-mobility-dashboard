"""
train_from_dwh.py — Pipeline MLOps complet connecté au DW réel
Preprocessing → Training → Evaluation → Saving → MLflow
"""
import mlflow
import numpy as np
import pandas as pd
import joblib, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import IsolationForest

# ── Connexion MLflow ──────────────────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("UrbanMobility_AI_KPI_DW")

print("=" * 60)
print("PIPELINE MLOPS — DONNEES REELLES DW dwurbanmobility")
print("=" * 60)

# ── ETAPE 1 : PREPROCESSING — Chargement DW ──────────────────────────────────
print("\n[1/4] PREPROCESSING — Chargement depuis SQL Server DW...")

from data.loader import load_transport, load_trafic

df_transport = load_transport('2019-01-01', '2022-12-31')
df_trafic    = load_trafic('2019-01-01', '2022-12-31')

print(f"  Transport chargé : {df_transport.shape[0]:,} lignes ✅")
print(f"  Trafic chargé    : {df_trafic.shape[0]:,} lignes ✅")

# Features Transport pour classification + régression
FEAT_TRANSPORT = ['heure', 'mois_num', 'charge_estimee', 'annule']
TARGET_CLF     = 'retard_s'   # Classification : retard élevé ou non
TARGET_REG     = 'retard_s'   # Régression : prédire le retard en secondes

# Nettoyage
df_t = df_transport[FEAT_TRANSPORT + [TARGET_CLF]].dropna()
df_t['charge_estimee'] = pd.to_numeric(df_t['charge_estimee'], errors='coerce').fillna(0)
df_t['annule']         = pd.to_numeric(df_t['annule'], errors='coerce').fillna(0)
df_t['heure']          = pd.to_numeric(df_t['heure'], errors='coerce').fillna(0)
df_t['mois_num']       = pd.to_numeric(df_t['mois_num'], errors='coerce').fillna(1)
df_t[TARGET_CLF]       = pd.to_numeric(df_t[TARGET_CLF], errors='coerce').fillna(0)
df_t = df_t.dropna()

# Features Trafic pour Isolation Forest
FEAT_TRAFIC = ['vitesse_kmh', 'congestion_index', 'temps_trajet_min']
df_tr = df_trafic[FEAT_TRAFIC].dropna()
df_tr = df_tr.apply(pd.to_numeric, errors='coerce').dropna()

# Préparer X, y
X = df_t[FEAT_TRANSPORT].values
y_reg = df_t[TARGET_REG].values
y_clf = (y_reg > np.percentile(y_reg, 70)).astype(int)  # Top 30% = retard élevé

X_train, X_test, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
_, _, y_reg_train, y_reg_test             = train_test_split(X, y_reg, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"  Features : {FEAT_TRANSPORT}")
print(f"  Train : {X_train.shape[0]:,} | Test : {X_test.shape[0]:,}")
print(f"  Taux retard élevé : {y_clf.mean()*100:.1f}%")
print("  Preprocessing terminé ✅")

os.makedirs("models", exist_ok=True)

# ── ETAPE 2 : TRAINING + EVALUATION + SAVING ─────────────────────────────────
print("\n[2/4] TRAINING — Entraînement des modèles sur données DW réelles...")

# ── Run 1 : LightGBM Classifier ──────────────────────────────────────────────
print("\n  Entraînement LightGBM Classifier (KPI RM)...")
with mlflow.start_run(run_name="LGB_Clf_DW"):
    mlflow.log_param("model",         "LightGBM")
    mlflow.log_param("source",        "DW_dwurbanmobility")
    mlflow.log_param("n_train",       X_train.shape[0])
    mlflow.log_param("n_estimators",  200)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("features",      str(FEAT_TRANSPORT))
    mlflow.log_param("target",        "retard_élevé (top 30%)")

    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05,
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train_sc, y_clf_train)
    preds = lgb_model.predict(X_test_sc)

    roc = round(roc_auc_score(y_clf_test, preds), 4)
    f1  = round(f1_score(y_clf_test, preds), 4)
    acc = round(accuracy_score(y_clf_test, preds), 4)

    mlflow.log_metric("roc_auc",  roc)
    mlflow.log_metric("f1",       f1)
    mlflow.log_metric("accuracy", acc)

    joblib.dump(lgb_model, "models/lgb_clf.pkl")
    print(f"    ROC-AUC={roc} | F1={f1} | Accuracy={acc}")
    print("    lgb_clf.pkl sauvegardé ✅")

# ── Run 2 : XGBoost Regressor ────────────────────────────────────────────────
print("\n  Entraînement XGBoost Regressor (prédiction retard_s)...")
with mlflow.start_run(run_name="XGBoost_Reg_DW"):
    mlflow.log_param("model",         "XGBoost")
    mlflow.log_param("source",        "DW_dwurbanmobility")
    mlflow.log_param("n_train",       X_train.shape[0])
    mlflow.log_param("n_estimators",  150)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("features",      str(FEAT_TRANSPORT))
    mlflow.log_param("target",        "retard_s (secondes)")

    xgb_model = xgb.XGBRegressor(
        n_estimators=150, learning_rate=0.1,
        random_state=42, verbosity=0
    )
    xgb_model.fit(X_train_sc, y_reg_train)
    preds_reg = xgb_model.predict(X_test_sc)

    rmse = round(np.sqrt(mean_squared_error(y_reg_test, preds_reg)), 4)
    r2   = round(r2_score(y_reg_test, preds_reg), 4)
    mae  = round(np.mean(np.abs(y_reg_test - preds_reg)), 4)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2",   r2)
    mlflow.log_metric("mae",  mae)

    joblib.dump(xgb_model, "models/xgb_reg.pkl")
    print(f"    RMSE={rmse} | R2={r2} | MAE={mae}")
    print("    xgb_reg.pkl sauvegardé ✅")

# ── Run 3 : Isolation Forest (sur données trafic réelles) ────────────────────
print("\n  Entraînement Isolation Forest (KPI SAN — données trafic réelles)...")
with mlflow.start_run(run_name="IsoForest_SAN_DW"):
    X_trafic = df_tr.values

    mlflow.log_param("model",         "IsolationForest")
    mlflow.log_param("source",        "DW_facttrafic")
    mlflow.log_param("n_train",       X_trafic.shape[0])
    mlflow.log_param("n_estimators",  200)
    mlflow.log_param("contamination", 0.05)
    mlflow.log_param("features",      str(FEAT_TRAFIC))

    iso = IsolationForest(
        n_estimators=200, contamination=0.05, random_state=42
    )
    iso.fit(X_trafic)

    scores     = iso.decision_function(X_trafic)
    n_anomalies = int(np.sum(iso.predict(X_trafic) == -1))

    mlflow.log_metric("anomaly_score_mean", round(float(np.mean(scores)), 4))
    mlflow.log_metric("n_anomalies",        n_anomalies)
    mlflow.log_metric("pct_anomalies",      round(n_anomalies / len(X_trafic) * 100, 2))

    joblib.dump(iso, "models/iso_forest.pkl")
    print(f"    Anomalies détectées : {n_anomalies:,} ({n_anomalies/len(X_trafic)*100:.1f}%)")
    print("    iso_forest.pkl sauvegardé ✅")

# ── ETAPE 3 : RESUME FINAL ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[4/4] PIPELINE COMPLET — RESUME FINAL")
print("=" * 60)
print(f"  Source données  : DW dwurbanmobility (SQL Server 2022)")
print(f"  Transport       : {df_transport.shape[0]:,} lignes réelles")
print(f"  Trafic          : {df_trafic.shape[0]:,} lignes réelles")
print(f"  lgb_clf.pkl     : ROC-AUC={roc} ✅")
print(f"  xgb_reg.pkl     : RMSE={rmse} ✅")
print(f"  iso_forest.pkl  : {n_anomalies:,} anomalies ✅")
print(f"  MLflow runs     : 3 runs loggés dans UrbanMobility_AI_KPI_DW ✅")
print("=" * 60)
print("\nOuvre MLflow : http://127.0.0.1:5000")
print("Expérience   : UrbanMobility_AI_KPI_DW")
