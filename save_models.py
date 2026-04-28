import joblib
import os
import numpy as np
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.makedirs("models", exist_ok=True)

# ── Données synthétiques pour entraîner ───────────────────────
np.random.seed(42)
n = 5000

X = np.column_stack([
    np.random.randint(0, 2, n),      # est_weekend
    np.random.randint(0, 24, n),     # heure
    np.random.uniform(10, 90, n),    # charge_estimee
    np.random.randint(0, 10, n),     # zone_id
    np.random.randint(0, 4, n),      # mode_encoded
    np.random.uniform(0, 50, n),     # pluie_mm
    np.random.randint(1, 13, n),     # mois_num
])

y_reg = (
    118 +
    (X[:, 1] > 7) * (X[:, 1] < 9) * 55 +
    (X[:, 2] - 44.7) * 0.3 +
    X[:, 5] * 2.5 +
    np.random.normal(0, 30, n)
)
y_clf = (y_reg >= 300).astype(int)

# ── LightGBM Classifier ───────────────────────────────────────
print("Entrainement LightGBM...")
lgb_model = Pipeline([
    ('sc', StandardScaler()),
    ('clf', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
])
lgb_model.fit(X, y_clf)
joblib.dump(lgb_model, "models/lgb_clf.pkl")
print("lgb_clf.pkl sauvegarde ✅")

# ── XGBoost Regressor ─────────────────────────────────────────
print("Entrainement XGBoost...")
xgb_model = Pipeline([
    ('sc', StandardScaler()),
    ('reg', xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0))
])
xgb_model.fit(X, y_reg)
joblib.dump(xgb_model, "models/xgb_reg.pkl")
print("xgb_reg.pkl sauvegarde ✅")

# ── Isolation Forest ──────────────────────────────────────────
print("Entrainement Isolation Forest...")
iso_model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)
iso_model.fit(X)
joblib.dump(iso_model, "models/iso_forest.pkl")
print("iso_forest.pkl sauvegarde ✅")

print("\nTous les modeles sauvegardes dans models/ ✅")
