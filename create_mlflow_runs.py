import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import joblib, os

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("UrbanMobility_AI_KPI")

np.random.seed(42)
n = 5000
X = np.column_stack([
    np.random.randint(0, 2, n),
    np.random.randint(0, 24, n),
    np.random.uniform(10, 90, n),
    np.random.randint(0, 10, n),
    np.random.randint(0, 4, n),
    np.random.uniform(0, 50, n),
    np.random.randint(1, 13, n),
])
y_reg = 118 + (X[:,1]>7)*(X[:,1]<9)*55 + X[:,2]*0.3 + np.random.normal(0,30,n)
y_clf = (y_reg >= 140).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2)

# Run 1 — Random Forest
with mlflow.start_run(run_name="RF_Clf"):
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    mlflow.log_metric("roc_auc", round(roc_auc_score(y_test, preds), 4))
    mlflow.log_metric("f1", round(f1_score(y_test, preds), 4))
    mlflow.log_metric("accuracy", round(accuracy_score(y_test, preds), 4))
    print("Run 1 RF ✅")

# Run 2 — LightGBM
with mlflow.start_run(run_name="LGB_Clf"):
    mlflow.log_param("model", "LightGBM")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("SMOTE", True)
    lgb_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    preds = lgb_model.predict(X_test)
    mlflow.log_metric("roc_auc", round(roc_auc_score(y_test, preds), 4))
    mlflow.log_metric("f1", round(f1_score(y_test, preds), 4))
    mlflow.log_metric("accuracy", round(accuracy_score(y_test, preds), 4))
    print("Run 2 LGB ✅")

# Run 3 — XGBoost Regressor
with mlflow.start_run(run_name="XGBoost_Reg"):
    from sklearn.metrics import mean_squared_error, r2_score
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y_reg, test_size=0.2)
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("n_estimators", 150)
    mlflow.log_param("learning_rate", 0.1)
    xgb_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, random_state=42, verbosity=0)
    xgb_model.fit(X_tr2, y_tr2)
    preds = xgb_model.predict(X_te2)
    mlflow.log_metric("rmse", round(np.sqrt(mean_squared_error(y_te2, preds)), 4))
    mlflow.log_metric("r2", round(r2_score(y_te2, preds), 4))
    mlflow.log_metric("mae", round(np.mean(np.abs(y_te2 - preds)), 4))
    print("Run 3 XGBoost ✅")

# Run 4 — Isolation Forest
with mlflow.start_run(run_name="IsolationForest_SAN"):
    from sklearn.metrics import silhouette_score
    mlflow.log_param("model", "IsolationForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("contamination", 0.05)
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(X)
    scores = iso.decision_function(X)
    mlflow.log_metric("anomaly_score_mean", round(float(np.mean(scores)), 4))
    mlflow.log_metric("n_anomalies", int(np.sum(iso.predict(X) == -1)))
    print("Run 4 IsolationForest ✅")

print("\n✅ 4 runs créés dans MLflow !")
print("Lance : .\venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlruns.db")
