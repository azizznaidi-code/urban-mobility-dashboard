# UrbanAI — Intelligence Artificielle pour la Gestion Urbaine & Transport
## Basé sur le DW `dwurbanmobility` (SQL Server 2022)
**Option ERP-BI · Encadrante : Manel Khamassi**

---

## Structure du projet

```
urban_ai/
├── app.py                              Application Streamlit principale
├── api_urbanai.py                      API FastAPI v4 (n8n ML Automation)
├── auth.py                             Authentification decideurs
├── config.py                           Parametres & colonnes DW
├── requirements.txt                    Dependances Python
├── predictions_urbanai.db              SQLite historique predictions
├── mlruns.db                           MLflow 12 runs
├── workflow1_cron_prediction.json      Workflow n8n Cron 1h
├── workflow2_webhook_eventdriven.json  Workflow n8n Webhook event-driven
├── workflow3_monitoring_kpi.json       Workflow n8n Monitoring + Email
├── README.md                           Documentation
│
├── data/
│   └── connector.py                   Connexion SQL Server + chargement DW
│
├── models/
│   ├── multi_horizon.py               Module 1 : Prediction Multi-Horizon (LightGBM)
│   ├── graph_model.py                 Module 2 : Graphe Spatio-Temporel (NetworkX)
│   ├── causal_ai.py                   Module 3 : IA Causale (DoWhy)
│   ├── whatif_optimizer.py            Module 4 : Simulation What-If (pymoo)
│   └── anomaly_detector.py            Module 5 : Detection Anomalies (IF + AE + SHAP)
│
└── pages/
    ├── prediction.py                  Prediction multi-horizon
    ├── anomaly.py                     Detection d anomalies
    ├── causal.py                      IA Causale
    ├── recommandation.py              Recommandation
    ├── nlp.py                         NLP Incidents
    ├── graph.py                       Graphe du reseau
    ├── whatif.py                      Simulation What-If
    ├── report.py                      Rapport XAI
    └── mlops_page.py                  MLOps & Drift
```

---

## Guide d Installation

### Prerequis systeme

```bash
python --version   # Python 3.11+
pip install -r requirements.txt
npm install -g n8n
```

### Connexion DW SQL Server

Fichier `.env` :
```env
DW_SERVER=NEGRAMARIEM\MSSQLSERVERM
DW_DATABASE=dwurbanmobility
DW_USERNAME=miming
DW_PASSWORD=miming
```

---

## Lancement — 3 Terminaux

### Terminal 1 — API FastAPI (port 8000)
```bash
cd urban_ai
venv\Scripts\activate
uvicorn api_urbanai:app --host 127.0.0.1 --port 8000 --reload
```
Verifier : http://127.0.0.1:8000/docs

### Terminal 2 — n8n Automation (port 5678)
```bash
n8n start
```
Ouvrir : http://localhost:5678

### Terminal 3 — Application Streamlit (port 8501)
```bash
streamlit run app.py
```
Ouvrir : http://localhost:8501

### MLflow UI (port 5000)
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```
Ouvrir : http://localhost:5000

---

## PART 1 — N8N ML Automation

### Architecture

```
DW dwurbanmobility (SQL Server 2022)
        ↓
api_urbanai.py (FastAPI port 8000)
  → /dw/transport/latest   (lit facttransport)
  → /dw/trafic/latest      (lit facttrfic)
  → /dw/kpi/realtime       (calcule tous KPI)
  → /predict/retard        (prediction KPI RM)
  → /predict/anomalie      (detection KPI SAN)
  → /kpi/status            (statut KPI)
  → /kpi/alertes           (KPI en alerte)
        ↓
n8n Workflows (port 5678)
  → Workflow 1 : Cron 1h
  → Workflow 2 : Webhook event-driven
  → Workflow 3 : Monitoring 30min
        ↓
Gmail Notifications (mariem.negra@esprit.tn)
        ↓
SQLite predictions_urbanai.db
```

### Workflow 1 — Prediction KPI RM (Cron toutes les heures)
**Fichier :** `workflow1_cron_prediction.json`

```
Cron 1h
  → GET /dw/transport/latest   (donnees reelles SQL Server)
  → Gerer erreur DW            (fallback si DW indisponible)
  → POST /predict/retard       (prediction ML KPI RM)
  → Verifier seuil KPI RM      (seuil 300s Fiche KPI)
  → Si alerte → ALERTE Log
  → Si normal → OK Log
```

**Critere grille :** C — Automated Inference Pipeline (scheduled Cron)

### Workflow 2 — Prediction Event-Driven (Webhook)
**Fichier :** `workflow2_webhook_eventdriven.json`
**URL Test :** `http://localhost:5678/webhook-test/urbanai-predict`

```
Webhook POST
  → GET /dw/transport/latest   (donnees reelles SQL Server)
  → POST /predict/retard       (prediction KPI RM)
  → POST /predict/anomalie     (detection KPI SAN)
  → Fusionner KPI RM et SAN
  → Repondre JSON au Webhook
```

**Test :**
```bash
curl -X POST "http://localhost:5678/webhook-test/urbanai-predict" \
  -H "Content-Type: application/json" \
  -d '{"zone_id":1,"heure":8,"charge_estimee":60}'
```

**Critere grille :** C — Event-Driven Automation (Webhook)

### Workflow 3 — Monitoring KPI + Alertes Email (Cron 30min)
**Fichier :** `workflow3_monitoring_kpi.json`

```
Cron 30min
  → GET /health              (API active ?)
  → GET /kpi/status          (statut KPI)
  → GET /predictions/stats   (stats ML)
  → Analyser KPI
  → Si alerte → Gmail Email automatique
  → Si normal → Log OK
```

**Email automatique :**
```
Destinataire : mariem.negra@esprit.tn
Sujet        : ALERTE KPI UrbanAI — N KPI en alerte
Corps        : Rapport KPI + actions recommandees
Source       : DW dwurbanmobility SQL Server
```

**Critere grille :** D — Robustness + Notifications

### Robustesse et Error Handling

```
Retries automatiques  : 3 tentatives, 2s entre chaque
Fallback DW           : valeurs KPI officielles si SQL Server indisponible
onError               : continueErrorOutput (workflow ne s arrete pas)
Logs SQLite           : historique predictions + alertes
Notifications Gmail   : email automatique si KPI en alerte
```

---

## PART 2 — Modeles ML (MLflow — 12 runs)

| Modele | Type | KPI | Metrique |
|--------|------|-----|----------|
| LightGBM + SMOTE | Classification | RM | ROC-AUC |
| Random Forest + SMOTE | Classification | RM | F1-Score |
| Lasso L1 | Regression | RM | RMSE=175s |
| Ridge L2 | Regression | RM | RMSE=175s |
| XGBoost | Regression | RM | RMSE=175s |
| K-Means | Clustering | Zones | Silhouette |
| Hierarchique Ward | Clustering | Zones | DBI |
| GMM | Clustering | Zones | Silhouette |
| ARIMA | Time Series | RM | MAPE |
| SARIMA | Time Series | RM | RMSE |
| LSTM PyTorch | Deep Learning | RM | MAPE |
| IF + Autoencoder | Anomalie | SAN | Score |

---

## KPI Surveilles — Fiches KPI Officielles

| KPI | Valeur Actuelle | Cible | Seuil Alerte | Decideur |
|-----|----------------|-------|-------------|----------|
| RM Retard Moyen | 118s | ≤ 300s | > 300s | President AOM |
| TA Taux Annulation | 1.99% | ≤ 1.5% | > 1.6% | Dir. Planification |
| CM Charge Moyenne | 44.7 pass. | ≥ 50 | < 25 | Dir. Offre Transport |
| IC Indice Congestion | 2.319 | ≤ 1.5 | > 2.0 | VP Mobilite |
| VM Vitesse Moyenne | 37.4 km/h | ≥ 30 | < 20 | Dir. Voirie |
| SU Satisfaction | 2.52/5 | ≥ 3.5 | < 3.0 | Dir. Clients |
| QA Qualite Air | 3.1 ATMO | ≤ 4 | > 6 | Dir. DREAL |
| TD Taux Delinquance | 1.02/1000 | ≤ 2.0 | > 5 | Prefet Securite |
| SAN Score Anomalie | 0.45 | < 0.4 | > 0.6 | CDO |

---

## Data Warehouse

```
Serveur  : NEGRAMARIEM\MSSQLSERVERM
Base     : dwurbanmobility
User     : miming
Periode  : 2019-2022
```

| Table | Lignes | KPI |
|-------|--------|-----|
| facttransport | 13 740 | RM, TA, CM |
| facttrfic | 51 769 | IC, VM, TTM |
| factexperienceusager | - | SU |
| facturbainncontext | - | QA, CO2 |
| factaccidents | - | TD |

---

## Grille Evaluation — Etat Final

| Critere | Statut | Fichiers |
|---------|--------|---------|
| A — Workflow Design | 100% | workflow1/2/3.json + README |
| B — ML Model Integration | 100% | api_urbanai.py + FastAPI |
| C — Automation Logic | 100% | Workflow 1 (Cron) + Workflow 2 (Webhook) |
| D — Robustness | 100% | Error handling + Gmail notifications |
| E — ETL Apache Airflow | En cours | Audit DAGs |
| F — Performance Power BI | En cours | SQL Profiler + Analyzer |

---

*UrbanAI — Option ERP-BI · Encadrante : Manel Khamassi*
*DW dwurbanmobility · SQL Server 2022 · Python 3.11*
