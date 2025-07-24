import pandas as pd
import joblib
from metrics import compute_classification_metrics, save_json

# Charger le modèle et les features
model_data = joblib.load("DATA/models/final_model.pkl")
model = model_data["model"]
features = model_data["features"]

# Charger le jeu de test
X_test = pd.read_parquet("DATA/processed/X_test.parquet")
y_test = pd.read_csv("DATA/processed/y_test.csv").squeeze().map({"Fully Paid": 0, "Charged Off": 1})

# Prédictions
y_pred = model.predict(X_test[features])
y_proba = model.predict_proba(X_test[features])[:, 1]

# Calcul des métriques
metrics = compute_classification_metrics(y_test, y_pred, y_proba)
print(metrics)

# Sauvegarde des métriques
save_json(metrics, "src/metrics_test.json")
print("✅ Métriques sauvegardées dans results/metrics_test.json")