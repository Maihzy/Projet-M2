import pandas as pd
from lightgbm import LGBMClassifier
from feature_selection import FEATURES_FINAL
from model_analysis import final_model_evaluation, seuil_analysis, save_model

# === Chargement des données ===
df_train = pd.read_csv("DATA/processed/train.csv")
df_test = pd.read_csv("DATA/processed/test.csv")

X_train = df_train[FEATURES_FINAL]
y_train = df_train["target"]

X_test = df_test[FEATURES_FINAL]
y_test = df_test["target"]

# === Entraînement du modèle final ===
model = LGBMClassifier(
    random_state=42,
    class_weight="balanced",
    n_estimators=100
)
model.fit(X_train, y_train)

# === Évaluation complète ===
final_model_evaluation(
    model,
    X_test,
    y_test,
    threshold=0.4,
    display_roc=True  # changer en False si nous ne voulons pas le graphe
)

#  === Analyse de seuil  === 
# seuil_analysis(model, X_test, y_test, step=0.05, plot_graph=True)

# === Sauvegarde du modèle ===
save_model(
    model,
    path="DATA/final_model.pkl",
    feature_names=FEATURES_FINAL
)

print("Modèle final enregistré dans 'DATA/final_model.pkl")
