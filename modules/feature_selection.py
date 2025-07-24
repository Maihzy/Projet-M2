import pandas as pd
from sklearn.feature_selection import RFE
from lightgbm import LGBMClassifier
import joblib

# === Méthode RFE (conservée mais optionnelle) ===

def rfe_lightgbm(X, y, n_features_to_select=20, random_state=42):
    """
    Sélection de variables par RFE avec un LGBMClassifier.

    X : DataFrame des variables
    y : Series de la cible
    n_features_to_select : nombre de variables à conserver
    random_state : graine aléatoire pour reproductibilité

    Retourne :
    - Liste des variables sélectionnées
    - Liste des variables éliminées
    """
    model = LGBMClassifier(random_state=random_state, class_weight="balanced")
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)

    selected_cols = X.columns[rfe.support_].tolist()
    eliminated_cols = X.columns[~rfe.support_].tolist()

    print(f"Variables sélectionnées ({len(selected_cols)}):")
    print(selected_cols)

    print(f"Variables éliminées ({len(eliminated_cols)}):")
    print(eliminated_cols)

    return selected_cols, eliminated_cols


# === Liste figée des 15 variables les plus importantes (selon le gain LightGBM) ===

FEATURES_FINAL = [
    'sub_grade_encoded', 'dti', 'annual_inc', 'int_rate', 'term_encoded',
    'loan_amnt', 'mort_acc', 'revol_util', 'revol_bal', 'open_acc',
    'employment_length_category_Unknown', 'credit_history_months',
    'total_acc', 'installment', 'home_ownership_RENT'
]

# Enregistrement (pour accès dans model_evaluation et tools)
joblib.dump(FEATURES_FINAL, "DATA/final_features.pkl")
