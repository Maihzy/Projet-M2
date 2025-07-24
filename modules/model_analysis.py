import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc as calc_auc
)



# Benchmark multi-modèles

class ModelBenchmark:
    def __init__(self, X_train, y_train, X_valid, y_valid):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.results = {}

    def run(self):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
            "LightGBM": LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1)
        }

        for name, model in models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', model)
            ])

            pipeline.fit(self.X_train, self.y_train)
            y_proba = pipeline.predict_proba(self.X_valid)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            self.results[name] = {
                "AUC": round(roc_auc_score(self.y_valid, y_proba), 4),
                "Précision": round(precision_score(self.y_valid, y_pred), 4),
                "Rappel": round(recall_score(self.y_valid, y_pred), 4),
                "F1-score": round(f1_score(self.y_valid, y_pred), 4)
            }

        print("Benchmark terminé. Résultats :")
        for model, scores in self.results.items():
            print(f"\n{model} :")
            for metric, value in scores.items():
                print(f"  {metric} : {value}")

    def get_results(self):
        return pd.DataFrame(self.results).T

    def plot_roc_curves(self):
        plt.figure(figsize=(8, 6))

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
            "LightGBM": LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1)
        }

        for name, model in models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', model)
            ])
            pipeline.fit(self.X_train, self.y_train)
            y_proba = pipeline.predict_proba(self.X_valid)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_valid, y_proba)
            roc_auc = calc_auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title('Courbes ROC - Modèles comparés')
        plt.legend()
        plt.grid()
        plt.show()


# Analyse du modèle final

def seuil_analysis(model, X_valid, y_valid, step=0.01, plot_graph=True):
    y_proba = model.predict_proba(X_valid)[:, 1]
    thresholds = np.arange(0, 1 + step, step)
    precisions, recalls, f1_scores, acceptance_rates = [], [], [], []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        precisions.append(precision_score(y_valid, y_pred, pos_label=1))
        recalls.append(recall_score(y_valid, y_pred, pos_label=1))
        f1_scores.append(f1_score(y_valid, y_pred, pos_label=1))
        acceptance_rates.append(y_pred.mean())

    df_results = pd.DataFrame({
        "threshold": thresholds,
        "precision": precisions,
        "recall": recalls,
        "f1_score": f1_scores,
        "acceptance_rate": acceptance_rates
    })

    if plot_graph:
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label='Precision')
        plt.plot(thresholds, recalls, label='Recall')
        plt.plot(thresholds, f1_scores, label='F1-score')
        plt.plot(thresholds, acceptance_rates, label='Acceptance rate', linestyle='dashed')
        plt.xlabel('Seuil de décision')
        plt.ylabel('Valeur des métriques')
        plt.title('Évolution des métriques selon le seuil')
        plt.legend()
        plt.grid()
        plt.show()

    return df_results


def plot_confusion_matrix(model, X, y_true, threshold=0.5):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non défaut", "Défaut"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matrice de confusion - Seuil {threshold}")
    plt.show()


def plot_feature_importance(model, feature_names, top_n=15, importance_type="gain"):
    importance = model.booster_.feature_importance(importance_type=importance_type)
    features = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by="Importance", ascending=False)

    top_features = features.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["Feature"], top_features["Importance"], color='steelblue')
    plt.xlabel(f"Importance ({importance_type})")
    plt.title(f"Top {top_n} variables importantes (LightGBM)")
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()


def final_model_evaluation(model, X, y_true, threshold=0.5, display_roc=False):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    auc_score = roc_auc_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    print("\n=== Évaluation finale ===")
    print(f"AUC : {auc_score:.4f}")
    print(f"Précision : {precision:.4f}")
    print(f"Rappel : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    plot_confusion_matrix(model, X, y_true, threshold)

    if display_roc:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = calc_auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title('Courbe ROC finale')
        plt.legend()
        plt.grid()
        plt.show()


def save_model(model, path, feature_names):
    joblib.dump({
        'model': model,
        'features': feature_names
    }, path)
    print(f"Modèle sauvegardé dans : {path}")
