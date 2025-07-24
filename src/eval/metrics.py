from __future__ import annotations
from typing import Dict
import json
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

def compute_classification_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "tn_fp_fn_tp": list(map(int, confusion_matrix(y_true, y_pred).ravel())),
    }

def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
