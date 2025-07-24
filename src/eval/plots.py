import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

def plot_roc(y_true, y_proba, path: str):
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()

def plot_confusion(y_true, y_pred, path: str):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
