import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


def get_predictions(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
    }
