import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def overall_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    return {'accuracy': acc, 'f1_macro': f1, 'confusion_matrix': cm}


def expected_calibration_error(probs, labels, n_bins=15):
    """Simple ECE implementation.

    probs: ndarray (n_samples, n_classes) predicted probabilities
    labels: ndarray (n_samples,) true labels
    """
    preds = np.argmax(probs, axis=1)
    confidences = probs.max(axis=1)
    ece = 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.sum() == 0:
            continue
        acc_bin = (preds[mask] == labels[mask]).mean()
        conf_bin = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(acc_bin - conf_bin)
    return float(ece)


def transition_error(y_true, y_pred):
    """Compute transition error: fraction of epochs where predicted transition indicator
    (whether next epoch label differs) mismatches the true transition indicator.
    """
    if len(y_true) < 2:
        return 0.0
    true_trans = (np.array(y_true[1:]) != np.array(y_true[:-1]))
    pred_trans = (np.array(y_pred[1:]) != np.array(y_pred[:-1]))
    return float(np.mean(true_trans != pred_trans))
