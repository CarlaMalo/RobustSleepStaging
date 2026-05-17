import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, precision_recall_curve


def get_predictions(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
    }


def plot_aucpr_multiclass(clf, X_val, y_val, X_st, y_st, class_ids, class_names):
    """Plot AUCPR for all classes (one-vs-rest) 
    Parameters

    clf : Already trained classifier 
    X_val, y_val : val features and labels
    X_st, y_st : test features and labels
    class_ids : [0, 1, 2, 3, 4]
    class_names : Human-readable class names ( ['Waker', 'N1', 'N2', 'N3', 'REM'])
    """
    # Binarize labels for one-vs-rest PR curves
    y_val_bin = np.asarray(label_binarize(y_val, classes=class_ids))
    y_st_bin = np.asarray(label_binarize(y_st, classes=class_ids))

    def aligned_proba(model, X, classes):
        proba = model.predict_proba(X)
        aligned = np.zeros((proba.shape[0], len(classes)), dtype=float)
        for col, cls in enumerate(model.classes_):
            if cls in classes:
                aligned[:, classes.index(cls)] = proba[:, col]
        return aligned

    proba_val = aligned_proba(clf, X_val, class_ids)
    proba_st = aligned_proba(clf, X_st, class_ids)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    datasets = [
        ('SC → SC (In-Distribution)', y_val_bin, proba_val, axes[0]),
        ('SC → ST (Distribution Shift)', y_st_bin, proba_st, axes[1]),
    ]

    for title, y_bin, y_score, ax in datasets:
        for i, class_name in enumerate(class_names):
            if y_bin[:, i].sum() == 0:
                ax.text(0.5, 0.5, f'No samples for {class_name}', ha='center', va='center', transform=ax.transAxes, fontsize=9)
                continue

            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
            ap = average_precision_score(y_bin[:, i], y_score[:, i])
            ax.plot(recall, precision, lw=2, label=f'{class_name} (AP={ap:.3f})')

        # Optional summary scores
        macro_ap = average_precision_score(y_bin, y_score, average='macro')
        micro_ap = average_precision_score(y_bin, y_score, average='micro')
        no_skill = y_bin.mean()
        ax.hlines(no_skill, 0, 1, colors='gray', linestyles='--', lw=1, label=f'No skill (pos rate={no_skill:.2f})')

        ax.set_title(f'{title}\nMacro AP={macro_ap:.3f} | Micro AP={micro_ap:.3f}')
        ax.set_xlabel('Recall')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(alpha=0.2)
        ax.legend(loc='lower left', fontsize=9)

    axes[0].set_ylabel('Precision')
    plt.tight_layout()
    plt.show()
