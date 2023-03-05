import numpy as np
import sklearn
from sklearn.metrics import (auc, confusion_matrix,
                             precision_recall_fscore_support, roc_curve)


def _compute_fbeta(precision, recall, beta=1.0):
    return (1 + beta**2) * precision * recall / (
        (beta**2) * precision + recall)


def compute_usual_metrics(gts, preds, beta=1.0, sample_weights=None):
    """Binary prediction only."""
    cfm = confusion_matrix(gts,
                           preds,
                           labels=[0, 1],
                           sample_weight=sample_weights)

    tn, fp, fn, tp = cfm.ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fbeta = _compute_fbeta(precision, recall, beta=beta)
    # frr = fp / (fp + tn)
    # far = fn / (fn + tp)  # 1 - recall
    # bacc_beta = _compute_fbeta(1 - frr, 1 - far, beta=beta)
    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'fbeta': fbeta,
        # 'bacc_beta': bacc_beta,
        # 'frr': frr,
        # 'far': far,
    }