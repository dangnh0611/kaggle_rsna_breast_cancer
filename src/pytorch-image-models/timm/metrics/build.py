import numpy as np
import sklearn
import torch
from _binary import compute_usual_metrics
from _pfbeta import pfbeta_np


def compute_metrics_over_thres(gts,
                               preds,
                               sample_weights=None,
                               thres_range=(0, 1, 0.01),
                               sort_by='fbeta'):
    if isinstance(gts, torch.Tensor):
        gts = gts.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    assert isinstance(gts, np.ndarray) and isinstance(preds, np.ndarray)
    assert len(preds) == len(gts)

    # log loss for pos/neg/overall
    pos_preds = preds[gts == 1]
    neg_preds = preds[gts == 0]
    pos_loss = sklearn.metrics.log_loss(np.ones_like(pos_preds),
                                        pos_preds,
                                        eps=1e-15,
                                        normalize=True,
                                        sample_weight=None,
                                        labels=[0, 1])
    neg_loss = sklearn.metrics.log_loss(np.zeros_like(neg_preds),
                                        neg_preds,
                                        eps=1e-15,
                                        normalize=True,
                                        sample_weight=None,
                                        labels=[0, 1])
    total_loss = sklearn.metrics.log_loss(gts,
                                          preds,
                                          eps=1e-15,
                                          normalize=True,
                                          sample_weight=None,
                                          labels=[0, 1])

    # Probabilistic-Fbeta
    pfbeta = pfbeta_np(gts, preds, beta=1.0)
    # AUC
    fpr, tpr, _thresholds = sklearn.metrics.roc_curve(gts, preds, pos_label=1)
    auc = sklearn.metrics.auc(fpr, tpr)

    # PER THRESHOLD METRIC
    per_thres_metrics = []
    for thres in np.arange(*thres_range):
        bin_preds = (preds > thres).astype(np.uint8)
        metric_at_thres = compute_usual_metrics(gts, bin_preds, beta=1.0)
        pfbeta_at_thres = pfbeta_np(gts, bin_preds, beta=1.0)
        metric_at_thres['pfbeta'] = pfbeta_at_thres

        if sample_weights is not None:
            w_metric_at_thres = compute_usual_metrics(gts, bin_preds, beta=1.0)
            w_metric_at_thres = {
                f'w_{k}': v
                for k, v in w_metric_at_thres.items()
            }
            metric_at_thres.update(w_metric_at_thres)
        per_thres_metrics.append((thres, metric_at_thres))

    per_thres_metrics.sort(key=lambda x: x[1][sort_by], reverse=True)

    # handle multiple thresholds with same scores
    top_score = per_thres_metrics[0][1][sort_by]
    same_scores = []
    for j, (thres, metric_at_thres) in enumerate(per_thres_metrics):
        if metric_at_thres[sort_by] == top_score:
            same_scores.append([j, thres, abs(thres - 0.5)])
        else:
            assert metric_at_thres[sort_by] < top_score
            break
    if len(same_scores) == 1:
        best_thres, best_metric = per_thres_metrics[0]
    else:
        # idx with threshold nearer 0.5 is better
        best_idx = np.argmin(np.array(same_scores)[:, 2])
        best_thres, best_metric = per_thres_metrics[best_idx]

    # best thres, best results, all results
    return {
        'best_thres': best_thres,
        'best_metric': best_metric,
        'all_metrics': per_thres_metrics,
        'pfbeta': pfbeta,
        'auc': auc,
        'pos_log_loss': pos_loss,
        'neg_log_loss': neg_loss,
        'log_loss': total_loss,
    }
