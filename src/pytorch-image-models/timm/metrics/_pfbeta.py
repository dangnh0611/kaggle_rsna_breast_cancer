import numpy as np

try:
    import tensorflow as tf
except:
    pass

try:
    from numba import jit, njit
except:
    pass


def pfbeta_py(gts, preds, beta=1):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(gts)):
        prediction = min(max(preds[idx], 0), 1)
        if (gts[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        ret = (1 + beta_squared) * (c_precision * c_recall) / (
            beta_squared * c_precision + c_recall)
        return ret
    else:
        return 0


def pfbeta_np(gts, preds, beta=1):
    preds = preds.clip(0, 1.)
    y_true_count = gts.sum()
    ctp = preds[gts == 1].sum()
    cfp = preds[gts == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        ret = (1 + beta_squared) * (c_precision * c_recall) / (
            beta_squared * c_precision + c_recall)
        return ret
    else:
        return 0.0


# First run will compile --> subsequent calls will be faster
# @jit(nopython=True)
@njit
def pfbeta_numba(gts, preds, beta=1):
    preds = preds.clip(0, 1.)
    y_true_count = gts.sum()
    ctp = preds[gts == 1].sum()
    cfp = preds[gts == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        ret = (1 + beta_squared) * (c_precision * c_recall) / (
            beta_squared * c_precision + c_recall)
        return ret
    else:
        return 0.0


def pfbeta_tf(gts, preds, beta=1):
    preds = tf.clip_by_value(preds, 0, 1)
    y_true_count = tf.reduce_sum(gts)
    ctp = tf.reduce_sum(preds[gts == 1])
    cfp = tf.reduce_sum(preds[gts == 0])
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        ret = (1 + beta_squared) * (c_precision * c_recall) / (
            beta_squared * c_precision + c_recall)
        return ret
    else:
        return 0.0


def pfbeta_torch(gts, preds, beta=1):
    preds = preds.clip(0, 1)
    y_true_count = gts.sum()
    ctp = preds[gts == 1].sum()
    cfp = preds[gts == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        ret = (1 + beta_squared) * (c_precision * c_recall) / (
            beta_squared * c_precision + c_recall)
        return ret
    else:
        return 0.0