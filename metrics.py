import numpy as np


def tp(y_true, y_pred):
    is_tp = np.logical_and(y_true == 1, y_pred == 1)
    return np.sum(is_tp)


def fp(y_true, y_pred):
    is_fp = np.logical_and(y_true == 0, y_pred == 1)
    return np.sum(is_fp)


def fn(y_true, y_pred):
    is_fn = np.logical_and(y_true == 1, y_pred == 0)
    return np.sum(is_fn)


def tn(y_true, y_pred):
    is_tn = np.logical_and(y_true == 0, y_pred == 0)
    return np.sum(is_tn)


def precision(y_true, y_pred):
    tp_ = tp(y_true, y_pred)
    fp_ = fp(y_true, y_pred)

    return tp_ / (tp_ + fp_)


def recall(y_true, y_pred):
    tp_ = tp(y_true, y_pred)
    fn_ = fn(y_true, y_pred)

    return tp_ / (tp_ + fn_)


def accuracy(y_true, y_pred):
    tp_ = tp(y_true, y_pred)
    tn_ = tn(y_true, y_pred)

    return (tp_ + tn_) / len(y_true)


def f1_score(y_true, y_pred):
    precision_ = precision(y_true, y_pred)
    recall_ = recall(y_true, y_pred)

    return 2 * precision_ * recall_ / (precision_ + recall_)


def mse(y_true, y_pred):
    return (1 / len(y_true)) * np.sum((y_true - y_pred) ** 2)
