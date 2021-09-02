import numpy as np


def tp(real_y, predicted_y):
    is_tp = np.logical_and(real_y == 1, predicted_y == 1)
    return np.sum(is_tp)


def fp(real_y, predicted_y):
    is_fp = np.logical_and(real_y == 0, predicted_y == 1)
    return np.sum(is_fp)


def fn(real_y, predicted_y):
    is_fn = np.logical_and(real_y == 1, predicted_y == 0)
    return np.sum(is_fn)


def tn(real_y, predicted_y):
    is_tn = np.logical_and(real_y == 0, predicted_y == 0)
    return np.sum(is_tn)


def precision(real_y, predicted_y):
    tp_ = tp(real_y, predicted_y)
    fp_ = fp(real_y, predicted_y)

    return tp_ / (tp_ + fp_)


def recall(real_y, predicted_y):
    tp_ = tp(real_y, predicted_y)
    fn_ = fn(real_y, predicted_y)

    return tp_ / (tp_ + fn_)


def accuracy(real_y, predicted_y):
    tp_ = tp(real_y, predicted_y)
    tn_ = tn(real_y, predicted_y)

    return (tp_ + tn_) / len(real_y)


def f1_score(real_y, predicted_y):
    precision_ = precision(real_y, predicted_y)
    recall_ = recall(real_y, predicted_y)

    return 2 * precision_ * recall_ / (precision_ + recall_)