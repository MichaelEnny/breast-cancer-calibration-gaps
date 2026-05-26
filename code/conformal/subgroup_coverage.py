"""Conformal prediction with per-subgroup coverage reporting."""

import numpy as np


def split_conformal_threshold(cal_scores, alpha=0.10):
    """Compute (1-alpha) quantile threshold from calibration nonconformity scores."""
    n    = len(cal_scores)
    q    = np.ceil((n + 1) * (1 - alpha)) / n
    q    = min(q, 1.0)
    return np.quantile(cal_scores, q)


def nonconformity_scores(y_prob):
    """1 - p_hat for binary classification."""
    return 1.0 - y_prob


def prediction_sets(y_prob, threshold):
    return y_prob >= (1.0 - threshold)


def subgroup_coverage(y_true, prediction_set, subgroup_labels, min_n=100):
    """
    Coverage per subgroup and Demographic Coverage Gap (DCG).
    Returns:
        coverage_per_group : dict
        dcg                : max(cov) - min(cov)
    """
    groups = np.unique(subgroup_labels)
    cov = {}
    for g in groups:
        mask = subgroup_labels == g
        if mask.sum() < min_n:
            continue
        cov[g] = float((prediction_set[mask] == y_true[mask]).mean())

    if len(cov) < 2:
        return cov, None

    vals = list(cov.values())
    dcg  = max(vals) - min(vals)
    return cov, dcg
