"""Demographic Calibration Gap Score (DCGS) -- range and spread versions.

Public API:
    compute_ece          -- Expected Calibration Error (equal-width bins)
    compute_dcgs         -- DCGS range + spread across subgroups
    bootstrap_dcgs_ci    -- Bootstrap 95% CI for DCGS range
    pairwise_ece_tests   -- Pairwise Wilcoxon signed-rank tests with Bonferroni correction
"""

import numpy as np
from itertools import combinations
from scipy.stats import wilcoxon


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error with equal-width binning."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true)


def compute_dcgs(y_true, y_prob, subgroup_labels, min_n=100, n_bins=10):
    """
    Compute DCGS across demographic subgroups.

    Returns:
        ece_per_group  : dict {group_label: ECE}
        dcgs_range     : max(ECE_k) - min(ECE_k)
        dcgs_spread    : std(ECE_k)
    """
    groups = np.unique(subgroup_labels)
    ece_per_group = {}

    for g in groups:
        mask = subgroup_labels == g
        if mask.sum() < min_n:
            continue
        ece_per_group[g] = compute_ece(y_true[mask], y_prob[mask], n_bins)

    if len(ece_per_group) < 2:
        return ece_per_group, None, None

    values = list(ece_per_group.values())
    dcgs_range  = max(values) - min(values)
    dcgs_spread = float(np.std(values))
    return ece_per_group, dcgs_range, dcgs_spread


def bootstrap_dcgs_ci(y_true, y_prob, subgroup_labels,
                      n_bootstrap=1000, alpha=0.05, min_n=100):
    """Bootstrap 95% CI for DCGS range."""
    rng = np.random.default_rng(42)
    n   = len(y_true)
    dcgs_samples = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        _, dcgs_r, _ = compute_dcgs(
            y_true[idx], y_prob[idx], subgroup_labels[idx], min_n=min_n
        )
        if dcgs_r is not None:
            dcgs_samples.append(dcgs_r)

    lo = np.percentile(dcgs_samples, 100 * alpha / 2)
    hi = np.percentile(dcgs_samples, 100 * (1 - alpha / 2))
    return lo, hi


def pairwise_ece_tests(y_true, y_prob, subgroup_labels,
                       n_bins=10, min_n=100, alpha=0.05):
    """Pairwise Wilcoxon signed-rank tests on per-bin calibration errors.

    Compares each pair of subgroups using per-bin absolute errors as the
    paired sample.  Bonferroni-corrects for the number of pairs.

    Returns a list of dicts, one per pair:
        group_a, group_b, statistic, p_raw, p_adj, significant
    """
    bins   = np.linspace(0, 1, n_bins + 1)
    groups = np.unique(subgroup_labels)

    # Build per-bin absolute error vector for each qualifying group
    group_errors = {}
    for g in groups:
        mask = subgroup_labels == g
        if mask.sum() < min_n:
            continue
        errs = []
        for i in range(n_bins):
            b_mask = mask & (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if b_mask.sum() == 0:
                errs.append(0.0)
            else:
                errs.append(abs(y_true[b_mask].mean() - y_prob[b_mask].mean()))
        group_errors[g] = np.array(errs)

    pairs   = list(combinations(sorted(group_errors.keys()), 2))
    n_pairs = max(len(pairs), 1)
    results = []

    for a, b in pairs:
        diff = group_errors[a] - group_errors[b]
        if np.all(diff == 0):
            stat, p_raw = 0.0, 1.0
        else:
            try:
                stat, p_raw = wilcoxon(group_errors[a], group_errors[b],
                                       zero_method="wilcox", alternative="two-sided")
            except ValueError:
                stat, p_raw = 0.0, 1.0

        p_adj = min(p_raw * n_pairs, 1.0)
        results.append({
            "group_a":     str(a),
            "group_b":     str(b),
            "statistic":   round(float(stat), 4),
            "p_raw":       round(float(p_raw), 6),
            "p_adj":       round(float(p_adj), 6),
            "significant": p_adj < alpha,
        })

    return results
