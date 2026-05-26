"""Generate reliability diagrams -- population-level, per-subgroup, and DCGS heatmap."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- safe for scripts
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

OUT = "figures"
os.makedirs(f"{OUT}/reliability_wisconsin", exist_ok=True)
os.makedirs(f"{OUT}/reliability_mimic",     exist_ok=True)


def _calibration_curve(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    frac_pos, mean_pred = [], []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        frac_pos.append(y_true[mask].mean())
        mean_pred.append(y_prob[mask].mean())
    return np.array(mean_pred), np.array(frac_pos)


def plot_reliability(y_true, y_prob_dict, title, save_path, n_bins=10):
    """Plot reliability diagrams for multiple calibration conditions."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect", lw=1)

    colors = {"No calib": "#e41a1c", "Platt": "#377eb8", "Isotonic": "#4daf4a",
              "Subgroup Platt": "#984ea3"}

    for label, y_prob in y_prob_dict.items():
        x, y = _calibration_curve(y_true, y_prob, n_bins)
        ax.plot(x, y, marker="o", ms=4, label=label,
                color=colors.get(label, "#888888"))

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_subgroup_reliability(y_true, y_prob, subgroup_labels,
                               title, save_path, n_bins=10):
    """Plot reliability curves for demographic subgroups on one figure."""
    groups  = np.unique(subgroup_labels)
    palette = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect", lw=1)

    for i, g in enumerate(groups):
        mask = subgroup_labels == g
        if mask.sum() < 50:
            continue
        x, y = _calibration_curve(y_true[mask], y_prob[mask], n_bins)
        ax.plot(x, y, marker="o", ms=4, label=str(g), color=palette[i % 10])

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_dcgs_heatmap(dcgs_df, value_col, title, save_path,
                      model_order=None, variant_order=None):
    """Heatmap of DCGS values: rows=models, columns=calibration variants.

    Args:
        dcgs_df      : DataFrame with columns [model, variant, <value_col>]
        value_col    : e.g. 'dcgs_range' or 'dcgs_spread'
        title        : figure title
        save_path    : output path (.png)
        model_order  : list of model names for row ordering
        variant_order: list of variant names for column ordering
    """
    if model_order is None:
        model_order = sorted(dcgs_df["model"].unique())
    if variant_order is None:
        variant_order = sorted(dcgs_df["variant"].unique())

    pivot = (
        dcgs_df[dcgs_df[value_col].notna()]
        .pivot_table(index="model", columns="variant",
                     values=value_col, aggfunc="mean")
        .reindex(index=model_order, columns=variant_order)
    )

    fig, ax = plt.subplots(
        figsize=(max(4, len(variant_order) * 1.6), max(3, len(model_order) * 0.9))
    )

    vmax = max(0.10, float(pivot.max().max()))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "calibration_gap", ["#f7fbff", "#fdae6b", "#d73027"]
    )
    im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label=value_col)

    ax.set_xticks(range(len(variant_order)))
    ax.set_xticklabels(variant_order, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=9)

    for i, m in enumerate(model_order):
        for j, v in enumerate(variant_order):
            val = pivot.loc[m, v] if (m in pivot.index and v in pivot.columns) else np.nan
            if pd.isna(val):
                continue
            weight = "bold" if val > 0.05 else "normal"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, fontweight=weight,
                    color="white" if val > 0.07 else "black")

    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel("Calibration variant", fontsize=9)
    ax.set_ylabel("Model", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")
