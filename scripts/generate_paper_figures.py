#!/usr/bin/env python3
"""
Generate all main-text figures for the paper.

Produces:
  paper/figures/fig1_datasets.pdf  -- Dataset overview (sample sizes)
  paper/figures/fig2_left.pdf      -- Correlation distributions before adjustment (K=0)
  paper/figures/fig2_right.pdf     -- Correlation distributions after adjustment (K=10)
  paper/figures/fig3_spectra.pdf   -- Eigenvalue spectra with power-law fits

Requirements:
  - Cached .npy data matrices in data/cache/cached_data/ (produced by run_analysis.py)
  - The crud package installed (pip install -e .)

Usage:
  python scripts/generate_paper_figures.py
"""

import os
import sys
import numpy as np
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ensure we can import code utilities
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

from crud.utils import (
    preprocess_for_pca_and_corr,
    pca_randomized,
    residual_corr_vals_after_k_pcs,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Dataset file names in data/cache/cached_data/ and display labels
DATASETS = {
    "Kay fMRI":   "Kay_fMRI",
    "Haxby fMRI": "Haxby_fMRI",
    "NHANES":     "NHANES",
    "CIFAR-10":   "CIFAR10",
    "Precinct":   "Precinct",
    "RNA-Seq":    "RNASeq",
    "GTEx":       "GTEx",
    "HEXACO":     "HEXACO",
    "Stringer":   "Stringer",
}

# Full dataset sizes (rows in cached .npy) for significance thresholds
FULL_N = {
    "Kay fMRI": 8_428,    "Haxby fMRI": 23_612,   "NHANES": 29_902,
    "CIFAR-10": 30_000,   "Precinct": 28_934,     "RNA-Seq": 10_071,
    "GTEx": 10_000,       "HEXACO": 22_786,       "Stringer": 7_018,
}

# Consistent colours across all plots
COLORS = {
    "Kay fMRI": "#1f77b4",   "Haxby fMRI": "#ff7f0e",  "NHANES": "#2ca02c",
    "CIFAR-10": "#d62728",   "Precinct": "#9467bd",     "RNA-Seq": "#8c564b",
    "GTEx": "#e377c2",       "HEXACO": "#7f7f7f",       "Stringer": "#bcbd22",
}

MAX_ROWS = 10_000   # Subsample rows for computational tractability
MAX_COLS = 500       # Subsample columns for correlation matrix size
SEED = 42            # For reproducibility of random subsampling
FIT_N = 40           # Number of leading eigenvalues used for power-law fit
KDE_BW = 0.05        # Bandwidth for kernel density estimates
FIG_DIR = os.path.join(REPO_ROOT, "paper", "figures")
CACHE_DIR = os.path.join(REPO_ROOT, "data", "cache", "cached_data")


def rcrit(n: int, alpha: float = 0.05) -> float:
    """Fisher z-transform significance threshold for |r| at given alpha."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    return z / np.sqrt(max(n - 3, 1))


# ---------------------------------------------------------------------------
# 1. Process phenotypic datasets
# ---------------------------------------------------------------------------

def load_phenotypic_datasets():
    """Load cached data, preprocess, PCA, and compute residual correlations."""
    results = {}
    for label, fname in DATASETS.items():
        path = os.path.join(CACHE_DIR, f"{fname}.npy")
        X = np.load(path)
        rng = np.random.RandomState(SEED)

        if X.shape[0] > MAX_ROWS:
            X = X[rng.choice(X.shape[0], MAX_ROWS, replace=False)]
        if X.shape[1] > MAX_COLS:
            X = X[:, rng.choice(X.shape[1], MAX_COLS, replace=False)]

        X_proc, _ = preprocess_for_pca_and_corr(X)
        cols = np.arange(X_proc.shape[1])
        evr, Vt = pca_randomized(X_proc, n_components=min(100, *X_proc.shape))

        vals_k0 = residual_corr_vals_after_k_pcs(X_proc, Vt, cols, 0)
        vals_k1 = residual_corr_vals_after_k_pcs(X_proc, Vt, cols, 1)
        vals_k10 = residual_corr_vals_after_k_pcs(X_proc, Vt, cols, 10)
        max_k50 = min(50, X_proc.shape[1] - 1)
        vals_k50 = residual_corr_vals_after_k_pcs(X_proc, Vt, cols, max_k50)

        results[label] = {
            "vals_k0": vals_k0,
            "vals_k1": vals_k1,
            "vals_k10": vals_k10,
            "vals_k50": vals_k50,
            "evr": evr,
            "n_full": FULL_N[label],
        }
        print(f"  {label}: shape={X_proc.shape}, "
              f"SD_k0={np.std(vals_k0):.3f}, SD_k1={np.std(vals_k1):.3f}, "
              f"SD_k10={np.std(vals_k10):.3f}, SD_k50={np.std(vals_k50):.3f}")

    return results


# ---------------------------------------------------------------------------
# 2. Figure generation
# ---------------------------------------------------------------------------

def make_fig1_datasets(outpath):
    """Bar chart of dataset sample sizes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = list(DATASETS.keys())
    n_vals = [FULL_N[l] for l in labels]

    bars = ax.bar(range(len(labels)), n_vals, color=[COLORS[l] for l in labels])
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Sample size $n$")
    ax.set_title("Datasets")

    for bar, n in zip(bars, n_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                f"{n:,}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}")


def make_fig2_left(results, outpath):
    """Distribution of pairwise correlations before adjustment (K=0)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.linspace(-0.5, 0.5, 500)

    for label in DATASETS:
        kde = gaussian_kde(results[label]["vals_k0"], bw_method=KDE_BW)
        ax.plot(x, kde(x), color=COLORS[label], linewidth=1.5, label=label)
        r_c = rcrit(results[label]["n_full"])
        ax.axvline(r_c, color=COLORS[label], linestyle=":", alpha=0.4, linewidth=1)
        ax.axvline(-r_c, color=COLORS[label], linestyle=":", alpha=0.4, linewidth=1)

    ax.set_xlabel("Pairwise correlation")
    ax.set_ylabel("Density")
    ax.set_title(r"Before adjustment ($K = 0$)")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, 0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}")


def make_fig2_right(results, outpath):
    """Distribution of residual correlations after K=10 PC adjustment."""
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.linspace(-0.5, 0.5, 500)

    for label in DATASETS:
        kde = gaussian_kde(results[label]["vals_k10"], bw_method=KDE_BW)
        ax.plot(x, kde(x), color=COLORS[label], linewidth=1.5, label=label)
        r_c = rcrit(results[label]["n_full"])
        ax.axvline(r_c, color=COLORS[label], linestyle=":", alpha=0.4, linewidth=1)
        ax.axvline(-r_c, color=COLORS[label], linestyle=":", alpha=0.4, linewidth=1)

    ax.set_xlabel(r"Residual correlation (after $K = 10$ PCs removed)")
    ax.set_ylabel("Density")
    ax.set_title(r"After adjustment ($K = 10$)")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, 0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}")


def make_fig3_spectra(results, outpath):
    """Eigenvalue spectra on log-log axes with power-law fits."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for label in DATASETS:
        evr = results[label]["evr"]
        n_comp = min(FIT_N, len(evr))
        ax.loglog(np.arange(1, len(evr) + 1), evr,
                  "o-", markersize=2, linewidth=1, color=COLORS[label])

        log_k = np.log(np.arange(1, n_comp + 1))
        log_evr = np.log(evr[:n_comp])
        valid = np.isfinite(log_evr)
        if valid.sum() > 2:
            slope, intercept = np.polyfit(log_k[valid], log_evr[valid], 1)
            ss_res = np.sum((log_evr[valid] - (slope * log_k[valid] + intercept)) ** 2)
            ss_tot = np.sum((log_evr[valid] - log_evr[valid].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
            fit_k = np.arange(1, n_comp + 1).astype(float)
            ax.loglog(fit_k, np.exp(intercept) * fit_k ** slope,
                      "--", color=COLORS[label], linewidth=1)
            ax.loglog([], [], "o-", color=COLORS[label], markersize=4, linewidth=1,
                      label=f"{label} (\u03b1={-slope:.2f}, R\u00b2={r2:.2f})")

    # 1/f reference line
    ev1_vals = [results[l]["evr"][0] for l in DATASETS]
    c1 = float(np.median(ev1_vals))
    f_ref = np.arange(1, FIT_N + 1, dtype=float)
    ax.loglog(f_ref, c1 / f_ref, color="k", linewidth=4, linestyle="-",
              label="1/f reference")

    ax.set_ylim(1e-5, None)
    ax.set_xlabel("Component rank $f$")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"Explained variance spectra with power-law fits (first {FIT_N} comps)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading phenotypic datasets...")
    results = load_phenotypic_datasets()

    print("\nGenerating figures...")
    make_fig1_datasets(os.path.join(FIG_DIR, "fig1_datasets.pdf"))
    make_fig2_left(results, os.path.join(FIG_DIR, "fig2_left.pdf"))
    make_fig2_right(results, os.path.join(FIG_DIR, "fig2_right.pdf"))
    make_fig3_spectra(results, os.path.join(FIG_DIR, "fig3_spectra.pdf"))

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
