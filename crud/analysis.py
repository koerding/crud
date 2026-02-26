"""Core per-dataset analysis and cross-dataset summaries.

This module implements the main analysis pipeline for studying universal data
properties relevant to causal inference. It examines how PCA eigenvalue spectra
follow power laws (lambda_k ~ k^{-alpha}) across diverse domains, creating
persistent background ("crud") correlations that resist removal by standard
dimensionality reduction.

Two main entry points:
  - analyze_dataset(): per-dataset pipeline producing Figures 2 (correlation
    distributions) and Figure 3 (spectra) data from the paper.
  - plot_cross_dataset_summaries(): cross-dataset overlay figures showing
    superimposed spectra, power-law fits, and residual correlation histograms.
"""

import gc
import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from crud.config import (
    EPS, SEED, MAX_ROWS_ANALYSIS, MAX_PCS_FIT, MAX_K_STORE,
    MAX_COLS_CORR, KS, BINS_CORR, BINS_RESID,
)
from crud.utils import (
    preprocess_for_pca_and_corr, corr_from_zscored, offdiag_vals_from_corr,
    pca_randomized, residual_corr_vals_after_k_pcs, fit_power_law_first_n,
)


def analyze_dataset(
    Xraw: np.ndarray,
    *,
    name: str,
    seed: int = SEED,
    max_rows: int = MAX_ROWS_ANALYSIS,
    max_pcs: int = MAX_PCS_FIT,
    max_k_store: int = MAX_K_STORE,
    max_cols_corr: int = MAX_COLS_CORR,
    ks: List[int] = KS,
) -> dict:
    """Per-dataset analysis pipeline: spectrum, correlations, and residuals.

    Pipeline steps:
      1. Subsample rows to max_rows (default MAX_ROWS_ANALYSIS) for tractability.
      2. Z-score columns (center + unit variance) via preprocess_for_pca_and_corr.
      3. Compute PCA eigenvalue spectrum (up to max_pcs components).
      4. Select a random feature subset of size max_cols_corr to avoid O(p^2) full
         correlation matrix computation.
      5. Compute correlation matrix on the feature subset, extract off-diagonal values.
      6. For each K in ks, remove K principal components and recompute residual
         correlations — this measures the "crud" that persists after dimensionality
         reduction. The crud scale sigma_K = sqrt(sum_{k>K} lambda_k^2) / sum_{k>K} lambda_k.

    Produces data for:
      - Figure 2 (paper): distribution of off-diagonal correlations before/after
        PC removal, showing persistent crud.
      - Figure 3 (paper): eigenvalue spectra on log-log axes revealing power-law.

    Args:
        Xraw: Raw data matrix, shape (n_samples, n_features).
        name: Human-readable dataset name for plot titles and logging.
        seed: RNG seed for reproducibility of subsampling.
        max_rows: Cap on number of rows used (subsampled without replacement).
        max_pcs: Maximum number of PCs to fit via randomized SVD (k_pca).
        max_k_store: Maximum number of PC loadings to retain in memory (k_store).
            We fit up to max_pcs components for the spectrum but only store
            max_k_store loadings for residualization, to limit memory usage.
        max_cols_corr: Size of the random feature subset for correlation computation.
            Avoids O(p^2) full correlation matrix when p is large.
        ks: List of K values at which to compute residual correlations (e.g., [0, 1, 5, 10, 50]).

    Returns:
        Dictionary with keys: n_full, p_full, n_used, p_used, zscore_diag,
        explained_variance_ratio, k_pca, top_pcs, k_store, corr_cols,
        resid_corr_vals_by_k, resid_corr_std_by_k.
    """
    rng = np.random.default_rng(seed)
    results = {}

    Xraw = np.asarray(Xraw)
    n_full, p_full = Xraw.shape
    results["n_full"] = int(n_full)
    results["p_full"] = int(p_full)

    # Step 1: Subsample rows if the dataset exceeds max_rows.
    # This keeps memory and compute manageable for large datasets.
    if n_full > max_rows:
        ridx = rng.choice(n_full, size=max_rows, replace=False)
        Xraw_use = Xraw[ridx]
    else:
        Xraw_use = Xraw

    # Step 2: Z-score each column (center to mean 0, scale to unit variance).
    # Also returns diagnostic info about column means/stds.
    Xz, diag = preprocess_for_pca_and_corr(Xraw_use)
    n, p = Xz.shape
    results["n_used"] = int(n)
    results["p_used"] = int(p)
    results["zscore_diag"] = diag

    print(f"[{name}] using n={n} (of {n_full}), p={p}")
    print(f"  zscore diag: mean|max={diag['col_mean_abs_max']:.3g}, "
          f"std med={diag['col_std_median']:.3g}, std min={diag['col_std_min']:.3g}")

    # Heatmap visualization: downsample to at most 2000 rows x 500 columns
    # for display purposes. Clip at 99th percentile for color range.
    max_rows_viz, max_cols_viz = 2000, 500
    r_viz = rng.choice(n, size=min(n, max_rows_viz), replace=False) if n > max_rows_viz else np.arange(n)
    c_viz = rng.choice(p, size=min(p, max_cols_viz), replace=False) if p > max_cols_viz else np.arange(p)

    Xviz = Xz[np.ix_(r_viz, c_viz)].copy()
    v = float(max(np.percentile(np.abs(Xviz), 99), 1e-6))
    Xviz = np.clip(Xviz, -v, v)

    plt.figure(figsize=(10, 6))
    plt.imshow(Xviz, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    plt.colorbar(label="z-score")
    plt.title(f"{name}: data heatmap (downsampled)")
    plt.xlabel("features (subset)"); plt.ylabel("samples (subset)")
    plt.show(); plt.close()
    del Xviz

    # Step 3: PCA spectrum via randomized SVD.
    # k_pca = number of components to compute (up to MAX_PCS_FIT).
    # This gives the eigenvalue spectrum lambda_k that we expect to follow
    # a power law: lambda_k ~ k^{-alpha}, with alpha in [0.63, 1.33].
    k_pca = int(min(max_pcs, n, p))
    evr, Vt = pca_randomized(Xz, k_pca, seed=seed)
    results["explained_variance_ratio"] = evr
    results["k_pca"] = int(k_pca)

    # k_store: we only retain the top k_store PC loadings (Vt rows) in memory.
    # k_pca may be large (e.g., 500) to get a full spectrum, but we only need
    # k_store (e.g., 50) loadings for residualization at each K in ks.
    # This saves memory when storing results across many datasets.
    k_store = int(min(max_k_store, Vt.shape[0]))
    top_pcs = Vt[:k_store].copy()
    results["top_pcs"] = top_pcs
    results["k_store"] = int(k_store)

    plt.figure(figsize=(8, 6))
    x = np.arange(1, len(evr) + 1)
    plt.loglog(x, evr, marker="o", linestyle="-", linewidth=1)
    plt.xlabel("principal component index"); plt.ylabel("explained variance ratio")
    plt.title(f"{name}: explained variance spectrum (log-log)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show(); plt.close()

    # Step 4: Correlations on a random feature subset.
    # cols: random subset of m <= max_cols_corr features. We compute the
    # correlation matrix only on this subset to avoid O(p^2) cost when p is
    # large (e.g., genomics with p > 20,000). The subset is fixed per dataset
    # (same seed) and reused for all residualization steps below.
    m = int(min(p, max_cols_corr))
    cols = rng.choice(p, size=m, replace=False) if p > m else np.arange(p)
    results["corr_cols"] = cols

    Xs = Xz[:, cols]
    S = corr_from_zscored(Xs)  # m x m correlation matrix from z-scored data

    plt.figure(figsize=(8, 6))
    plt.imshow(S, aspect="auto", cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f"{name}: correlation matrix (subset m={m})")
    plt.show(); plt.close()

    # Extract off-diagonal correlation values (the "crud" at K=0, before any
    # PC removal). These form the baseline distribution for Figure 2.
    vals0 = offdiag_vals_from_corr(S)
    results["resid_corr_vals_by_k"] = {0: vals0}
    results["resid_corr_std_by_k"] = {0: float(np.std(vals0))}

    plt.figure(figsize=(8, 6))
    plt.hist(vals0, bins=BINS_CORR, alpha=0.7, edgecolor="black")
    plt.xlabel("correlation (off-diagonal)"); plt.ylabel("count")
    plt.title(f"{name}: correlation histogram (subset)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show(); plt.close()

    # Step 5: Residual correlations after removing K principal components.
    # For each K in ks, project out the top-K PCs from the z-scored data and
    # recompute correlations on the feature subset. The std of these residual
    # correlations measures the "crud scale" sigma_K — how much background
    # correlation persists after removing K PCs. Due to the power-law spectrum,
    # sigma_K decreases slowly with K (the crud is hard to remove).
    for k in ks:
        if int(k) == 0:
            continue
        vals = residual_corr_vals_after_k_pcs(Xz, top_pcs, cols, int(k))
        results["resid_corr_vals_by_k"][int(k)] = vals
        results["resid_corr_std_by_k"][int(k)] = float(np.std(vals))

    # Show residual correlation histogram at K=10 (or the largest K available).
    k_show = 10 if 10 in results["resid_corr_vals_by_k"] else sorted(results["resid_corr_vals_by_k"].keys())[-1]
    vals_r = results["resid_corr_vals_by_k"][k_show]
    plt.figure(figsize=(8, 6))
    plt.hist(vals_r, bins=BINS_RESID, alpha=0.7, edgecolor="black")
    plt.xlabel("residual correlation (off-diagonal, subset)"); plt.ylabel("count")
    plt.title(f"{name}: residual correlation histogram (k={k_show} PCs removed)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show(); plt.close()

    # Free large arrays to reduce memory pressure across multiple datasets.
    del Xz, Xs, S, Vt, Xraw_use
    gc.collect()
    return results


def plot_cross_dataset_summaries(all_results: Dict[str, dict]) -> List[dict]:
    """Produce cross-dataset overlay figures and summary tables.

    Creates the key comparative visualizations for the paper:
      1. Superimposed eigenvalue spectra on log-log axes — reveals that all
         datasets share approximate power-law form lambda_k ~ k^{-alpha}.
      2. Power-law fits with reference 1/f line — fits alpha and R^2 on the
         first FIT_N=40 components (matching the paper), overlays fitted lines,
         and adds a 1/f reference scaled to the median first eigenvalue.
      3. Residual correlation histograms at K=0, 1, 10 — shows how the
         distribution of off-diagonal correlations evolves as PCs are removed.
      4. Table of residual correlation std at each K.
      5. Table of approximate correlation significance thresholds (Fisher z).

    Args:
        all_results: Dict mapping dataset name -> result dict from analyze_dataset().

    Returns:
        List of table dicts with keys {title, headers, rows} suitable for
        embedding in the HTML report gallery.
    """
    tables = []

    # (1) Superimposed explained-variance spectra (log-log).
    # This is the central figure showing universal power-law behavior.
    plt.figure(figsize=(9, 6))
    for name, res in all_results.items():
        evr = res["explained_variance_ratio"]
        x = np.arange(1, len(evr) + 1)
        plt.loglog(x, evr, linewidth=2, label=name, alpha=0.9)
    plt.ylim(1e-6, None)
    plt.xlabel("principal component rank"); plt.ylabel("explained variance ratio")
    plt.title("Explained-variance spectra (log-log), all datasets")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(); plt.show(); plt.close()

    # (2) Power-law fits with reference line.
    # FIT_N=40: fit the power law lambda_k ~ k^{-alpha} on the first 40
    # components, matching the paper's methodology. Y_BOTTOM filters out
    # near-zero eigenvalues that would distort the log-log fit.
    FIT_N = 40
    Y_BOTTOM = 1e-6
    fit_table = []
    ev1_list = []

    plt.figure(figsize=(9, 6))
    for name, res in all_results.items():
        evr = np.asarray(res["explained_variance_ratio"], dtype=float)
        f = np.arange(1, len(evr) + 1, dtype=float)
        alpha, r2 = fit_power_law_first_n(evr, n_fit=FIT_N, y_bottom=Y_BOTTOM)
        fit_table.append((name, alpha, r2))

        (line,) = plt.loglog(f, evr, linewidth=2, alpha=0.85,
                             label=f"{name} (\u03b1\u2248{alpha:.2f}, R\u00b2={r2:.2f})")
        color = line.get_color()

        n_fit = int(min(FIT_N, len(evr)))
        f_fit = f[:n_fit]
        y_fit = evr[:n_fit]
        mask = np.isfinite(y_fit) & (y_fit > Y_BOTTOM)
        if np.isfinite(alpha) and mask.sum() >= 8:
            x_log = np.log10(f_fit[mask])
            y_log = np.log10(y_fit[mask])
            slope, intercept = np.polyfit(x_log, y_log, 1)
            c = 10 ** intercept
            plt.loglog(f_fit[mask], c * (f_fit[mask] ** slope),
                       linestyle="--", linewidth=2, alpha=0.95, color=color)

        if np.isfinite(evr[0]) and evr[0] > 0:
            ev1_list.append(float(evr[0]))

    # 1/f reference line: uses the median of first eigenvalues across datasets
    # as the scaling constant, so the line passes through a "typical" EV(1).
    # This provides a visual anchor for alpha=1 (pure 1/f spectrum).
    if ev1_list:
        c1 = float(np.median(ev1_list))
        f_ref = np.arange(1, FIT_N + 1, dtype=float)
        plt.loglog(f_ref, c1 / f_ref, color="k", linewidth=4, linestyle="-", label="1/f reference")

    plt.ylim(Y_BOTTOM, None)
    plt.xlabel("component rank f"); plt.ylabel("EV(f)")
    plt.title(f"Explained variance spectra with power-law fits (first {FIT_N} comps)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(); plt.show(); plt.close()

    # Sort datasets by alpha (ascending) for the summary table.
    fit_table = sorted(fit_table, key=lambda t: (np.nan_to_num(t[1], nan=999)))
    tables.append({
        "title": f"Power-law fits (first {FIT_N} components)",
        "headers": ["Dataset", "\u03b1", "R\u00b2"],
        "rows": [[name, f"{alpha:.3f}", f"{r2:.3f}"] for name, alpha, r2 in fit_table],
    })

    # (3) Overlay residual-correlation histograms at K=0, 1, 10.
    # These show how the distribution of pairwise correlations changes as
    # leading PCs are removed. The paper's key finding: even after removing
    # many PCs, substantial background correlation ("crud") persists due to
    # the power-law tail of the spectrum.
    def _plot_residual_histograms_for_k(k: int, title_suffix: str):
        plt.figure(figsize=(9, 6))
        for name, res in all_results.items():
            vals = res["resid_corr_vals_by_k"][int(k)]
            plt.hist(vals, bins=BINS_RESID, density=True, alpha=0.25, label=name)
        plt.xlabel("residual correlation (off-diagonal, subset)"); plt.ylabel("density")
        plt.title(f"Residual correlation distributions, {title_suffix}")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend(); plt.show(); plt.close()

    _plot_residual_histograms_for_k(0, "baseline (k=0)")
    _plot_residual_histograms_for_k(1, "after removing 1 PC")
    _plot_residual_histograms_for_k(10, "after removing 10 PCs")

    # (4) Table: std of off-diagonal correlations
    resid_headers = ["Dataset"] + [f"k={k}" for k in KS]
    resid_rows = []
    for name in sorted(all_results.keys()):
        res = all_results[name]
        row_vals = [res["resid_corr_std_by_k"][int(k)] for k in KS]
        resid_rows.append([name] + [f"{v:.4g}" for v in row_vals])
    tables.append({
        "title": "Std of off-diagonal correlations after removing k PCs",
        "headers": resid_headers,
        "rows": resid_rows,
    })

    # (5) Approximate correlation significance thresholds.
    # rcrit_approx: Fisher z-transform approximation for the critical |r|
    # at alpha=0.05 (two-sided). Under H0 (rho=0), the Fisher z-transform
    # of r is approximately N(0, 1/(n-3)), so the threshold is
    # z_{0.025} / sqrt(n-3) = 1.96 / sqrt(n-3).
    # This contextualizes the residual correlations: values above r_crit
    # would be individually "significant" (though not corrected for
    # multiple comparisons across all feature pairs).
    def rcrit_approx(n: int, z: float = 1.96) -> float:
        return float(z / math.sqrt(max(n - 3, 1)))

    rcrit_rows = []
    for name in sorted(all_results.keys()):
        n = int(all_results[name]["n_used"])
        rcrit_rows.append([name, str(n), f"{rcrit_approx(n):.4f}"])
    tables.append({
        "title": "Approx 95% |r| threshold (subset correlations)",
        "headers": ["Dataset", "n_used", "r_crit"],
        "rows": rcrit_rows,
    })

    # Print tables to stdout too
    for t in tables:
        print(f"\n{t['title']}")
        print("  ".join(h.rjust(12) for h in t["headers"]))
        print("-" * (14 * len(t["headers"])))
        for row in t["rows"]:
            print("  ".join(c.rjust(12) for c in row))

    return tables
