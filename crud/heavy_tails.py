"""Heavy-tail robustness checks for PCA power-law spectra.

This module implements stress tests showing that the observed power-law
eigenvalue spectra (lambda_k ~ k^{-alpha}) are NOT driven by heavy-tailed
entries in the data matrices. Results are reported in the paper's Appendix.

The key idea: if heavy-tailed marginals were responsible for the power-law
spectrum, then transformations that remove heavy tails (winsorizing, row
normalization, rank-Gaussianization) should substantially change the fitted
power-law exponent alpha. In practice, alpha is remarkably stable across
all transformations, confirming that the spectral structure is a genuine
property of the correlation/covariance structure, not an artifact of outliers.

Transformations tested:
  1. Original data (baseline)
  2. Clipped/winsorized at 99.9th percentile
  3. Row-normalized to unit L2 norm
  4. Rank-Gaussianized (Van der Waerden scores)
  5. i.i.d. Gaussian null (no structure baseline)
"""

import gc
import math
from typing import Callable, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from crud.config import EPS, SEED, MAX_ROWS_ANALYSIS, MAX_PCS_FIT
from crud.utils import preprocess_for_pca_and_corr, pca_randomized, fit_power_law_first_n


def evr_topk_randomized(X: np.ndarray, k: int, *, seed: int = SEED) -> np.ndarray:
    """Convenience wrapper: preprocess (z-score) + randomized PCA in one call.

    Takes a raw data matrix, z-scores it, computes the top-k explained variance
    ratios via randomized SVD, and returns them. Used throughout the stress test
    to get comparable spectra from differently-transformed data matrices.

    Args:
        X: Raw data matrix, shape (n, p).
        k: Number of principal components to compute.
        seed: RNG seed for randomized SVD reproducibility.

    Returns:
        1-D array of explained variance ratios, length min(k, n, p).
    """
    Xz, _ = preprocess_for_pca_and_corr(X)
    n, p = Xz.shape
    kk = int(min(k, n, p))
    evr, _ = pca_randomized(Xz, kk, seed=seed)
    del Xz; gc.collect()
    return evr


def hill_tail_index(x: np.ndarray, top_frac: float = 0.05) -> Optional[float]:
    """Hill estimator for the Pareto tail index mu of |x| values.

    The Hill estimator is a standard tool for estimating the tail exponent
    of a heavy-tailed distribution. For a Pareto tail P(|X| > t) ~ t^{-mu}:
      - mu > 2: finite variance (light enough tails)
      - mu < 2: infinite variance (genuinely heavy tails)

    The estimator uses the top `top_frac` fraction of |x| values (default 5%,
    but called with 2% in run_heavy_tail_stress_test). Specifically:
      hill = mean(log(x_tail / x_min))
      mu = 1 / hill

    where x_tail are the top-k order statistics and x_min is the smallest
    among them (the threshold).

    Args:
        x: 1-D array of values (absolute values are taken internally).
        top_frac: Fraction of sorted |x| values to use as the tail.

    Returns:
        Estimated tail index mu, or None if insufficient data (< 100 values)
        or degenerate input.
    """
    x = np.asarray(x, dtype=float)
    x = np.abs(x)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size < 100:
        return None
    x = np.sort(x)
    # Use at least 50 tail values, or top_frac of the data
    k = int(max(50, top_frac * x.size))
    tail = x[-k:]       # top-k values (sorted ascending)
    xk = tail[0]        # threshold: smallest value in the tail
    if xk <= 0:
        return None
    # Hill estimator: mean of log-ratios in the tail
    hill = np.mean(np.log(tail / xk))
    if hill <= 0:
        return None
    return float(1.0 / hill)  # mu = 1/hill


def clip_entries(X: np.ndarray, q: float = 0.999) -> np.ndarray:
    """Winsorize (clip) matrix entries at the q-th and (1-q)-th percentiles.

    If the power-law eigenvalue spectrum were driven by outlier entries,
    clipping at the 99.9th percentile should substantially change the
    fitted alpha. In practice, alpha barely changes, confirming that
    individual extreme values are not responsible for the spectral shape.

    Args:
        X: Data matrix, shape (n, p).
        q: Quantile for clipping (default 0.999 = 99.9th percentile).
            Values below the (1-q) quantile or above the q quantile are clipped.

    Returns:
        Clipped copy of X as float32.
    """
    X = np.asarray(X, dtype=np.float32)
    lo = float(np.quantile(X, 1 - q))
    hi = float(np.quantile(X, q))
    return np.clip(X, lo, hi)


def row_normalize_l2(X: np.ndarray) -> np.ndarray:
    """Project each row onto the unit sphere (L2 normalization).

    Removes row-level magnitude variation, which could arise from heavy-tailed
    row norms. After this transformation, all rows have unit L2 norm, so the
    spectrum reflects only directional (angular) structure in the data.

    Args:
        X: Data matrix, shape (n, p).

    Returns:
        Row-normalized copy of X as float32. Each row has L2 norm ~1
        (EPS added to denominator for numerical safety).
    """
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + EPS)


def rank_gaussianize(X: np.ndarray) -> np.ndarray:
    """Van der Waerden rank-based Gaussianization of all matrix entries.

    Applies the transformation: rank -> uniform(0,1) -> Phi^{-1} -> Gaussian,
    where Phi^{-1} is the standard normal quantile function (implemented via
    erfinv). This destroys ALL marginal non-Gaussianity (heavy tails, skewness,
    kurtosis) while perfectly preserving rank correlations (Spearman rho).

    If the power-law spectrum were an artifact of non-Gaussian marginals,
    rank-Gaussianization would eliminate it. The paper shows alpha is
    essentially unchanged, confirming the spectral structure lives in the
    correlation/rank structure, not in marginal distributional shape.

    Steps:
      1. Flatten the matrix to a 1-D array.
      2. Compute ranks (0-indexed) via argsort.
      3. Map ranks to uniform: u = (rank + 0.5) / N  (midpoint correction).
      4. Apply Phi^{-1}(u) = sqrt(2) * erfinv(2u - 1) to get standard normal.
      5. Reshape back to original matrix shape.

    Args:
        X: Data matrix, shape (n, p).

    Returns:
        Rank-Gaussianized matrix as float32, same shape as X.
    """
    from scipy.special import erfinv
    X = np.asarray(X, dtype=np.float32)
    flat = X.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(flat), dtype=np.float64)
    u = (ranks + 0.5) / len(flat)           # uniform(0,1) via midpoint rule
    z = math.sqrt(2.0) * erfinv(2.0 * u - 1.0)  # Phi^{-1}(u)
    return z.reshape(X.shape).astype(np.float32)


def quick_gaussian_null(n: int, p: int, k: int, *, seed: int = SEED) -> Optional[np.ndarray]:
    """Generate an i.i.d. Gaussian matrix and compute its PCA spectrum.

    This provides a "no structure" baseline: the Marchenko-Pastur distribution.
    An i.i.d. Gaussian matrix has NO power-law spectrum — its eigenvalues follow
    the Marchenko-Pastur law, which looks qualitatively different on a log-log
    plot. Comparing against this null confirms that the observed power-law
    spectra reflect genuine data structure, not a statistical artifact of
    matrix dimensionality.

    Dimensions are capped at 5000 x 2000 for computational speed.

    Args:
        n: Number of rows in the real dataset (used as target, capped at 5000).
        p: Number of columns in the real dataset (used as target, capped at 2000).
        k: Number of PCA components to compute.
        seed: RNG seed.

    Returns:
        Explained variance ratio array, or None if dimensions are too small (< 50).
    """
    MAX_N, MAX_P = 5000, 2000
    nn = int(min(n, MAX_N))
    pp = int(min(p, MAX_P))
    if nn < 50 or pp < 50:
        return None
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((nn, pp)).astype(np.float32)
    return evr_topk_randomized(G, k=min(k, nn, pp), seed=seed)


def _fmt(x):
    """Format a numeric value for table display: 3 significant figures, or 'NA'."""
    if x is None:
        return "NA"
    if isinstance(x, float) and not np.isfinite(x):
        return "NA"
    return f"{x:.3g}"


def run_heavy_tail_stress_test(dataset_loaders: Dict[str, Callable]) -> List[dict]:
    """Run the full heavy-tail robustness battery across all datasets.

    For each dataset, computes the PCA eigenvalue spectrum under 4 transformations
    (original, clipped, row-normalized, rank-Gaussianized) plus a Gaussian null.
    Fits the power-law exponent alpha to each spectrum. If alpha is similar across
    all transformations, heavy-tailed entries are NOT driving the power law — the
    spectral structure is intrinsic to the correlation/covariance matrix.

    Also computes the Hill tail index mu for the raw entries and row-normalized
    entries. mu > 2 indicates finite variance; mu < 2 indicates genuinely
    heavy-tailed entries (infinite variance).

    Args:
        dataset_loaders: Dict mapping dataset name -> callable that returns
            the raw data matrix X when called with visualize=False.

    Returns:
        List containing one table dict with columns for dataset dimensions,
        Hill tail indices, and (alpha, R^2) under each transformation.
    """
    CLIP_Q = 0.999   # Winsorize at 99.9th percentile
    FIT_K = 500       # Compute up to 500 PCA components for the spectrum

    summary = []
    for name, loader in dataset_loaders.items():
        print(f"\n[heavy-tail] {name}")
        X = loader(visualize=False)

        rng = np.random.default_rng(SEED)
        if X.shape[0] > MAX_ROWS_ANALYSIS:
            X = X[rng.choice(X.shape[0], size=MAX_ROWS_ANALYSIS, replace=False)]

        n, p = X.shape
        k = int(min(FIT_K, MAX_PCS_FIT, n, p))
        if k < 50:
            print("  skip (too small)")
            continue

        # Transformation 1: Original data (baseline spectrum)
        evr0 = evr_topk_randomized(X, k, seed=SEED)
        a0, r20 = fit_power_law_first_n(evr0, n_fit=min(40, len(evr0)), y_bottom=1e-12)

        # Transformation 2: Clipped/winsorized entries
        Xc = clip_entries(X, q=CLIP_Q)
        evr_clip = evr_topk_randomized(Xc, k, seed=SEED)
        ac, r2c = fit_power_law_first_n(evr_clip, n_fit=min(40, len(evr_clip)), y_bottom=1e-12)

        # Transformation 3: Row-normalized to unit L2 norm
        Xr = row_normalize_l2(X)
        evr_row = evr_topk_randomized(Xr, k, seed=SEED)
        ar, r2r = fit_power_law_first_n(evr_row, n_fit=min(40, len(evr_row)), y_bottom=1e-12)

        # Transformation 4: Rank-Gaussianized (Van der Waerden scores)
        try:
            Xg = rank_gaussianize(X)
            evr_rank = evr_topk_randomized(Xg, k, seed=SEED)
            ag, r2g = fit_power_law_first_n(evr_rank, n_fit=min(40, len(evr_rank)), y_bottom=1e-12)
        except Exception as e:
            print("  rank-gauss failed:", e)
            evr_rank = None
            ag, r2g = (float("nan"), float("nan"))

        # Hill tail index on raw entries (top 2% of |x|) and row-normalized entries.
        # mu > 2 => finite variance; mu < 2 => heavy tails (infinite variance).
        mu_entry = hill_tail_index(X.reshape(-1), top_frac=0.02)
        mu_row = hill_tail_index(row_normalize_l2(X).reshape(-1), top_frac=0.02)

        summary.append({
            "dataset": name, "n": n, "p": p,
            "mu_entry": mu_entry, "mu_row": mu_row,
            "alpha_orig": a0, "r2_orig": r20,
            "alpha_clip": ac, "r2_clip": r2c,
            "alpha_row": ar, "r2_row": r2r,
            "alpha_rank": ag, "r2_rank": r2g,
        })

        f = np.arange(1, len(evr0) + 1, dtype=float)
        plt.figure(figsize=(9, 6))
        plt.loglog(f, evr0, linewidth=2, label=f"orig (\u03b1={a0:.2f}, R\u00b2={r20:.2f})")
        plt.loglog(f, evr_clip, linestyle="--", linewidth=2,
                   label=f"clip@{CLIP_Q:.3f} (\u03b1={ac:.2f}, R\u00b2={r2c:.2f})")
        plt.loglog(f, evr_row, linestyle=":", linewidth=2,
                   label=f"row-norm (\u03b1={ar:.2f}, R\u00b2={r2r:.2f})")
        if evr_rank is not None:
            plt.loglog(f, evr_rank, linestyle="-.", linewidth=2,
                       label=f"rank-gauss (\u03b1={ag:.2f}, R\u00b2={r2g:.2f})")

        # Overlay the i.i.d. Gaussian null spectrum for comparison.
        # This shows what a "structureless" matrix looks like — no power law.
        evr_g = quick_gaussian_null(n, p, k=k, seed=SEED)
        if evr_g is not None:
            plt.loglog(f, evr_g, linewidth=1, alpha=0.7, label="Gaussian null (capped size)")

        plt.ylim(1e-6, None)
        plt.xlabel("component rank f"); plt.ylabel("EV(f)")
        title = f"{name}: heavy-tail stress test"
        if mu_entry is not None:
            title += f" | Hill \u03bc(entry)\u2248{mu_entry:.2f}"
        if mu_row is not None:
            title += f", \u03bc(row)\u2248{mu_row:.2f}"
        plt.title(title)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(); plt.show(); plt.close()

        del X; gc.collect()

    hdr = ["Dataset", "n", "p", "\u03bc_entry", "\u03bc_row",
           "\u03b1_orig", "R\u00b2_orig", "\u03b1_clip", "R\u00b2_clip",
           "\u03b1_row", "R\u00b2_row", "\u03b1_rank", "R\u00b2_rank"]
    rows = []
    for s in summary:
        rows.append([
            s["dataset"], str(s["n"]), str(s["p"]),
            _fmt(s["mu_entry"]), _fmt(s["mu_row"]),
            _fmt(s["alpha_orig"]), _fmt(s["r2_orig"]),
            _fmt(s["alpha_clip"]), _fmt(s["r2_clip"]),
            _fmt(s["alpha_row"]), _fmt(s["r2_row"]),
            _fmt(s["alpha_rank"]), _fmt(s["r2_rank"]),
        ])

    table = {"title": "Heavy-tail stress test (fits on first ~40 comps)", "headers": hdr, "rows": rows}
    # Print to stdout
    print(f"\n{table['title']}")
    print(" | ".join(hdr))
    print("-" * 120)
    for row in rows:
        print(" | ".join(row))

    return [table]
