"""Null-model comparisons showing empirical spectra differ from structureless nulls.

Two null models are used:
- Gaussian i.i.d. null: entries drawn from N(0,1). Its spectrum follows the
  Marchenko-Pastur law (flat bulk), serving as a baseline for completely
  structureless data.
- Column-permutation null: each column is independently permuted. This
  preserves marginal distributions of every feature but destroys all
  cross-feature dependence (covariance structure).

Comparing empirical eigenvalue spectra and eigenvector IPR against these nulls
demonstrates that real datasets carry non-trivial correlation structure that
cannot be attributed to marginal effects alone.
"""

import gc
from typing import Callable, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from crud.config import SEED
from crud.utils import preprocess_for_pca_and_corr, pca_randomized


def ipr_from_components(Vt: np.ndarray) -> np.ndarray:
    """Compute the Inverse Participation Ratio (IPR) for each eigenvector.

    IPR = sum_j v_{kj}^4 measures eigenvector localization:
      - IPR ~ 1/p  means the eigenvector is delocalized (spread evenly across
        all p features), consistent with Assumption A.1 ("Haar-like
        eigenvectors / delocalization") in the paper.
      - IPR ~ 1    means the eigenvector is localized on a single variable.

    Parameters
    ----------
    Vt : (K, p) array
        Right singular vectors (rows are eigenvectors).

    Returns
    -------
    (K,) array of IPR values, one per component.
    """
    Vt = np.asarray(Vt, dtype=np.float32)
    # IPR_k = sum_j (v_{kj})^4 — fourth moment of each row of Vt
    return np.sum(Vt ** 4, axis=1).astype(np.float32)


def pca_topk_components(X: np.ndarray, k: int, *, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: z-score X, then return top-k explained variance ratios and Vt.

    Parameters
    ----------
    X : (n, p) array — raw data matrix.
    k : int — number of principal components to extract.
    seed : int — random seed for the randomized SVD solver.

    Returns
    -------
    evr : (k,) array of explained variance ratios.
    Vt  : (k, p) array of right singular vectors.
    """
    Xz, _ = preprocess_for_pca_and_corr(X)
    k = int(min(k, Xz.shape[0], Xz.shape[1]))
    evr, Vt = pca_randomized(Xz, k, seed=seed)
    del Xz; gc.collect()  # free the (potentially large) z-scored matrix
    return evr, Vt


def gaussian_null_like(n: int, p: int, *, seed: int = SEED) -> np.ndarray:
    """Generate an i.i.d. N(0,1) matrix of the same shape as the data.

    This is the Marchenko-Pastur reference null: a completely structureless
    matrix whose eigenvalue spectrum converges to the Marchenko-Pastur
    distribution as n, p -> infinity with p/n -> gamma.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)).astype(np.float32)


def column_permutation_null(X: np.ndarray, *, seed: int = SEED) -> np.ndarray:
    """Independently permute each column of X.

    This preserves the marginal distribution of every feature but destroys
    all cross-feature dependence (covariance structure). Any eigenvalue
    structure remaining in the permuted matrix is due solely to finite-sample
    noise and marginal effects, not genuine correlations.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    n, p = X.shape
    Xp = np.empty_like(X)
    for j in range(p):
        # Permute rows of column j independently of all other columns
        idx = rng.permutation(n)
        Xp[:, j] = X[idx, j]
    return Xp


def run_null_model_comparisons(dataset_loaders: Dict[str, Callable]) -> dict:
    """For each dataset, compare the empirical spectrum and IPR against nulls.

    For every dataset provided by *dataset_loaders*, this function:
      1. Caps the matrix at MAX_N_NULL rows x MAX_P_NULL cols for speed.
      2. Computes the empirical PCA spectrum and eigenvector IPR.
      3. Computes the same quantities for a Gaussian i.i.d. null (Marchenko-
         Pastur reference) and for a column-permutation null (averaged over
         N_PERM=3 independent permutations).
      4. Produces log-log spectrum plots and IPR plots comparing all three.

    Returns
    -------
    dict mapping dataset name -> {"evr_emp", "ipr_emp", "evr_gauss",
    "ipr_gauss", "evr_perm", "ipr_perm"}.
    """
    NULL_TOPK = 300   # max number of PCs to extract per dataset
    N_PERM = 3        # number of column-permutation replicates to average
    MAX_N_NULL = 5000  # cap rows for computational tractability
    MAX_P_NULL = 2000  # cap columns for computational tractability

    null_results = {}

    for name, loader in dataset_loaders.items():
        print(f"\n[nulls] {name}")
        X = loader(visualize=False)

        # --- Subsample rows and columns if the matrix is too large ---
        rng = np.random.default_rng(SEED)
        if X.shape[0] > MAX_N_NULL:
            X = X[rng.choice(X.shape[0], size=MAX_N_NULL, replace=False)]
        if X.shape[1] > MAX_P_NULL:
            cols = rng.choice(X.shape[1], size=MAX_P_NULL, replace=False)
            X = X[:, cols]

        n, p = X.shape
        k = int(min(NULL_TOPK, n, p))
        if k < 50:
            print("  skip (too small)")
            continue

        # --- Empirical spectrum and IPR ---
        evr_emp, Vt_emp = pca_topk_components(X, k, seed=SEED)
        ipr_emp = ipr_from_components(Vt_emp)

        # --- Gaussian i.i.d. null (Marchenko-Pastur reference) ---
        G = gaussian_null_like(n, p, seed=SEED)
        evr_g, Vt_g = pca_topk_components(G, k, seed=SEED + 1)
        ipr_g = ipr_from_components(Vt_g)

        # --- Column-permutation null (averaged over N_PERM replicates) ---
        evr_perm_list, ipr_perm_list = [], []
        for t in range(N_PERM):
            Xp = column_permutation_null(X, seed=SEED + 10 + t)
            evr_p, Vt_p = pca_topk_components(Xp, k, seed=SEED + 20 + t)
            evr_perm_list.append(evr_p)
            ipr_perm_list.append(ipr_from_components(Vt_p))
            del Xp, Vt_p; gc.collect()

        # Average the permutation replicates for a smoother reference curve
        evr_perm = np.mean(np.stack(evr_perm_list, axis=0), axis=0)
        ipr_perm = np.mean(np.stack(ipr_perm_list, axis=0), axis=0)

        null_results[name] = {
            "evr_emp": evr_emp, "ipr_emp": ipr_emp,
            "evr_gauss": evr_g, "ipr_gauss": ipr_g,
            "evr_perm": evr_perm, "ipr_perm": ipr_perm,
        }

        # --- Log-log spectrum plot (eigenvalue spectrum comparison) ---
        f = np.arange(1, k + 1)

        plt.figure(figsize=(9, 6))
        plt.loglog(f, evr_emp, linewidth=2, label="empirical")
        plt.loglog(f, evr_perm, linewidth=2, linestyle="--", label=f"col-perm (mean of {N_PERM})")
        plt.loglog(f, evr_g, linewidth=2, linestyle=":", label="gaussian")
        plt.xlabel("component rank"); plt.ylabel("explained variance ratio")
        plt.title(f"{name}: PCA spectrum vs nulls (capped n={n}, p={p})")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(); plt.show(); plt.close()

        # --- IPR plot (eigenvector localization comparison) ---
        plt.figure(figsize=(9, 6))
        plt.semilogy(f, ipr_emp, linewidth=2, label="empirical")
        plt.semilogy(f, ipr_perm, linewidth=2, linestyle="--", label=f"col-perm (mean of {N_PERM})")
        plt.semilogy(f, ipr_g, linewidth=2, linestyle=":", label="gaussian")
        plt.xlabel("component rank"); plt.ylabel("IPR (\u2211 v^4)")
        plt.title(f"{name}: IPR vs nulls (capped n={n}, p={p})")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(); plt.show(); plt.close()

        del X, Vt_emp, Vt_g; gc.collect()

    print("\nNull-model results available for:", list(null_results.keys()))
    return null_results
