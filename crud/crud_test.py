"""Crud-aware calibration test for adjusted associations.

Implements the crud-aware p-value framework. Given a data matrix X and target
feature pairs, removes K principal components and calibrates adjusted
correlations against the empirical background of all (or a Monte Carlo sample
of) feature pairs.

Two interfaces:

1. Full test (requires raw data matrix)::

    from crud import crud_test
    result = crud_test(X, [(0, 1), (2, 3)], K=10)
    print(result.summary())

2. Parametric z-test (requires only summary statistics)::

    from crud import crud_z_test
    result = crud_z_test(r=0.15, n=200, crud_sd=0.05)
    print(result.summary())
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.utils.extmath import randomized_svd

from crud.config import EPS


# -- Utility functions (use package versions where possible, keep private
#    residualize/pairwise helpers that are specific to crud_test) --

def _safe_zscore_columns(X: np.ndarray, *, eps: float = EPS) -> np.ndarray:
    """Z-score each column. Constant columns get unit scale."""
    # NOTE: This duplicates utils.safe_zscore_columns intentionally so that
    # crud_test.py remains fully self-contained and importable without pulling
    # in the rest of the crud.utils module.
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    Xc = X - mu
    sd = Xc.std(axis=0, ddof=1, keepdims=True).astype(np.float32)
    sd = np.where(sd < eps, 1.0, sd)
    Xz = Xc / (sd + eps)
    Xz = Xz - Xz.mean(axis=0, keepdims=True)
    return Xz.astype(np.float32)


def _corr_from_zscored(Xz: np.ndarray) -> np.ndarray:
    """Correlation matrix from already z-scored columns."""
    n = Xz.shape[0]
    S = (Xz.T @ Xz).astype(np.float64) / max(n - 1, 1)
    S = S.astype(np.float32)
    np.fill_diagonal(S, 1.0)
    return S


def _pca_top_k(Xz: np.ndarray, K: int, *, seed: int = 42) -> np.ndarray:
    """Return Vk (K, p) -- right singular vectors for the top K PCs."""
    # NOTE: Uses sklearn's randomized_svd directly (not utils.pca_randomized)
    # because we only need Vt (the right singular vectors) and do not need
    # explained variance ratios. This avoids importing the full utils module.
    n, p = Xz.shape
    k = int(min(K, n, p))
    if k <= 0:
        return np.zeros((0, p), dtype=np.float32)
    _, _, Vt = randomized_svd(Xz, n_components=k, random_state=seed)
    return Vt.astype(np.float32)


def _residualize(Xz: np.ndarray, Vk: np.ndarray) -> np.ndarray:
    """Residualize Xz against PCs in Vk, then re-standardize.

    Algebra:
      scores = Xz @ Vk'          — project onto the top-K PC subspace
      resid  = Xz - scores @ Vk  — subtract the projection
             = Xz - Xz @ Vk' @ Vk
             = (I - Vk' @ Vk) @ Xz   (orthogonal projection onto complement)

    After projection, columns are re-centered and re-standardized so that
    downstream correlations are comparable across different K values.
    """
    if Vk.shape[0] == 0:
        resid = Xz.copy()
    else:
        # Project Xz onto the subspace spanned by Vk (top-K right singular vectors)
        scores = (Xz @ Vk.T).astype(np.float32)        # (n, K): PC scores
        # Subtract the projection: resid = Xz - Xz @ Vk' @ Vk = (I - Vk'Vk) @ Xz
        resid = (Xz - scores @ Vk).astype(np.float32)
    # Re-center and re-standardize each column after residualization
    resid = resid - resid.mean(axis=0, keepdims=True)
    resid = resid / (resid.std(axis=0, ddof=1, keepdims=True) + EPS)
    return resid.astype(np.float32)


def _pairwise_corrs(Xz: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
    """Compute correlations for specific pairs from z-scored data."""
    n = Xz.shape[0]
    corrs = np.empty(len(pairs), dtype=np.float64)
    for idx, (i, j) in enumerate(pairs):
        corrs[idx] = float(np.dot(Xz[:, i], Xz[:, j])) / max(n - 1, 1)
    return corrs


# -- Result dataclass --


@dataclass
class CrudTestResult:
    """Result of a crud-aware calibration test."""

    target_pairs: list
    adjusted_correlations: np.ndarray
    crud_pvalues: np.ndarray
    crud_percentiles: np.ndarray
    crud_pvalue_se: np.ndarray
    background_sd: float
    sampling_sd: float
    K: int
    n_reference_pairs: int
    n_samples: int

    def summary(self) -> str:
        """Return a readable summary table."""
        lines = [
            f"Crud-aware test  (K={self.K}, n={self.n_samples}, "
            f"M={self.n_reference_pairs:,} reference pairs)",
            f"Background SD of reference correlations: {self.background_sd:.4f}",
            f"  - finite-sample component (1/sqrt(n-1)):  {self.sampling_sd:.4f}",
            f"  - crud component (background_sd^2 - sampling^2): "
            f"{max(0, self.background_sd**2 - self.sampling_sd**2)**.5:.4f}",
            "",
            f"{'pair':>12s}  {'adj_corr':>9s}  {'crud_pval':>10s}  "
            f"{'pval_SE':>9s}  {'percentile':>10s}",
            "-" * 60,
        ]
        for k, (i, j) in enumerate(self.target_pairs):
            lines.append(
                f"  ({i:4d},{j:4d})  "
                f"{self.adjusted_correlations[k]:+9.4f}  "
                f"{self.crud_pvalues[k]:10.6f}  "
                f"{self.crud_pvalue_se[k]:9.6f}  "
                f"{self.crud_percentiles[k]:10.4f}"
            )
        return "\n".join(lines)


# -- Core function --


def crud_test(
    X: np.ndarray,
    target_pairs: list,
    K: int = 10,
    *,
    M: Optional[int] = None,
    strata: Optional[np.ndarray] = None,
    seed: int = 42,
) -> CrudTestResult:
    """Crud-aware calibration test for adjusted associations.

    Parameters
    ----------
    X : (n, p) array
        Raw data matrix (will be z-scored internally).
    target_pairs : list of (int, int)
        Feature index pairs to evaluate.
    K : int
        Number of principal components to remove.
    M : int or None
        Number of Monte Carlo reference pairs. If None and p*(p-1)/2 <= 500_000,
        uses all pairs (full correlation matrix). Otherwise defaults to 500_000.
    strata : (p,) array or None
        Optional stratum labels for features.
    seed : int
        Random seed for PCA and Monte Carlo sampling.

    Returns
    -------
    CrudTestResult
    """
    X = np.asarray(X, dtype=np.float32)
    n, p = X.shape
    target_pairs = [(int(i), int(j)) for i, j in target_pairs]

    # 1. Z-score each column (mean 0, unit variance)
    Xz = _safe_zscore_columns(X)

    # 2. PCA + residualize: remove top-K principal components
    Vk = _pca_top_k(Xz, K, seed=seed)
    resid = _residualize(Xz, Vk)

    # 3. Compute adjusted correlations for the user-specified target pairs
    target_corrs = _pairwise_corrs(resid, target_pairs)

    # 4. Build the reference (null) distribution of adjusted correlations.
    #    Three code paths depending on data size and user options:
    #    (a) stratified:   when strata labels are provided
    #    (b) full matrix:  when p*(p-1)/2 <= 500k (compute full corr matrix)
    #    (c) Monte Carlo:  sample M random pairs for large p
    total_pairs = p * (p - 1) // 2
    use_full = (M is None) and (total_pairs <= 500_000)

    # --- Code path (a): Stratified reference distribution ---
    # When strata are provided, the reference distribution for each target
    # pair (ti, tj) is built from pairs within the same stratum combination,
    # ensuring a fair comparison against structurally similar features.
    if strata is not None:
        strata = np.asarray(strata)
        crud_pvalues = np.empty(len(target_pairs), dtype=np.float64)
        n_refs_used = []
        all_ref_abs = []

        for k, (ti, tj) in enumerate(target_pairs):
            si, sj = strata[ti], strata[tj]
            idx_i = np.where(strata == si)[0]
            idx_j = np.where(strata == sj)[0]

            # Build candidate reference pairs from the same stratum combination
            if si == sj:
                candidates = [
                    (a, b)
                    for ii, a in enumerate(idx_i)
                    for b in idx_i[ii + 1 :]
                ]
            else:
                candidates = [(a, b) for a in idx_i for b in idx_j if a < b]

            if len(candidates) == 0:
                crud_pvalues[k] = 1.0
                n_refs_used.append(0)
                continue

            rng = np.random.default_rng(seed + k)
            m_eff = len(candidates) if M is None else min(M, len(candidates))
            if m_eff < len(candidates):
                chosen = rng.choice(len(candidates), size=m_eff, replace=False)
                ref_pairs = [candidates[c] for c in chosen]
            else:
                ref_pairs = candidates

            ref_corrs = _pairwise_corrs(resid, ref_pairs)
            ref_abs = np.abs(ref_corrs)
            all_ref_abs.append(ref_abs)
            n_refs_used.append(len(ref_pairs))

            # Add-one correction: p_hat = (1 + count) / (M + 1)
            # This prevents p=0 and makes the empirical p-value super-uniform
            # under H0 (see Theorem A.3, "Parametric crud-aware test").
            count = np.sum(ref_abs >= abs(target_corrs[k]))
            crud_pvalues[k] = (1 + count) / (len(ref_pairs) + 1)

        n_ref = int(np.median(n_refs_used)) if n_refs_used else 0
        bg_sd = float(np.std(np.concatenate(all_ref_abs))) if all_ref_abs else 0.0

    # --- Code path (b): Full correlation matrix ---
    # When p is small enough that p*(p-1)/2 <= 500k, we compute the full
    # residualized correlation matrix and use all upper-triangular entries
    # as the reference distribution. This is exact (no Monte Carlo noise).
    elif use_full:
        S = _corr_from_zscored(resid)
        iu = np.triu_indices_from(S, k=1)
        all_vals = S[iu]           # all unique off-diagonal correlations
        all_abs = np.abs(all_vals)
        bg_sd = float(np.std(all_vals))  # background SD = sigma_crud estimate
        n_ref = len(all_vals)

        crud_pvalues = np.empty(len(target_pairs), dtype=np.float64)
        for k, (ti, tj) in enumerate(target_pairs):
            # Add-one correction: p_hat = (1 + count) / (M + 1)
            # Prevents p=0 and ensures super-uniformity under H0.
            count = np.sum(all_abs >= abs(target_corrs[k]))
            crud_pvalues[k] = (1 + count) / (n_ref + 1)

    # --- Code path (c): Monte Carlo sampling of reference pairs ---
    # For large p where the full correlation matrix is too expensive,
    # sample M random feature pairs (default 500k) to approximate the
    # reference distribution.
    else:
        if M is None:
            M = 500_000
        rng = np.random.default_rng(seed)
        if M >= total_pairs:
            # M exceeds the total number of pairs — just use all of them
            iu = np.triu_indices(p, k=1)
            ref_pairs = list(zip(iu[0].tolist(), iu[1].tolist()))
            M = len(ref_pairs)
        else:
            # Map linear indices in [0, total_pairs) to (row, col) pairs
            # using the inverse of the upper-triangular indexing formula
            linear_idx = rng.choice(total_pairs, size=M, replace=False)
            rows = np.empty(M, dtype=np.int64)
            cols = np.empty(M, dtype=np.int64)
            for idx_pos in range(M):
                li = int(linear_idx[idx_pos])
                r = int(np.floor(p - 0.5 - np.sqrt((p - 0.5) ** 2 - 2 * li)))
                c = li - r * p + r * (r + 1) // 2 + r + 1
                rows[idx_pos] = r
                cols[idx_pos] = c
            ref_pairs = list(zip(rows.tolist(), cols.tolist()))

        ref_corrs = _pairwise_corrs(resid, ref_pairs)
        ref_abs = np.abs(ref_corrs)
        bg_sd = float(np.std(ref_corrs))  # background SD = sigma_crud estimate
        n_ref = len(ref_pairs)

        crud_pvalues = np.empty(len(target_pairs), dtype=np.float64)
        for k in range(len(target_pairs)):
            # Add-one correction: p_hat = (1 + count) / (M + 1)
            # Prevents p=0 and ensures super-uniformity under H0.
            count = np.sum(ref_abs >= abs(target_corrs[k]))
            crud_pvalues[k] = (1 + count) / (n_ref + 1)

    crud_percentiles = 1.0 - crud_pvalues

    # Binomial SE of the empirical p-value: SE = sqrt(p_hat * (1 - p_hat) / M).
    # This quantifies the Monte Carlo uncertainty in p_hat itself.
    crud_pvalue_se = np.sqrt(crud_pvalues * (1 - crud_pvalues) / max(n_ref, 1))

    # Finite-sample component of the null SD: 1/sqrt(n-1)
    sampling_sd = 1.0 / max(n - 1, 1) ** 0.5

    return CrudTestResult(
        target_pairs=target_pairs,
        adjusted_correlations=target_corrs,
        crud_pvalues=crud_pvalues,
        crud_percentiles=crud_percentiles,
        crud_pvalue_se=crud_pvalue_se,
        background_sd=bg_sd,
        sampling_sd=sampling_sd,
        K=K,
        n_reference_pairs=n_ref,
        n_samples=n,
    )


# -- Parametric crud-aware test (summary-statistic version) --


@dataclass
class CrudZResult:
    """Result of the parametric crud-aware z-test."""

    r: np.ndarray
    n: int
    crud_sd: float
    sampling_sd: float
    total_sd: float
    z: np.ndarray
    crud_pvalues: np.ndarray
    crud_percentiles: np.ndarray
    classical_pvalues: np.ndarray

    def summary(self) -> str:
        lines = [
            f"Parametric crud-aware z-test  (n={self.n})",
            f"  crud SD (domain background):  {self.crud_sd:.4f}",
            f"  sampling SD (1/sqrt(n-1)):    {self.sampling_sd:.4f}",
            f"  total null SD:                {self.total_sd:.4f}",
            "",
            f"  {'r':>9s}  {'z_crud':>8s}  {'crud_pval':>10s}  "
            f"{'percentile':>10s}  {'classical_p':>11s}",
            "-" * 60,
        ]
        r = np.atleast_1d(self.r)
        for k in range(len(r)):
            lines.append(
                f"  {r[k]:+9.4f}  "
                f"{self.z[k]:8.3f}  "
                f"{self.crud_pvalues[k]:10.6f}  "
                f"{self.crud_percentiles[k]:10.4f}  "
                f"{self.classical_pvalues[k]:11.6f}"
            )
        return "\n".join(lines)


def crud_z_test(
    r: float | np.ndarray,
    n: int,
    crud_sd: float,
) -> CrudZResult:
    """Parametric crud-aware significance test from summary statistics.

    Parameters
    ----------
    r : float or array
        Observed (adjusted) correlation(s).
    n : int
        Sample size.
    crud_sd : float
        SD of background correlations in the domain.

    Returns
    -------
    CrudZResult
    """
    from scipy.stats import norm

    r_arr = np.atleast_1d(np.asarray(r, dtype=np.float64))

    # Finite-sample component: sampling_sd = 1/sqrt(n-1)
    sampling_sd = 1.0 / max(n - 1, 1) ** 0.5

    # Total null SD combines crud background and sampling noise (paper formula):
    #   total_sd^2 = sigma_crud^2 + 1/(n-1)
    # Under H0 the adjusted correlation is N(0, total_sd^2).
    total_sd = (crud_sd ** 2 + sampling_sd ** 2) ** 0.5

    # Crud-aware z-statistic (paper formula):
    #   z_crud = |r| / sqrt(sigma_crud^2 + 1/(n-1))
    z = np.abs(r_arr) / total_sd
    # Two-sided p-value from the standard normal
    crud_pvalues = 2.0 * norm.sf(z)
    crud_percentiles = 1.0 - crud_pvalues

    # Classical z-test: the special case where sigma_crud = 0, so
    #   z_classical = |r| * sqrt(n-1)   (only sampling variance in denominator)
    z_classical = np.abs(r_arr) * max(n - 1, 1) ** 0.5
    classical_pvalues = 2.0 * norm.sf(z_classical)

    return CrudZResult(
        r=r_arr,
        n=n,
        crud_sd=crud_sd,
        sampling_sd=sampling_sd,
        total_sd=total_sd,
        z=z,
        crud_pvalues=crud_pvalues,
        crud_percentiles=crud_percentiles,
        classical_pvalues=classical_pvalues,
    )


# -- CLI --


def main():
    parser = argparse.ArgumentParser(
        description="Crud-aware calibration test for adjusted associations."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to data file (.npy or .csv)",
    )
    parser.add_argument(
        "--pairs",
        required=True,
        help='Target pairs as "i,j;k,l;..." (0-indexed)',
    )
    parser.add_argument("--K", type=int, default=10, help="PCs to remove (default: 10)")
    parser.add_argument(
        "--M", type=int, default=None, help="Monte Carlo reference pairs (default: auto)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Load data
    path = args.data
    if path.endswith(".npy"):
        X = np.load(path)
    elif path.endswith(".csv"):
        import pandas as pd

        X = pd.read_csv(path).select_dtypes(include=[np.number]).dropna().to_numpy()
    else:
        print(f"Unsupported file format: {path}", file=sys.stderr)
        sys.exit(1)

    # Parse pairs
    target_pairs = []
    for tok in args.pairs.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        i, j = tok.split(",")
        target_pairs.append((int(i.strip()), int(j.strip())))

    if not target_pairs:
        print("No target pairs specified.", file=sys.stderr)
        sys.exit(1)

    result = crud_test(X, target_pairs, K=args.K, M=args.M, seed=args.seed)
    print(result.summary())


if __name__ == "__main__":
    main()
