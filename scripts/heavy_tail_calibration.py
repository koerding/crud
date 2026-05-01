"""Quantify heavy-tail miscalibration of the parametric crud-aware z-test.

For each of two heavy-tailed datasets (NHANES, RNASeq), we:
  1. Z-score, then residualize off the top K=10 PCs.
  2. Compute the empirical distribution of off-diagonal residual correlations
     across uniformly sampled feature pairs.
  3. Estimate sigma_K = SD of that distribution.
  4. For target |r| in {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}, compute:
        - parametric p = 2 * (1 - Phi(|r| / sqrt(sigma_K^2 + 1/(n-1))))
        - empirical  p = fraction of off-diagonal pairs with |r_resid| >= |r|
     and the ratio (empirical / parametric).

We print a small table that the paper can quote.
"""

from pathlib import Path
import numpy as np
from scipy.stats import norm, kurtosis, skew

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "cached_data"
DATASETS = [("NHANES", "NHANES.npy"), ("RNASeq", "RNASeq.npy")]
K = 10
RNG = np.random.default_rng(42)


def zscore_columns(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return ((X - mu) / sd).astype(np.float32)


def residualize_topk(Xz: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Project out top-k right singular components."""
    n, p = Xz.shape
    # randomized SVD top-k via numpy linalg on covariance (small p datasets here)
    # NHANES: p=165, RNASeq: p=387 — full SVD is fine.
    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    Vk = Vt[:k]  # (k, p)
    coef = Xz @ Vk.T  # (n, k)
    return Xz - coef @ Vk


def offdiag_corr_all(Xr: np.ndarray) -> np.ndarray:
    """Compute all off-diagonal feature-pair correlations from residualized matrix."""
    # corrcoef rows of Xr.T  -> p x p correlation
    C = np.corrcoef(Xr.T)
    iu = np.triu_indices(C.shape[0], k=1)
    return C[iu].astype(np.float64)


def report(name: str, X: np.ndarray, K: int):
    print(f"\n=== {name}  (n={X.shape[0]}, p={X.shape[1]})  K={K} ===")
    Xz = zscore_columns(X)
    Xr = residualize_topk(Xz, K)
    rs = offdiag_corr_all(Xr)
    print(f"  pairs analyzed = {len(rs):,}")
    sigma_K = float(rs.std(ddof=0))
    sk = float(skew(rs))
    ku = float(kurtosis(rs, fisher=True))  # excess kurtosis
    print(f"  sigma_K (empirical)    = {sigma_K:.4f}")
    print(f"  skewness (empirical)   = {sk:+.3f}")
    print(f"  excess kurtosis (emp.) = {ku:+.3f}")
    n = X.shape[0]
    sampling_sd = 1.0 / np.sqrt(max(n - 1, 1))
    total_sd = np.sqrt(sigma_K**2 + sampling_sd**2)
    print(f"  total parametric SD    = {total_sd:.4f}  "
          f"(sampling SD = {sampling_sd:.4f})")
    print()
    print(f"  {'|r|':>5}  {'param p':>10}  {'empir p':>10}  {'ratio E/P':>10}")
    print("  " + "-" * 42)
    targets = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    abs_r = np.abs(rs)
    rows = []
    for t in targets:
        param_p = 2.0 * norm.sf(t / total_sd)
        emp_p = float((abs_r >= t).mean())
        ratio = emp_p / param_p if param_p > 0 else float("inf")
        print(f"  {t:>5.2f}  {param_p:>10.5f}  {emp_p:>10.5f}  {ratio:>10.2f}")
        rows.append((t, param_p, emp_p, ratio))
    return {
        "name": name,
        "n": n,
        "p": X.shape[1],
        "sigma_K": sigma_K,
        "skew": sk,
        "ex_kurt": ku,
        "rows": rows,
    }


def main():
    results = []
    for name, fname in DATASETS:
        path = CACHE_DIR / fname
        X = np.load(path).astype(np.float32)
        results.append(report(name, X, K))
    print("\n\n=== Summary (empirical / parametric tail-mass ratios) ===")
    print(f"{'dataset':>8}  {'sigma_K':>8}  {'kurt':>6}  "
          f"{'|r|=0.10':>10}  {'|r|=0.20':>10}  {'|r|=0.30':>10}")
    for r in results:
        rd = dict((row[0], row) for row in r["rows"])
        print(f"{r['name']:>8}  {r['sigma_K']:>8.4f}  {r['ex_kurt']:>6.1f}  "
              f"{rd[0.10][3]:>10.2f}  {rd[0.20][3]:>10.2f}  {rd[0.30][3]:>10.2f}")


if __name__ == "__main__":
    main()
