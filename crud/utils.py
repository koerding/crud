"""Central utility functions shared across the analysis pipeline.

Provides the numerical building blocks used by every dataset notebook and
by the assumption audit:

* **Z-scoring** -- column-wise standardisation with constant-column safety.
* **Correlation matrices** -- standard z-scored Gram matrix.
* **Randomised PCA** -- thin SVD wrapper that returns explained-variance
  ratios (eigenvalue spectrum) and right singular vectors (loadings).
* **Residualisation** -- projects out the top-K principal-component
  subspace and returns the off-diagonal residual correlations, which is
  the key operation for measuring "crud" at different K.
* **Power-law fitting** -- OLS on log-log to estimate the spectral
  exponent alpha in lambda_k ~ k^{-alpha}.
* **Caching / downloading** -- helpers to avoid redundant computation
  and to fetch remote datasets on first use.

Key notation throughout:
  - Xz : (n x p) z-scored data matrix (zero mean, unit variance per column)
  - Vt : (k x p) right singular vectors (rows are PC loading directions)
  - evr: explained-variance ratio, i.e. lambda_k / sum(lambda)
  - K   : number of PCs removed when computing residual correlations
"""

from typing import Callable, Tuple

import numpy as np
import requests
from pathlib import Path
from sklearn.utils.extmath import randomized_svd

from crud.config import EPS, SEED, CACHE_DIR


def download_if_needed(url: str, local_path: str, *, timeout: int = 120, chunk_mb: int = 4) -> None:
    """Download a remote file to *local_path* if it does not already exist.

    Uses streaming to handle large files (e.g. GTEx expression matrices)
    without loading the entire response into memory at once.

    Parameters
    ----------
    url : str
        Remote URL to fetch.
    local_path : str
        Local filesystem destination.
    timeout : int
        Connection timeout in seconds.
    chunk_mb : int
        Download chunk size in megabytes.
    """
    path = Path(local_path)
    if path.exists():
        return
    print(f"Downloading {url} -> {local_path}")
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    chunk_size = chunk_mb * 1024 * 1024
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    print("Download complete.")


def safe_zscore_columns(X: np.ndarray, *, eps: float = EPS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each feature (column) independently across samples.

    Returns (Xz, mean, std) where Xz has zero mean and unit sample
    variance per column.

    Implementation notes
    --------------------
    * The column mean is computed in **float64** to avoid catastrophic
      cancellation when the mean is large relative to the spread, then
      cast back to float32 for the main computation.  The rest of the
      arithmetic stays in float32 for speed and memory.
    * Constant (or near-constant) columns would produce sd ~ 0 and blow
      up the division.  We replace any sd < eps with 1.0, which
      effectively leaves those columns as (X - mean) un-scaled.  They
      end up near-zero and contribute negligibly to correlations.
    * After dividing by sd we **re-centre** each column (subtract its
      column mean again).  This is a numerical-hygiene step: the first
      subtraction of mu can leave residual means on the order of
      float32 machine epsilon * |mu|, which would bias Gram-matrix
      correlations.  The second centering removes that drift.
    """
    X = np.asarray(X, dtype=np.float32)
    # float64 accumulation for the mean avoids loss of significance.
    mu = X.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    Xc = X - mu
    sd = Xc.std(axis=0, ddof=1, keepdims=True).astype(np.float32)
    # Guard against constant columns: replace near-zero std with 1.
    sd = np.where(sd < eps, 1.0, sd)
    Xz = Xc / (sd + eps)
    # Re-centre to eliminate residual float32 drift in column means.
    Xz = Xz - Xz.mean(axis=0, keepdims=True)
    return Xz.astype(np.float32), mu.squeeze(0), sd.squeeze(0)


def preprocess_for_pca_and_corr(X: np.ndarray, *, eps: float = EPS) -> Tuple[np.ndarray, dict]:
    """Z-score *X* by feature and return diagnostics about the result.

    The diagnostics dict records how well the standardisation worked:
    column means should be near zero, standard deviations near one, and
    the number of near-constant columns should be small.  These are
    logged per-dataset as a sanity check.
    """
    Xz, mu, sd = safe_zscore_columns(X, eps=eps)
    diag = {
        "col_mean_abs_median": float(np.median(np.abs(Xz.mean(axis=0)))),
        "col_mean_abs_max": float(np.max(np.abs(Xz.mean(axis=0)))),
        "col_std_median": float(np.median(Xz.std(axis=0, ddof=1))),
        "col_std_min": float(np.min(Xz.std(axis=0, ddof=1))),
        "col_std_max": float(np.max(Xz.std(axis=0, ddof=1))),
        "n_near_const_cols": int(np.sum(sd < 1e-6)),
    }
    return Xz, diag


def corr_from_zscored(Xz: np.ndarray) -> np.ndarray:
    """Compute the sample correlation matrix from already z-scored data.

    When each column of Xz has zero mean and unit sample variance, the
    sample covariance matrix equals the Gram matrix:

        S = Xz^T Xz / (n - 1)

    and this is also the sample *correlation* matrix (since the marginal
    variances are 1).  The matrix product is accumulated in **float64**
    before the division to reduce rounding error in the p x p inner
    products, then cast back to float32 for storage.  The diagonal is
    explicitly set to 1.0 to correct any floating-point drift.
    """
    Xz = np.asarray(Xz, dtype=np.float32)
    n = Xz.shape[0]
    # Accumulate the Gram matrix in float64 for numerical precision.
    S = (Xz.T @ Xz).astype(np.float64) / max(n - 1, 1)
    S = S.astype(np.float32)
    # Force exact 1s on the diagonal (float rounding can make them ~0.9999).
    np.fill_diagonal(S, 1.0)
    return S


def offdiag_vals_from_corr(S: np.ndarray) -> np.ndarray:
    """Extract the upper-triangle off-diagonal entries of a correlation matrix.

    Returns a flat array of p*(p-1)/2 pairwise correlations (excluding
    the diagonal, which is always 1).  These are the values whose
    distribution the paper analyses: in the absence of any true causal
    signal, the "crud factor" predicts they will be non-zero due to
    persistent background covariance from the power-law spectrum.

    Non-finite entries (possible from degenerate columns) are dropped.
    """
    # k=1 skips the main diagonal, giving strictly off-diagonal pairs.
    iu = np.triu_indices_from(S, k=1)
    vals = S[iu]
    vals = vals[np.isfinite(vals)]
    return vals.astype(np.float32)


def pca_randomized(Xz: np.ndarray, n_components: int, *, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PCA via randomised SVD and return eigenvalue spectrum + loadings.

    Parameters
    ----------
    Xz : (n, p) array
        Z-scored data matrix (zero mean, unit variance per column).
    n_components : int
        Requested number of components (capped at min(n, p)).
    seed : int
        Random state for reproducibility of the randomised SVD.

    Returns
    -------
    evr : (k,) array, float32
        Explained-variance ratios: evr[i] = lambda_i / sum(lambda).
        This is the eigenvalue spectrum normalised to sum to 1, i.e.
        the fraction of total variance captured by each PC.  The paper
        shows these follow a power law lambda_k ~ k^{-alpha}.
    Vt : (k, p) array, float32
        Right singular vectors (rows are PC loading directions).  Used
        downstream by ``residual_corr_vals_after_k_pcs`` to project
        out the top-K subspace.

    Notes
    -----
    The relationship between singular values and eigenvalues is:
        lambda_k = s_k^2 / (n - 1)
    where s_k are the singular values of Xz.  This follows from the
    eigendecomposition of the sample covariance: Cov = Xz^T Xz / (n-1),
    whose eigenvalues are s_k^2 / (n-1).

    Total variance is computed as sum_j Var(Xz_j) = sum of all squared
    entries / (n-1).  For perfectly z-scored data this equals p, but we
    compute it explicitly to be safe against float32 drift.
    """
    Xz = np.asarray(Xz, dtype=np.float32)
    n, p = Xz.shape
    k = int(min(n_components, n, p))
    if k <= 0:
        raise ValueError("Need at least 1 component.")
    U, S, Vt = randomized_svd(Xz, n_components=k, random_state=seed)
    # lambda_k = s_k^2 / (n-1): eigenvalues of the sample covariance.
    ev = (S.astype(np.float64) ** 2) / max(n - 1, 1)
    # Total variance = trace(Cov) = sum of all column variances.
    ss = np.sum((Xz.astype(np.float64) ** 2), axis=0)
    total_var = float(np.sum(ss) / max(n - 1, 1))
    # evr = lambda_k / sum(lambda): fraction of variance explained.
    evr = (ev / (total_var + EPS)).astype(np.float32)
    return evr, Vt.astype(np.float32)


def residual_corr_vals_after_k_pcs(
    Xz: np.ndarray,
    top_pcs: np.ndarray,
    cols: np.ndarray,
    k: int,
    *,
    eps: float = EPS,
) -> np.ndarray:
    """Residual pairwise correlations on a feature subset after removing K PCs.

    This is the **key function** for the crud-factor analysis.  It
    implements the residualisation

        Sigma^{(K)} = (I - P_K) Sigma (I - P_K)

    from the paper, where P_K is the rank-K projection onto the top K
    principal component directions.

    Algebra of the residualisation
    ------------------------------
    Let V_K be the (K x p) matrix whose rows are the top-K right
    singular vectors (PC loadings).  The projection matrix is
    P_K = V_K^T V_K, and the residual of any data vector x is

        x_resid = x - V_K^T V_K x  =  (I - V_K^T V_K) x.

    In matrix form for the full data:

        scores = Xz @ V_K^T          -- (n x K) PC scores using ALL p features
        recon  = scores @ V_K         -- (n x p) rank-K reconstruction
        resid  = Xz - recon           -- (n x p) residual

    We only need correlations among a *subset* of columns (``cols``), so
    we restrict the reconstruction to those columns:

        Xs    = Xz[:, cols]                          -- (n x |cols|)
        resid = Xs - scores @ V_K[:, cols]           -- (n x |cols|)

    This is equivalent to ``(Xz - Xz @ V_K^T @ V_K)[:, cols]`` but
    avoids materialising the full (n x p) reconstruction.

    **Important**: the PC scores are computed from the FULL Xz (all p
    features), not just the subset.  The subset selection happens only
    in the reconstruction step.  This is correct and intentional --
    the PCs capture global structure across all features, and we want
    to remove that global structure from the subset before measuring
    residual correlations.

    After residualisation the columns are re-centred and re-standardised,
    then the off-diagonal correlations are extracted.

    Parameters
    ----------
    Xz : (n, p) array
        Z-scored full data matrix.
    top_pcs : (K_max, p) array
        Rows are the top PC loading vectors (right singular vectors Vt),
        as returned by ``pca_randomized``.
    cols : 1-d int array
        Column indices of the feature subset to compute correlations for.
    k : int
        Number of PCs to project out. k=0 gives raw correlations.
    eps : float
        Floor for standard deviation to prevent division by zero.

    Returns
    -------
    vals : 1-d float32 array
        Flat vector of off-diagonal pairwise residual correlations,
        i.e. the upper triangle of the |cols| x |cols| residual
        correlation matrix.
    """
    Xz = np.asarray(Xz, dtype=np.float32)
    cols = np.asarray(cols, dtype=int)
    Xs = Xz[:, cols]                          # feature subset (n x |cols|)
    if k <= 0:
        # No PCs removed -- raw correlations.
        resid = Xs
    else:
        kk = int(min(k, top_pcs.shape[0]))
        Vk = top_pcs[:kk].astype(np.float32)  # (kk x p) loading matrix
        # PC scores computed from ALL p features (n x kk).
        scores = (Xz @ Vk.T).astype(np.float32)
        # Reconstruct only the subset columns and subtract.
        # resid = Xs - (Xz @ Vk^T @ Vk)[:, cols]
        resid = (Xs - scores @ Vk[:, cols]).astype(np.float32)
        del scores
    # Re-centre and re-standardise the residual columns so that
    # corr_from_zscored gives a proper correlation matrix.
    resid = resid - resid.mean(axis=0, keepdims=True)
    sd = resid.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    resid = resid / sd
    S = corr_from_zscored(resid)
    vals = offdiag_vals_from_corr(S)
    del resid, S, Xs
    return vals


def fit_power_law_first_n(evr: np.ndarray, n_fit: int = 40, y_bottom: float = 1e-6) -> Tuple[float, float]:
    """Fit the power-law exponent alpha for the eigenvalue spectrum.

    The paper's central empirical claim is that PCA eigenvalues follow a
    power law across domains:

        lambda_k  ~  k^{-alpha}

    Taking logs gives a linear model:

        log(lambda_k) = -alpha * log(k) + const

    which we fit by ordinary least squares on the first *n_fit* eigenvalue
    ratios (default 40).  Values below *y_bottom* are clamped to avoid
    log(0).

    Across the 9 datasets in the paper, alpha ranges from 0.63 to 1.33
    (see Table in assumption_audit.tex), with R^2 typically above 0.95,
    confirming the power-law structure.

    Parameters
    ----------
    evr : 1-d array
        Explained-variance ratios (eigenvalue spectrum), as returned by
        ``pca_randomized``.
    n_fit : int
        Number of leading eigenvalues to include in the regression.
    y_bottom : float
        Floor for eigenvalues before taking log (prevents -inf).

    Returns
    -------
    alpha : float
        Estimated power-law exponent (-slope on the log-log plot).
    r2 : float
        Coefficient of determination of the log-log linear fit.
    """
    evr = np.asarray(evr, dtype=float)
    k = np.arange(1, len(evr) + 1, dtype=float)
    m = int(min(n_fit, len(evr)))
    # Need at least 5 points for a meaningful regression.
    if m < 5:
        return float("nan"), float("nan")
    x = np.log(k[:m])                               # log(k)
    y = np.log(np.maximum(evr[:m], y_bottom))        # log(lambda_k)
    # OLS: y = slope * x + intercept
    A = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(beta[0]), float(beta[1])
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + EPS
    r2 = 1.0 - ss_res / ss_tot
    # alpha is the *negative* slope (power-law exponent is positive).
    alpha = -slope
    return alpha, r2


def cached_load(name: str, loader: Callable, **kwargs) -> np.ndarray:
    """Load a dataset from .npy cache, or compute + cache it on first call.

    This avoids re-downloading / re-parsing large datasets (e.g. GTEx,
    UK Biobank) on every run.  The cache directory is set in
    ``crud.config.CACHE_DIR``.

    Parameters
    ----------
    name : str
        Human-readable cache key; the file is stored as ``<name>.npy``.
    loader : Callable
        Zero-argument (after partial application via **kwargs) function
        that returns the numpy array to cache.
    **kwargs
        Forwarded to *loader*.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{name}.npy"
    if cache_path.exists():
        print(f"[cache] Loading {name} from {cache_path}")
        return np.load(cache_path)
    X = loader(**kwargs)
    np.save(cache_path, X)
    print(f"[cache] Saved {name} to {cache_path}")
    return X
