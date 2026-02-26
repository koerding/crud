"""
Assumption audit for Theorems A.1 ("Crud scale under top-K PC removal")
and A.2 ("Association-only decisions cannot be reliable below the crud scale").

Runs seven diagnostic tests on every cached dataset to verify the
mathematical assumptions behind the sigma_K formula.  Tests 1--6
correspond directly to the six tests reported in the paper's
Appendix (Section "Empirical audit of assumptions").  Test 7 is a
supplementary stability check not reported in the paper.

The seven tests are:
  1) Power-law eigenvalue decay -- AIC/BIC comparison of power-law vs
     exponential fit to the top eigenvalues; local log-log slope diagnostic.
     [Paper Test 1]
  2) Eigenvector delocalization -- checks Assumption A.1 ("Haar-like
     eigenvectors / delocalization") via IPR, coherence, entropy, leverage,
     and QQ-plots of sqrt(p)*v_{ik} against N(0,1).
     [Paper Test 2]
  3) Predicted sigma_K vs empirical residual correlation SD -- the central
     test of Theorem A.1: does the spectral formula accurately predict
     the standard deviation of off-diagonal residual correlations?
     [Paper Test 3]
  4) Diagonal concentration -- checks that the coefficient of variation
     of residual variances across variables is small (i.e. the diagonal
     of Sigma^(K) concentrates), an intermediate step in the proof.
     [Paper Test 4]
  5) Normality of residual correlations -- tests Theorem A.2: the
     off-diagonal entries of the residual correlation matrix should be
     approximately Gaussian (QQ-plot, skewness, kurtosis, tail quantiles).
     [Paper Test 5]
  6) Cross-component covariance negligibility -- checks that the products
     v_{ik}*v_{jk} across components k are approximately uncorrelated for
     random variable pairs (i,j), a condition needed for the variance
     calculation in the Theorem A.1 proof.
     [Paper Test 6]
  7) Split-half PC stability -- splits samples in half and checks
     reproducibility of principal subspaces (via subspace angles) and
     of the sigma_K estimate across the two halves.
     [Supplementary; not in paper]

Usage:
    .venv/bin/python -u assumption_audit.py
"""

import gc
import math
import os, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Ensure CWD is repo root so relative data paths resolve correctly.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.utils.extmath import randomized_svd

from crud.config import EPS, SEED, CACHE_DIR
from crud.utils import safe_zscore_columns

# ---------------------------------------------------------------------------
# Constants (audit-specific overrides)
# ---------------------------------------------------------------------------
MAX_ROWS = 10_000
FIG_DIR = Path("figures_audit")
FIG_FORMAT = "png"

# ---------------------------------------------------------------------------
# Figure auto-save (audit uses different dir and filename prefix)
# ---------------------------------------------------------------------------
_fig_counter = 0


def install_autosave_show(fig_dir: Path, fmt: str = "png") -> None:
    global _fig_counter
    fig_dir.mkdir(parents=True, exist_ok=True)
    if getattr(plt.show, "__name__", "") == "show_and_save":
        return
    old_show = plt.show

    def show_and_save(*args, **kwargs):
        global _fig_counter
        _fig_counter += 1
        fig = plt.gcf()
        out = fig_dir / f"audit_{_fig_counter:03d}.{fmt}"
        fig.savefig(out, format=fmt, bbox_inches="tight")
        return old_show(*args, **kwargs)

    plt.show = show_and_save


install_autosave_show(FIG_DIR, FIG_FORMAT)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_subsample(name: str, max_rows: int = MAX_ROWS, seed: int = SEED) -> np.ndarray:
    X = np.load(CACHE_DIR / f"{name}.npy")
    n = X.shape[0]
    if n > max_rows:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(n, max_rows, replace=False)]
    return X.astype(np.float32)


def zscore_columns(X: np.ndarray) -> np.ndarray:
    Xz, _mu, _sd = safe_zscore_columns(X)
    return Xz


def pca(Xz: np.ndarray, k: int, seed: int = SEED):
    """Compute top-k PCA via randomized SVD.

    Returns a 3-tuple (evr, Vt, raw_eigenvalues).  Note: this returns
    three values unlike ``pca_randomized`` in ``crud.utils`` which
    returns only two (evr, Vt).  The raw eigenvalues are needed by
    several audit tests (e.g. test_sigma_prediction, test_split_stability).
    """
    n, p = Xz.shape
    k = min(k, n, p)
    U, S, Vt = randomized_svd(Xz.astype(np.float32), n_components=k, random_state=seed)
    ev = (S.astype(np.float64) ** 2) / max(n - 1, 1)
    total_var = float(np.sum(Xz.astype(np.float64) ** 2) / max(n - 1, 1))
    evr = (ev / (total_var + EPS)).astype(np.float64)
    return evr, Vt, ev


# ===================================================================
# TEST 1: Power-law eigenvalue decay
# ===================================================================
# Checks whether the empirical eigenvalue spectrum is better described
# by a power law (lambda_k ~ k^{-alpha}) than by an exponential
# (lambda_k ~ exp(-beta*k)).  Uses AIC/BIC for model selection and
# reports the local log-log slope as a diagnostic.

def test_powerlaw(evr: np.ndarray, name: str, max_fit: int = 40) -> dict:
    """Local slope, power-law vs exponential vs broken power-law fits.

    Fits on the first max_fit components (default 40), matching the main
    analysis.  Empirical eigenvalue spectra follow a power law well over
    this range; including very high-index components where the spectrum
    flattens or becomes noisy would bias the comparison against the
    power-law model.
    """
    evr = np.asarray(evr, dtype=np.float64)
    k = np.arange(1, len(evr) + 1, dtype=np.float64)
    m = min(max_fit, len(evr))

    # --- Local slope (finite difference in log-log) ---
    # Show local slope over a wider range for diagnostic purposes,
    # but fit only on [1, m].
    m_diag = min(200, len(evr))
    log_k_diag = np.log(k[:m_diag])
    log_ev_diag = np.log(np.maximum(evr[:m_diag], 1e-15))
    local_slope_diag = np.diff(log_ev_diag) / np.diff(log_k_diag)

    log_k = np.log(k[:m])
    log_ev = np.log(np.maximum(evr[:m], 1e-15))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(k[1:m_diag], local_slope_diag, linewidth=1, alpha=0.6, label="all")
    plt.plot(k[1:m], local_slope_diag[:m-1], linewidth=2, label=f"fit range (1–{m})")
    median_slope = np.median(local_slope_diag[:m-1])
    plt.axhline(median_slope, color="r", linestyle="--",
                label=f"median(1–{m})={median_slope:.2f}")
    plt.axvline(m, color="gray", linestyle=":", linewidth=1, label=f"fit boundary k={m}")
    plt.xlabel("component k"); plt.ylabel("local slope d(log λ)/d(log k)")
    plt.title(f"{name}: local power-law slope")
    plt.legend(fontsize=8); plt.grid(True, linestyle="--", linewidth=0.5)

    # --- Model comparison: power-law vs exponential (fit range only) ---
    # Both models are linearised in log-space and fit by OLS:
    #   Power law:    log(evr) = a + b*log(k)   =>  evr ~ k^b
    #   Exponential:  log(evr) = a + b*k         =>  evr ~ exp(b*k)
    # AIC and BIC are computed from the residual sum of squares.
    x_pl = log_k
    x_exp = k[:m]
    y = log_ev
    mask = np.isfinite(y)

    # Power law fit
    A_pl = np.vstack([x_pl[mask], np.ones(mask.sum())]).T
    beta_pl, res_pl, *_ = np.linalg.lstsq(A_pl, y[mask], rcond=None)
    yhat_pl = beta_pl[0] * x_pl[mask] + beta_pl[1]
    ss_res_pl = float(np.sum((y[mask] - yhat_pl) ** 2))

    # R² for power law
    ss_tot = float(np.sum((y[mask] - y[mask].mean()) ** 2)) + EPS
    r2_pl = 1.0 - ss_res_pl / ss_tot

    # Exponential fit
    A_exp = np.vstack([x_exp[mask], np.ones(mask.sum())]).T
    beta_exp, res_exp, *_ = np.linalg.lstsq(A_exp, y[mask], rcond=None)
    yhat_exp = beta_exp[0] * x_exp[mask] + beta_exp[1]
    ss_res_exp = float(np.sum((y[mask] - yhat_exp) ** 2))

    n_pts = int(mask.sum())
    k_params = 2
    aic_pl = n_pts * np.log(ss_res_pl / n_pts + EPS) + 2 * k_params
    aic_exp = n_pts * np.log(ss_res_exp / n_pts + EPS) + 2 * k_params
    bic_pl = n_pts * np.log(ss_res_pl / n_pts + EPS) + k_params * np.log(n_pts)
    bic_exp = n_pts * np.log(ss_res_exp / n_pts + EPS) + k_params * np.log(n_pts)

    # Plot: show full spectrum but highlight fit range
    m_plot = min(200, len(evr))
    plt.subplot(1, 2, 2)
    plt.loglog(k[:m_plot], evr[:m_plot], "k-", linewidth=1, alpha=0.4, label="data (full)")
    plt.loglog(k[:m], evr[:m], "k-", linewidth=2, label=f"data (fit range 1–{m})")
    plt.loglog(k[:m][mask], np.exp(yhat_pl), "r--", linewidth=1.5,
               label=f"power law (α={-beta_pl[0]:.2f}, R²={r2_pl:.3f})")
    plt.loglog(k[:m][mask], np.exp(yhat_exp), "b:", linewidth=1.5,
               label=f"exponential (AIC={aic_exp:.0f})")
    plt.axvline(m, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("k"); plt.ylabel("λ_k / total var")
    plt.title(f"{name}: model comparison (fit on first {m} PCs)")
    plt.legend(fontsize=8); plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout(); plt.show(); plt.close()

    return {
        "alpha": -beta_pl[0],
        "r2": r2_pl,
        "aic_powerlaw": aic_pl, "aic_exponential": aic_exp,
        "bic_powerlaw": bic_pl, "bic_exponential": bic_exp,
        "powerlaw_preferred": aic_pl < aic_exp,
    }


# ===================================================================
# TEST 2: Eigenvector delocalization
# ===================================================================
# Checks Assumption A.1 ("Haar-like eigenvectors / delocalization"):
# delocalized across all p variables.  For truly Haar-distributed
# eigenvectors each entry scales as 1/sqrt(p).  Diagnostics:
#   - Coherence: max|v_{ik}| should be close to 1/sqrt(p)
#   - IPR (inverse participation ratio): sum(v_{ik}^4), small => spread
#   - Entropy effective dimension: exp(H), should be close to p
#   - QQ-plot of sqrt(p)*v_{ik} vs N(0,1)
#   - Leverage scores: sum_k v_{ik}^2, should be uniform ~ K/p

def test_delocalization(Vt: np.ndarray, name: str, max_k: int = 50) -> dict:
    """Coherence, IPR, entropy, entry QQ, leverage scores.

    Tests Assumption A.1 ("Haar-like eigenvectors / delocalization").
    """
    Vt = Vt[:max_k].astype(np.float64)
    K, p = Vt.shape

    # --- Coherence ---
    coherence = float(np.max(np.abs(Vt)))
    ideal_coherence = 1.0 / np.sqrt(p)

    # --- IPR per component ---
    ipr = np.sum(Vt ** 4, axis=1)  # (K,)
    effective_support = 1.0 / (ipr + EPS)
    ideal_ipr = 1.0 / p

    # --- Entropy per component ---
    v2 = Vt ** 2
    v2 = np.maximum(v2, EPS)
    entropy = -np.sum(v2 * np.log(v2), axis=1)
    effective_dim = np.exp(entropy)

    # --- Leverage scores ---
    leverage = np.sum(Vt ** 2, axis=0)  # sum over components for each variable
    # Under uniform delocalization, leverage_i ≈ K/p for all i

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"{name}: eigenvector delocalization diagnostics", fontsize=14)

    # IPR
    ax = axes[0, 0]
    ax.semilogy(np.arange(1, K + 1), ipr, "o-", markersize=3)
    ax.axhline(ideal_ipr, color="r", linestyle="--", label=f"ideal 1/p = {ideal_ipr:.2e}")
    ax.set_xlabel("PC index k"); ax.set_ylabel("IPR(k)")
    ax.set_title(f"IPR (coherence μ={coherence:.4f}, ideal={ideal_coherence:.4f})")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", linewidth=0.5)

    # Effective support
    ax = axes[0, 1]
    ax.plot(np.arange(1, K + 1), effective_support, "o-", markersize=3)
    ax.axhline(p, color="r", linestyle="--", label=f"p = {p}")
    ax.set_xlabel("PC index k"); ax.set_ylabel("1/IPR(k) (effective support)")
    ax.set_title("Effective support per PC")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", linewidth=0.5)

    # Entropy effective dimension
    ax = axes[0, 2]
    ax.plot(np.arange(1, K + 1), effective_dim, "o-", markersize=3)
    ax.axhline(p, color="r", linestyle="--", label=f"p = {p}")
    ax.set_xlabel("PC index k"); ax.set_ylabel("exp(H_k)")
    ax.set_title("Entropy-based effective dimension")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", linewidth=0.5)

    # Entry distribution: QQ plot of sqrt(p)*v_{ik} vs N(0,1) for first few PCs
    ax = axes[1, 0]
    for pc_idx in [0, 4, 9, min(K - 1, 19)]:
        entries = np.sqrt(p) * Vt[pc_idx, :]
        qq = stats.probplot(entries, dist="norm")
        ax.plot(qq[0][0], qq[0][1], ".", markersize=1, label=f"PC{pc_idx+1}")
    lims = ax.get_xlim()
    ax.plot(lims, lims, "k--", linewidth=0.5)
    ax.set_xlabel("theoretical quantiles"); ax.set_ylabel("sample quantiles")
    ax.set_title("QQ: √p · v_ik vs N(0,1)")
    ax.legend(fontsize=7); ax.grid(True, linestyle="--", linewidth=0.5)

    # Entry histogram for PC1
    ax = axes[1, 1]
    entries_pc1 = np.sqrt(p) * Vt[0, :]
    ax.hist(entries_pc1, bins=60, density=True, alpha=0.7, edgecolor="black")
    xg = np.linspace(-4, 4, 200)
    ax.plot(xg, stats.norm.pdf(xg), "r-", linewidth=2, label="N(0,1)")
    ax.set_xlabel("√p · v_{i,1}"); ax.set_ylabel("density")
    ax.set_title(f"PC1 entry distribution (kurtosis={stats.kurtosis(entries_pc1):.2f})")
    ax.legend(); ax.grid(True, linestyle="--", linewidth=0.5)

    # Leverage scores
    ax = axes[1, 2]
    lev_sorted = np.sort(leverage)[::-1]
    ax.semilogy(np.arange(1, p + 1), lev_sorted, linewidth=1)
    ax.axhline(K / p, color="r", linestyle="--", label=f"uniform = K/p = {K/p:.4f}")
    ax.set_xlabel("variable rank"); ax.set_ylabel("leverage score")
    ax.set_title("Leverage scores (sorted)")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", linewidth=0.5)

    plt.tight_layout(); plt.show(); plt.close()

    return {
        "coherence": coherence,
        "ideal_coherence": ideal_coherence,
        "coherence_ratio": coherence / ideal_coherence,
        "median_ipr": float(np.median(ipr)),
        "ideal_ipr": ideal_ipr,
        "median_effective_support": float(np.median(effective_support)),
        "median_entropy_dim": float(np.median(effective_dim)),
        "p": p,
        "leverage_cv": float(np.std(leverage) / (np.mean(leverage) + EPS)),
    }


# ===================================================================
# TEST 3: Predicted sigma_K vs measured residual correlation SD
# ===================================================================
# The central prediction of Theorem A.1 ("Crud scale under top-K PC removal"):
# components, the standard deviation of the off-diagonal residual
# correlations is predicted by sigma_K = sqrt(sum_{k>K} lambda_k^2)
# / sum_{k>K} lambda_k.  This test computes both the spectral
# prediction and the empirical SD for several values of K, then
# measures their agreement (scatter plot + Pearson correlation).

def test_sigma_prediction(
    Xz: np.ndarray, ev: np.ndarray, Vt: np.ndarray, name: str,
    ks: List[int] = [0, 1, 2, 5, 10, 20, 50, 100],
    n_pairs: int = 500, seed: int = SEED,
) -> dict:
    """Compare spectral sigma_K prediction to empirical residual correlation SD.

    For each K, computes:
      - predicted sigma_K from the tail eigenvalues (Theorem A.1, "Crud scale")
      - empirical SD of off-diagonal entries of the residual correlation
        matrix after projecting out the top-K PCs
    High correlation between the two validates the theorem.
    """
    n, p = Xz.shape
    rng = np.random.default_rng(seed)
    m = min(p, 500)
    cols = rng.choice(p, m, replace=False) if p > m else np.arange(p)

    predicted = []
    empirical = []
    ks_used = []

    for K in ks:
        if K >= len(ev):
            continue
        ks_used.append(K)

        # Predicted: sigma_K = sqrt(sum_{k>K} lambda_k^2) / sum_{k>K} lambda_k
        tail_ev = ev[K:]
        s1 = float(np.sum(tail_ev))
        s2 = float(np.sum(tail_ev ** 2))
        sigma_pred = np.sqrt(s2) / (s1 + EPS)
        predicted.append(sigma_pred)

        # Empirical: residual correlations after removing top-K PCs
        Xs = Xz[:, cols].copy()
        if K > 0:
            kk = min(K, Vt.shape[0])
            V_k = Vt[:kk].astype(np.float32)
            scores = (Xz @ V_k.T).astype(np.float32)
            Xs = Xs - (scores @ V_k[:, cols]).astype(np.float32)
            del scores
        # Re-standardize
        Xs -= Xs.mean(axis=0, keepdims=True)
        sd = Xs.std(axis=0, ddof=1, keepdims=True)
        sd = np.where(sd < EPS, 1.0, sd)
        Xs = Xs / (sd + EPS)
        # Correlation matrix
        S = (Xs.T @ Xs).astype(np.float64) / max(n - 1, 1)
        np.fill_diagonal(S, np.nan)
        offdiag = S[~np.isnan(S)]
        empirical.append(float(np.std(offdiag)))

    predicted = np.array(predicted)
    empirical = np.array(empirical)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ks_used, predicted, "rs--", markersize=6, label="predicted σ_K (from spectrum)")
    plt.plot(ks_used, empirical, "bo-", markersize=6, label="empirical SD(ρ^(K)_ij)")
    plt.xlabel("K (PCs removed)"); plt.ylabel("σ_K")
    plt.title(f"{name}: predicted vs empirical residual correlation SD")
    plt.legend(); plt.grid(True, linestyle="--", linewidth=0.5)

    plt.subplot(1, 2, 2)
    lo = min(predicted.min(), empirical.min()) * 0.8
    hi = max(predicted.max(), empirical.max()) * 1.2
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=0.5)
    plt.plot(predicted, empirical, "go", markersize=8)
    for i, K in enumerate(ks_used):
        plt.annotate(f"K={K}", (predicted[i], empirical[i]), fontsize=7,
                     xytext=(4, 4), textcoords="offset points")
    plt.xlabel("predicted σ_K"); plt.ylabel("empirical SD")
    plt.title(f"{name}: scatter (predicted vs empirical)")
    plt.grid(True, linestyle="--", linewidth=0.5)

    plt.tight_layout(); plt.show(); plt.close()

    corr = float(np.corrcoef(predicted, empirical)[0, 1]) if len(ks_used) > 2 else float("nan")
    return {
        "ks": ks_used,
        "predicted": predicted.tolist(),
        "empirical": empirical.tolist(),
        "correlation": corr,
        "max_ratio": float(np.max(empirical / (predicted + EPS))),
        "min_ratio": float(np.min(empirical / (predicted + EPS))),
    }


# ===================================================================
# TEST 4: Diagonal concentration
# ===================================================================
# Checks whether the diagonal of the residual covariance Sigma^(K)
# concentrates (i.e. all variables have approximately equal residual
# variance after removing K PCs).  A small coefficient of variation
# (CV) of the per-variable residual variances indicates concentration,
# which is used in the proof of Theorem A.1 ("Crud scale") when normalising to a
# correlation matrix.

def test_diagonal_concentration(
    Xz: np.ndarray, Vt: np.ndarray, name: str,
    ks: List[int] = [0, 1, 5, 10, 50, 100],
) -> dict:
    """Check CV of residual variances across variables after removing K PCs.

    A small CV means the diagonal of Sigma^(K) concentrates -- all
    variables carry roughly equal residual variance, so the residual
    covariance and correlation matrices are close up to a scalar.
    """
    n, p = Xz.shape
    results_cv = {}

    fig, axes = plt.subplots(1, min(len(ks), 6), figsize=(4 * min(len(ks), 6), 4))
    if len(ks) == 1:
        axes = [axes]

    for idx, K in enumerate(ks):
        if K >= min(Vt.shape[0], p):
            continue
        Xr = Xz.copy()
        if K > 0:
            kk = min(K, Vt.shape[0])
            V_k = Vt[:kk].astype(np.float32)
            scores = (Xz @ V_k.T).astype(np.float32)
            Xr = Xr - (scores @ V_k).astype(np.float32)
            del scores

        resid_var = np.var(Xr, axis=0, ddof=1).astype(np.float64)
        cv = float(np.std(resid_var) / (np.mean(resid_var) + EPS))
        results_cv[K] = cv

        if idx < len(axes):
            ax = axes[idx]
            ax.hist(resid_var, bins=60, alpha=0.7, edgecolor="black")
            ax.axvline(np.mean(resid_var), color="r", linestyle="--")
            ax.set_title(f"K={K}, CV={cv:.3f}")
            ax.set_xlabel("residual variance")
            ax.set_ylabel("count")

    plt.suptitle(f"{name}: residual variance distribution across variables", fontsize=12)
    plt.tight_layout(); plt.show(); plt.close()

    return results_cv


# ===================================================================
# TEST 5: Normality of residual correlations
# ===================================================================
# Tests Theorem A.2 ("Association-only decisions"): the off-diagonal entries of the residual
# correlation matrix (after removing K PCs) should be approximately
# Gaussian.  Diagnostics: QQ-plot against N(0,1), skewness (should
# be near 0), excess kurtosis (should be near 0), and tail quantile
# comparison (empirical vs Gaussian at 95/99/99.9-th percentiles).

def test_normality(
    Xz: np.ndarray, Vt: np.ndarray, name: str,
    ks: List[int] = [0, 10, 50],
    max_cols: int = 500, seed: int = SEED,
) -> dict:
    """QQ plot, tail quantiles, skewness, kurtosis of off-diagonal residual correlations."""
    n, p = Xz.shape
    rng = np.random.default_rng(seed)
    m = min(p, max_cols)
    cols = rng.choice(p, m, replace=False) if p > m else np.arange(p)

    results = {}

    fig, axes = plt.subplots(2, len(ks), figsize=(6 * len(ks), 10))
    if len(ks) == 1:
        axes = axes.reshape(-1, 1)

    for idx, K in enumerate(ks):
        if K >= min(Vt.shape[0], p):
            continue

        Xs = Xz[:, cols].copy()
        if K > 0:
            kk = min(K, Vt.shape[0])
            V_k = Vt[:kk].astype(np.float32)
            scores = (Xz @ V_k.T).astype(np.float32)
            Xs = Xs - (scores @ V_k[:, cols]).astype(np.float32)
            del scores
        Xs -= Xs.mean(axis=0, keepdims=True)
        sd = Xs.std(axis=0, ddof=1, keepdims=True)
        sd = np.where(sd < EPS, 1.0, sd)
        Xs /= (sd + EPS)

        S = (Xs.T @ Xs).astype(np.float64) / max(n - 1, 1)
        iu = np.triu_indices_from(S, k=1)
        vals = S[iu]
        vals = vals[np.isfinite(vals)]

        skew = float(stats.skew(vals))
        kurt = float(stats.kurtosis(vals))
        sd_vals = float(np.std(vals))

        # Tail comparison
        empirical_quantiles = {}
        for q in [95, 99, 99.9]:
            emp_q = float(np.percentile(np.abs(vals), q))
            norm_q = float(stats.norm.ppf((1 + q / 100) / 2) * sd_vals)
            empirical_quantiles[q] = (emp_q, norm_q)

        results[K] = {
            "skewness": skew, "excess_kurtosis": kurt,
            "sd": sd_vals, "tail_quantiles": empirical_quantiles,
        }

        # QQ plot
        ax = axes[0, idx]
        qq = stats.probplot(vals / (sd_vals + EPS), dist="norm")
        ax.plot(qq[0][0], qq[0][1], ".", markersize=0.5, alpha=0.3)
        lims = [min(qq[0][0].min(), qq[0][1].min()), max(qq[0][0].max(), qq[0][1].max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_title(f"K={K}: QQ (skew={skew:.3f}, kurt={kurt:.3f})")
        ax.set_xlabel("theoretical"); ax.set_ylabel("sample")
        ax.grid(True, linestyle="--", linewidth=0.5)

        # Histogram with normal overlay
        ax = axes[1, idx]
        ax.hist(vals, bins=80, density=True, alpha=0.7, edgecolor="black")
        xg = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(xg, stats.norm.pdf(xg, loc=np.mean(vals), scale=sd_vals),
                "r-", linewidth=2, label="N(μ,σ²)")
        ax.set_xlabel("ρ^(K)_ij"); ax.set_ylabel("density")
        ax.set_title(f"K={K}: |T| quantiles 95%: {empirical_quantiles[95][0]:.4f} vs {empirical_quantiles[95][1]:.4f}")
        ax.legend(fontsize=8)

    plt.suptitle(f"{name}: normality of residual correlations", fontsize=13)
    plt.tight_layout(); plt.show(); plt.close()

    return results


# ===================================================================
# TEST 6: Cross-component covariance
# ===================================================================
# Checks whether the products v_{ik}*v_{jk} are approximately
# uncorrelated across components k for random variable pairs (i,j).
# This is needed in the variance calculation of the Theorem A.1 ("Crud scale")
# proof: if these products were correlated, the variance formula
# for the sum over tail eigenvalues would need a correction term.
# A small mean autocorrelation confirms the assumption.

def test_cross_component_cov(
    Xz: np.ndarray, Vt: np.ndarray, name: str,
    max_k: int = 50, n_pairs: int = 2000, seed: int = SEED,
) -> dict:
    """Estimate correlation of v_{ik}*v_{jk} across k for random pairs (i,j).

    For each sampled variable pair, computes the lag-1 autocorrelation
    of the product sequence (v_{ik}*v_{jk})_{k=1..K}.  If eigenvectors
    are Haar-like, these products should be approximately uncorrelated
    across k, yielding autocorrelations clustered near zero.
    """
    n, p = Xz.shape
    K = min(max_k, Vt.shape[0])
    Vt_use = Vt[:K].astype(np.float64)
    rng = np.random.default_rng(seed)

    # Sample random variable pairs
    pairs_i = rng.choice(p, n_pairs)
    pairs_j = rng.choice(p, n_pairs)
    # Avoid same-variable pairs
    mask = pairs_i != pairs_j
    pairs_i = pairs_i[mask]
    pairs_j = pairs_j[mask]

    # For each pair, compute products v_ik * v_jk across k
    # Then compute autocorrelation of these products across k
    cross_corrs = []
    for ii, jj in zip(pairs_i[:min(1000, len(pairs_i))],
                      pairs_j[:min(1000, len(pairs_j))]):
        products = Vt_use[:, ii] * Vt_use[:, jj]  # length K
        if products.std() < EPS:
            continue
        # Correlation between adjacent component products
        if len(products) > 2:
            r = np.corrcoef(products[:-1], products[1:])[0, 1]
            if np.isfinite(r):
                cross_corrs.append(r)

    cross_corrs = np.array(cross_corrs)

    plt.figure(figsize=(8, 4))
    plt.hist(cross_corrs, bins=60, alpha=0.7, edgecolor="black", density=True)
    plt.axvline(0, color="r", linestyle="--")
    plt.xlabel("corr(v_ik·v_jk, v_i(k+1)·v_j(k+1)) across k")
    plt.ylabel("density")
    plt.title(f"{name}: cross-component covariance (mean={np.mean(cross_corrs):.4f}, "
              f"|mean|={np.abs(np.mean(cross_corrs)):.4f})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout(); plt.show(); plt.close()

    return {
        "mean_cross_corr": float(np.mean(cross_corrs)),
        "abs_mean": float(np.abs(np.mean(cross_corrs))),
        "std_cross_corr": float(np.std(cross_corrs)),
        "fraction_gt_0.1": float((np.abs(cross_corrs) > 0.1).mean()),
    }


# ===================================================================
# TEST 7: Split-half PC stability
# ===================================================================
# Checks that the principal components and sigma_K estimates are
# reproducible across independent halves of the data.  If the
# eigenvalue spectrum and eigenvectors are stable (small subspace
# angles, similar sigma_K), the theoretical predictions are not
# artifacts of overfitting to a particular sample split.

def test_split_stability(
    Xz: np.ndarray, name: str,
    ks: List[int] = [1, 5, 10, 20, 50],
    seed: int = SEED,
) -> dict:
    """Split rows in half, compute PCs in each, measure subspace angles and sigma_K stability.

    Subspace overlap is measured via the singular values of V_a^T V_b
    (principal angles).  sigma_K stability is the relative difference
    between the half-A and half-B sigma_K estimates, normalised by
    the full-sample estimate.
    """
    n, p = Xz.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    half = n // 2
    Xz_a = Xz[perm[:half]]
    Xz_b = Xz[perm[half:2 * half]]

    max_k = max(ks)
    k_fit = min(max_k, half, p)
    evr_a, Vt_a, ev_a = pca(Xz_a, k_fit, seed=seed)
    evr_b, Vt_b, ev_b = pca(Xz_b, k_fit, seed=seed + 1)
    evr_full, Vt_full, ev_full = pca(Xz, k_fit, seed=seed + 2)

    subspace_overlaps = {}
    sigma_stability = {}

    for K in ks:
        if K > k_fit:
            continue

        # Principal angles between top-K subspaces
        Va = Vt_a[:K].T  # (p, K)
        Vb = Vt_b[:K].T  # (p, K)
        M = Va.T @ Vb
        svals = np.linalg.svd(M, compute_uv=False)
        svals = np.clip(svals, 0, 1)
        angles_deg = np.arccos(svals) * 180 / np.pi
        subspace_overlaps[K] = {
            "mean_angle_deg": float(np.mean(angles_deg)),
            "max_angle_deg": float(np.max(angles_deg)),
            "mean_cosine": float(np.mean(svals)),
        }

        # sigma_K from each half
        tail_a = ev_a[K:]; tail_b = ev_b[K:]
        sig_a = np.sqrt(np.sum(tail_a ** 2)) / (np.sum(tail_a) + EPS)
        sig_b = np.sqrt(np.sum(tail_b ** 2)) / (np.sum(tail_b) + EPS)
        sig_full = np.sqrt(np.sum(ev_full[K:] ** 2)) / (np.sum(ev_full[K:]) + EPS)
        sigma_stability[K] = {
            "sigma_half_a": float(sig_a),
            "sigma_half_b": float(sig_b),
            "sigma_full": float(sig_full),
            "relative_diff": float(abs(sig_a - sig_b) / (sig_full + EPS)),
        }

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Spectrum comparison
    ax = axes[0]
    x = np.arange(1, k_fit + 1)
    ax.loglog(x, evr_a[:k_fit], "b-", alpha=0.7, label="half A")
    ax.loglog(x, evr_b[:k_fit], "r-", alpha=0.7, label="half B")
    ax.loglog(x, evr_full[:k_fit], "k--", alpha=0.9, label="full")
    ax.set_xlabel("k"); ax.set_ylabel("explained variance ratio")
    ax.set_title(f"{name}: split-half spectra")
    ax.legend(); ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Subspace angles
    ax = axes[1]
    ks_plot = sorted(subspace_overlaps.keys())
    mean_cos = [subspace_overlaps[K]["mean_cosine"] for K in ks_plot]
    ax.plot(ks_plot, mean_cos, "go-", markersize=6)
    ax.set_xlabel("K (subspace dimension)"); ax.set_ylabel("mean cosine similarity")
    ax.set_title("Split-half subspace overlap")
    ax.set_ylim(0, 1.05); ax.grid(True, linestyle="--", linewidth=0.5)

    # sigma_K stability
    ax = axes[2]
    sig_a_list = [sigma_stability[K]["sigma_half_a"] for K in ks_plot]
    sig_b_list = [sigma_stability[K]["sigma_half_b"] for K in ks_plot]
    sig_f_list = [sigma_stability[K]["sigma_full"] for K in ks_plot]
    ax.plot(ks_plot, sig_a_list, "b^--", markersize=6, label="half A")
    ax.plot(ks_plot, sig_b_list, "rv--", markersize=6, label="half B")
    ax.plot(ks_plot, sig_f_list, "ks-", markersize=6, label="full")
    ax.set_xlabel("K"); ax.set_ylabel("σ_K")
    ax.set_title("Split-half σ_K stability")
    ax.legend(); ax.grid(True, linestyle="--", linewidth=0.5)

    plt.suptitle(f"{name}: split-half PC stability", fontsize=13)
    plt.tight_layout(); plt.show(); plt.close()

    return {"subspace_overlaps": subspace_overlaps, "sigma_stability": sigma_stability}


# ===================================================================
# Main
# ===================================================================

def main():
    datasets = sorted([p.stem for p in CACHE_DIR.glob("*.npy")])
    print(f"Found cached datasets: {datasets}")

    all_tables = []  # for HTML

    # Collect summary rows
    powerlaw_rows = []
    deloc_rows = []
    sigma_rows = []
    diagconc_rows = []
    normality_rows = []
    crosscomp_rows = []
    stability_rows = []

    for name in datasets:
        print(f"\n{'='*60}")
        print(f"  DATASET: {name}")
        print(f"{'='*60}")

        X = load_and_subsample(name)
        Xz = zscore_columns(X)
        n, p = Xz.shape
        # Fit up to 300 principal components, matching the paper's claim of
        # "up to 300 components" for spectrum characterisation.  Capped by n
        # and p for small datasets.
        k_fit = min(300, n, p)
        evr, Vt, ev = pca(Xz, k_fit, seed=SEED)

        print(f"  shape: {X.shape} -> z-scored: {Xz.shape}, fitting {k_fit} PCs")

        # Test 1
        print(f"\n  [1] Power-law decay")
        pl = test_powerlaw(evr, name)
        print(f"      α={pl['alpha']:.3f}, R²={pl['r2']:.3f}, AIC(PL)={pl['aic_powerlaw']:.0f}, "
              f"AIC(exp)={pl['aic_exponential']:.0f}, preferred={'power-law' if pl['powerlaw_preferred'] else 'exponential'}")
        powerlaw_rows.append([name, f"{pl['alpha']:.3f}", f"{pl['r2']:.3f}",
                              f"{pl['aic_powerlaw']:.0f}", f"{pl['aic_exponential']:.0f}",
                              "power-law" if pl["powerlaw_preferred"] else "exponential"])

        # Test 2
        print(f"\n  [2] Eigenvector delocalization")
        dl = test_delocalization(Vt, name)
        print(f"      coherence={dl['coherence']:.4f} (ideal={dl['ideal_coherence']:.4f}, ratio={dl['coherence_ratio']:.1f}x)")
        print(f"      median IPR={dl['median_ipr']:.2e} (ideal={dl['ideal_ipr']:.2e})")
        print(f"      median effective support={dl['median_effective_support']:.0f} / {dl['p']}")
        deloc_rows.append([name, f"{dl['coherence']:.4f}", f"{dl['ideal_coherence']:.4f}",
                           f"{dl['coherence_ratio']:.1f}x",
                           f"{dl['median_effective_support']:.0f}", str(dl["p"]),
                           f"{dl['leverage_cv']:.3f}"])

        # Test 3
        print(f"\n  [3] Predicted σ_K vs empirical")
        sig = test_sigma_prediction(Xz, ev, Vt, name)
        print(f"      correlation: {sig['correlation']:.4f}")
        sigma_rows.append([name, f"{sig['correlation']:.4f}",
                           f"{sig['min_ratio']:.2f}", f"{sig['max_ratio']:.2f}"])

        # Test 4
        print(f"\n  [4] Diagonal concentration")
        dc = test_diagonal_concentration(Xz, Vt, name)
        for K, cv in dc.items():
            print(f"      K={K}: CV of residual variance = {cv:.4f}")
        dc_vals = [f"{dc.get(K, float('nan')):.3f}" for K in [0, 1, 10, 50, 100]]
        diagconc_rows.append([name] + dc_vals)

        # Test 5
        print(f"\n  [5] Normality of residual correlations")
        norm = test_normality(Xz, Vt, name)
        for K, r in norm.items():
            print(f"      K={K}: skew={r['skewness']:.4f}, excess_kurt={r['excess_kurtosis']:.4f}")
        norm0 = norm.get(0, norm.get(min(norm.keys()), {}))
        norm10 = norm.get(10, {})
        normality_rows.append([name,
                               f"{norm0.get('skewness', float('nan')):.3f}",
                               f"{norm0.get('excess_kurtosis', float('nan')):.3f}",
                               f"{norm10.get('skewness', float('nan')):.3f}",
                               f"{norm10.get('excess_kurtosis', float('nan')):.3f}"])

        # Test 6
        print(f"\n  [6] Cross-component covariance")
        cc = test_cross_component_cov(Xz, Vt, name)
        print(f"      mean cross-corr: {cc['mean_cross_corr']:.4f}, |mean|: {cc['abs_mean']:.4f}")
        crosscomp_rows.append([name, f"{cc['mean_cross_corr']:.4f}",
                               f"{cc['abs_mean']:.4f}", f"{cc['fraction_gt_0.1']:.3f}"])

        # Test 7
        print(f"\n  [7] Split-half PC stability")
        stab = test_split_stability(Xz, name)
        for K, s in stab["subspace_overlaps"].items():
            print(f"      K={K}: mean cosine={s['mean_cosine']:.4f}, "
                  f"σ_K diff={stab['sigma_stability'][K]['relative_diff']:.4f}")
        stab5 = stab["subspace_overlaps"].get(5, {})
        stab50 = stab["subspace_overlaps"].get(50, {})
        sig5 = stab["sigma_stability"].get(5, {})
        sig50 = stab["sigma_stability"].get(50, {})
        stability_rows.append([name,
                               f"{stab5.get('mean_cosine', float('nan')):.3f}",
                               f"{stab50.get('mean_cosine', float('nan')):.3f}",
                               f"{sig5.get('relative_diff', float('nan')):.4f}",
                               f"{sig50.get('relative_diff', float('nan')):.4f}"])

        del X, Xz, Vt, ev, evr
        gc.collect()

    # Build tables
    all_tables = [
        {
            "title": "Test 1: Power-law eigenvalue decay (fit on first 40 PCs)",
            "headers": ["Dataset", "\u03b1", "R\u00b2", "AIC(power-law)", "AIC(exponential)", "Preferred"],
            "rows": powerlaw_rows,
        },
        {
            "title": "Test 2: Eigenvector delocalization",
            "headers": ["Dataset", "Coherence", "Ideal (1/\u221ap)", "Ratio",
                         "Med. eff. support", "p", "Leverage CV"],
            "rows": deloc_rows,
        },
        {
            "title": "Test 3: Predicted \u03c3_K vs empirical SD correlation",
            "headers": ["Dataset", "Correlation", "Min ratio", "Max ratio"],
            "rows": sigma_rows,
        },
        {
            "title": "Test 4: Diagonal concentration (CV of residual variance)",
            "headers": ["Dataset", "K=0", "K=1", "K=10", "K=50", "K=100"],
            "rows": diagconc_rows,
        },
        {
            "title": "Test 5: Normality of residual correlations",
            "headers": ["Dataset", "Skew(K=0)", "Kurt(K=0)", "Skew(K=10)", "Kurt(K=10)"],
            "rows": normality_rows,
        },
        {
            "title": "Test 6: Cross-component covariance negligibility",
            "headers": ["Dataset", "Mean cross-corr", "|Mean|", "Frac |r|>0.1"],
            "rows": crosscomp_rows,
        },
        {
            "title": "Test 7: Split-half PC stability",
            "headers": ["Dataset", "Cosine(K=5)", "Cosine(K=50)",
                         "\u0394\u03c3(K=5)", "\u0394\u03c3(K=50)"],
            "rows": stability_rows,
        },
    ]

    # Print all tables
    for t in all_tables:
        print(f"\n{t['title']}")
        print("  ".join(h.rjust(16) for h in t["headers"]))
        print("-" * (18 * len(t["headers"])))
        for row in t["rows"]:
            print("  ".join(c.rjust(16) for c in row))

    # Build HTML
    build_html(all_tables)


def build_html(tables: List[dict], fig_dir: Path = FIG_DIR,
               out_path: str = "assumption_audit.html") -> None:
    import base64
    figs = sorted(fig_dir.glob(f"*.{FIG_FORMAT}"))

    parts = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        "<title>Assumption Audit</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;background:#1a1a1a;color:#eee;"
        "max-width:1400px;margin:0 auto;padding:20px}",
        "h1{text-align:center}",
        "h2{margin-top:40px;border-bottom:1px solid #555;padding-bottom:6px}",
        ".fig{background:#fff;border-radius:8px;margin:24px 0;padding:12px;text-align:center}",
        ".fig img{max-width:100%;height:auto}",
        ".fig p{color:#333;font-size:14px;margin:8px 0 0}",
        "table{border-collapse:collapse;margin:20px auto;font-size:13px}",
        "th,td{border:1px solid #555;padding:6px 12px;text-align:right}",
        "th{background:#333;color:#eee}",
        "td{background:#222}",
        "td:first-child,th:first-child{text-align:left}",
        "</style></head><body>",
        "<h1>Assumption Audit for Theorems A.1 &amp; A.2</h1>",
    ]

    if figs:
        parts.append("<h2>Diagnostic Figures</h2>")
        for fig_path in figs:
            data = base64.b64encode(fig_path.read_bytes()).decode("ascii")
            mime = "image/png" if FIG_FORMAT == "png" else f"image/{FIG_FORMAT}"
            parts.append(f'<div class="fig">')
            parts.append(f'<img src="data:{mime};base64,{data}"/>')
            parts.append(f'<p>{fig_path.name}</p>')
            parts.append(f'</div>')

    if tables:
        parts.append("<h2>Summary Tables</h2>")
        for tbl in tables:
            parts.append(f"<h3>{tbl['title']}</h3>")
            parts.append("<table>")
            parts.append("<tr>" + "".join(f"<th>{h}</th>" for h in tbl["headers"]) + "</tr>")
            for row in tbl["rows"]:
                parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
            parts.append("</table>")

    parts.append("</body></html>")
    Path(out_path).write_text("\n".join(parts))
    n_figs = len(figs)
    n_tables = len(tables)
    print(f"\nWrote {out_path} with {n_figs} figures and {n_tables} tables.")


if __name__ == "__main__":
    main()
