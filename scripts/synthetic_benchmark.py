"""Synthetic benchmark: crud-aware test vs. classical, Bonferroni, BH-FDR.

Data-generating process:
  - p = 200 features
  - L = 10 latent factors, each with sparse loadings on ~40 features
  - E = 50 randomly placed direct edges, each pair (i,j) shares an extra
    Gaussian source contributing to both i and j
  - n = 2000 samples, multivariate Gaussian

Truth: the E direct-edge pairs. All other pairs are "background" (correlated
through shared latent factors but not by a direct mechanism).

Methods compared:
  1. Classical:   |r| > 1.96 / sqrt(n - 1)
  2. Bonferroni:  classical at alpha / M, M = p(p-1)/2
  3. BH-FDR:      sort p-values, threshold at i*q/M
  4. Crud-aware empirical (K=10): |r_resid| > 95th percentile of |r_resid|
                                   over off-diagonal pairs
  5. Crud-aware empirical (K=10), 99th percentile

Metrics: number of direct edges recovered (TP), false discoveries (FP),
precision = TP/(TP+FP), recall = TP/(TP+FN), F1.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


P = 200
L = 10
E = 50           # number of direct edges
N = 2000
LOADING_NONZERO = 40   # how many features each latent factor loads on
LATENT_LOADING_SD = 0.40
DIRECT_STRENGTH = 0.40
NOISE_SD = 1.0
K_ADJUST = 10
ALPHA = 0.05
RNG = np.random.default_rng(20260509)


def generate_data():
    # Latent factor loadings
    A = np.zeros((L, P))
    for l in range(L):
        idx = RNG.choice(P, size=LOADING_NONZERO, replace=False)
        A[l, idx] = RNG.normal(0, LATENT_LOADING_SD, size=LOADING_NONZERO)

    # Pick direct edges (unordered pairs i<j)
    pair_indices = []
    seen = set()
    while len(pair_indices) < E:
        i = int(RNG.integers(0, P))
        j = int(RNG.integers(0, P))
        if i == j:
            continue
        a, b = min(i, j), max(i, j)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        pair_indices.append((a, b))

    # Sample latent factors and direct-edge sources
    Y_latent = RNG.normal(0, 1, size=(N, L))
    direct_sources = RNG.normal(0, DIRECT_STRENGTH, size=(N, E))
    noise = RNG.normal(0, NOISE_SD, size=(N, P))

    X = noise + Y_latent @ A
    for k, (i, j) in enumerate(pair_indices):
        X[:, i] += direct_sources[:, k]
        X[:, j] += direct_sources[:, k]

    # Truth mask: 1 if (i,j) is a direct edge, 0 otherwise
    truth = np.zeros((P, P), dtype=bool)
    for i, j in pair_indices:
        truth[i, j] = True
        truth[j, i] = True
    return X, truth, pair_indices


def zscore(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def residualize_topk(Xz, k):
    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    Vk = Vt[:k]
    return Xz - (Xz @ Vk.T) @ Vk


def upper_pairs(M):
    iu = np.triu_indices(M.shape[0], k=1)
    return iu, M[iu]


def evaluate(flagged, truth_pairs, total_pairs):
    """flagged: boolean mask over pairs (length M); truth_pairs: indices (length M)."""
    tp = int(np.sum(flagged & truth_pairs))
    fp = int(np.sum(flagged & (~truth_pairs)))
    fn = int(np.sum((~flagged) & truth_pairs))
    flagged_count = int(np.sum(flagged))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "flagged": flagged_count,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


def bh_fdr(pvalues, q):
    M = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    crit = (np.arange(1, M + 1) * q / M)
    below = sorted_p <= crit
    if not np.any(below):
        return np.zeros(M, dtype=bool)
    threshold = sorted_p[np.where(below)[0].max()]
    return pvalues <= threshold


def efron_empirical_null(zvals):
    """Efron-style empirical null. Estimate (mu0, sigma0) robustly so the
    fit is not pulled by heavy tails / non-null signal:
        mu0 = median(z)
        sigma0 = IQR(z) / 1.349   (robust SD; 1.349 is the IQR-to-SD factor
                                    for a Gaussian)
    Then return the recalibrated two-sided p-values under N(mu0, sigma0).
    This captures Efron's central-matching idea: the bulk of z is mostly
    null, and the IQR of the bulk gives a scale that resists outliers.
    """
    mu0 = float(np.median(zvals))
    q25, q75 = np.percentile(zvals, [25, 75])
    iqr = float(q75 - q25)
    sigma0 = iqr / 1.349
    z_recal = (zvals - mu0) / sigma0
    p_recal = 2 * norm.sf(np.abs(z_recal))
    return p_recal, mu0, sigma0


def main():
    X, truth, edge_list = generate_data()
    print(f"DGP: p={P}, n={N}, L={L} latent factors, E={E} direct edges")
    print(f"Total off-diagonal pairs: {P*(P-1)//2}\n")

    Xz = zscore(X)

    # Classical: pairwise correlation, full data
    R_full = np.corrcoef(Xz.T)
    iu, r_full = upper_pairs(R_full)
    truth_pairs = upper_pairs(truth)[1]

    # Residualized correlation at K=10
    Xr = residualize_topk(Xz, K_ADJUST)
    R_resid = np.corrcoef(Xr.T)
    _, r_resid = upper_pairs(R_resid)

    # 1. Classical |r| > 1.96 / sqrt(n-1)
    z_crit = norm.ppf(1 - ALPHA / 2)
    classical_thresh = z_crit / np.sqrt(N - 1)
    flagged_classical = np.abs(r_full) > classical_thresh

    # 2. Bonferroni
    M = len(r_full)
    bonf_thresh = norm.ppf(1 - ALPHA / (2 * M)) / np.sqrt(N - 1)
    flagged_bonf = np.abs(r_full) > bonf_thresh

    # 3. BH-FDR at q = 0.05 against the theoretical N(0,1) null
    z_classical = r_full * np.sqrt(N - 1)
    p_classical = 2 * norm.sf(np.abs(z_classical))
    flagged_bh = bh_fdr(p_classical, q=ALPHA)

    # 3b. Efron empirical-null + BH-FDR
    p_efron, mu0, sigma0 = efron_empirical_null(z_classical)
    flagged_efron_bh = bh_fdr(p_efron, q=ALPHA)
    print(f"Efron empirical null: mu0={mu0:.3f}, sigma0={sigma0:.3f}")

    # 4. Crud-aware empirical at K=10, 95th percentile
    sigma_K = float(r_resid.std(ddof=0))
    print(f"Empirical sigma_K (K={K_ADJUST}) = {sigma_K:.4f}")
    crud95 = np.percentile(np.abs(r_resid), 95)
    crud99 = np.percentile(np.abs(r_resid), 99)
    flagged_crud95 = np.abs(r_resid) > crud95
    flagged_crud99 = np.abs(r_resid) > crud99

    # Also: crud-aware applied to RAW (K=0) correlations using residual sigma_K?
    # The "applied test" version: flag the (i,j) target pair using r_full[i,j]
    # against the K=0 empirical null (raw correlations):
    sigma_0 = float(r_full.std(ddof=0))
    raw95 = np.percentile(np.abs(r_full), 95)
    flagged_crud_raw95 = np.abs(r_full) > raw95

    methods = [
        ("Classical $|r|>1.96/\\sqrt{n-1}$",        flagged_classical),
        ("Bonferroni",                              flagged_bonf),
        (f"BH-FDR (theoretical null, $q={ALPHA}$)", flagged_bh),
        (f"Efron empirical null + BH ($q={ALPHA}$)",flagged_efron_bh),
        (f"Crud-aware, 95th pct (K=0)",             flagged_crud_raw95),
        (f"Crud-aware, 95th pct (K={K_ADJUST})",    flagged_crud95),
        (f"Crud-aware, 99th pct (K={K_ADJUST})",    flagged_crud99),
    ]

    print(f"\n{'Method':<35} {'Flagged':>8} {'TP':>5} {'FP':>5} "
          f"{'Prec':>6} {'Recall':>7} {'F1':>6}")
    print("-" * 80)
    rows = []
    for name, flag in methods:
        m = evaluate(flag, truth_pairs, total_pairs=M)
        rows.append((name, m))
        print(f"{name:<35} {m['flagged']:>8} {m['TP']:>5} {m['FP']:>5} "
              f"{m['precision']:>6.3f} {m['recall']:>7.3f} {m['F1']:>6.3f}")

    print(f"\nPrior P(direct edge) = E / M = {E}/{M} = {E/M:.4f}")
    print(f"Empirical sigma_K (K=0)  = {sigma_0:.4f}")
    print(f"Empirical sigma_K (K={K_ADJUST}) = {sigma_K:.4f}")

    return rows


if __name__ == "__main__":
    main()
