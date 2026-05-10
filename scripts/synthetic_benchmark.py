"""Synthetic benchmark: crud-aware test vs. classical, Bonferroni,
BH-FDR, BY-FDR (dependence-aware), and Efron empirical null.

Data-generating process per replicate:
  - p = 200 features
  - L = 10 latent factors, each with sparse loadings on ~40 features
  - E = 50 randomly placed direct edges, each pair (i,j) shares an extra
    Gaussian source contributing to both i and j
  - n = 2000 samples, multivariate Gaussian

Truth: the E direct-edge pairs.

We run N_SEEDS replicates and report mean +/- SE for each metric.

Methods:
  1. Classical:                 |r| > 1.96 / sqrt(n - 1)
  2. Bonferroni:                classical at alpha / M, M = p(p-1)/2
  3. BH-FDR (theoretical null): standard BH at q
  4. BY-FDR (dependent BH):     BH at q' = q / sum_{i=1}^M 1/i
  5. Efron empirical null + BH: recalibrate p-values under N(mu0, sigma0)
                                where mu0=median(z), sigma0=IQR(z)/1.349
  6. Crud-aware K=0, 95th pct
  7. Crud-aware K=K_ADJUST, 95th pct
  8. Crud-aware K=K_ADJUST, 99th pct

Metrics: TP, FP, precision, recall, F1.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


P = 200
L = 10
E = 50
N = 2000
LOADING_NONZERO = 40
LATENT_LOADING_SD = 0.40
DIRECT_STRENGTH = 0.40
NOISE_SD = 1.0
K_ADJUST = 10
ALPHA = 0.05
N_SEEDS = 50
BASE_SEED = 20260510


def generate_data(rng):
    A = np.zeros((L, P))
    for l in range(L):
        idx = rng.choice(P, size=LOADING_NONZERO, replace=False)
        A[l, idx] = rng.normal(0, LATENT_LOADING_SD, size=LOADING_NONZERO)

    pair_indices = []
    seen = set()
    while len(pair_indices) < E:
        i = int(rng.integers(0, P))
        j = int(rng.integers(0, P))
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        pair_indices.append((a, b))

    Y_latent = rng.normal(0, 1, size=(N, L))
    direct_sources = rng.normal(0, DIRECT_STRENGTH, size=(N, E))
    noise = rng.normal(0, NOISE_SD, size=(N, P))

    X = noise + Y_latent @ A
    for k, (i, j) in enumerate(pair_indices):
        X[:, i] += direct_sources[:, k]
        X[:, j] += direct_sources[:, k]

    truth = np.zeros((P, P), dtype=bool)
    for i, j in pair_indices:
        truth[i, j] = True
        truth[j, i] = True
    return X, truth


def zscore(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def residualize_topk(Xz, k):
    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    Vk = Vt[:k]
    return Xz - (Xz @ Vk.T) @ Vk


def upper_pairs(M):
    iu = np.triu_indices(M.shape[0], k=1)
    return iu, M[iu]


def evaluate(flagged, truth_pairs):
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
    crit = np.arange(1, M + 1) * q / M
    below = sorted_p <= crit
    if not np.any(below):
        return np.zeros(M, dtype=bool)
    threshold = sorted_p[np.where(below)[0].max()]
    return pvalues <= threshold


def by_fdr(pvalues, q):
    """Benjamini-Yekutieli, BH at adjusted level q' = q / sum_{i=1}^M 1/i."""
    M = len(pvalues)
    harmonic = float(np.sum(1.0 / np.arange(1, M + 1)))
    return bh_fdr(pvalues, q / harmonic)


def efron_empirical_null(zvals):
    mu0 = float(np.median(zvals))
    q25, q75 = np.percentile(zvals, [25, 75])
    sigma0 = float(q75 - q25) / 1.349
    z_recal = (zvals - mu0) / sigma0
    p_recal = 2 * norm.sf(np.abs(z_recal))
    return p_recal, mu0, sigma0


def run_one_seed(seed):
    rng = np.random.default_rng(seed)
    X, truth = generate_data(rng)
    Xz = zscore(X)
    R_full = np.corrcoef(Xz.T)
    iu, r_full = upper_pairs(R_full)
    truth_pairs = upper_pairs(truth)[1]

    Xr = residualize_topk(Xz, K_ADJUST)
    R_resid = np.corrcoef(Xr.T)
    _, r_resid = upper_pairs(R_resid)

    z_crit = norm.ppf(1 - ALPHA / 2)
    flagged_classical = np.abs(r_full) > z_crit / np.sqrt(N - 1)

    M = len(r_full)
    bonf_thresh = norm.ppf(1 - ALPHA / (2 * M)) / np.sqrt(N - 1)
    flagged_bonf = np.abs(r_full) > bonf_thresh

    z_classical = r_full * np.sqrt(N - 1)
    p_classical = 2 * norm.sf(np.abs(z_classical))
    flagged_bh = bh_fdr(p_classical, q=ALPHA)
    flagged_by = by_fdr(p_classical, q=ALPHA)

    p_efron, mu0, sigma0 = efron_empirical_null(z_classical)
    flagged_efron = bh_fdr(p_efron, q=ALPHA)

    crud95_raw = np.percentile(np.abs(r_full), 95)
    flagged_crud_raw = np.abs(r_full) > crud95_raw

    crud95 = np.percentile(np.abs(r_resid), 95)
    crud99 = np.percentile(np.abs(r_resid), 99)
    flagged_crud95 = np.abs(r_resid) > crud95
    flagged_crud99 = np.abs(r_resid) > crud99

    sigma_K = float(r_resid.std(ddof=0))
    sigma_0 = float(r_full.std(ddof=0))

    methods = [
        ("Classical |r|>1.96/sqrt(n-1)", flagged_classical),
        ("Bonferroni",                    flagged_bonf),
        ("BH-FDR (theoretical null)",     flagged_bh),
        ("BY-FDR (dependent BH)",         flagged_by),
        ("Efron empirical null + BH",     flagged_efron),
        ("Crud-aware K=0, 95%ile",        flagged_crud_raw),
        ("Crud-aware K=10, 95%ile",       flagged_crud95),
        ("Crud-aware K=10, 99%ile",       flagged_crud99),
    ]

    results = {name: evaluate(flag, truth_pairs) for name, flag in methods}
    return results, sigma_K, sigma_0, mu0, sigma0


def main():
    print(f"Running {N_SEEDS} seeds, p={P}, n={N}, L={L}, E={E}")
    print(f"Reporting mean +/- SE across seeds.\n")

    all_results = []
    sigmas_K = []
    sigmas_0 = []
    efron_mu0 = []
    efron_sigma0 = []
    for s in range(N_SEEDS):
        seed = BASE_SEED + s
        res, sk, s0, m0, e_sigma = run_one_seed(seed)
        all_results.append(res)
        sigmas_K.append(sk)
        sigmas_0.append(s0)
        efron_mu0.append(m0)
        efron_sigma0.append(e_sigma)

    method_names = list(all_results[0].keys())
    metrics = ["flagged", "TP", "FP", "precision", "recall", "F1"]

    print(f"sigma_K (K=0)  : mean = {np.mean(sigmas_0):.4f}, SE = {np.std(sigmas_0)/np.sqrt(N_SEEDS):.4f}")
    print(f"sigma_K (K=10) : mean = {np.mean(sigmas_K):.4f}, SE = {np.std(sigmas_K)/np.sqrt(N_SEEDS):.4f}")
    print(f"Efron mu0      : mean = {np.mean(efron_mu0):+.3f}, SE = {np.std(efron_mu0)/np.sqrt(N_SEEDS):.3f}")
    print(f"Efron sigma0   : mean = {np.mean(efron_sigma0):.3f}, SE = {np.std(efron_sigma0)/np.sqrt(N_SEEDS):.3f}")
    print()

    header = f"{'Method':<32} " + " ".join(f"{m:>15}" for m in metrics)
    print(header)
    print("-" * len(header))
    for name in method_names:
        vals = {m: np.array([r[name][m] for r in all_results], dtype=float) for m in metrics}
        row = f"{name:<32} "
        for m in metrics:
            arr = vals[m]
            mean = np.nanmean(arr)
            se = np.nanstd(arr) / np.sqrt(N_SEEDS)
            if m in ("flagged", "TP", "FP"):
                row += f"{mean:>7.0f}+-{se:<5.0f} "
            else:
                row += f"{mean:>8.3f}+-{se:<6.3f}"
        print(row)


if __name__ == "__main__":
    main()
