"""Alternative adjustment methods and positive control analysis (paper Appendix).

This module implements the comparison of PCA-based crud adjustment against three
alternative strategies, as described in the paper's Appendix. The goal is to show
that the broad residual correlations (the "crud factor") are not an artifact of
the specific PCA adjustment method. If different adjustment approaches all yield
similar residual correlation spreads, then the phenomenon is a genuine property
of the data's eigenvalue spectrum, not a quirk of PCA.

Alternative methods implemented:
  1. Random covariate adjustment -- regress on randomly chosen covariates
     (adjust_random_covariates).
  2. Random projection adjustment -- project out random Gaussian directions
     (adjust_random_projections).
  3. Lasso-based adjustment -- for each feature, Lasso-select predictors from
     all other features (adjust_lasso).

The module also provides positive control analysis (run_positive_control):
known biologically or psychologically meaningful variable pairs (HEXACO
within-facet pairs, NHANES biomarker groups) should remain distinguishable
from the crud background even after adjustment -- i.e., they should appear
in the upper tail (>95th percentile) of the residual correlation distribution.

Dimensions tested for the alternative methods: {10, 50, 100}.
Feature subset used for correlation measurement: 300 columns.
"""

import gc
import math
import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from crud.config import EPS, SEED, MAX_ROWS_ANALYSIS
from crud.utils import (
    preprocess_for_pca_and_corr, corr_from_zscored, offdiag_vals_from_corr,
    pca_randomized, residual_corr_vals_after_k_pcs,
)
from crud.loaders import load_hexaco_data


def _residual_corr_sd_from_residuals(
    Xresid: np.ndarray, cols: np.ndarray, *, eps: float = EPS,
) -> float:
    """Standardize pre-computed residuals and return SD of off-diagonal correlations.

    This is a shared helper used by all alternative adjustment functions. After
    an adjustment method produces a residual matrix (original data minus the
    adjustment's projection), this function:
      1. Selects the feature subset specified by `cols`.
      2. Re-standardizes each column to zero mean and unit variance (since
         the adjustment may have changed column scales).
      3. Computes the full pairwise correlation matrix from the z-scored residuals.
      4. Extracts off-diagonal correlation values and returns their standard
         deviation -- the key summary statistic measuring residual correlation
         spread (i.e., how much "crud" remains after adjustment).

    Parameters
    ----------
    Xresid : np.ndarray, shape (n_samples, p_features)
        Residual matrix after some adjustment has been applied.
    cols : np.ndarray of int
        Indices of the feature subset to compute correlations over.
    eps : float
        Small constant to avoid division by zero when standardizing.

    Returns
    -------
    float
        Standard deviation of all off-diagonal residual correlations.
    """
    # Select the feature subset and cast to float32 for memory efficiency.
    Xs = Xresid[:, cols].astype(np.float32)
    # Re-center each column (adjustment may shift means).
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    # Compute per-column standard deviation with Bessel's correction.
    sd = Xs.std(axis=0, ddof=1, keepdims=True)
    # Guard against near-zero variance columns (constant after adjustment).
    sd = np.where(sd < eps, 1.0, sd)
    # Z-score the residuals so correlation = simple dot product / n.
    Xs = Xs / (sd + eps)
    # Compute the correlation matrix from z-scored data.
    S = corr_from_zscored(Xs)
    # Extract the upper triangle (off-diagonal) correlation values.
    vals = offdiag_vals_from_corr(S)
    del Xs, S
    return float(np.std(vals))


def adjust_random_covariates(
    Xz: np.ndarray, cols: np.ndarray, m: int, *, seed: int = SEED,
    n_draws: int = 5, eps: float = EPS,
) -> float:
    """Random covariate adjustment: OLS regression on m randomly chosen features.

    This is the "random covariate adjustment" from the paper's Appendix.
    Instead of projecting out the top PCA directions, we regress each feature
    on m randomly selected *other* features and take residuals. If PCA
    adjustment is merely fitting noise, then regressing on arbitrary covariates
    should produce a similar reduction in residual correlation spread.

    Implementation uses QR factorization of the covariate matrix for numerical
    stability (equivalent to OLS but avoids forming X^T X explicitly). The
    projection is: Xresid = X - Q @ Q^T @ X, where Q comes from the thin QR
    decomposition of the covariate matrix.

    Results are averaged over n_draws=5 independent random draws of covariates
    to reduce variance from any single random selection.

    Parameters
    ----------
    Xz : np.ndarray, shape (n, p)
        Z-scored data matrix (all features).
    cols : np.ndarray of int
        Feature subset indices for measuring residual correlation spread.
    m : int
        Number of random covariates to regress on (tested at 10, 50, 100).
    seed : int
        Random seed for reproducibility.
    n_draws : int
        Number of independent random draws to average over (default 5).
    eps : float
        Numerical stability constant.

    Returns
    -------
    float
        Mean SD of off-diagonal residual correlations, averaged over draws.
    """
    n, p = Xz.shape
    rng = np.random.default_rng(seed)
    sds = []
    # Build the set of "other" features not in the correlation subset.
    col_set = set(cols.tolist())
    other = np.array([j for j in range(p) if j not in col_set], dtype=int)
    # Fallback: if fewer non-subset features than m, draw from all features.
    if len(other) < m:
        other = np.arange(p, dtype=int)
    for draw in range(n_draws):
        # Randomly select m covariate indices (without replacement).
        cov_idx = rng.choice(other, size=min(m, len(other)), replace=False)
        Z = Xz[:, cov_idx].astype(np.float32)
        # QR factorization for numerically stable projection (instead of normal equations).
        Q, R = np.linalg.qr(Z, mode="reduced")
        Q = Q.astype(np.float32)
        # Project all features onto the column space of the random covariates.
        proj = Q @ (Q.T @ Xz.astype(np.float32))
        # Residuals = original data minus the projection onto random covariates.
        Xresid = Xz.astype(np.float32) - proj
        sd_val = _residual_corr_sd_from_residuals(Xresid, cols, eps=eps)
        sds.append(sd_val)
        del Z, Q, R, proj, Xresid
    # Average over random draws to smooth out randomness.
    return float(np.mean(sds))


def adjust_random_projections(
    Xz: np.ndarray, cols: np.ndarray, k: int, *, seed: int = SEED,
    n_draws: int = 5, eps: float = EPS,
) -> float:
    """Random projection adjustment: project out k random Gaussian directions.

    Tests whether the *structured* PCA directions matter or whether projecting
    out *any* k-dimensional subspace reduces residual correlations similarly.
    If PCA and random projections produce comparable residual correlation
    spreads, it would suggest that dimensionality reduction alone (not the
    specific eigenstructure) drives the phenomenon. In practice, random
    projections tend to be less effective, supporting the importance of the
    power-law eigenvalue structure.

    Random directions are drawn from a standard Gaussian in p dimensions,
    then orthonormalized via QR decomposition to form an orthonormal basis
    for the random k-dimensional subspace.

    Parameters
    ----------
    Xz : np.ndarray, shape (n, p)
        Z-scored data matrix.
    cols : np.ndarray of int
        Feature subset indices for correlation measurement.
    k : int
        Number of random directions to project out (tested at 10, 50, 100).
    seed : int
        Random seed for reproducibility.
    n_draws : int
        Number of independent random draws to average over (default 5).
    eps : float
        Numerical stability constant.

    Returns
    -------
    float
        Mean SD of off-diagonal residual correlations, averaged over draws.
    """
    n, p = Xz.shape
    rng = np.random.default_rng(seed)
    sds = []
    for draw in range(n_draws):
        # Draw a random p x k Gaussian matrix (each column is a random direction in R^p).
        G = rng.standard_normal((p, k)).astype(np.float32)
        # Orthonormalize via QR to get k orthonormal random directions.
        Q, _ = np.linalg.qr(G, mode="reduced")
        # Project data onto the random subspace: scores = X @ Q, then reconstruct.
        scores = (Xz.astype(np.float32) @ Q)
        proj = scores @ Q.T
        # Residuals = data minus its projection onto the random subspace.
        Xresid = Xz.astype(np.float32) - proj
        sd_val = _residual_corr_sd_from_residuals(Xresid, cols, eps=eps)
        sds.append(sd_val)
        del G, Q, scores, proj, Xresid
    # Average over random draws.
    return float(np.mean(sds))


def adjust_lasso(
    Xz: np.ndarray, cols: np.ndarray, *, seed: int = SEED,
    alpha: float = 0.1, max_iter: int = 500, eps: float = EPS,
) -> float:
    """Lasso-based adjustment: data-driven sparse predictor selection.

    For each target feature in `cols`, fit a Lasso regression using *all other*
    features as potential predictors. The L1 penalty (alpha=0.1) encourages
    sparsity, so only the most predictive covariates are selected for each
    feature. The residuals (target minus Lasso prediction) are then used to
    compute residual correlations.

    Unlike PCA and random projections (which apply a single global projection
    to all features), Lasso selects a *different* set of predictors for each
    target. This makes it a fully data-driven, feature-specific adjustment.
    The paper uses this as the strongest alternative: if even Lasso-selected
    predictors cannot eliminate the residual correlation spread, the crud
    phenomenon is robust.

    Parameters
    ----------
    Xz : np.ndarray, shape (n, p)
        Z-scored data matrix.
    cols : np.ndarray of int
        Feature subset indices to adjust and measure.
    seed : int
        Random seed for Lasso solver reproducibility.
    alpha : float
        L1 regularization strength (default 0.1).
    max_iter : int
        Maximum Lasso solver iterations.
    eps : float
        Numerical stability constant.

    Returns
    -------
    float
        SD of off-diagonal residual correlations after Lasso adjustment.
    """
    # Lazy import to avoid requiring sklearn at module load time.
    from sklearn.linear_model import Lasso
    n, p = Xz.shape
    # Start with a copy; we'll overwrite columns in `cols` with their residuals.
    Xresid = Xz.astype(np.float32).copy()
    lasso = Lasso(alpha=alpha, max_iter=max_iter, random_state=seed,
                  warm_start=False)
    for j in cols:
        # Build a boolean mask selecting all features *except* the current target.
        mask = np.ones(p, dtype=bool)
        mask[j] = False
        # Use float64 for Lasso fitting (sklearn expects it for numerical stability).
        Z = Xz[:, mask].astype(np.float64)
        y = Xz[:, j].astype(np.float64)
        lasso.fit(Z, y)
        yhat = lasso.predict(Z).astype(np.float32)
        # Store the residual: original value minus Lasso-predicted value.
        Xresid[:, j] = Xz[:, j].astype(np.float32) - yhat
    return _residual_corr_sd_from_residuals(Xresid, cols, eps=eps)


def _hexaco_facet_groups(col_names: list) -> Dict[str, List[int]]:
    """Map HEXACO questionnaire items to personality facets by name prefix.

    HEXACO column names follow the pattern "<FacetName><ItemNumber>", e.g.,
    "Honesty1", "Honesty2", "Emotionality1", etc. This function groups items
    by their facet name prefix so we can identify within-facet pairs for
    positive control testing. Items within the same facet should be more
    correlated than the crud background, since they measure the same
    psychological construct.

    Parameters
    ----------
    col_names : list of str
        Column names from the HEXACO dataset.

    Returns
    -------
    dict mapping facet name (str) to list of column indices (int).
    """
    groups = {}
    for i, name in enumerate(col_names):
        # Match pattern: capital letter + letters, followed by digits at end.
        # E.g., "Honesty1" -> group "Honesty", "Emotionality3" -> "Emotionality".
        m = re.match(r'^([A-Z][A-Za-z]+)\d+$', name)
        if m:
            facet = m.group(1)
            groups.setdefault(facet, []).append(i)
    return groups


def _nhanes_biomarker_groups(col_names: list) -> Dict[str, List[int]]:
    """Map NHANES variable names to known biomarker categories.

    NHANES variables are identified by standardized codes (e.g., LBXTC for
    total cholesterol). This function assigns variables to clinically
    meaningful groups based on exact or prefix matching:

      - Lipids: total cholesterol, HDL, triglycerides, LDL
      - Glucose: fasting glucose, glycohemoglobin (HbA1c)
      - Liver: ALT, AST, total bilirubin, albumin, ALP
      - Renal: blood urea nitrogen, creatinine
      - Hematology: RBC count, hemoglobin, hematocrit, MCV
      - BloodPressure: systolic and diastolic readings (3 measurements each)
      - Anthropometry: weight, height, BMI, waist circumference

    Variables within the same biomarker group should be biologically
    correlated (e.g., systolic and diastolic blood pressure). These serve as
    positive controls: their pairwise correlations should remain in the upper
    tail (>95th percentile) of the crud-adjusted distribution.

    Parameters
    ----------
    col_names : list of str
        Column names from the merged NHANES dataset.

    Returns
    -------
    dict mapping group name (str) to list of column indices (int).
        Only groups with >= 2 matched variables are included.
    """
    # Define the biomarker groups and their associated NHANES variable codes.
    group_defs = {
        "Lipids": ["LBXTC", "LBDHDD", "LBXTR", "LBDLDL"],
        "Glucose": ["LBXGLU", "LBXGH"],
        "Liver": ["LBXSATSI", "LBXSASSI", "LBXSTB", "LBXSAL", "LBXSAPSI"],
        "Renal": ["LBXSBU", "LBXSCR"],
        "Hematology": ["LBXRBCSI", "LBXHGB", "LBXHCT", "LBXMCVSI"],
        "BloodPressure": ["BPXSY1", "BPXDI1", "BPXSY2", "BPXDI2", "BPXSY3", "BPXDI3"],
        "Anthropometry": ["BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"],
    }
    # Build a lookup from column name to column index.
    name_to_idx = {n: i for i, n in enumerate(col_names)}
    groups = {}
    for gname, prefixes in group_defs.items():
        idxs = []
        for prefix in prefixes:
            # Match either exact name or prefix (handles suffixed variants).
            for cn, ci in name_to_idx.items():
                if cn == prefix or cn.startswith(prefix):
                    if ci not in idxs:
                        idxs.append(ci)
        # Only keep groups with at least 2 variables (need pairs for correlation).
        if len(idxs) >= 2:
            groups[gname] = idxs
    return groups


def _within_group_pairs(groups: Dict[str, List[int]]) -> List[Tuple[int, int, str]]:
    """Generate all unique (i, j) pairs within each group for positive control testing.

    For each group of related variables, enumerates all unique unordered pairs.
    These within-group pairs represent the "positive controls" -- variable pairs
    that we *expect* to be meaningfully correlated (e.g., two items from the
    same HEXACO personality facet, or systolic and diastolic blood pressure).

    Parameters
    ----------
    groups : dict mapping group name to list of column indices.

    Returns
    -------
    list of (i, j, group_name) tuples, where i < j are column indices and
    group_name identifies which group the pair belongs to.
    """
    pairs = []
    for gname, idxs in groups.items():
        # Enumerate all unique pairs (i, j) with i < j within this group.
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pairs.append((idxs[a], idxs[b], gname))
    return pairs


def run_positive_control() -> List[dict]:
    """Positive control analysis: known meaningful pairs should have high crud percentiles.

    This function validates that our crud adjustment framework does not "wash out"
    genuinely meaningful correlations. Two positive control datasets are used:

      1. **HEXACO**: Within-facet item pairs from a personality questionnaire.
         Items measuring the same facet (e.g., Honesty1 and Honesty2) share a
         latent psychological construct and should be more correlated than the
         background crud level.

      2. **NHANES**: Within-group biomarker pairs from the national health survey.
         Variables in the same biomarker category (e.g., systolic and diastolic
         blood pressure) share biological pathways and should remain distinguishable.

    For each positive control pair, we compute its residual correlation after
    PCA adjustment and its percentile rank in the full distribution of all
    pairwise residual correlations. A successful positive control means most
    within-group pairs land above the 95th percentile of the crud distribution
    (i.e., their correlations are not explained by the background eigenstructure).

    Tests are run at K=0 (no adjustment), K=10, and K=20 PCA components removed.

    Returns
    -------
    list of dict
        Each dict contains 'title', 'headers', and 'rows' for a summary table.
    """

    results_tables = []
    # Test at K=0 (raw), K=10, K=20 PCA components removed.
    K_VALUES = [0, 10, 20]

    # ----------------------------------------------------------------
    # HEXACO positive control: within-facet item pairs
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Positive control: HEXACO (within-facet pairs)")
    print("=" * 60)

    # Try to load HEXACO with column names (needed for facet grouping).
    extract_dir = Path("data/cache/HEXACO_extracted")
    data_path = extract_dir / "HEXACO" / "data.csv"
    if data_path.exists():
        # Try tab-separated first, fall back to comma-separated.
        df = pd.read_csv(data_path, sep="\t")
        if df.shape[1] <= 1:
            df = pd.read_csv(data_path)
        df_num = df.select_dtypes(include=[np.number]).dropna()
        col_names = list(df_num.columns)
        X = df_num.to_numpy(dtype=np.float32)
    else:
        # Fallback loader without column names; facet grouping will use generic names.
        X = load_hexaco_data()
        col_names = [f"V{i}" for i in range(X.shape[1])]

    # Subsample rows if dataset is very large (memory/speed guard).
    rng = np.random.default_rng(SEED)
    if X.shape[0] > MAX_ROWS_ANALYSIS:
        X = X[rng.choice(X.shape[0], size=MAX_ROWS_ANALYSIS, replace=False)]
    # Z-score and prepare for PCA.
    Xz, _ = preprocess_for_pca_and_corr(X)
    n, p = Xz.shape

    # Compute PCA once; we'll use the top components for adjustment at various K.
    k_pca = min(300, n, p)
    evr, Vt = pca_randomized(Xz, k_pca, seed=SEED)
    top_pcs = Vt[:min(100, Vt.shape[0])].copy()

    # Identify within-facet pairs (the positive control set).
    facet_groups = _hexaco_facet_groups(col_names)
    within_pairs = _within_group_pairs(facet_groups)
    print(f"  Found {len(facet_groups)} facets, {len(within_pairs)} within-facet pairs")

    hexaco_rows = []
    for K in K_VALUES:
        all_cols = np.arange(p)
        # Compute residuals after removing top-K PCA components.
        if K <= 0:
            # K=0: no adjustment (raw correlations).
            resid = Xz.copy()
        else:
            # Project out the top K principal components.
            kk = int(min(K, top_pcs.shape[0]))
            Vk = top_pcs[:kk].astype(np.float32)
            scores = (Xz @ Vk.T).astype(np.float32)
            resid = (Xz - scores @ Vk).astype(np.float32)
        # Re-standardize residuals to zero mean and unit variance.
        resid = resid - resid.mean(axis=0, keepdims=True)
        resid = resid / (resid.std(axis=0, ddof=1, keepdims=True) + EPS)
        # Compute full correlation matrix of residuals.
        S = corr_from_zscored(resid)

        # Extract all off-diagonal correlations (the "crud distribution").
        all_vals = offdiag_vals_from_corr(S)
        all_abs = np.abs(all_vals)

        # For each within-facet pair, compute its residual correlation and
        # its percentile rank in the background (crud) distribution.
        within_corrs = []
        within_pcts = []
        for i, j, gname in within_pairs:
            r = float(S[i, j])
            # Fraction of all pairs with |correlation| >= this pair's |r|.
            # A small value means this pair is in the upper tail (strong signal).
            pct = float(np.mean(all_abs >= abs(r)))
            within_corrs.append(abs(r))
            within_pcts.append(pct)

        # Summarize: median correlation, median crud percentile, fraction in top 5%/1%.
        median_within_corr = float(np.median(within_corrs))
        median_within_pct = float(np.median(within_pcts))
        frac_top5 = float(np.mean(np.array(within_pcts) < 0.05))
        frac_top1 = float(np.mean(np.array(within_pcts) < 0.01))
        bg_sd = float(np.std(all_vals))

        print(f"  K={K}: median |r|={median_within_corr:.3f}, "
              f"median crud-pval={median_within_pct:.4f}, "
              f"frac in top 5%={frac_top5:.2f}, top 1%={frac_top1:.2f}, "
              f"bg SD={bg_sd:.4f}")
        hexaco_rows.append({
            "K": K, "median_r": median_within_corr,
            "median_crud_p": median_within_pct,
            "frac_top5": frac_top5, "frac_top1": frac_top1,
            "bg_sd": bg_sd, "n_pairs": len(within_pairs),
        })

    # Plot: overlay histograms of within-facet |r| (red) vs all-pairs |r| (gray)
    # for each value of K. The positive control passes if the red distribution
    # is visibly shifted to the right of the gray crud background.
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(5 * len(K_VALUES), 4))
    if len(K_VALUES) == 1:
        axes = [axes]
    for ax, K, row in zip(axes, K_VALUES, hexaco_rows):
        # Recompute residuals for this K (same logic as above).
        if K <= 0:
            resid = Xz.copy()
        else:
            kk = int(min(K, top_pcs.shape[0]))
            Vk = top_pcs[:kk].astype(np.float32)
            scores = (Xz @ Vk.T).astype(np.float32)
            resid = (Xz - scores @ Vk).astype(np.float32)
        resid = resid - resid.mean(axis=0, keepdims=True)
        resid = resid / (resid.std(axis=0, ddof=1, keepdims=True) + EPS)
        S = corr_from_zscored(resid)
        all_vals = offdiag_vals_from_corr(S)
        within_r = [abs(float(S[i, j])) for i, j, _ in within_pairs]

        # Gray histogram: all pairwise |r| (the crud background distribution).
        ax.hist(np.abs(all_vals), bins=80, density=True, alpha=0.5,
                color="gray", label="All pairs (crud null)")
        # Red histogram: within-facet |r| (should be in the right tail).
        ax.hist(within_r, bins=30, density=True, alpha=0.7,
                color="red", label="Within-facet pairs")
        ax.set_xlabel("|residual correlation|")
        ax.set_ylabel("Density")
        ax.set_title(f"HEXACO, K={K}")
        ax.legend(fontsize=8)
    plt.suptitle("Positive control: within-facet pairs vs crud background", fontsize=13)
    plt.tight_layout()
    plt.show(); plt.close()

    # ----------------------------------------------------------------
    # NHANES positive control: within-biomarker-group variable pairs
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Positive control: NHANES (biomarker groups)")
    print("=" * 60)

    nhanes_rows = []
    try:
        # Load and merge NHANES data from multiple survey cycles and data files.
        cycles = [("2011-2012", "G"), ("2013-2014", "H"), ("2015-2016", "I")]
        file_stems = ["DEMO", "BMX", "BPX", "CBC", "BIOPRO", "TCHOL", "HDL", "TRIGLY", "GLU", "GHB"]
        out_dir = Path("data/cache/nhanes_xpt")

        def _read_xpt(path):
            df = pd.read_sas(str(path), format="xport")
            df.columns = [str(c) for c in df.columns]
            return df

        # Merge all NHANES data files within each cycle on SEQN (respondent ID),
        # then concatenate across cycles.
        all_dfs = []
        for cycle, suffix in cycles:
            dfs = []
            for stem in file_stems:
                fname = f"{cycle}_{stem}_{suffix}.xpt"
                local = out_dir / fname
                if local.exists():
                    df = _read_xpt(local)
                    if "SEQN" in df.columns:
                        df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce").astype("Int64")
                        dfs.append(df)
            if dfs:
                # Left-join all data files on SEQN within this cycle.
                merged = dfs[0]
                for df in dfs[1:]:
                    merged = merged.merge(df, on="SEQN", how="left", suffixes=("", "_dup"))
                    # Drop duplicate columns from overlapping variable names.
                    dup_cols = [c for c in merged.columns if c.endswith("_dup")]
                    if dup_cols:
                        merged = merged.drop(columns=dup_cols)
                merged["NHANES_CYCLE"] = cycle
                all_dfs.append(merged)

        if all_dfs:
            all_df = pd.concat(all_dfs, ignore_index=True)
            all_df = all_df.dropna(subset=["SEQN"]).copy()
            # Drop ID, survey design, and sample weight columns (not biomarkers).
            drop_cols = {"SEQN", "NHANES_CYCLE", "SDMVPSU", "SDMVSTRA"}
            wt_cols = [c for c in all_df.columns if c.startswith("WT")]
            all_df = all_df.drop(columns=list(drop_cols.intersection(all_df.columns)) + wt_cols, errors="ignore")
            # Keep only numeric columns with at least one non-NaN value.
            num_df = all_df.select_dtypes(include=[np.number]).copy()
            num_df = num_df.loc[:, num_df.notna().any(axis=0)]
            # Impute missing values with column medians (simple imputation).
            num_df = num_df.fillna(num_df.median(axis=0, skipna=True))
            nh_col_names = list(num_df.columns)
            X_nh = num_df.to_numpy(dtype=np.float32)
            print(f"  Loaded NHANES with column names: {X_nh.shape}, {len(nh_col_names)} cols")
        else:
            raise FileNotFoundError("No NHANES XPT files found")
    except Exception as e:
        print(f"  Could not reload NHANES with column names: {e}")
        print("  Skipping NHANES positive control.")
        nh_col_names = []
        X_nh = None

    if X_nh is not None:
        # Subsample and z-score NHANES data, same as HEXACO above.
        rng_nh = np.random.default_rng(SEED)
        if X_nh.shape[0] > MAX_ROWS_ANALYSIS:
            X_nh = X_nh[rng_nh.choice(X_nh.shape[0], size=MAX_ROWS_ANALYSIS, replace=False)]
        Xz_nh, _ = preprocess_for_pca_and_corr(X_nh)
        n_nh, p_nh = Xz_nh.shape

        # PCA for NHANES data.
        k_pca_nh = min(300, n_nh, p_nh)
        evr_nh, Vt_nh = pca_randomized(Xz_nh, k_pca_nh, seed=SEED)
        top_pcs_nh = Vt_nh[:min(100, Vt_nh.shape[0])].copy()

        # Identify within-biomarker-group pairs (the NHANES positive control set).
        bio_groups = _nhanes_biomarker_groups(nh_col_names)
        bio_pairs = _within_group_pairs(bio_groups)
        print(f"  Found {len(bio_groups)} biomarker groups, {len(bio_pairs)} within-group pairs")
        for gname, idxs in bio_groups.items():
            names = [nh_col_names[i] for i in idxs]
            print(f"    {gname}: {names}")

        # Same analysis as HEXACO: for each K, compute residual correlations
        # and check whether within-group biomarker pairs land in the upper tail.
        for K in K_VALUES:
            # Compute residuals after removing top-K PCA components.
            if K <= 0:
                resid = Xz_nh.copy()
            else:
                kk = int(min(K, top_pcs_nh.shape[0]))
                Vk = top_pcs_nh[:kk].astype(np.float32)
                scores = (Xz_nh @ Vk.T).astype(np.float32)
                resid = (Xz_nh - scores @ Vk).astype(np.float32)
            # Re-standardize residuals.
            resid = resid - resid.mean(axis=0, keepdims=True)
            resid = resid / (resid.std(axis=0, ddof=1, keepdims=True) + EPS)
            S = corr_from_zscored(resid)
            # Background (crud) distribution of all pairwise correlations.
            all_vals = offdiag_vals_from_corr(S)
            all_abs = np.abs(all_vals)

            # Compute percentile rank of each within-group pair.
            within_corrs = []
            within_pcts = []
            for i, j, gname in bio_pairs:
                # Bounds check (in case column mapping didn't match all variables).
                if i < p_nh and j < p_nh:
                    r = float(S[i, j])
                    pct = float(np.mean(all_abs >= abs(r)))
                    within_corrs.append(abs(r))
                    within_pcts.append(pct)

            # Summarize positive control results for this K.
            if within_corrs:
                median_within_corr = float(np.median(within_corrs))
                median_within_pct = float(np.median(within_pcts))
                frac_top5 = float(np.mean(np.array(within_pcts) < 0.05))
                frac_top1 = float(np.mean(np.array(within_pcts) < 0.01))
            else:
                median_within_corr = median_within_pct = frac_top5 = frac_top1 = float("nan")
            bg_sd = float(np.std(all_vals))

            print(f"  K={K}: median |r|={median_within_corr:.3f}, "
                  f"median crud-pval={median_within_pct:.4f}, "
                  f"frac in top 5%={frac_top5:.2f}, top 1%={frac_top1:.2f}, "
                  f"bg SD={bg_sd:.4f}")
            nhanes_rows.append({
                "K": K, "median_r": median_within_corr,
                "median_crud_p": median_within_pct,
                "frac_top5": frac_top5, "frac_top1": frac_top1,
                "bg_sd": bg_sd, "n_pairs": len(within_corrs),
            })

    # Build and print a combined summary table for both datasets.
    print("\n  --- Positive Control Summary ---")
    print(f"  {'Dataset':<10} {'K':>3} {'Med |r|':>8} {'Med p_crud':>11} "
          f"{'Top 5%':>7} {'Top 1%':>7} {'BG SD':>7}")
    print("  " + "-" * 60)
    for row in hexaco_rows:
        print(f"  {'HEXACO':<10} {row['K']:>3} {row['median_r']:>8.3f} "
              f"{row['median_crud_p']:>11.4f} {row['frac_top5']:>7.0%} "
              f"{row['frac_top1']:>7.0%} {row['bg_sd']:>7.4f}")
    for row in nhanes_rows:
        print(f"  {'NHANES':<10} {row['K']:>3} {row['median_r']:>8.3f} "
              f"{row['median_crud_p']:>11.4f} {row['frac_top5']:>7.0%} "
              f"{row['frac_top1']:>7.0%} {row['bg_sd']:>7.4f}")

    # Build structured output tables for downstream reporting.
    hdr = ["Dataset", "K", "Median |r|", "Median p_crud",
           "Frac top 5%", "Frac top 1%", "Background SD", "N pairs"]
    rows_out = []
    for row in hexaco_rows + nhanes_rows:
        ds = "HEXACO" if row in hexaco_rows else "NHANES"
        rows_out.append([
            ds, str(row["K"]), f"{row['median_r']:.3f}",
            f"{row['median_crud_p']:.4f}", f"{row['frac_top5']:.0%}",
            f"{row['frac_top1']:.0%}", f"{row['bg_sd']:.4f}",
            str(row["n_pairs"]),
        ])
    results_tables.append({
        "title": "Positive control: known-strong pairs vs crud background",
        "headers": hdr, "rows": rows_out,
    })

    del X, Xz
    if X_nh is not None:
        del X_nh, Xz_nh
    gc.collect()
    return results_tables


def run_alternative_adjustments(dataset_loaders: Dict[str, Callable]) -> List[dict]:
    """Main comparison function: residual correlation SD under four adjustment methods.

    For each dataset provided by `dataset_loaders`, this function computes
    the standard deviation of off-diagonal residual correlations after applying
    four different adjustment strategies:

      1. PCA adjustment (the paper's primary method) -- project out top-K PCs.
      2. Random covariate adjustment -- regress on m randomly chosen features.
      3. Random projection adjustment -- project out k random Gaussian directions.
      4. Lasso adjustment -- per-feature Lasso-selected predictors (alpha=0.1).

    Methods 1-3 are tested at dimensions {10, 50, 100}. Method 4 is data-driven
    (no dimension parameter). A feature subset of 300 columns is used for
    correlation measurement to keep computation tractable.

    The function produces:
      - Bar charts comparing methods at each dimension across datasets.
      - Per-dataset panels showing how residual correlation SD changes with
        adjustment dimension for each method.
      - Summary tables for inclusion in the paper's Appendix.

    If all methods yield similar residual correlation spreads, this confirms
    that the persistent background correlations are a genuine data property
    (driven by the power-law eigenvalue spectrum), not an artifact of PCA.

    Parameters
    ----------
    dataset_loaders : dict mapping dataset name to loader callable.
        Each loader should accept `visualize=False` and return an np.ndarray.

    Returns
    -------
    list of dict
        Each dict contains 'title', 'headers', and 'rows' for a summary table.
    """
    # Dimensions at which to compare adjustment methods.
    ADJ_DIMS = [10, 50, 100]
    # Number of columns to use for correlation measurement (subset for speed).
    MAX_COLS = 300

    summary_rows = []
    all_method_curves = {}  # dataset -> {method -> {dim -> SD}}

    for name, loader in dataset_loaders.items():
        print(f"\n[alt-adjust] {name}")
        # Load and preprocess the dataset.
        X = loader(visualize=False)
        rng = np.random.default_rng(SEED)
        if X.shape[0] > MAX_ROWS_ANALYSIS:
            X = X[rng.choice(X.shape[0], size=MAX_ROWS_ANALYSIS, replace=False)]
        Xz, _ = preprocess_for_pca_and_corr(X)
        n, p = Xz.shape
        print(f"  n={n}, p={p}")

        # Select a random subset of 300 columns for correlation measurement.
        # Using a subset keeps the correlation matrix computation tractable.
        m = min(p, MAX_COLS)
        cols = rng.choice(p, size=m, replace=False) if p > m else np.arange(p)

        # Compute PCA (used for PCA adjustment and as baseline comparison).
        k_pca = min(300, n, p)
        evr, Vt = pca_randomized(Xz, k_pca, seed=SEED)
        top_pcs = Vt[:min(100, Vt.shape[0])].copy()

        row = {"dataset": name, "n": n, "p": p}
        method_results = {}

        # --- Method 1: PCA adjustment (the paper's primary method) ---
        pca_sds = {}
        for K in ADJ_DIMS:
            if K > top_pcs.shape[0]:
                continue
            vals = residual_corr_vals_after_k_pcs(Xz, top_pcs, cols, K)
            pca_sds[K] = float(np.std(vals))
            del vals
        method_results["PCA"] = pca_sds
        print(f"  PCA:       {pca_sds}")

        # --- Method 2: Random covariate regression ---
        rcov_sds = {}
        for m_cov in ADJ_DIMS:
            # Skip if there are not enough non-subset features for regression.
            if m_cov >= p - len(cols):
                if m_cov >= p:
                    continue
            sd = adjust_random_covariates(Xz, cols, m_cov, seed=SEED)
            rcov_sds[m_cov] = sd
        method_results["Random covariates"] = rcov_sds
        print(f"  RandCov:   {rcov_sds}")

        # --- Method 3: Random projection adjustment ---
        rproj_sds = {}
        for K in ADJ_DIMS:
            # Need K < min(n, p) for the random projection to be well-defined.
            if K >= min(n, p):
                continue
            sd = adjust_random_projections(Xz, cols, K, seed=SEED)
            rproj_sds[K] = sd
        method_results["Random projections"] = rproj_sds
        print(f"  RandProj:  {rproj_sds}")

        # --- Method 4: Lasso adjustment (data-driven, no dimension parameter) ---
        try:
            lasso_sd = adjust_lasso(Xz, cols, seed=SEED, alpha=0.1)
            method_results["Lasso"] = {"data-driven": lasso_sd}
            print(f"  Lasso:     {lasso_sd:.4f}")
        except Exception as e:
            print(f"  Lasso failed: {e}")
            method_results["Lasso"] = {}

        all_method_curves[name] = method_results

        # Collect results into a flat row for the summary table.
        for K in ADJ_DIMS:
            row[f"PCA_{K}"] = pca_sds.get(K, float("nan"))
            row[f"RandCov_{K}"] = rcov_sds.get(K, float("nan"))
            row[f"RandProj_{K}"] = rproj_sds.get(K, float("nan"))
        row["Lasso"] = method_results.get("Lasso", {}).get("data-driven", float("nan"))
        summary_rows.append(row)

        # Free memory before processing the next dataset.
        del X, Xz, Vt, top_pcs, evr
        gc.collect()

    # --- Plotting: bar charts comparing methods at each adjustment dimension ---
    # One figure per dimension K, with grouped bars for PCA, random covariates,
    # and random projections across all datasets.
    for K in ADJ_DIMS:
        plt.figure(figsize=(12, 5))
        datasets = [r["dataset"] for r in summary_rows]
        methods = ["PCA", "Random covariates", "Random projections"]
        x = np.arange(len(datasets))
        width = 0.22
        colors = ["#2196F3", "#FF9800", "#4CAF50"]

        for i, (method, color) in enumerate(zip(methods, colors)):
            if method == "PCA":
                key = f"PCA_{K}"
            elif method == "Random covariates":
                key = f"RandCov_{K}"
            else:
                key = f"RandProj_{K}"
            vals = [r.get(key, float("nan")) for r in summary_rows]
            plt.bar(x + i * width, vals, width, label=method, color=color, alpha=0.85)

        plt.xticks(x + width, datasets, rotation=30, ha="right")
        plt.ylabel("SD of residual correlations")
        plt.title(f"Residual correlation spread: adjustment methods at dimension {K}")
        plt.legend()
        plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
        plt.tight_layout(); plt.show(); plt.close()

    # --- Per-dataset panels: SD vs adjustment dimension curves ---
    # Each subplot shows one dataset, with lines for PCA, random covariates,
    # and random projections as a function of dimension, plus a horizontal
    # line for Lasso (which has no dimension parameter).
    n_ds = len(summary_rows)
    ncols_fig = min(4, n_ds)
    nrows_fig = math.ceil(n_ds / ncols_fig)
    fig, axes = plt.subplots(nrows_fig, ncols_fig, figsize=(5 * ncols_fig, 4 * nrows_fig),
                             squeeze=False)
    for idx, name in enumerate(all_method_curves):
        ax = axes[idx // ncols_fig][idx % ncols_fig]
        mr = all_method_curves[name]
        # Plot dimension-based methods as line curves.
        for method, style in [("PCA", "o-"), ("Random covariates", "s--"),
                               ("Random projections", "^:")]:
            if method in mr and mr[method]:
                ks = sorted(mr[method].keys())
                vals = [mr[method][k] for k in ks]
                ax.plot(ks, vals, style, markersize=5, linewidth=1.5, label=method)
        # Plot Lasso as a horizontal line (single data-driven value, no dimension).
        if "Lasso" in mr and mr["Lasso"]:
            lasso_val = list(mr["Lasso"].values())[0]
            ax.axhline(lasso_val, color="red", linestyle="-.", linewidth=1.5,
                       label=f"Lasso (\u03b1=0.1)")
        ax.set_xlabel("adjustment dimension")
        ax.set_ylabel("SD(residual corr)")
        ax.set_title(name)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", linewidth=0.5)
    # Hide unused subplot axes.
    for idx in range(n_ds, nrows_fig * ncols_fig):
        axes[idx // ncols_fig][idx % ncols_fig].set_visible(False)
    fig.suptitle("Residual correlation spread across adjustment methods", fontsize=14)
    plt.tight_layout(); plt.show(); plt.close()

    # --- Build structured tables for reporting (one per dimension, plus a combined summary) ---
    tables = []
    # Per-dimension comparison tables.
    for K in ADJ_DIMS:
        hdr = ["Dataset", f"PCA (K={K})", f"Rand.\\ cov (m={K})", f"Rand.\\ proj (K={K})"]
        rows = []
        for r in summary_rows:
            rows.append([
                r["dataset"],
                f"{r.get(f'PCA_{K}', float('nan')):.4f}",
                f"{r.get(f'RandCov_{K}', float('nan')):.4f}",
                f"{r.get(f'RandProj_{K}', float('nan')):.4f}",
            ])
        tables.append({
            "title": f"Residual correlation SD: alternative adjustments (dimension {K})",
            "headers": hdr, "rows": rows,
        })

    # Combined summary table: PCA, RandCov, RandProj at dim=10 and dim=50, plus Lasso.
    hdr_all = ["Dataset", "PCA (10)", "RandCov (10)", "RandProj (10)",
               "PCA (50)", "RandCov (50)", "RandProj (50)", "Lasso"]
    rows_all = []
    for r in summary_rows:
        rows_all.append([
            r["dataset"],
            f"{r.get('PCA_10', float('nan')):.4f}",
            f"{r.get('RandCov_10', float('nan')):.4f}",
            f"{r.get('RandProj_10', float('nan')):.4f}",
            f"{r.get('PCA_50', float('nan')):.4f}",
            f"{r.get('RandCov_50', float('nan')):.4f}",
            f"{r.get('RandProj_50', float('nan')):.4f}",
            f"{r.get('Lasso', float('nan')):.4f}",
        ])
    tables.append({
        "title": "Residual correlation SD: all methods summary",
        "headers": hdr_all, "rows": rows_all,
    })

    for t in tables:
        print(f"\n{t['title']}")
        print(" | ".join(t["headers"]))
        print("-" * 100)
        for row in t["rows"]:
            print(" | ".join(row))

    return tables
