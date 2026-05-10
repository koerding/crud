"""Compute sigma_K on the T2D-restricted NHANES sub-cohort.

T2D criterion: HbA1c (LBXGH) >= 6.5%, following standard ADA cutoff.

Mirrors the NHANES loader in crud/loaders.py but subsets to T2D before
preprocessing. Reports sigma_K at K=10 on the T2D sub-cohort, alongside
the full-cohort number for comparison.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/cache/nhanes_xpt")
CYCLES = [("2011-2012", "G"), ("2013-2014", "H"), ("2015-2016", "I")]
FILE_STEMS = ["DEMO", "BMX", "BPX", "CBC", "BIOPRO", "TCHOL", "HDL", "TRIGLY", "GLU", "GHB"]
T2D_HBA1C_THRESHOLD = 6.5
K = 10


def _read_xpt(path: Path) -> pd.DataFrame:
    df = pd.read_sas(str(path), format="xport")
    df.columns = [str(c) for c in df.columns]
    return df


def load_cycle(cycle: str, suffix: str) -> pd.DataFrame:
    dfs = []
    for stem in FILE_STEMS:
        path = CACHE_DIR / f"{cycle}_{stem}_{suffix}.xpt"
        df = _read_xpt(path)
        if "SEQN" not in df.columns:
            raise ValueError(f"{path} missing SEQN")
        df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce").astype("Int64")
        dfs.append(df)
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="SEQN", how="left", suffixes=("", "_dup"))
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_dup")])
    merged["NHANES_CYCLE"] = cycle
    return merged


def main():
    all_df = pd.concat([load_cycle(c, s) for c, s in CYCLES], axis=0, ignore_index=True)
    all_df = all_df.dropna(subset=["SEQN"]).copy()

    print(f"Total NHANES participants pooled: {len(all_df)}")
    print(f"Has LBXGH (HbA1c)? {('LBXGH' in all_df.columns)}")
    if "LBXGH" not in all_df.columns:
        raise SystemExit("HbA1c column missing — cannot identify T2D")

    n_with_hba1c = int(all_df["LBXGH"].notna().sum())
    print(f"Participants with HbA1c measured: {n_with_hba1c}")

    t2d_mask = all_df["LBXGH"] >= T2D_HBA1C_THRESHOLD
    n_t2d = int(t2d_mask.sum())
    print(f"T2D subjects (HbA1c >= {T2D_HBA1C_THRESHOLD}): {n_t2d}")

    # Drop survey-design and non-numeric columns, mirror loader
    drop_cols = {"SEQN", "NHANES_CYCLE", "SDMVPSU", "SDMVSTRA"}
    wt_cols = [c for c in all_df.columns if c.startswith("WT")]
    feat_df = all_df.drop(
        columns=list(drop_cols.intersection(all_df.columns)) + wt_cols, errors="ignore"
    )
    num_df = feat_df.select_dtypes(include=[np.number]).copy()
    num_df = num_df.loc[:, num_df.notna().any(axis=0)]
    num_df_full = num_df.fillna(num_df.median(axis=0, skipna=True))

    print(f"Full-cohort feature matrix: {num_df_full.shape}")

    # T2D restriction
    num_df_t2d = num_df.loc[t2d_mask].copy()
    # Refill T2D-only medians (so T2D sub-cohort is its own population)
    num_df_t2d = num_df_t2d.fillna(num_df_t2d.median(axis=0, skipna=True))
    print(f"T2D-only feature matrix: {num_df_t2d.shape}")

    X_full = num_df_full.to_numpy(dtype=np.float64)
    X_t2d = num_df_t2d.to_numpy(dtype=np.float64)

    def sigma_K(X, k):
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
        # Drop columns with zero variance after subsetting
        std = Xz.std(0)
        keep = std > 1e-6
        Xz = Xz[:, keep]
        U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
        Vk = Vt[:k]
        Xr = Xz - (Xz @ Vk.T) @ Vk
        C = np.corrcoef(Xr.T)
        iu = np.triu_indices(C.shape[0], k=1)
        rs = C[iu]
        return float(rs.std(ddof=0)), Xz.shape

    sk_full, shape_full = sigma_K(X_full, K)
    sk_t2d, shape_t2d = sigma_K(X_t2d, K)

    print()
    print(f"Full NHANES (n={shape_full[0]}, p={shape_full[1]}): sigma_K(K={K}) = {sk_full:.4f}")
    print(f"T2D-restricted (n={shape_t2d[0]}, p={shape_t2d[1]}): sigma_K(K={K}) = {sk_t2d:.4f}")
    print(f"Ratio T2D/full: {sk_t2d / sk_full:.3f}")

    # Folate verdict at the new sigma_K
    r_folate = 0.02
    n_folate = 8000
    for sk in (sk_full, sk_t2d):
        sample_sd = 1.0 / np.sqrt(n_folate - 1)
        total = (sk * sk + sample_sd * sample_sd) ** 0.5
        z = abs(r_folate) / total
        from scipy.stats import norm
        p = 2 * norm.sf(z)
        print(f"  Folate r={r_folate}, n={n_folate}: sigma_K={sk:.4f} -> z={z:.3f}, p_crud={p:.3f}")


if __name__ == "__main__":
    main()
