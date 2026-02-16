"""
Universal data properties relevant to causal inference.

Loads real datasets (medical, neuroscience, social science, images),
computes PCA spectra, correlation structure, residual correlations,
and produces cross-dataset summary plots.

Includes optional heavy-tail stress tests and null-model comparisons.

Usage:
    python fuzzy_domains_clean_working_v2.py
"""

import os
import gc
import json
import math
import tarfile
import zipfile
import pickle
from pathlib import Path
from typing import Dict, Callable, Tuple, List, Optional

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.utils.extmath import randomized_svd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPS = 1e-12
SEED = 42
MAX_ROWS_ANALYSIS = 10_000
MAX_PCS_FIT = 500
MAX_K_STORE = 100
MAX_COLS_CORR = 500
KS = [0, 1, 2, 5, 10, 20, 50]
BINS_CORR = 100
BINS_RESID = 100

AUTOSAVE_FIGS = True
FIG_DIR = Path("figures")
FIG_FORMAT = "png"
CACHE_DIR = Path("cached_data")

# ---------------------------------------------------------------------------
# Auto-save every plt.show() to disk
# ---------------------------------------------------------------------------
_fig_counter = 0


def install_autosave_show(fig_dir: Path, fmt: str = "png") -> None:
    global _fig_counter
    fig_dir.mkdir(parents=True, exist_ok=True)

    if getattr(plt.show, "__name__", "") == "show_and_save":
        print(f"plt.show already wrapped; saving to {fig_dir.resolve()} as .{fmt}")
        return

    old_show = plt.show

    def show_and_save(*args, **kwargs):
        global _fig_counter
        _fig_counter += 1
        fig = plt.gcf()
        out = fig_dir / f"fig_{_fig_counter:03d}.{fmt}"
        fig.savefig(out, format=fmt, bbox_inches="tight")
        return old_show(*args, **kwargs)

    plt.show = show_and_save
    print(f"Auto-saving figures on plt.show() to: {fig_dir.resolve()} as .{fmt}")


if AUTOSAVE_FIGS:
    install_autosave_show(FIG_DIR, FIG_FORMAT)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def download_if_needed(url: str, local_path: str, *, timeout: int = 120, chunk_mb: int = 4) -> None:
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
    """Z-score each feature (column) across samples. Returns (Xz, mean, std)."""
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    Xc = X - mu
    sd = Xc.std(axis=0, ddof=1, keepdims=True).astype(np.float32)
    sd = np.where(sd < eps, 1.0, sd)
    Xz = Xc / (sd + eps)
    Xz = Xz - Xz.mean(axis=0, keepdims=True)
    return Xz.astype(np.float32), mu.squeeze(0), sd.squeeze(0)


def preprocess_for_pca_and_corr(X: np.ndarray, *, eps: float = EPS) -> Tuple[np.ndarray, dict]:
    """Return z-scored-by-feature matrix (float32) and diagnostics."""
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
    """Correlation matrix for already z-scored columns."""
    Xz = np.asarray(Xz, dtype=np.float32)
    n = Xz.shape[0]
    S = (Xz.T @ Xz).astype(np.float64) / max(n - 1, 1)
    S = S.astype(np.float32)
    np.fill_diagonal(S, 1.0)
    return S


def corr_poisson_corrected(Xraw_subset: np.ndarray, *, eps: float = EPS) -> np.ndarray:
    """Poisson-noise-corrected correlation matrix for count-like data.

    For Poisson data, Var(x_j) = mean(x_j), so the diagonal of the sample
    covariance is inflated by the Poisson noise floor. We subtract diag(mean)
    from the covariance before normalizing to correlation. Off-diagonal entries
    are unaffected (independent Poisson noise doesn't create spurious covariance).
    """
    X = np.asarray(Xraw_subset, dtype=np.float64)
    n, p = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(n - 1, 1)       # sample covariance
    C_corr = C.copy()
    # subtract estimated Poisson noise variance from diagonal
    np.fill_diagonal(C_corr, np.maximum(np.diag(C) - mu, eps))
    # normalize to correlation
    sd = np.sqrt(np.maximum(np.diag(C_corr), eps))
    S = C_corr / np.outer(sd, sd)
    np.fill_diagonal(S, 1.0)
    return S.astype(np.float32)


def is_count_like(Xraw: np.ndarray) -> bool:
    """Heuristic: non-negative, >15% zeros, right-skewed."""
    flat = Xraw.ravel()
    if (flat < -1e-6).sum() > 0.01 * flat.size:
        return False
    frac_zero = (np.abs(flat) < 1e-8).sum() / flat.size
    if frac_zero < 0.15:
        return False
    if np.median(flat) <= 0:
        return True  # heavily zero-inflated
    skew = (np.mean(flat) - np.median(flat)) / (np.std(flat) + 1e-12)
    return skew > 0.05


def offdiag_vals_from_corr(S: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(S, k=1)
    vals = S[iu]
    vals = vals[np.isfinite(vals)]
    return vals.astype(np.float32)


def pca_randomized(Xz: np.ndarray, n_components: int, *, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Randomized SVD PCA on standardized data."""
    Xz = np.asarray(Xz, dtype=np.float32)
    n, p = Xz.shape
    k = int(min(n_components, n, p))
    if k <= 0:
        raise ValueError("Need at least 1 component.")
    U, S, Vt = randomized_svd(Xz, n_components=k, random_state=seed)
    ev = (S.astype(np.float64) ** 2) / max(n - 1, 1)
    ss = np.sum((Xz.astype(np.float64) ** 2), axis=0)
    total_var = float(np.sum(ss) / max(n - 1, 1))
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
    """Off-diagonal residual correlations on a feature subset after removing k PCs."""
    Xz = np.asarray(Xz, dtype=np.float32)
    cols = np.asarray(cols, dtype=int)
    Xs = Xz[:, cols]
    if k <= 0:
        resid = Xs
    else:
        kk = int(min(k, top_pcs.shape[0]))
        Vk = top_pcs[:kk].astype(np.float32)
        scores = (Xz @ Vk.T).astype(np.float32)
        resid = (Xs - scores @ Vk[:, cols]).astype(np.float32)
        del scores
    resid = resid - resid.mean(axis=0, keepdims=True)
    resid = resid / (resid.std(axis=0, ddof=1, keepdims=True) + eps)
    S = corr_from_zscored(resid)
    vals = offdiag_vals_from_corr(S)
    del resid, S, Xs
    return vals


def fit_power_law_first_n(evr: np.ndarray, n_fit: int = 40, y_bottom: float = 1e-6) -> Tuple[float, float]:
    """Fit log(evr) ~ -alpha log(k) on first n_fit points above y_bottom."""
    evr = np.asarray(evr, dtype=float)
    k = np.arange(1, len(evr) + 1, dtype=float)
    m = int(min(n_fit, len(evr)))
    if m < 5:
        return float("nan"), float("nan")
    x = np.log(k[:m])
    y = np.log(np.maximum(evr[:m], y_bottom))
    A = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(beta[0]), float(beta[1])
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + EPS
    r2 = 1.0 - ss_res / ss_tot
    alpha = -slope
    return alpha, r2


def cached_load(name: str, loader: Callable, **kwargs) -> np.ndarray:
    """Load from .npy cache if available, otherwise run loader and cache the result."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{name}.npy"
    if cache_path.exists():
        print(f"[cache] Loading {name} from {cache_path}")
        return np.load(cache_path)
    X = loader(**kwargs)
    np.save(cache_path, X)
    print(f"[cache] Saved {name} to {cache_path}")
    return X


# ---------------------------------------------------------------------------
# Dataset loaders — each returns a samples x features numpy array
# ---------------------------------------------------------------------------

def load_nhanes_data(*, visualize: bool = False) -> np.ndarray:
    cycles = [("2011-2012", "G"), ("2013-2014", "H"), ("2015-2016", "I")]
    file_stems = ["DEMO", "BMX", "BPX", "CBC", "BIOPRO", "TCHOL", "HDL", "TRIGLY", "GLU", "GHB"]
    out_dir = Path("nhanes_xpt")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _download_xpt(url: str, path: Path) -> None:
        if path.exists():
            return
        print(f"Downloading {url} -> {path}")
        r = requests.get(url, stream=True, timeout=180)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        head = path.read_bytes()[:80]
        if b"HEADER RECORD" not in head:
            snippet = head.decode("latin1", errors="replace")
            raise ValueError(f"{path} does not look like XPT. First bytes: {snippet!r}")

    def _read_xpt(path: Path) -> pd.DataFrame:
        df = pd.read_sas(str(path), format="xport")
        df.columns = [str(c) for c in df.columns]
        return df

    def _load_cycle(cycle: str, suffix: str) -> pd.DataFrame:
        begin_year = cycle.split("-")[0]
        base = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{begin_year}/DataFiles/"
        dfs = []
        for stem in file_stems:
            fname = f"{stem}_{suffix}.xpt"
            local = out_dir / f"{cycle}_{stem}_{suffix}.xpt"
            _download_xpt(base + fname, local)
            df = _read_xpt(local)
            if "SEQN" not in df.columns:
                raise ValueError(f"{fname} missing SEQN; cannot merge.")
            df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce").astype("Int64")
            dfs.append(df)
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on="SEQN", how="left", suffixes=("", "_dup"))
            dup_cols = [c for c in merged.columns if c.endswith("_dup")]
            if dup_cols:
                merged = merged.drop(columns=dup_cols)
        merged["NHANES_CYCLE"] = cycle
        return merged

    all_df = pd.concat([_load_cycle(c, s) for c, s in cycles], axis=0, ignore_index=True)
    all_df = all_df.dropna(subset=["SEQN"]).copy()

    drop_cols = {"SEQN", "NHANES_CYCLE", "SDMVPSU", "SDMVSTRA"}
    wt_cols = [c for c in all_df.columns if c.startswith("WT")]
    all_df = all_df.drop(columns=list(drop_cols.intersection(all_df.columns)) + wt_cols, errors="ignore")

    num_df = all_df.select_dtypes(include=[np.number]).copy()
    num_df = num_df.loc[:, num_df.notna().any(axis=0)]
    num_df = num_df.fillna(num_df.median(axis=0, skipna=True))

    X = num_df.to_numpy(dtype=np.float32)
    print(f"NHANES pooled matrix shape: {X.shape} (rows=participants, cols=features)")
    return X


def load_physionet2012_icu_data(*, visualize: bool = False) -> np.ndarray:
    URL_A = "https://physionet.org/files/challenge-2012/1.0.0/set-a.zip?download=1"
    URL_B = "https://physionet.org/files/challenge-2012/1.0.0/set-b.zip?download=1"
    local_a = "physionet2012_set-a.zip"
    local_b = "physionet2012_set-b.zip"
    extract_dir = Path("physionet2012_extracted")

    download_if_needed(URL_A, local_a, timeout=180)
    download_if_needed(URL_B, local_b, timeout=180)

    extract_dir.mkdir(parents=True, exist_ok=True)
    if not (extract_dir / "set-a").exists():
        with zipfile.ZipFile(local_a, "r") as z:
            z.extractall(extract_dir)
    if not (extract_dir / "set-b").exists():
        with zipfile.ZipFile(local_b, "r") as z:
            z.extractall(extract_dir)

    DESCRIPTORS = ["Age", "Gender", "Height", "ICUType", "Weight"]
    TIME_SERIES = [
        "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol", "Creatinine",
        "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "Mg",
        "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH",
        "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT",
        "Urine", "WBC", "Weight",
    ]
    N_HOURS = 48

    paths = sorted((extract_dir / "set-a").glob("*.txt")) + sorted((extract_dir / "set-b").glob("*.txt"))
    max_patients = int(os.environ.get("P12_MAX_PATIENTS", "2000"))
    if max_patients > 0:
        paths = paths[:max_patients]

    n_patients = len(paths)
    ts_index = {v: i for i, v in enumerate(TIME_SERIES)}
    D = len(DESCRIPTORS) + len(TIME_SERIES) * N_HOURS
    X = np.zeros((n_patients, D), dtype=np.float32)

    def _time_to_hour(t: str) -> int:
        try:
            h = int(str(t).split(":", 1)[0])
        except Exception:
            return 0
        return max(0, min(N_HOURS - 1, h))

    for i, pth in enumerate(paths):
        if (i + 1) % 250 == 0:
            print(f"Parsed {i+1}/{n_patients}")
        df = pd.read_csv(pth)
        if df.empty:
            continue
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna(subset=["Value"])
        df = df[df["Value"] >= 0]
        df["hour"] = df["Time"].map(_time_to_hour)

        for j, name in enumerate(DESCRIPTORS):
            sub = df[df["Parameter"] == name]
            if not sub.empty:
                X[i, j] = float(sub.iloc[0]["Value"])

        ts = df[df["Parameter"].isin(TIME_SERIES)]
        if not ts.empty:
            g = ts.groupby(["Parameter", "hour"])["Value"].mean()
            base = len(DESCRIPTORS)
            for (param, hour), val in g.items():
                col = base + ts_index[param] * N_HOURS + int(hour)
                X[i, col] = float(val)

    keep = np.any(X != 0, axis=0)
    X = X[:, keep]
    print(f"PhysioNet 2012 matrix shape: {X.shape}")
    return X


def load_precinct_data(*, visualize: bool = False) -> np.ndarray:
    url = "https://www.dropbox.com/scl/fi/1u1jcibxke28nx9dfimmu/PrecinctData.tab?rlkey=9ejk9scli1uq0bblgeyh4heyc&dl=1"
    local_file = "PrecinctData.tab"
    download_if_needed(url, local_file, timeout=180)

    df = pd.read_csv(local_file, sep="\t")
    if visualize:
        missing = df.isnull().sum().sort_values(ascending=False).head(20)
        missing.plot(kind="bar", figsize=(12, 4))
        plt.title("Precinct: top 20 columns by missing values")
        plt.ylabel("missing count")
        plt.tight_layout()
        plt.show()
        plt.close()

    df = df.loc[:, df.isnull().sum() <= len(df) * 0.5]
    df = df.dropna()
    df = df.select_dtypes(include=["number"])

    X = df.to_numpy(dtype=np.float32)
    print(f"Precinct matrix shape: {X.shape}")
    return X


def load_rnaseq_data(*, visualize: bool = False) -> np.ndarray:
    url = "https://www.dropbox.com/scl/fi/6pcw5lztnp9l0pcvjfiyt/medians.csv?rlkey=5997l65mvqczuqhbl87uqutfk&dl=1"
    local_file = "medians.csv"
    download_if_needed(url, local_file, timeout=180)

    df = pd.read_csv(local_file)
    df.set_index(df.columns[0], inplace=True)
    df = df.loc[df.sum(axis=1) > 0]
    df = df.loc[:, df.sum(axis=0) > 0]

    X = df.to_numpy(dtype=np.float32)
    print(f"RNASeq matrix shape: {X.shape}")
    return X


def load_hexaco_data(*, visualize: bool = False) -> np.ndarray:
    url = "https://www.dropbox.com/scl/fi/gpis8v7ojcwegqqco9ede/HEXACO.zip?rlkey=tnupayuu8bpwfgw8i50xtbub8&dl=1"
    zip_path = "HEXACO.zip"
    extract_dir = Path("HEXACO_extracted")
    download_if_needed(url, zip_path, timeout=180)

    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    data_path = extract_dir / "HEXACO" / "data.csv"
    df = pd.read_csv(data_path, sep="\t")
    if df.shape[1] <= 1:
        df = pd.read_csv(data_path)

    df_num = df.select_dtypes(include=[np.number]).dropna()
    X = df_num.to_numpy(dtype=np.float32)
    print(f"HEXACO matrix shape: {X.shape}")
    return X


def load_pda360_data(*, visualize: bool = False) -> np.ndarray:
    url = "https://www.dropbox.com/scl/fi/53s5j5pbwkwxgjm3ywi9s/360PDA.tab?rlkey=3lrpa8zumxxbn5y5hkrfubi1i&dl=1"
    local_file = "360PDA.tab"
    download_if_needed(url, local_file, timeout=180)

    df = pd.read_csv(local_file, sep="\t")
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    df = df.select_dtypes(include=[np.number]).dropna()

    X = df.to_numpy(dtype=np.float32)
    print(f"PDA360 matrix shape: {X.shape}")
    return X


def load_stringer_data(*, top_neurons: int = 6000, visualize: bool = False) -> np.ndarray:
    """Returns (timebins x neurons). Selects neurons by highest variance."""
    fname = "stringer_spontaneous.npy"
    url = "https://osf.io/dpqaj/download"

    if not os.path.isfile(fname):
        print("Downloading Stringer dataset...")
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        Path(fname).write_bytes(r.content)
        print("Download complete.")

    dat = np.load(fname, allow_pickle=True).item()

    if visualize:
        plt.figure(figsize=(15, 3))
        ax = plt.subplot(1, 5, 1); plt.plot(dat["pupilArea"][:500, 0]); ax.set(xlabel="Time", ylabel="Pupil Area")
        ax = plt.subplot(1, 5, 2); plt.plot(dat["pupilCOM"][:500, :]); ax.set(xlabel="Time", ylabel="Pupil COM")
        ax = plt.subplot(1, 5, 3); plt.plot(dat["beh_svd_time"][:500, 0]); ax.set(xlabel="Time", ylabel="Face SVD 0")
        ax = plt.subplot(1, 5, 4); plt.plot(dat["beh_svd_time"][:500, 1]); ax.set(xlabel="Time", ylabel="Face SVD 1")
        ax = plt.subplot(1, 5, 5); plt.scatter(dat["beh_svd_time"][:, 0], dat["beh_svd_time"][:, 1], s=1); ax.set(xlabel="SVD0", ylabel="SVD1")
        plt.tight_layout(); plt.show(); plt.close()

    sresp = np.asarray(dat["sresp"], dtype=np.float32)
    v = np.var(sresp, axis=1, dtype=np.float64)
    k = int(min(top_neurons, sresp.shape[0]))

    idx = np.argpartition(v, -k)[-k:]
    idx = idx[np.argsort(v[idx])[::-1]]

    X = sresp[idx, :].T
    print("Original sresp shape (neurons x timebins):", sresp.shape)
    print("Selected shape (timebins x neurons):", X.shape)
    return X


def load_haxby_data(
    *,
    visualize: bool = False,
    subject: int = 1,
    top_voxels: int = 60000,
    max_timepoints: int = None,
    seed: int = 0,
    data_dir: str = "fmri_haxby_cache",
    eps: float = 1e-12,
) -> np.ndarray:
    """Haxby 2001 fMRI (single subject). Returns (n_voxels_selected x n_timepoints)."""
    try:
        import nibabel as nib
        from nilearn.datasets import fetch_haxby
        from nilearn.masking import compute_epi_mask, apply_mask
        from nilearn.image import index_img, mean_img
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "nilearn", "nibabel"])
        import nibabel as nib
        from nilearn.datasets import fetch_haxby
        from nilearn.masking import compute_epi_mask, apply_mask
        from nilearn.image import index_img, mean_img

    os.makedirs(data_dir, exist_ok=True)

    if subject < 1 or subject > 6:
        raise ValueError("subject must be in {1,2,3,4,5,6}.")

    ds = fetch_haxby(data_dir=data_dir, subjects=[subject], fetch_stimuli=False, verbose=1)
    func_path = ds.func[0]

    vol0 = index_img(func_path, 0)
    mask_img = compute_epi_mask(vol0)

    X_tv = apply_mask(func_path, mask_img).astype(np.float32)
    T, V = X_tv.shape

    if max_timepoints is not None:
        Tcap = int(min(T, max_timepoints))
        X_tv = X_tv[:Tcap, :]
        T, V = X_tv.shape

    if top_voxels is not None and top_voxels < V:
        X0 = X_tv - X_tv.mean(axis=0, keepdims=True)
        var = (X0 * X0).mean(axis=0)
        rng = np.random.default_rng(seed)
        jitter = rng.standard_normal(size=var.shape).astype(np.float32) * eps
        k = int(top_voxels)
        idx = np.argpartition(-(var + jitter), k - 1)[:k]
        idx = np.sort(idx)
        X_tv = X_tv[:, idx]
        T, V = X_tv.shape

    X = X_tv.T.copy()

    if visualize:
        plt.figure(figsize=(10, 3))
        plt.plot(X_tv[:min(400, T), :min(6, V)])
        plt.title(f"Haxby fMRI subj={subject}: first timepoints, first few selected voxels")
        plt.xlabel("time"); plt.ylabel("signal")
        plt.tight_layout(); plt.show(); plt.close()

        mimg = mean_img(func_path)
        mdata = mimg.get_fdata()
        z = mdata.shape[2] // 2
        plt.figure(figsize=(6, 5))
        plt.imshow(mdata[:, :, z].T, origin="lower", aspect="auto")
        plt.title("Mean EPI (mid-slice)"); plt.axis("off")
        plt.tight_layout(); plt.show(); plt.close()

    print(f"Haxby fMRI matrix shape (voxels x timepoints): {X.shape}")
    return X


def load_kay_fmri_data(*, visualize: bool = False) -> np.ndarray:
    """Kay et al. natural images fMRI — 8428 voxels x 1750 stimuli.

    Data hosted by Neuromatch Academy on OSF. Single subject, V1-V4 + lateral occipital.
    Returns (voxels x stimuli) matrix, so samples=voxels (>8k), features=stimuli (1750).
    """
    npz_url = "https://osf.io/ymnjv/download"
    npz_path = "kay_images.npz"
    download_if_needed(npz_url, npz_path, timeout=300)

    with np.load(npz_path) as dobj:
        dat = dict(**dobj)

    # dat contains 'responses' (1750, 8428) and 'responses_test' (120, 8428)
    # Combine train+test for a bigger matrix, transpose to (voxels x stimuli)
    resp_train = dat["responses"]       # (1750, 8428)
    resp_test = dat["responses_test"]   # (120, 8428)
    resp = np.concatenate([resp_train, resp_test], axis=0)  # (1870, 8428)
    X = resp.T.astype(np.float32)   # (8428, 1870) — voxels x stimuli

    if visualize:
        plt.figure(figsize=(10, 3))
        plt.plot(X[:6, :].T)
        plt.title("Kay fMRI: first 6 voxels across stimuli")
        plt.xlabel("stimulus index"); plt.ylabel("response")
        plt.tight_layout(); plt.show(); plt.close()

    print(f"Kay fMRI matrix shape (voxels x stimuli): {X.shape}")
    return X


def load_fmri_data(
    visualize: bool = False,
    *,
    subject: Optional[str] = None,
    min_timepoints: int = 200,
    max_timepoints: Optional[int] = None,
    top_voxels: Optional[int] = 60000,
    seed: int = 0,
    data_dir: str = "openneuro_ds000030_cache",
) -> np.ndarray:
    """OpenNeuro ds000030 (single subject, single run) -> (voxels x timepoints)."""
    try:
        import nibabel as nib
        from nilearn.datasets import fetch_ds000030_urls, fetch_openneuro_dataset
        from nilearn.masking import compute_epi_mask, apply_mask
        from nilearn.image import index_img
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "nilearn", "nibabel"])
        import nibabel as nib
        from nilearn.datasets import fetch_ds000030_urls, fetch_openneuro_dataset
        from nilearn.masking import compute_epi_mask, apply_mask
        from nilearn.image import index_img

    os.makedirs(data_dir, exist_ok=True)

    urls_obj = fetch_ds000030_urls(data_dir=data_dir, verbose=1)
    if isinstance(urls_obj, dict) and "urls" in urls_obj:
        urls = list(urls_obj["urls"])
    elif hasattr(urls_obj, "urls"):
        urls = list(urls_obj.urls)
    elif isinstance(urls_obj, (list, tuple)):
        urls = list(urls_obj)
    else:
        raise TypeError(f"Unexpected return type from fetch_ds000030_urls: {type(urls_obj)}")

    func_urls = [u for u in urls if ("/func/" in u and u.endswith("_bold.nii.gz"))]
    if not func_urls:
        raise RuntimeError("Could not find any *_bold.nii.gz func files in ds000030 URL index.")

    if subject is None:
        for u in func_urls:
            parts = u.split("/")
            sub = next((p for p in parts if p.startswith("sub-")), None)
            if sub is not None:
                subject = sub
                break
    if subject is None:
        raise RuntimeError("Could not infer a subject id from ds000030 URLs.")
    if not subject.startswith("sub-"):
        subject = f"sub-{subject}"

    sub_urls = [u for u in func_urls if f"/{subject}/" in u]
    if not sub_urls:
        raise RuntimeError(f"No func BOLD urls found for subject={subject} in ds000030.")

    chosen_path = None
    sub_urls = sorted(sub_urls, key=lambda s: (len(s), s))

    for u in sub_urls:
        ds = fetch_openneuro_dataset(urls=[u], data_dir=data_dir, verbose=1)
        if isinstance(ds, dict) and "files" in ds:
            files = ds["files"]
        elif hasattr(ds, "files"):
            files = ds.files
        elif isinstance(ds, (list, tuple)):
            files = list(ds)
        else:
            files = []
            if hasattr(ds, "data_dir"):
                base = os.path.basename(u)
                for root, _, fnames in os.walk(ds.data_dir):
                    if base in fnames:
                        files = [os.path.join(root, base)]
                        break

        if not files:
            continue

        fpath = files[0]
        try:
            img = nib.load(fpath)
            shape = img.shape
            if len(shape) != 4:
                continue
            T = int(shape[3])
        except Exception:
            continue

        if T >= int(min_timepoints):
            chosen_path = fpath
            break

    if chosen_path is None:
        raise RuntimeError(
            f"Could not find any single-run BOLD file for {subject} with >= {min_timepoints} timepoints."
        )

    vol0 = index_img(chosen_path, 0)
    mask_img = compute_epi_mask(vol0)
    X_tv = apply_mask(chosen_path, mask_img).astype(np.float32)
    T, V = X_tv.shape

    if max_timepoints is not None:
        Tcap = int(min(T, max_timepoints))
        X_tv = X_tv[:Tcap, :]
        T = Tcap

    if top_voxels is not None and int(top_voxels) < V:
        X0 = X_tv - X_tv.mean(axis=0, keepdims=True)
        var = (X0 * X0).mean(axis=0)
        rng = np.random.default_rng(seed)
        jitter = rng.standard_normal(size=var.shape).astype(np.float32) * 1e-12
        score = var + jitter
        k = int(top_voxels)
        idx = np.argpartition(-score, k - 1)[:k]
        idx.sort()
        X_tv = X_tv[:, idx]
        V = X_tv.shape[1]

    X = X_tv.T.copy()

    if visualize:
        plt.figure(figsize=(10, 3))
        plt.plot(X_tv[:min(400, T), :min(5, V)])
        plt.title(f"ds000030 {subject}: example voxel timecourses (T={T}, V={V})")
        plt.xlabel("time"); plt.ylabel("signal")
        plt.tight_layout(); plt.show(); plt.close()

    print(f"OpenNeuro ds000030: picked {os.path.basename(chosen_path)}")
    print(f"Matrix shape (voxels x timepoints): {X.shape}")
    return X


def load_cifar10_data(
    *,
    visualize: bool = False,
    max_train_batches: int = 2,
    include_test: bool = True,
    max_images: Optional[int] = None,
) -> np.ndarray:
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = "cifar-10-python.tar.gz"
    extract_dir = Path("cifar10_extracted")
    batch_dir = extract_dir / "cifar-10-batches-py"

    download_if_needed(url, tar_path, timeout=180)
    if not batch_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

    batches = []
    max_train_batches = int(max(1, min(5, max_train_batches)))
    for i in range(1, max_train_batches + 1):
        with open(batch_dir / f"data_batch_{i}", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        batches.append(d[b"data"])

    if include_test:
        with open(batch_dir / "test_batch", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        batches.append(d[b"data"])

    X = np.concatenate(batches, axis=0).astype(np.float32) / 255.0
    if max_images is not None:
        X = X[:int(max_images)]

    print(f"CIFAR-10 matrix shape: {X.shape} (images x 3072)")
    return X


# ---------------------------------------------------------------------------
# Core per-dataset analysis
# ---------------------------------------------------------------------------

def analyze_dataset(
    Xraw: np.ndarray,
    *,
    name: str,
    seed: int = SEED,
    max_rows: int = MAX_ROWS_ANALYSIS,
    max_pcs: int = MAX_PCS_FIT,
    max_k_store: int = MAX_K_STORE,
    max_cols_corr: int = MAX_COLS_CORR,
    ks: List[int] = KS,
) -> dict:
    rng = np.random.default_rng(seed)
    results = {}

    Xraw = np.asarray(Xraw)
    n_full, p_full = Xraw.shape
    results["n_full"] = int(n_full)
    results["p_full"] = int(p_full)

    if n_full > max_rows:
        ridx = rng.choice(n_full, size=max_rows, replace=False)
        Xraw_use = Xraw[ridx]
    else:
        Xraw_use = Xraw

    Xz, diag = preprocess_for_pca_and_corr(Xraw_use)
    n, p = Xz.shape
    results["n_used"] = int(n)
    results["p_used"] = int(p)
    results["zscore_diag"] = diag

    print(f"[{name}] using n={n} (of {n_full}), p={p}")
    print(f"  zscore diag: mean|max={diag['col_mean_abs_max']:.3g}, "
          f"std med={diag['col_std_median']:.3g}, std min={diag['col_std_min']:.3g}")

    # Heatmap (downsample for display)
    max_rows_viz, max_cols_viz = 2000, 500
    r_viz = rng.choice(n, size=min(n, max_rows_viz), replace=False) if n > max_rows_viz else np.arange(n)
    c_viz = rng.choice(p, size=min(p, max_cols_viz), replace=False) if p > max_cols_viz else np.arange(p)

    Xviz = Xz[np.ix_(r_viz, c_viz)].copy()
    v = float(max(np.percentile(np.abs(Xviz), 99), 1e-6))
    Xviz = np.clip(Xviz, -v, v)

    plt.figure(figsize=(10, 6))
    plt.imshow(Xviz, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    plt.colorbar(label="z-score")
    plt.title(f"{name}: data heatmap (downsampled)")
    plt.xlabel("features (subset)"); plt.ylabel("samples (subset)")
    plt.show(); plt.close()
    del Xviz

    # PCA spectrum
    k_pca = int(min(max_pcs, n, p))
    evr, Vt = pca_randomized(Xz, k_pca, seed=seed)
    results["explained_variance_ratio"] = evr
    results["k_pca"] = int(k_pca)

    k_store = int(min(max_k_store, Vt.shape[0]))
    top_pcs = Vt[:k_store].copy()
    results["top_pcs"] = top_pcs
    results["k_store"] = int(k_store)

    plt.figure(figsize=(8, 6))
    x = np.arange(1, len(evr) + 1)
    plt.loglog(x, evr, marker="o", linestyle="-", linewidth=1)
    plt.xlabel("principal component index"); plt.ylabel("explained variance ratio")
    plt.title(f"{name}: explained variance spectrum (log-log)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show(); plt.close()

    # Correlations on a fixed feature subset
    m = int(min(p, max_cols_corr))
    cols = rng.choice(p, size=m, replace=False) if p > m else np.arange(p)
    results["corr_cols"] = cols

    Xs = Xz[:, cols]
    S = corr_from_zscored(Xs)

    plt.figure(figsize=(8, 6))
    plt.imshow(S, aspect="auto", cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f"{name}: correlation matrix (subset m={m})")
    plt.show(); plt.close()

    vals0 = offdiag_vals_from_corr(S)
    results["resid_corr_vals_by_k"] = {0: vals0}
    results["resid_corr_std_by_k"] = {0: float(np.std(vals0))}

    plt.figure(figsize=(8, 6))
    plt.hist(vals0, bins=BINS_CORR, alpha=0.7, edgecolor="black")
    plt.xlabel("correlation (off-diagonal)"); plt.ylabel("count")
    plt.title(f"{name}: correlation histogram (subset)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show(); plt.close()

    # Residual correlations after removing PCs
    for k in ks:
        if int(k) == 0:
            continue
        vals = residual_corr_vals_after_k_pcs(Xz, top_pcs, cols, int(k))
        results["resid_corr_vals_by_k"][int(k)] = vals
        results["resid_corr_std_by_k"][int(k)] = float(np.std(vals))

    k_show = 10 if 10 in results["resid_corr_vals_by_k"] else sorted(results["resid_corr_vals_by_k"].keys())[-1]
    vals_r = results["resid_corr_vals_by_k"][k_show]
    plt.figure(figsize=(8, 6))
    plt.hist(vals_r, bins=BINS_RESID, alpha=0.7, edgecolor="black")
    plt.xlabel("residual correlation (off-diagonal, subset)"); plt.ylabel("count")
    plt.title(f"{name}: residual correlation histogram (k={k_show} PCs removed)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show(); plt.close()

    del Xz, Xs, S, Vt, Xraw_use
    gc.collect()
    return results


# ---------------------------------------------------------------------------
# Cross-dataset summary plots
# ---------------------------------------------------------------------------

def plot_cross_dataset_summaries(all_results: Dict[str, dict]) -> List[dict]:
    """Returns a list of table dicts: {title, headers, rows} for HTML embedding."""
    tables = []

    # (1) Superimposed explained-variance spectra
    plt.figure(figsize=(9, 6))
    for name, res in all_results.items():
        evr = res["explained_variance_ratio"]
        x = np.arange(1, len(evr) + 1)
        plt.loglog(x, evr, linewidth=2, label=name, alpha=0.9)
    plt.ylim(1e-6, None)
    plt.xlabel("principal component rank"); plt.ylabel("explained variance ratio")
    plt.title("Explained-variance spectra (log-log), all datasets")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(); plt.show(); plt.close()

    # (2) Power-law fits with reference line
    FIT_N = 40
    Y_BOTTOM = 1e-6
    fit_table = []
    ev1_list = []

    plt.figure(figsize=(9, 6))
    for name, res in all_results.items():
        evr = np.asarray(res["explained_variance_ratio"], dtype=float)
        f = np.arange(1, len(evr) + 1, dtype=float)
        alpha, r2 = fit_power_law_first_n(evr, n_fit=FIT_N, y_bottom=Y_BOTTOM)
        fit_table.append((name, alpha, r2))

        (line,) = plt.loglog(f, evr, linewidth=2, alpha=0.85,
                             label=f"{name} (\u03b1\u2248{alpha:.2f}, R\u00b2={r2:.2f})")
        color = line.get_color()

        n_fit = int(min(FIT_N, len(evr)))
        f_fit = f[:n_fit]
        y_fit = evr[:n_fit]
        mask = np.isfinite(y_fit) & (y_fit > Y_BOTTOM)
        if np.isfinite(alpha) and mask.sum() >= 8:
            x_log = np.log10(f_fit[mask])
            y_log = np.log10(y_fit[mask])
            slope, intercept = np.polyfit(x_log, y_log, 1)
            c = 10 ** intercept
            plt.loglog(f_fit[mask], c * (f_fit[mask] ** slope),
                       linestyle="--", linewidth=2, alpha=0.95, color=color)

        if np.isfinite(evr[0]) and evr[0] > 0:
            ev1_list.append(float(evr[0]))

    if ev1_list:
        c1 = float(np.median(ev1_list))
        f_ref = np.arange(1, FIT_N + 1, dtype=float)
        plt.loglog(f_ref, c1 / f_ref, color="k", linewidth=4, linestyle="-", label="1/f reference")

    plt.ylim(Y_BOTTOM, None)
    plt.xlabel("component rank f"); plt.ylabel("EV(f)")
    plt.title(f"Explained variance spectra with power-law fits (first {FIT_N} comps)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(); plt.show(); plt.close()

    fit_table = sorted(fit_table, key=lambda t: (np.nan_to_num(t[1], nan=999)))
    tables.append({
        "title": f"Power-law fits (first {FIT_N} components)",
        "headers": ["Dataset", "\u03b1", "R\u00b2"],
        "rows": [[name, f"{alpha:.3f}", f"{r2:.3f}"] for name, alpha, r2 in fit_table],
    })

    # (3) Overlay residual-correlation histograms
    def _plot_residual_histograms_for_k(k: int, title_suffix: str):
        plt.figure(figsize=(9, 6))
        for name, res in all_results.items():
            vals = res["resid_corr_vals_by_k"][int(k)]
            plt.hist(vals, bins=BINS_RESID, density=True, alpha=0.25, label=name)
        plt.xlabel("residual correlation (off-diagonal, subset)"); plt.ylabel("density")
        plt.title(f"Residual correlation distributions, {title_suffix}")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend(); plt.show(); plt.close()

    _plot_residual_histograms_for_k(0, "baseline (k=0)")
    _plot_residual_histograms_for_k(1, "after removing 1 PC")
    _plot_residual_histograms_for_k(10, "after removing 10 PCs")

    # (4) Table: std of off-diagonal correlations
    resid_headers = ["Dataset"] + [f"k={k}" for k in KS]
    resid_rows = []
    for name in sorted(all_results.keys()):
        res = all_results[name]
        row_vals = [res["resid_corr_std_by_k"][int(k)] for k in KS]
        resid_rows.append([name] + [f"{v:.4g}" for v in row_vals])
    tables.append({
        "title": "Std of off-diagonal correlations after removing k PCs",
        "headers": resid_headers,
        "rows": resid_rows,
    })

    # (5) Approx correlation significance thresholds
    def rcrit_approx(n: int, z: float = 1.96) -> float:
        return float(z / math.sqrt(max(n - 3, 1)))

    rcrit_rows = []
    for name in sorted(all_results.keys()):
        n = int(all_results[name]["n_used"])
        rcrit_rows.append([name, str(n), f"{rcrit_approx(n):.4f}"])
    tables.append({
        "title": "Approx 95% |r| threshold (subset correlations)",
        "headers": ["Dataset", "n_used", "r_crit"],
        "rows": rcrit_rows,
    })

    # Print tables to stdout too
    for t in tables:
        print(f"\n{t['title']}")
        print("  ".join(h.rjust(12) for h in t["headers"]))
        print("-" * (14 * len(t["headers"])))
        for row in t["rows"]:
            print("  ".join(c.rjust(12) for c in row))

    return tables


# ---------------------------------------------------------------------------
# Heavy-tail stress test
# ---------------------------------------------------------------------------

def evr_topk_randomized(X: np.ndarray, k: int, *, seed: int = SEED) -> np.ndarray:
    Xz, _ = preprocess_for_pca_and_corr(X)
    n, p = Xz.shape
    kk = int(min(k, n, p))
    evr, _ = pca_randomized(Xz, kk, seed=seed)
    del Xz; gc.collect()
    return evr


def hill_tail_index(x: np.ndarray, top_frac: float = 0.05) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    x = np.abs(x)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size < 100:
        return None
    x = np.sort(x)
    k = int(max(50, top_frac * x.size))
    tail = x[-k:]
    xk = tail[0]
    if xk <= 0:
        return None
    hill = np.mean(np.log(tail / xk))
    if hill <= 0:
        return None
    return float(1.0 / hill)


def clip_entries(X: np.ndarray, q: float = 0.999) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    lo = float(np.quantile(X, 1 - q))
    hi = float(np.quantile(X, q))
    return np.clip(X, lo, hi)


def row_normalize_l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + EPS)


def rank_gaussianize(X: np.ndarray) -> np.ndarray:
    from scipy.special import erfinv
    X = np.asarray(X, dtype=np.float32)
    flat = X.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(flat), dtype=np.float64)
    u = (ranks + 0.5) / len(flat)
    z = math.sqrt(2.0) * erfinv(2.0 * u - 1.0)
    return z.reshape(X.shape).astype(np.float32)


def quick_gaussian_null(n: int, p: int, k: int, *, seed: int = SEED) -> Optional[np.ndarray]:
    MAX_N, MAX_P = 5000, 2000
    nn = int(min(n, MAX_N))
    pp = int(min(p, MAX_P))
    if nn < 50 or pp < 50:
        return None
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((nn, pp)).astype(np.float32)
    return evr_topk_randomized(G, k=min(k, nn, pp), seed=seed)


def _fmt(x):
    if x is None:
        return "NA"
    if isinstance(x, float) and not np.isfinite(x):
        return "NA"
    return f"{x:.3g}"


def run_heavy_tail_stress_test(dataset_loaders: Dict[str, Callable]) -> List[dict]:
    """Returns list of table dicts for HTML embedding."""
    CLIP_Q = 0.999
    FIT_K = 500

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

        evr0 = evr_topk_randomized(X, k, seed=SEED)
        a0, r20 = fit_power_law_first_n(evr0, n_fit=min(40, len(evr0)), y_bottom=1e-12)

        Xc = clip_entries(X, q=CLIP_Q)
        evr_clip = evr_topk_randomized(Xc, k, seed=SEED)
        ac, r2c = fit_power_law_first_n(evr_clip, n_fit=min(40, len(evr_clip)), y_bottom=1e-12)

        Xr = row_normalize_l2(X)
        evr_row = evr_topk_randomized(Xr, k, seed=SEED)
        ar, r2r = fit_power_law_first_n(evr_row, n_fit=min(40, len(evr_row)), y_bottom=1e-12)

        try:
            Xg = rank_gaussianize(X)
            evr_rank = evr_topk_randomized(Xg, k, seed=SEED)
            ag, r2g = fit_power_law_first_n(evr_rank, n_fit=min(40, len(evr_rank)), y_bottom=1e-12)
        except Exception as e:
            print("  rank-gauss failed:", e)
            evr_rank = None
            ag, r2g = (float("nan"), float("nan"))

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


# ---------------------------------------------------------------------------
# Null-model comparisons
# ---------------------------------------------------------------------------

def ipr_from_components(Vt: np.ndarray) -> np.ndarray:
    Vt = np.asarray(Vt, dtype=np.float32)
    return np.sum(Vt ** 4, axis=1).astype(np.float32)


def pca_topk_components(X: np.ndarray, k: int, *, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    Xz, _ = preprocess_for_pca_and_corr(X)
    k = int(min(k, Xz.shape[0], Xz.shape[1]))
    evr, Vt = pca_randomized(Xz, k, seed=seed)
    del Xz; gc.collect()
    return evr, Vt


def gaussian_null_like(n: int, p: int, *, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)).astype(np.float32)


def column_permutation_null(X: np.ndarray, *, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    n, p = X.shape
    Xp = np.empty_like(X)
    for j in range(p):
        idx = rng.permutation(n)
        Xp[:, j] = X[idx, j]
    return Xp


def run_null_model_comparisons(dataset_loaders: Dict[str, Callable]) -> dict:
    NULL_TOPK = 300
    N_PERM = 3
    MAX_N_NULL = 5000
    MAX_P_NULL = 2000

    null_results = {}

    for name, loader in dataset_loaders.items():
        print(f"\n[nulls] {name}")
        X = loader(visualize=False)

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

        evr_emp, Vt_emp = pca_topk_components(X, k, seed=SEED)
        ipr_emp = ipr_from_components(Vt_emp)

        G = gaussian_null_like(n, p, seed=SEED)
        evr_g, Vt_g = pca_topk_components(G, k, seed=SEED + 1)
        ipr_g = ipr_from_components(Vt_g)

        evr_perm_list, ipr_perm_list = [], []
        for t in range(N_PERM):
            Xp = column_permutation_null(X, seed=SEED + 10 + t)
            evr_p, Vt_p = pca_topk_components(Xp, k, seed=SEED + 20 + t)
            evr_perm_list.append(evr_p)
            ipr_perm_list.append(ipr_from_components(Vt_p))
            del Xp, Vt_p; gc.collect()

        evr_perm = np.mean(np.stack(evr_perm_list, axis=0), axis=0)
        ipr_perm = np.mean(np.stack(ipr_perm_list, axis=0), axis=0)

        null_results[name] = {
            "evr_emp": evr_emp, "ipr_emp": ipr_emp,
            "evr_gauss": evr_g, "ipr_gauss": ipr_g,
            "evr_perm": evr_perm, "ipr_perm": ipr_perm,
        }

        f = np.arange(1, k + 1)

        plt.figure(figsize=(9, 6))
        plt.loglog(f, evr_emp, linewidth=2, label="empirical")
        plt.loglog(f, evr_perm, linewidth=2, linestyle="--", label=f"col-perm (mean of {N_PERM})")
        plt.loglog(f, evr_g, linewidth=2, linestyle=":", label="gaussian")
        plt.xlabel("component rank"); plt.ylabel("explained variance ratio")
        plt.title(f"{name}: PCA spectrum vs nulls (capped n={n}, p={p})")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(); plt.show(); plt.close()

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


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(all_results: Dict[str, dict]) -> None:
    if AUTOSAVE_FIGS and FIG_DIR.exists():
        out_tgz = "figures_vector.tar.gz"
        with tarfile.open(out_tgz, "w:gz") as tar:
            tar.add(str(FIG_DIR), arcname=FIG_DIR.name)
        print("Wrote", out_tgz)
    else:
        print("No figures directory to bundle (or autosave disabled).")

    ckpt = {
        "config": {
            "SEED": SEED,
            "MAX_ROWS_ANALYSIS": MAX_ROWS_ANALYSIS,
            "MAX_PCS_FIT": MAX_PCS_FIT,
            "MAX_K_STORE": MAX_K_STORE,
            "MAX_COLS_CORR": MAX_COLS_CORR,
            "KS": KS,
        },
        "datasets": {},
    }

    for name, res in all_results.items():
        ckpt["datasets"][name] = {
            "n_full": res["n_full"],
            "p_full": res["p_full"],
            "n_used": res["n_used"],
            "p_used": res["p_used"],
            "zscore_diag": res["zscore_diag"],
            "k_pca": res["k_pca"],
            "explained_variance_ratio": res["explained_variance_ratio"].tolist(),
            "resid_corr_std_by_k": res["resid_corr_std_by_k"],
        }

    with open("analysis_checkpoint.json", "w") as f:
        json.dump(ckpt, f, indent=2)
    print("Wrote analysis_checkpoint.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset_loaders: Dict[str, Callable[..., np.ndarray]] = {
        # "PHYSIONET": load_physionet2012_icu_data,
        "Kay_fMRI": load_kay_fmri_data,
        "Haxby_fMRI": load_haxby_data,
        "NHANES": load_nhanes_data,
        "CIFAR10": load_cifar10_data,
        "Precinct": load_precinct_data,
        "RNASeq": load_rnaseq_data,
        "HEXACO": load_hexaco_data,
        # "PDA360": load_pda360_data,
        "Stringer": load_stringer_data,
    }

    # --- Phase 1: Load and analyze each dataset ---
    # First load caches all raw matrices to cached_data/*.npy so re-runs are fast.
    all_results: Dict[str, dict] = {}
    for name, loader in dataset_loaders.items():
        print(f"\n=== Loading dataset: {name} ===")
        X = cached_load(name, loader, visualize=True)
        all_results[name] = analyze_dataset(X, name=name, seed=SEED)
        print(f"=== Done: {name} ===")
        del X; gc.collect()

    print("\nAll datasets processed:", list(all_results.keys()))

    # --- Phase 2: Cross-dataset summaries ---
    tables = plot_cross_dataset_summaries(all_results)

    # --- Phase 3: Heavy-tail stress test ---
    # Use cached loaders so we don't re-download
    cached_loaders = {name: (lambda n=name, l=loader, **kw: cached_load(n, l))
                      for name, loader in dataset_loaders.items()}
    tables += run_heavy_tail_stress_test(cached_loaders)

    # --- Phase 4: Null-model comparisons ---
    run_null_model_comparisons(cached_loaders)

    # --- Phase 5: Save ---
    save_artifacts(all_results)
    build_html_gallery(tables=tables)


def build_html_gallery(
    fig_dir: Path = FIG_DIR,
    out_path: str = "figures.html",
    tables: Optional[List[dict]] = None,
) -> None:
    """Compile all figures + tables into a single HTML file."""
    import base64
    figs = sorted(fig_dir.glob(f"*.{FIG_FORMAT}"))
    if not figs and not tables:
        print("No figures or tables to compile.")
        return

    parts = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        "<title>Analysis Figures &amp; Tables</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;background:#1a1a1a;color:#eee;"
        "max-width:1200px;margin:0 auto;padding:20px}",
        "h1{text-align:center}",
        "h2{margin-top:40px;border-bottom:1px solid #555;padding-bottom:6px}",
        ".fig{background:#fff;border-radius:8px;margin:24px 0;padding:12px;text-align:center}",
        ".fig img{max-width:100%;height:auto}",
        ".fig p{color:#333;font-size:14px;margin:8px 0 0}",
        "table{border-collapse:collapse;margin:20px auto;font-size:14px}",
        "th,td{border:1px solid #555;padding:6px 12px;text-align:right}",
        "th{background:#333;color:#eee}",
        "td{background:#222}",
        "td:first-child,th:first-child{text-align:left}",
        "</style></head><body>",
        "<h1>Analysis Figures &amp; Tables</h1>",
    ]

    # Figures
    if figs:
        parts.append("<h2>Figures</h2>")
        for fig_path in figs:
            data = base64.b64encode(fig_path.read_bytes()).decode("ascii")
            mime = "image/png" if FIG_FORMAT == "png" else f"image/{FIG_FORMAT}"
            parts.append(f'<div class="fig">')
            parts.append(f'<img src="data:{mime};base64,{data}"/>')
            parts.append(f'<p>{fig_path.name}</p>')
            parts.append(f'</div>')

    # Tables
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
    n_tables = len(tables) if tables else 0
    print(f"Wrote {out_path} with {n_figs} figures and {n_tables} tables.")


if __name__ == "__main__":
    main()
