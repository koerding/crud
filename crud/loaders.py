"""Dataset loaders for the nine primary datasets analysed in the paper.

Each loader returns an (n, p) float32 matrix where rows are observations and
columns are features, matching the convention used throughout the package.
Datasets span medicine (NHANES), genomics (GTEx, RNA-Seq), psychology (HEXACO),
political science (Precinct), computer vision (CIFAR-10), and neuroscience
(Kay fMRI, Haxby fMRI, Stringer).

Several loaders download data on first use and cache locally. All loaders
accept a ``visualize`` keyword argument that, when True, produces exploratory
plots during loading.

"""

import os
import pickle
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from crud.config import EPS
from crud.utils import download_if_needed, safe_zscore_columns


def load_nhanes_data(*, visualize: bool = False) -> np.ndarray:
    """Load NHANES (National Health and Nutrition Examination Survey) data.

    Pools three survey cycles (2011-2016) across demographics, anthropometry,
    blood pressure, blood counts, biochemistry, lipids, and glucose panels.
    Returns (n_participants, p_features) ≈ (29902, 165).

    Missing values are filled with column medians; survey design columns
    (weights, PSU, strata) and non-numeric columns are dropped.
    """
    cycles = [("2011-2012", "G"), ("2013-2014", "H"), ("2015-2016", "I")]
    file_stems = ["DEMO", "BMX", "BPX", "CBC", "BIOPRO", "TCHOL", "HDL", "TRIGLY", "GLU", "GHB"]
    out_dir = Path("data/cache/nhanes_xpt")
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


def load_precinct_data(*, visualize: bool = False) -> np.ndarray:
    """Load precinct-level voting and census demographics.

    Returns (n_precincts, p_features) ≈ (28934, 83). Columns with >50%
    missing values are dropped; remaining rows with any missing are dropped.
    """
    url = "https://www.dropbox.com/scl/fi/1u1jcibxke28nx9dfimmu/PrecinctData.tab?rlkey=9ejk9scli1uq0bblgeyh4heyc&dl=1"
    local_file = "data/raw/PrecinctData.tab"
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
    """Load Allen Institute mouse brain RNA-Seq (median expression per gene).

    Returns (n_cell_types, p_genes) ≈ (10071, 387). Zero-sum rows and
    columns are removed.
    """
    url = "https://www.dropbox.com/scl/fi/6pcw5lztnp9l0pcvjfiyt/medians.csv?rlkey=5997l65mvqczuqhbl87uqutfk&dl=1"
    local_file = "data/raw/medians.csv"
    download_if_needed(url, local_file, timeout=180)

    df = pd.read_csv(local_file)
    df.set_index(df.columns[0], inplace=True)
    df = df.loc[df.sum(axis=1) > 0]
    df = df.loc[:, df.sum(axis=0) > 0]

    X = df.to_numpy(dtype=np.float32)
    print(f"RNASeq matrix shape: {X.shape}")
    return X


def load_gtex_data(
    *,
    tissue: str = "Muscle - Skeletal",
    top_genes: int = 10000,
    min_tpm: float = 1.0,
    visualize: bool = False,
) -> np.ndarray:
    """Load GTEx v8 gene expression (TPM), filtered to one tissue.

    Pipeline: download TPM matrix + sample metadata → filter to tissue →
    keep genes with median TPM ≥ min_tpm → log2(TPM+1) transform →
    select top_genes by variance → transpose to (genes, samples).

    Returns (top_genes, n_samples) ≈ (10000, ~803) for skeletal muscle.
    The transpose puts genes as rows (large n) and samples as columns
    (small p), consistent with the paper's convention (n=10000, p=803).
    """
    import gzip

    tpm_url = (
        "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/"
        "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    )
    meta_url = (
        "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/"
        "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
    )

    gtex_dir = Path("data/cache/gtex_cache")
    gtex_dir.mkdir(parents=True, exist_ok=True)

    tpm_local = gtex_dir / "gene_tpm.gct.gz"
    meta_local = gtex_dir / "sample_attributes.txt"

    download_if_needed(tpm_url, str(tpm_local), timeout=600, chunk_mb=8)
    download_if_needed(meta_url, str(meta_local), timeout=120)

    meta = pd.read_csv(meta_local, sep="\t", low_memory=False)
    tissue_samples = set(meta.loc[meta["SMTSD"] == tissue, "SAMPID"].values)
    if not tissue_samples:
        avail = sorted(meta["SMTSD"].dropna().unique())
        raise ValueError(
            f"Tissue '{tissue}' not found. Available: {avail}"
        )
    print(f"GTEx: {len(tissue_samples)} samples for tissue '{tissue}'")

    print("Reading GTEx TPM matrix (this may take a minute)...")
    with gzip.open(tpm_local, "rt") as f:
        f.readline()  # skip "#1.2"
        f.readline()  # skip dimensions line
        header = f.readline().strip().split("\t")

    sample_cols = [c for c in header[2:] if c in tissue_samples]
    if len(sample_cols) < 10:
        raise ValueError(
            f"Only {len(sample_cols)} samples found for tissue '{tissue}'. "
            f"Need at least 10."
        )
    usecols = ["Name", "Description"] + sample_cols
    print(f"  Keeping {len(sample_cols)} sample columns out of {len(header)-2} total")

    df = pd.read_csv(
        tpm_local,
        sep="\t",
        skiprows=2,
        usecols=usecols,
        compression="gzip",
        low_memory=False,
    )
    gene_ids = df["Name"].values
    gene_names = df["Description"].values
    df = df.drop(columns=["Name", "Description"])

    print(f"  Raw matrix: {df.shape[0]} genes x {df.shape[1]} samples")

    X = df.values.astype(np.float64)

    median_tpm = np.median(X, axis=1)
    keep_genes = median_tpm >= min_tpm
    X = X[keep_genes]
    gene_ids = gene_ids[keep_genes]
    gene_names = gene_names[keep_genes]
    print(f"  After min_tpm={min_tpm} filter: {X.shape[0]} genes")

    X = np.log2(X + 1.0)  # log-transform to stabilize variance

    X = X.T  # now (samples, genes) — intermediate orientation

    variances = np.var(X, axis=0)
    top_idx = np.argsort(variances)[::-1][:top_genes]
    X = X[:, top_idx]
    print(f"  After top-{top_genes} variance filter: {X.shape}")

    X = X.T.astype(np.float32)  # (genes x samples) — large n, small p
    print(f"GTEx matrix shape: {X.shape} (genes x samples, tissue='{tissue}')")
    return X


def load_hexaco_data(*, visualize: bool = False) -> np.ndarray:
    """Load HEXACO personality inventory responses.

    Returns (n_respondents, p_items) ≈ (22786, 242). Items are Likert-scale
    questionnaire responses across 6 personality domains with 4 facets each.
    Used as a positive control: within-facet item pairs should show strong
    correlations that survive crud-aware calibration.
    """
    url = "https://www.dropbox.com/scl/fi/gpis8v7ojcwegqqco9ede/HEXACO.zip?rlkey=tnupayuu8bpwfgw8i50xtbub8&dl=1"
    zip_path = "data/raw/HEXACO.zip"
    extract_dir = Path("data/cache/HEXACO_extracted")
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


def load_stringer_data(*, top_neurons: int = 6000, visualize: bool = False) -> np.ndarray:
    """Load Stringer et al. spontaneous mouse V1 calcium imaging data.

    Returns (n_timebins, p_neurons) ≈ (7018, 6000). Selects the top
    neurons by variance from a larger population. Original data is
    (neurons × timebins); we transpose so rows = observations.
    """
    fname = "data/raw/stringer_spontaneous.npy"
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
    data_dir: str = "data/cache/fmri_haxby_cache",
    eps: float = 1e-12,
) -> np.ndarray:
    """Load Haxby 2001 fMRI dataset (single subject).

    Preprocessing: linear detrend per voxel, DCT high-pass filter
    (cutoff ~0.01 Hz), global signal regression. Top voxels selected
    by variance. Returns (n_voxels, n_timepoints) ≈ (23612, 1452).
    Rows = voxels (features as observations) for large-n convention.
    """
    try:
        import nibabel as nib
        from nilearn.datasets import fetch_haxby
        from nilearn.masking import compute_epi_mask, apply_mask
        from nilearn.image import index_img, mean_img
    except ImportError:
        raise ImportError(
            "Haxby loader requires nilearn and nibabel. "
            "Install them with: pip install nilearn nibabel"
        )

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

    # --- Preprocessing: remove scanner drift & global signal ---
    # 1) Linear detrend per voxel (removes slow drift)
    t_axis = np.arange(T, dtype=np.float32)
    t_mean = t_axis.mean()
    t_centered = t_axis - t_mean
    t_var = (t_centered ** 2).sum()
    for j in range(V):
        col = X_tv[:, j]
        slope = (t_centered * (col - col.mean())).sum() / (t_var + 1e-12)
        X_tv[:, j] = col - slope * t_centered - col.mean()

    # 2) High-pass: remove low-frequency drift via DCT basis (cutoff ~0.01 Hz)
    from scipy.fft import dct, idct
    cutoff_period_trs = 40
    n_remove = max(1, T // cutoff_period_trs)
    for j in range(V):
        coeffs = dct(X_tv[:, j], type=2, norm="ortho")
        coeffs[:n_remove] = 0.0
        X_tv[:, j] = idct(coeffs, type=2, norm="ortho")

    # 3) Regress out global mean signal (shared scanner/physio noise)
    global_mean = X_tv.mean(axis=1, keepdims=True)
    gm_var = (global_mean ** 2).sum() + 1e-12
    for j in range(V):
        col = X_tv[:, j]
        beta = (col * global_mean.ravel()).sum() / gm_var
        X_tv[:, j] = col - beta * global_mean.ravel()

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
    """Load Kay/Vim-1 natural images fMRI dataset.

    Returns (n_voxels, p_stimuli) ≈ (8428, 1870). Concatenates training
    (1750 stimuli) and test (120 stimuli) responses. Rows = voxels.
    """
    npz_url = "https://osf.io/ymnjv/download"
    npz_path = "data/raw/kay_images.npz"
    download_if_needed(npz_url, npz_path, timeout=300)

    with np.load(npz_path) as dobj:
        dat = dict(**dobj)

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


def load_cifar10_data(
    *,
    visualize: bool = False,
    max_train_batches: int = 2,
    include_test: bool = True,
    max_images: Optional[int] = None,
) -> np.ndarray:
    """Load CIFAR-10 image dataset as flattened pixel vectors.

    Returns (n_images, p_pixels) ≈ (30000, 3072) where 3072 = 32×32×3 RGB.
    Pixel values normalized to [0, 1]. Uses first 2 training batches + test
    batch by default.
    """
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = "data/raw/cifar-10-python.tar.gz"
    extract_dir = Path("data/cache/cifar10_extracted")
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
