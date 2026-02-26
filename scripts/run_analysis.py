"""
Main entry point for the full analysis pipeline.

Loads real datasets (medical, neuroscience, social science, images),
computes PCA spectra, correlation structure, residual correlations,
and produces cross-dataset summary plots.

The pipeline runs in six phases:
  Phase 1 -- Load and analyze each of the 9 datasets individually
             (spectrum fit, residual correlations, sigma_K prediction).
  Phase 2 -- Cross-dataset summary figures (Figures 2-3 in the paper):
             eigenvalue spectra overlay, sigma_K comparison, etc.
  Phase 3 -- Heavy-tail robustness checks (paper Appendix): verifies that
             conclusions hold when data are transformed to heavier tails.
  Phase 4 -- Alternative adjustment comparison (paper Appendix): compares
             PCA-based adjustment to ridge, PEER, and SVA methods.
  Phase 5 -- Null-model comparisons: tests whether Wishart / MP null models
             reproduce the observed spectral and correlation structure.
  Phase 6 -- Save artifacts (numpy results dict) and build an HTML gallery
             with all figures and summary tables.

Usage:
    python run_analysis.py
"""

import gc
import os, sys
from typing import Dict, Callable

import numpy as np

# Ensure CWD is repo root so relative data paths resolve correctly.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

from crud.config import SEED
from crud.utils import cached_load
from crud.loaders import (
    load_kay_fmri_data,
    load_haxby_data,
    load_nhanes_data,
    load_cifar10_data,
    load_precinct_data,
    load_rnaseq_data,
    load_gtex_data,
    load_hexaco_data,
    load_stringer_data,
)
from crud.analysis import analyze_dataset, plot_cross_dataset_summaries
from crud.heavy_tails import run_heavy_tail_stress_test
from crud.adjustments import run_alternative_adjustments
from crud.null_models import run_null_model_comparisons
from crud.reporting import save_artifacts, build_html_gallery


def main():
    dataset_loaders: Dict[str, Callable[..., np.ndarray]] = {
        "Kay_fMRI": load_kay_fmri_data,
        "Haxby_fMRI": load_haxby_data,
        "NHANES": load_nhanes_data,
        "CIFAR10": load_cifar10_data,
        "Precinct": load_precinct_data,
        "RNASeq": load_rnaseq_data,
        "GTEx": load_gtex_data,
        "HEXACO": load_hexaco_data,
        "Stringer": load_stringer_data,
    }

    # --- Phase 1: Load and analyze each dataset ---
    # Each dataset is loaded via cached_load (downloads once, then serves
    # from disk), analysed with the full per-dataset pipeline, and freed
    # immediately to keep memory usage bounded.
    all_results: Dict[str, dict] = {}
    for name, loader in dataset_loaders.items():
        print(f"\n=== Loading dataset: {name} ===")
        X = cached_load(name, loader, visualize=True)
        all_results[name] = analyze_dataset(X, name=name, seed=SEED)
        print(f"=== Done: {name} ===")
        del X; gc.collect()

    print("\nAll datasets processed:", list(all_results.keys()))

    # --- Phase 2: Cross-dataset summary figures (Figures 2-3 in paper) ---
    tables = plot_cross_dataset_summaries(all_results)

    # --- Phase 3: Heavy-tail stress test (paper Appendix) ---
    # Build a dict of cached loader callables for phases 3-5.  The lambda
    # uses default-argument binding (n=name, l=loader) so that each closure
    # captures the correct per-iteration values rather than all sharing the
    # final loop variable -- a classic Python closure-in-a-loop gotcha.
    cached_loaders = {name: (lambda n=name, l=loader, **kw: cached_load(n, l))
                      for name, loader in dataset_loaders.items()}
    tables += run_heavy_tail_stress_test(cached_loaders)

    # --- Phase 4: Alternative adjustment methods (paper Appendix) ---
    tables += run_alternative_adjustments(cached_loaders)

    # --- Phase 5: Null-model comparisons ---
    # Generates Wishart / Marchenko-Pastur null data and checks whether
    # the observed spectral and correlation patterns are merely artifacts
    # of high dimensionality.
    run_null_model_comparisons(cached_loaders)

    # --- Phase 6: Save artifacts + HTML gallery ---
    save_artifacts(all_results)
    build_html_gallery(tables=tables)


if __name__ == "__main__":
    main()
