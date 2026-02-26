"""Package-wide constants and figure auto-save setup.

This module centralizes every tuneable parameter for the analysis
pipeline -- sample-size caps, PCA limits, histogram bins, output paths,
etc. -- so that notebooks and scripts import a single source of truth.

It also monkey-patches ``matplotlib.pyplot.show`` at import time (when
AUTOSAVE_FIGS is True) so that every figure displayed during an
interactive session is also saved to disk as a numbered PNG.  This is
useful for reproducibility: the exact figures that appear in a notebook
are persisted without explicit ``savefig`` calls.
"""

from pathlib import Path

# Force the non-interactive Agg backend so figure rendering works in
# headless environments (CI, remote servers) without an X display.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Numerical floor added to denominators (standard deviations, total
# variance, etc.) to prevent division by zero.
EPS = 1e-12

# Global random seed for reproducibility of randomised SVD and any
# random sub-sampling.
SEED = 42

# Maximum number of samples (rows) used in the main analysis loop.
# Datasets larger than this are randomly sub-sampled to keep runtime
# and memory manageable.
MAX_ROWS_ANALYSIS = 10_000

# Maximum number of principal components to compute via randomised SVD
# in the main analysis.  The assumption_audit script uses a separate
# cap of min(300, n, p), matching the paper's claim of fitting "up to
# 300 components".  This higher limit (500) allows the main pipeline to
# capture more of the spectrum, but only the top MAX_K_STORE components
# are retained for residualisation.
MAX_PCS_FIT = 500

# Number of top PC loading vectors actually stored and used later for
# residualisation (i.e., computing residual correlations after removing
# K PCs).  Keeping 100 is sufficient because KS goes up to 50 at most,
# and storing all 500 would waste memory.
MAX_K_STORE = 100

# Maximum number of features (columns) included in pairwise-correlation
# analyses.  When p > MAX_COLS_CORR a random subset of columns is used,
# because the p x p correlation matrix would otherwise be too large.
MAX_COLS_CORR = 500

# The set of K values at which residual correlations are computed.
# For each K in KS, the top K principal components are projected out and
# the distribution of off-diagonal correlations in the residual is
# recorded.  K = 0 means raw (no PCs removed); increasing K shows how
# the "crud" correlation mass shrinks as dominant latent factors are
# removed.
KS = [0, 1, 2, 5, 10, 20, 50]

# Number of histogram bins for plotting raw and residual correlation
# distributions, respectively.
BINS_CORR = 100
BINS_RESID = 100

# --- Figure and cache output paths ----------------------------------------

# When True, every call to plt.show() also saves the current figure to
# disk (see install_autosave_show below).
AUTOSAVE_FIGS = True

# Directory where auto-saved figures are written.
FIG_DIR = Path("figures")

# Image format for saved figures (e.g. "png", "pdf", "svg").
FIG_FORMAT = "png"

# Directory for .npy caches of downloaded / preprocessed data matrices,
# managed by utils.cached_load().
CACHE_DIR = Path("data/cache/cached_data")

# ---------------------------------------------------------------------------
# Auto-save every plt.show() to disk
# ---------------------------------------------------------------------------

# Module-level counter that persists across calls; each plt.show()
# increments it so figures are saved as fig_001.png, fig_002.png, etc.
_fig_counter = 0


def install_autosave_show(fig_dir: Path, fmt: str = "png") -> None:
    """Monkey-patch ``plt.show`` so that it saves the current figure first.

    After patching, every call to ``plt.show()`` will:
      1. Increment a global counter.
      2. Save the current figure to *fig_dir*/fig_NNN.<fmt>.
      3. Delegate to the original ``plt.show`` for normal display.

    The patch is idempotent: if ``plt.show`` has already been replaced
    (detected by checking ``__name__``), it prints a notice and returns.

    Parameters
    ----------
    fig_dir : Path
        Directory to write figure files into (created if absent).
    fmt : str
        Image format string passed to ``fig.savefig``.
    """
    global _fig_counter
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Guard against double-wrapping (e.g. when the module is reloaded
    # or imported from multiple entry-points).
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


# --- Module-level side effect ---------------------------------------------
# When this module is first imported (which happens on any
# ``from crud.config import ...``), the auto-save hook is installed
# immediately so that all downstream figure-producing code benefits
# without any explicit setup.
if AUTOSAVE_FIGS:
    install_autosave_show(FIG_DIR, FIG_FORMAT)
