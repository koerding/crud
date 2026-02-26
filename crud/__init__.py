"""
crud -- universal data properties for causal inference.

The "crud" package implements the empirical and theoretical tools described in
the paper.  Its core contribution is showing that power-law eigenvalue spectra
-- observed universally across medical, neuroscience, social-science, and image
datasets -- create persistent background correlations (Correlations from
Unobserved Dimensions, or "crud") that fundamentally limit what association-only
causal inference can achieve.

Key exports
-----------
crud_test : function
    Permutation-based empirical calibration test.  Given a dataset and a set
    of variable pairs, it checks whether observed partial correlations (after
    removing K principal components) are distinguishable from the background
    crud distribution predicted by the eigenvalue spectrum.
crud_z_test : function
    Parametric (z-score) variant of the crud test using the spectral
    sigma_K formula from Theorem A.1 ("Crud scale under top-K PC
    removal"), avoiding the need for permutation resampling.
CrudTestResult, CrudZResult : namedtuples
    Structured result containers for the two tests above.
analyze_dataset : function
    Full per-dataset analysis pipeline: PCA spectrum fitting, residual
    correlation analysis, sigma_K prediction, and diagnostic plots.
"""

from crud.crud_test import crud_test, crud_z_test, CrudTestResult, CrudZResult
from crud.analysis import analyze_dataset

__all__ = [
    "crud_test",
    "crud_z_test",
    "CrudTestResult",
    "CrudZResult",
    "analyze_dataset",
]
