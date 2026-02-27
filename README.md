# Crud, 1/f spectra, and calibration for small-effect observational claims

Code and paper for studying background correlations ("crud") in large observational datasets across biomedicine, neuroscience, psychology, genomics, politics, and vision.

## Structure

```
paper/          LaTeX source for the manuscript
crud/           Python package (loaders, analysis, crud-aware tests)
scripts/        Entry points for reproducing results
  run_analysis.py           Full analysis pipeline (all 9 datasets)
  generate_paper_figures.py Reproduce the 4 main-text figures
  assumption_audit.py       Appendix B diagnostics
  crud_test.py              CLI wrapper for crud-aware z-test
webapp/         Static interactive webapp (open webapp/index.html)
data/           Downloaded datasets and caches (gitignored)
  raw/          Original downloads
  cache/        Processed matrices and intermediate files
```

## Webapp

Live demo: https://koerding.github.io/crud/

Local demo: open `webapp/index.html` in a browser.

## Links

Repository: https://github.com/koerding/crud

Paper (draft PDF): https://raw.githubusercontent.com/koerding/crud/master/paper/crud.pdf

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline (downloads data on first run)
python scripts/run_analysis.py

# Regenerate paper figures from cached data
python scripts/generate_paper_figures.py

# Compile the paper
cd paper && pdflatex crud && bibtex crud && pdflatex crud && pdflatex crud
```

## Crud-aware z-test

Given a sample correlation `r`, sample size `n`, and domain crud scale `sigma_crud`:

```python
from crud.crud_test import crud_z_test
result = crud_z_test(r=0.15, n=200, sigma_crud=0.15)
print(result)  # z_crud=1.0, p_crud=0.37
```

Table 1 in the paper reports `sigma_crud` values for nine domains at various levels of PC adjustment.
