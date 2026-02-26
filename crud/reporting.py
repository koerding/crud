"""Artifact saving and HTML gallery generation.

Provides two main utilities:
- save_artifacts: saves an analysis checkpoint as a JSON file containing
  configuration and per-dataset summary statistics, and bundles all saved
  figures into a compressed tar.gz archive.
- build_html_gallery: compiles all figures as base64-embedded images plus
  optional summary tables into a single self-contained HTML file that can
  be viewed in any browser without external dependencies.
"""

import json
import tarfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from crud.config import (
    AUTOSAVE_FIGS, FIG_DIR, FIG_FORMAT, SEED,
    MAX_ROWS_ANALYSIS, MAX_PCS_FIT, MAX_K_STORE, MAX_COLS_CORR, KS,
)


def save_artifacts(all_results: Dict[str, dict]) -> None:
    """Save analysis checkpoint as JSON and bundle figures into tar.gz.

    The JSON checkpoint stores the global configuration (SEED, KS, size caps,
    etc.) and per-dataset summaries (dimensions, z-score diagnostics, explained
    variance ratios, residual correlation SD by K). This allows resuming or
    reviewing analyses without re-running the full pipeline.

    The tar.gz archive packages the entire figures directory for easy
    distribution alongside the HTML gallery.
    """
    # --- Bundle figures into a compressed tar archive ---
    if AUTOSAVE_FIGS and FIG_DIR.exists():
        out_tgz = "data/cache/figures_vector.tar.gz"
        with tarfile.open(out_tgz, "w:gz") as tar:
            tar.add(str(FIG_DIR), arcname=FIG_DIR.name)
        print("Wrote", out_tgz)
    else:
        print("No figures directory to bundle (or autosave disabled).")

    # --- Write JSON checkpoint with config + per-dataset summaries ---
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


def build_html_gallery(
    fig_dir: Path = FIG_DIR,
    out_path: str = "figures.html",
    tables: Optional[List[dict]] = None,
) -> None:
    """Compile all figures and summary tables into a single self-contained HTML file.

    Each figure is read from disk and base64-encoded directly into an <img> tag,
    so the resulting HTML has no external dependencies and can be opened in any
    browser. Optional *tables* are rendered as HTML <table> elements below the
    figures section.

    Parameters
    ----------
    fig_dir : Path
        Directory containing saved figure files (e.g., PNG images).
    out_path : str
        Output path for the generated HTML file.
    tables : list of dict, optional
        Each dict has keys "title" (str), "headers" (list of str), and
        "rows" (list of list) for rendering an HTML table.
    """
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
