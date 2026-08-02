#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_software_versions.py

Collects the exact version of every package used across the CXCR3/zinc
pipeline (fig1-fig4, FigS4-S7, run_all_zinc_analyses.py, stats_utils.py),
plus the Python interpreter itself, and prints a clean, copy-pasteable
block for the manuscript's Methods/Software section.

Run this in the SAME environment you actually used to produce the results
(e.g. the Spyder spyder-runtime environment) -- running it anywhere else
(including this sandbox) would report the wrong versions.

Usage:
    python get_software_versions.py
or, in the Spyder/IPython console:
    %run get_software_versions.py
"""

import sys
import platform
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Every package actually imported somewhere in the pipeline. Grouped by
# role so the printed output doubles as a quick sanity check that nothing
# was missed -- add a line here if a future script imports something new.
PACKAGES = {
    "Core scientific stack": [
        "numpy",
        "pandas",
        "scipy",
    ],
    "Single-cell / genomics": [
        "scanpy",
        "anndata",
        "harmonypy",
    ],
    "Statistics": [
        "statsmodels",
    ],
    "Plotting": [
        "matplotlib",
        "seaborn",
    ],
    "I/O": [
        "yaml",       # PyYAML -- import name differs from package name
        "h5py",       # anndata's underlying .h5ad backend
    ],
}

# import name -> PyPI package name, only where they differ (for the report)
PYPI_NAME = {
    "yaml": "PyYAML",
}


def get_version(import_name: str):
    """Import a package and return its __version__, or an error string."""
    try:
        module = __import__(import_name)
    except ImportError as e:
        return None, f"NOT INSTALLED ({e})"
    for attr in ("__version__", "version"):
        v = getattr(module, attr, None)
        if v is not None:
            return str(v), None
    # fall back to importlib.metadata (works even if the module itself
    # doesn't expose __version__, e.g. some minimal packages)
    try:
        from importlib.metadata import version as _pkg_version
        return _pkg_version(import_name), None
    except Exception:
        return None, "version not found (checked __version__ and importlib.metadata)"


def main() -> None:
    print("=" * 70)
    print("SOFTWARE VERSIONS -- run in the SAME environment used for analysis")
    print("=" * 70)
    print()
    print(f"Python: {platform.python_version()}  ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print()

    results = {}
    for group, import_names in PACKAGES.items():
        print(f"-- {group} --")
        for import_name in import_names:
            display_name = PYPI_NAME.get(import_name, import_name)
            version, error = get_version(import_name)
            if version is not None:
                print(f"  {display_name}: {version}")
                results[display_name] = version
            else:
                print(f"  {display_name}: {error}")
                results[display_name] = None
        print()

    # ---- Manuscript-ready copy-paste block -----------------------------
    print("=" * 70)
    print("COPY-PASTE BLOCK FOR MANUSCRIPT (edit wording as needed):")
    print("=" * 70)
    print()
    ordered_display_names = [
        PYPI_NAME.get(n, n) for names in PACKAGES.values() for n in names
    ]
    parts = []
    for name in ordered_display_names:
        v = results.get(name)
        if v:
            parts.append(f"{name} {v}")
    sentence = (
        f"All analyses were performed in Python {platform.python_version()} "
        f"using {', '.join(parts[:-1])}, and {parts[-1]}."
    )
    print(sentence)
    print()

    # ---- Markdown table, for Supplementary Methods ----------------------
    print("Markdown table version (for Supplementary Methods):")
    print()
    print("| Package | Version |")
    print("|---|---|")
    print(f"| Python | {platform.python_version()} |")
    for name in ordered_display_names:
        v = results.get(name)
        print(f"| {name} | {v if v else 'NOT FOUND -- check environment'} |")
    print()

    missing = [name for name in ordered_display_names if not results.get(name)]
    if missing:
        print("!" * 70)
        print(f"WARNING: could not find a version for: {', '.join(missing)}")
        print("These packages may not be installed in THIS environment, or may")
        print("be imported under a different name than listed above. Check")
        print("manually with e.g. `import <name>; <name>.__version__`.")
        print("!" * 70)


if __name__ == "__main__":
    main()