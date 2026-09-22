#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py
==========
One entry point for the whole CXCR3/zinc analysis.

  Stage 0  pre-flight checks on the per-sample .h5ad files and the config.
           These catch the classes of problem that have already bitten this
           project once each, and that are invisible once the data are merged:
             - lost gene names (CSF_10 had positional var_names, so every
               symbol lookup silently failed)
             - duplicated input matrices (four samples shared two files)
             - tissue labels contradicting the filename
             - datasets.yaml registering samples that do not exist, or
               missing samples that do
           Any failure aborts before a single figure is produced.

  Stage 1  every analysis script, in dependency order, from one merged object.

It writes output/run_manifest.json recording the git commit, package versions,
pre-flight results and per-step status, so you can state that all reported
values came from one run of one commit and show it.

Usage
-----
    python run_all.py --checks-only      # stage 0 only
    python run_all.py --list             # show the steps
    python run_all.py --dry-run
    python run_all.py                    # checks, then everything
    python run_all.py --skip-checks      # pipeline only
    python run_all.py --keep-going       # do not stop on a failed step
    python run_all.py --from fig2        # resume
    python run_all.py --only fig2 FigS6
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
STEPS = [
    dict(key="build",       path="run_all_zinc_analyses.py",
         note="loads datasets, harmonises, writes merged_bcells.h5ad"),
    dict(key="counts",      path="cell_numbers.py",
         note="per-sample counts + supplement_per_patient_counts.csv (Table S1)"),
    dict(key="qc_pipeline", path="output/qc/qc_pipeline.py",
         note="per-sample QC, cluster stability, metadata audit, QC plots"),
    dict(key="qc",          path="output/qc/qc.py",
         note="cohort-level QC figures from the merged object"),
    dict(key="fig1",        path="output/figures/fig1_umap_overview/plot_figure1_umap_bar.py",
         note="Figure 1"),
    dict(key="fig2",        path="output/figures/fig2_cxcr3_zinc/fig2.py",
         note="Figure 2"),
    dict(key="fig3",        path="output/figures/fig3_csf_vs_pb/fig3.py",
         note="Figure 3"),
    dict(key="fig4",        path="output/figures/fig4_patient_level/Fig4.py",
         note="Figure 4"),
    dict(key="FigS4",       path="output/figures/supp/FigS4.py", note="Figure S4"),
    dict(key="FigS5",       path="output/figures/supp/FigS5.py", note="Figure S5"),
    dict(key="FigS6",       path="output/figures/supp/FigS6.py", note="Figure S6"),
    dict(key="FigS7",       path="output/figures/supp/FigS7.py", note="Figure S7"),
    dict(key="sexcheck",    path="sex_stratified_sensitivity_check.py",
         note="sex-stratified sensitivity analysis"),
    dict(key="zncheck",     path="checkZnExpression.py",
         note="zinc expression diagnostics (Table S2 source)"),
    dict(key="geneset",     path="geneset.py", optional=True,
         note="exploratory marker screen (not in the manuscript)"),
]

VERSION_PKGS = ["numpy", "pandas", "scipy", "scanpy", "anndata", "statsmodels",
                "harmonypy", "matplotlib", "seaborn", "h5py", "sklearn", "yaml"]

# a gene every B-cell sample must carry, used to prove symbols survived
SENTINEL_GENE = "MS4A1"


# ---------------------------------------------------------------------------
def repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def git_info(root: str) -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=root, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    dirty = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "describe": run("describe", "--tags", "--always", "--dirty"),
            "uncommitted_changes": bool(dirty)}


def package_versions() -> dict:
    from importlib import metadata
    dist = {"sklearn": "scikit-learn", "yaml": "PyYAML"}
    out = {"python": sys.version.split()[0]}
    for n in VERSION_PKGS:
        try:
            out[n] = metadata.version(dist.get(n, n))
        except Exception:
            out[n] = "not installed"
    return out


# ---------------------------------------------------------------------------
# Stage 0
# ---------------------------------------------------------------------------
def file_md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:12]


def preflight(root: str) -> dict:
    import anndata as ad

    proc = os.path.join(root, "data", "processed")
    files = sorted(glob.glob(os.path.join(proc, "*", "*.h5ad")))
    # Same exclusions as qc_pipeline.discover_samples(): combined objects and
    # files marked bad are not per-sample matrices.
    def is_sample(p):
        stem = os.path.splitext(os.path.basename(p))[0].lower()
        return "processed" not in stem and "bad" not in stem
    
    skipped = [os.path.basename(p) for p in files if not is_sample(p)]
    files = [p for p in files if is_sample(p)]
    if skipped:
        print(f"  [skip] not per-sample objects: {', '.join(skipped)}")
    problems, rows = [], []

    print("=" * 78)
    print("STAGE 0  pre-flight checks")
    print("=" * 78)
    if not files:
        problems.append(f"no .h5ad files under {proc}")
        print(f"  [FAIL] {problems[-1]}")
        return {"ok": False, "problems": problems, "samples": []}

    print(f"  {len(files)} sample files under data/processed\n")
    by_md5 = defaultdict(list)

    for f in files:
        sid = os.path.splitext(os.path.basename(f))[0]
        ds = os.path.basename(os.path.dirname(f))
        try:
            a = ad.read_h5ad(f)
        except Exception as e:
            problems.append(f"{sid}: unreadable ({e})")
            print(f"  [FAIL] {sid}: unreadable")
            continue

        names_ok = SENTINEL_GENE in a.var_names
        n_mt = sum(str(g).upper().startswith("MT-") for g in a.var_names)
        uniq = bool(a.var_names.is_unique)
        tissue = "<missing>"
        if "tissue" in a.obs.columns:
            import pandas as pd
            vals = pd.unique(a.obs["tissue"].dropna().astype(str))
            tissue = "|".join(sorted(map(str, vals))) if len(vals) else "<empty>"
        expect = "CSF" if sid.upper().startswith("CSF") or sid.upper().endswith("_CSF") else "PB"
        md5 = file_md5(f)
        by_md5[md5].append(sid)

        if not names_ok:
            problems.append(f"{sid}: gene symbols missing from var_names "
                            f"(first: {list(a.var_names[:3])})")
        if n_mt == 0:
            problems.append(f"{sid}: no MT- genes found — gene names likely wrong")
        if not uniq:
            problems.append(f"{sid}: var_names not unique (var_names_make_unique not applied)")
        if tissue != expect:
            problems.append(f"{sid}: tissue is {tissue!r} but filename implies {expect!r}")

        rows.append(dict(sample=sid, dataset=ds, n_obs=int(a.n_obs), n_vars=int(a.n_vars),
                         names_ok=names_ok, n_mt=n_mt, tissue=tissue, md5=md5))
        flag = "ok  " if (names_ok and n_mt and uniq and tissue == expect) else "FAIL"
        print(f"  [{flag}] {sid:22s} {a.n_obs:6d} cells  {a.n_vars:6d} genes  "
              f"MT={n_mt:3d}  tissue={tissue}")
        del a

    for md5, ids in by_md5.items():
        if len(ids) > 1:
            problems.append(f"duplicate matrices ({md5}): {', '.join(ids)}")

    # config vs disk
    cfg_path = os.path.join(root, "config", "datasets.yaml")
    if os.path.exists(cfg_path):
        try:
            import yaml
            cfg = yaml.safe_load(open(cfg_path))
            on_disk = {r["sample"] for r in rows}
            listed = set()
            def walk(node):
                if isinstance(node, dict):
                    for k, v in node.items():
                        if isinstance(v, dict) and "condition" in v:
                            listed.add(str(k))
                        walk(v)
            walk(cfg)
            if listed:
                ghost = sorted(listed - on_disk)
                unreg = sorted(on_disk - listed)
                if ghost:
                    problems.append(f"in datasets.yaml but not on disk: {ghost}")
                if unreg:
                    print(f"\n  [note] on disk but not registered: {unreg}")
        except Exception as e:
            print(f"\n  [note] could not parse datasets.yaml ({e})")

    print()
    if problems:
        print(f"  {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  all pre-flight checks passed")
    print()
    return {"ok": not problems, "problems": problems, "samples": rows}


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------
def run_step(step, root, log_dir, i, total) -> dict:
    name, rel = step["key"], step["path"]
    abspath = os.path.join(root, rel)
    log_path = os.path.join(log_dir, f"{i:02d}_{name}.log")
    rel_log = os.path.relpath(log_path, root)
    print("=" * 78)
    print(f"[{i}/{total}] {name}  ({rel})")
    print("=" * 78, flush=True)

    if not os.path.exists(abspath):
        print(f"  !! not found: {rel}\n", flush=True)
        return dict(step=name, path=rel, status="missing", seconds=0.0, log=rel_log)

    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLBACKEND", "Agg")

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# {name} ({rel})\n# {datetime.now().isoformat()}\n\n")
        proc = subprocess.Popen([sys.executable, abspath], cwd=root, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write("   " + line)
            log.write(line)
        proc.wait()
    dt = time.time() - t0
    status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"   -> {status} in {dt:.1f}s   log: {rel_log}\n", flush=True)
    return dict(step=name, path=rel, status=status, seconds=round(dt, 1),
                returncode=proc.returncode, log=rel_log)


# ---------------------------------------------------------------------------
def main() -> int:
    root = repo_root()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--checks-only", action="store_true")
    ap.add_argument("--skip-checks", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--from", dest="start", metavar="STEP")
    ap.add_argument("--only", nargs="+", metavar="STEP")
    ap.add_argument("--skip", nargs="+", metavar="STEP", default=[])
    ap.add_argument("--include-optional", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"repo root: {root}\n")
        for i, s in enumerate(STEPS, 1):
            tag = "  [optional]" if s.get("optional") else ""
            print(f"  {i:2d}. {s['key']:<12s} {s['path']}{tag}\n      {s['note']}")
        return 0

    keys = [s["key"].lower() for s in STEPS]

    def resolve(n):
        if n.lower() not in keys:
            sys.exit(f"[fatal] unknown step {n!r}. Known: {', '.join(k['key'] for k in STEPS)}")
        return n.lower()

    selected = list(STEPS)
    if args.only:
        want = {resolve(n) for n in args.only}
        selected = [s for s in selected if s["key"].lower() in want]
    else:
        if args.start:
            selected = STEPS[keys.index(resolve(args.start)):]
        if not args.include_optional:
            selected = [s for s in selected if not s.get("optional")]
    if args.skip:
        drop = {resolve(n) for n in args.skip}
        selected = [s for s in selected if s["key"].lower() not in drop]

    git, pkgs = git_info(root), package_versions()
    print(f"repo root : {root}")
    print(f"git       : {git['describe'] or '(no git)'}"
          f"{'  [UNCOMMITTED CHANGES]' if git['uncommitted_changes'] else ''}")
    print(f"python    : {pkgs['python']}  pandas {pkgs.get('pandas')}  "
          f"scanpy {pkgs.get('scanpy')}  anndata {pkgs.get('anndata')}\n")
    if git["uncommitted_changes"]:
        print("[warn] working tree is dirty; the manifest will name a commit that is")
        print("       not exactly what ran. Commit first if you will cite this run.\n")

    if args.dry_run:
        if not args.skip_checks:
            print("  would run stage 0 pre-flight checks")
        for i, s in enumerate(selected, 1):
            print(f"  would run [{i}/{len(selected)}] {s['key']}: {s['path']}")
        return 0

    checks = None
    if not args.skip_checks:
        checks = preflight(root)
        if not checks["ok"]:
            print("[stop] pre-flight failed. Fix the problems above, or rerun with")
            print("       --skip-checks if you understand why they are acceptable.")
            return 1
    if args.checks_only:
        return 0 if (checks is None or checks["ok"]) else 1

    log_dir = os.path.join(root, "output", "logs")
    os.makedirs(log_dir, exist_ok=True)
    started, results, failed = datetime.now(timezone.utc), [], False
    t_all = time.time()

    for i, s in enumerate(selected, 1):
        r = run_step(s, root, log_dir, i, len(selected))
        results.append(r)
        if r["status"] != "ok":
            failed = True
            if not args.keep_going:
                print(f"[stop] {s['key']} failed. Resume with --from {s['key']} "
                      f"--skip-checks, or use --keep-going.\n")
                break

    total_s = time.time() - t_all
    manifest = dict(started_utc=started.isoformat(),
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    total_seconds=round(total_s, 1), git=git, versions=pkgs,
                    command=" ".join(sys.argv), preflight=checks,
                    steps=results, all_ok=not failed)
    mpath = os.path.join(root, "output", "run_manifest.json")
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 78 + "\nSUMMARY\n" + "=" * 78)
    for r in results:
        print(f"  {r['step']:<12s} {r['status']:<20s} {r['seconds']:>7.1f}s   {r['log']}")
    print(f"\n  total {total_s/60:.1f} min\n  manifest: {os.path.relpath(mpath, root)}")
    if failed:
        print("\n  [!] a step failed — do not cite this run.")
    else:
        print("\n  all steps completed. Tag before quoting any numbers:")
        print("      git tag corrigendum-v1 && git push --tags")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())