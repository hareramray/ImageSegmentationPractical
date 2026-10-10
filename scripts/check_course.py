"""Validate notebook structure, then optionally execute each in a fresh kernel.

Run with the project interpreter. Outputs go to reports/executed by default.
--in-place also saves checked outputs into the learning notebooks themselves.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--pattern", default="*.ipynb")
    parser.add_argument("--timeout", default=240, type=int, help="Timeout per code cell, seconds")
    parser.add_argument("--report", default=None, help="Report stem inside reports/ (default: validation or structure)")
    args = parser.parse_args()
    report_name = args.report or ("validation" if args.execute else "structure")
    paths = sorted((ROOT / "notebooks").glob(args.pattern))
    if not paths:
        parser.error("No notebooks match the pattern")
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    executed_dir = report_dir / "executed"
    executed_dir.mkdir(exist_ok=True)
    # Reduce thread oversubscription in small classroom workloads.
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    results = []
    for path in paths:
        start = time.perf_counter()
        record = {"notebook": path.name, "status": "pending"}
        notebook = None
        try:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            code_cells = [c for c in notebook.cells if c.cell_type == "code"]
            record["code_cells"] = len(code_cells)
            for idx, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    compile(cell.source, f"{path.name}:cell{idx}", "exec")
            if args.execute:
                # Every notebook receives its own brand-new Python process.
                for cell in code_cells:
                    cell.outputs = []
                    cell.execution_count = None
                NotebookClient(notebook, timeout=args.timeout, kernel_name="image-segmentation",
                               resources={"metadata": {"path": str(path.parent)}},
                               allow_errors=False).execute()
                record["executed_cells"] = sum(c.execution_count is not None for c in code_cells)
                record["figures"] = sum("image/png" in o.get("data", {})
                                         for c in code_cells for o in c.outputs)
                if record["executed_cells"] != len(code_cells):
                    raise AssertionError("Some code cells did not execute")
                nbformat.write(notebook, executed_dir / path.name)
                if args.in_place:
                    nbformat.write(notebook, path)
                record["status"] = "passed"
            else:
                record["status"] = "structure passed (not executed)"
        except Exception as exc:
            record["status"] = "FAILED"
            record["error"] = str(exc)
            if notebook is not None:
                nbformat.write(notebook, executed_dir / path.name)
        record["seconds"] = round(time.perf_counter() - start, 2)
        results.append(record)
        print(f"{record['status']}: {path.name} ({record['seconds']}s)", flush=True)
        if "error" in record:
            print(record["error"], flush=True)
    versions = {"python": platform.python_version()}
    for package in ["numpy", "scipy", "matplotlib", "scikit-image", "scikit-learn", "pandas",
                    "Pillow", "torch", "torchvision", "jupyterlab", "nbformat", "nbclient"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    compute = {"platform": platform.system()}
    try:
        import torch
        compute["cuda_available"] = torch.cuda.is_available()
        compute["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        compute["cuda_available"] = False
    report = {"mode": "execute" if args.execute else "structure", "versions": versions,
              "compute": compute,
              "optional_downloads": "Not enabled by this runner; defaults only", "results": results}
    (report_dir / f"{report_name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Notebook validation", "", f"Mode: **{report['mode']}**.", "",
             "Each executed notebook used a fresh kernel. Default settings only; optional Oxford",
             "Pet downloads and pretrained transfer-learning runs were not enabled.", "",
             "| Notebook | Status | Seconds | Code cells | Figures |", "|---|---|---:|---:|---:|"]
    for row in results:
        lines.append(f"| [{row['notebook']}](../notebooks/{row['notebook']}) | {row['status']} | "
                     f"{row['seconds']} | {row.get('code_cells', 0)} | {row.get('figures', '—')} |")
    lines += ["", "## Environment", "", "```json", json.dumps({**compute, **versions}, indent=2), "```", ""]
    (report_dir / f"{report_name}.md").write_text("\n".join(lines), encoding="utf-8")
    return int(any(r["status"] == "FAILED" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
