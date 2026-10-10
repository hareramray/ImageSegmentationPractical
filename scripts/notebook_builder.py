"""Small standard-library notebook authoring helpers; no generated cell magic."""
import hashlib
import json
import textwrap
from pathlib import Path


def _source(text):
    return textwrap.dedent(text).strip() + "\n"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _source(text)}


def bootstrap():
    return code('''
        from pathlib import Path
        import sys
        PROJECT_ROOT = next(
            (p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "learning" / "data.py").is_file()), None
        )
        if PROJECT_ROOT is None:
            raise FileNotFoundError("Open this notebook inside the complete ImageSegmentationPractical folder.")
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        import numpy as np
        import matplotlib.pyplot as plt
        plt.rcParams.update({"figure.figsize": (10, 3.5), "image.cmap": "gray",
                             "axes.spines.top": False, "axes.spines.right": False})
        print("Project:", PROJECT_ROOT.name)
    ''')


def write_notebook(path, cells, title):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for index, cell in enumerate(cells):
        cell["id"] = hashlib.sha256(f"{path.name}:{index}:{cell['source']}".encode()).hexdigest()[:12]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Image Segmentation (.venv)", "language": "python", "name": "image-segmentation"},
            "language_info": {"name": "python", "version": "3.13"},
            "title": title,
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path.as_posix()} ({len(cells)} cells)")
