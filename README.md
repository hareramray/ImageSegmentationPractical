# Image segmentation: learn by doing

A 13-notebook practical course for someone comfortable with Python and new to image segmentation. Start with masks and simple rules, build and evaluate learned models, then practice advanced training and inference workflows.

Every lesson has explanations before code, visual results, experiments, exercises with worked answers, common mistakes, and a completion checklist. Default examples use small generated or bundled images, so no dataset or model download is required. Lesson 10 provides an optional real-data and pretrained-model path. Building expertise also requires completing the exercises, working with real annotations, and evaluating on independent data.

## Start on this computer

The project environment has been created in `.venv`, JupyterLab is installed, and the **Image Segmentation (.venv)** kernel is registered within that environment. It reuses your existing Python 3.13 PyTorch installation through `--system-site-packages`; new scientific and Jupyter packages were installed inside the project environment. PyTorch detects your **NVIDIA GeForce RTX 5050 Laptop GPU**.

1. Double-click **[start_notebooks.cmd](start_notebooks.cmd)**, or run it from PowerShell:

   ```powershell
   .\start_notebooks.cmd
   ```

2. Open **[00_start_here.ipynb](notebooks/00_start_here.ipynb)** if it does not open automatically.
3. Select **Image Segmentation (.venv)** as the kernel. Run cells with **Shift+Enter**.
4. Continue in numeric order. Before finishing each lesson, choose **Kernel → Restart Kernel and Run All Cells**.

The launcher starts a local JupyterLab server and opens your browser. Keep its terminal open while using the notebooks; stop it with Ctrl+C when finished. The notebooks include saved outputs from the checked default runs so you can inspect the examples before rerunning them.

## Learning route

Study times include reading and experimentation; they are estimates, not execution timings. Plan extra sessions for the exercises and capstone.

| Notebook | Practical outcome | Study time |
|---|---|---|
| [00 · Start here](notebooks/00_start_here.ipynb) | Check the kernel/GPU, run a baseline, record an experiment | 25–40 min |
| [01 · Images, masks, and coordinates](notebooks/01_images_masks_coordinates.ipynb) | Inspect, resize, annotate, overlay, and export masks correctly | 45–60 min |
| [02 · Thresholding](notebooks/02_thresholding.ipynb) | Compare manual, Otsu, and local thresholds | 45–60 min |
| [03 · Morphology and components](notebooks/03_morphology_components.ipynb) | Clean masks, count objects, measure regions | 45–60 min |
| [04 · Edges and watershed](notebooks/04_edges_watershed.ipynb) | Split touching objects and diagnose seed failures | 60–75 min |
| [05 · Clustering and superpixels](notebooks/05_clustering_superpixels.ipynb) | Group pixels using color and spatial information | 60–75 min |
| [06 · Metrics and error analysis](notebooks/06_metrics_error_analysis.ipynb) | Calculate IoU/Dice, inspect boundaries, tune on validation data | 60–90 min |
| [07 · Classical machine learning](notebooks/07_classical_machine_learning.ipynb) | Train a pixel classifier and test domain shift | 60–90 min |
| [08 · PyTorch segmentation basics](notebooks/08_pytorch_segmentation_basics.ipynb) | Understand tensor contracts and train a segmentation network | 60–90 min |
| [09 · U-Net training](notebooks/09_unet_training.ipynb) | Train, validate, checkpoint, and reload a U-Net | 90–120 min |
| [10 · Real data and transfer learning](notebooks/10_real_data_transfer_learning.ipynb) | Validate image/mask pairs; optionally fine-tune on Oxford Pets | 90 min + optional training |
| [11 · Instances, panoptic labels, and prompting](notebooks/11_instance_panoptic_prompting.ipynb) | Match individual objects, measure instance errors, use point prompts | 90–120 min |
| [12 · Advanced training and deployment](notebooks/12_advanced_training_deployment.ipynb) | Compare losses, blend tiles, use TTA, inspect uncertainty, export predictions | 2–3 h + capstone |

**Suggested pace:** one or two lessons per week. Complete 00–03 for basic mask handling, 04–07 for classical methods and evaluation, and 08–12 for deep learning and advanced practice. Spend a separate session on each lesson's exercises. The final capstone should use a genuinely separate test set.

## How to learn from each notebook

1. Read the next explanation and predict the output before running the code.
2. Run the default example, inspect the plots, and explain the errors in your own words.
3. Change one parameter at a time and record what changed.
4. Try the exercises before expanding the worked answers.
5. Restart and run all cells, then complete the checklist.

Notebook 00 includes an experiment log you can adapt. Keep the random seed fixed when isolating a parameter effect; use several seeds when checking whether a conclusion is reliable. Use training data to fit a model, validation data to make decisions, and the final test set only after freezing those decisions. Group related images by patient, video, specimen, or scene before splitting.

## GPU and runtime settings

Deep learning lessons automatically select CUDA when available and fall back to CPU. `FAST_RUN=True` uses small data and short training for quick feedback. After the default works, follow each lesson's settings to increase training. A demonstration run is not a benchmark or a fully converged real-data model.

- If GPU memory runs out, reduce batch size or image size and restart the kernel.
- Keep `num_workers=0` for these Windows notebook examples.
- CUDA inference timings require synchronization; notebook 12 demonstrates it.
- A fixed seed improves repeatability but does not guarantee identical results across devices and library versions.
- Default toy images intentionally make some methods look strong. Shifted colors and real-data extensions expose their limitations.

Your existing `triton-windows` package may print a “Failed to find CUDA” **toolkit** warning when an optimizer is first created. These lessons use ordinary PyTorch operations, and actual CUDA computation is checked in lesson 00. A passing device check plus successful GPU training distinguishes that optional compiler warning from a GPU runtime failure; the notebooks do not use `torch.compile`.

## Recreate or repair the environment

On this computer, `.venv` already exists. If only the kernel entry is missing, run:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name image-segmentation --display-name "Image Segmentation (.venv)"
```

For a new machine, use Python 3.13 (the tested version) and run these commands from the project root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install a compatible pair of `torch` and `torchvision` using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) for your Windows/NVIDIA driver setup. Use `.\.venv\Scripts\python.exe -m pip` in place of the selector's `pip` or `pip3`. Then register the kernel using the command above. Do not reinstall a CPU-only build over a working CUDA build. The exact tested PyTorch and package versions are recorded in [reports/validation.md](reports/validation.md).

If you intentionally want to reuse an existing working PyTorch installation, create the environment with `py -3.13 -m venv --system-site-packages .venv` instead. This shares access to global packages, so it is less isolated; the initial setup on this computer uses that option to avoid another large PyTorch download.

You do not need to activate the environment or change PowerShell execution policy when using the explicit interpreter commands. The `python` command on this computer points to a Windows Store alias; the launcher and commands above use the actual project interpreter.

## Data and downloads

- `learning/data.py` generates labeled disks and rectangles with fixed random seeds. It uses no external data.
- Classical notebooks also use small images bundled with scikit-image. These images have no task-specific ground truth unless explicitly stated.
- Lesson 10 first creates a tiny on-disk dataset to exercise the complete loader without a download. It then offers explicitly enabled Oxford-IIIT Pet data and pretrained torchvision weights. Read its configuration cell and dataset terms before enabling those paths. They use network, disk space, and more training time.
- Oxford Pet trimaps have raw IDs 1 = foreground, 2 = background, 3 = unclassified. The optional pipeline remaps these to 1 = pet, 0 = background, 255 = ignored. Ignore pixels must be excluded from both loss and evaluation.
- Keep masks as integer IDs. Colored overlays are for display and must not become training labels.

## Files and verification

```text
notebooks/              The 13 lessons, in study order
learning/               Inspectable data and PyTorch helpers
scripts/                Notebook builders and execution checker
requirements.txt        Scientific/Jupyter dependencies (PyTorch installed separately)
start_notebooks.cmd     Windows launcher
data/                   Created only by lessons that use disk data
artifacts/              Generated masks, models, and experiments
reports/                Execution reports and checked notebook copies
```

Keep the complete project folder together. Each notebook creates its own required data and model; you can restart it independently. Outputs use lesson-specific directories and rerunning a lesson may replace that lesson's generated demonstration files. Put your own projects and irreplaceable data in separate directories.

Check notebook structure:

```powershell
.\.venv\Scripts\python.exe scripts/check_course.py
```

Execute every default notebook in a fresh kernel and save checked copies under `reports/executed/`:

```powershell
.\.venv\Scripts\python.exe scripts/check_course.py --execute
```

Add `--in-place` to update outputs in `notebooks/` too. Optional downloads remain disabled. The builders in `scripts/build_*.py` are authoring tools; running them recreates lesson notebooks and clears their outputs, so avoid running a builder after making personal notebook edits.

Additional offline edge checks are available with `.\.venv\Scripts\python.exe scripts/check_torch_contracts.py`. They cover ignored labels, odd U-Net image sizes, and torchvision classifier adaptation with random weights. They do not validate pretrained accuracy or the optional dataset download.

## Continue beyond the course

Complete the capstone in notebook 12 using a task you care about: plants, manufactured parts, documents, or everyday objects. Define an annotation policy, keep a held-out test set, compare a simple baseline with a learned model, inspect the worst cases, and explain the tradeoffs. For bigger next steps, follow the official [Torchvision instance-segmentation tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html) or explore [Segment Anything](https://github.com/facebookresearch/segment-anything) after understanding the prompt-based example in lesson 11. Those larger pretrained systems are extensions, not implementations hidden inside the toy examples.
