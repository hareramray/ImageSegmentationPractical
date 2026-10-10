"""Build the two opening practical notebooks."""
from pathlib import Path
from notebook_builder import md, code, write_notebook, bootstrap

ROOT = Path(__file__).resolve().parents[1]

cells = [
md('''
# 00 · Start here: your segmentation laboratory

**Level:** orientation · **Study time:** 25–40 minutes · **Prerequisite:** comfortable with Python.

Image segmentation assigns a label to every pixel. You will move from simple rules to learned
models, then examine where those models fail. This course assumes Python knowledge, but no
segmentation knowledge. You have an NVIDIA GPU; the deep learning lessons use it when PyTorch
can access CUDA. Classical image processing usually stays on the CPU.

**By the end:** select the right kernel, inspect a labeled image, run a baseline, and keep a useful
experiment record. All default examples run locally without dataset downloads. The real-data
and pretrained sections in lesson 10 are explicit opt-ins. Small examples teach the workflow;
they do not establish real-world accuracy or professional expertise.
'''),
md('''
## 1. Use Jupyter as a laboratory

Select **Image Segmentation (.venv)** as the kernel. A notebook has Markdown cells (explanation)
and code cells (Python). **Shift+Enter** runs a cell and moves forward; **Ctrl+Enter** runs it in place.
Run from top to bottom. Variables survive between cells until you restart the kernel.

For each lesson: **predict → run → inspect → change one parameter → explain**. At the end, use
**Kernel → Restart Kernel and Run All Cells**. This catches accidental dependence on an earlier
experiment. Each notebook starts fresh and creates its own required data; none requires a prior
notebook's saved model. The `learning/` directory contains small readable utilities; keep it beside
`notebooks/`. Open the complete project folder, rather than moving a notebook on its own.
'''),
bootstrap(),
md('''
## 2. Check the environment

`sys.executable` identifies the Python interpreter used by this notebook. If an import fails,
check the kernel first. Installing into a different interpreter does not fix the active kernel.
Package version numbers may differ from the recorded environment; the execution report records
the versions actually tested for this course.
'''),
code('''
import sys
import importlib.metadata as metadata
from IPython.display import display
import pandas as pd

print("Python:", sys.version.split()[0])
print("Interpreter:", sys.executable)
packages = ["numpy", "scipy", "matplotlib", "scikit-image", "scikit-learn", "pandas", "torch", "torchvision"]
rows = []
for package in packages:
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = "Not installed (torch/torchvision needed from lesson 08)"
    rows.append({"package": package, "version": version})
display(pd.DataFrame(rows))
'''),
md('''
## 3. Confirm GPU computation, not just installation

PyTorch must be built with CUDA support, and the NVIDIA driver must support that build. A local
CUDA toolkit is not required for normal operations using a compatible prebuilt PyTorch wheel.
The cell below performs an actual matrix multiplication on the selected device. A GPU improves
larger training jobs; tiny examples may be faster on the CPU because transfers have overhead.
'''),
code('''
try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch:", torch.__version__, "| CUDA build:", torch.version.cuda)
    print("Selected device:", DEVICE)
    if DEVICE.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        print("GPU:", properties.name)
        print("VRAM (GiB):", round(properties.total_memory / 1024**3, 1))
    a = torch.ones((32, 32), device=DEVICE)
    b = a @ a
    assert torch.allclose(b, torch.full_like(b, 32))
    print("Actual computation passed on", b.device)
except ImportError:
    DEVICE = "cpu"
    print("PyTorch is optional for lessons 01–07 and 11; see README before lesson 08.")
'''),
md('''
## 4. Meet the data and the three segmentation tasks

**Semantic segmentation:** which category occupies each pixel? All disks share class 1.
**Instance segmentation:** which individual object occupies each pixel? Two disks have different
object IDs. **Panoptic segmentation:** every pixel has a semantic category, and countable objects
also have an instance ID; lesson 11 constructs this representation.

Our toy generator returns an RGB image, a semantic mask, and an instance mask. The generator
knows the exact answer, called **ground truth**. Real ground truth comes from annotation and can
contain uncertainty. IDs are categories, not brightness values. The RGB colors used to display a
mask are only a visualization. Foreground means any object of interest; background is class 0.
'''),
code('''
from learning.data import make_scene, CLASS_NAMES, palette_mask

image, semantic, instances = make_scene(seed=7, size=128)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0].imshow(image); axes[0].set_title("RGB input")
axes[1].imshow(palette_mask(semantic)); axes[1].set_title("Semantic: 0, 1, 2")
axes[2].imshow(instances, cmap="tab20", interpolation="nearest"); axes[2].set_title("Instances: object IDs")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
print("Classes:", dict(enumerate(CLASS_NAMES)))
print("Image:", image.shape, image.dtype, "| Mask:", semantic.shape, semantic.dtype)
'''),
md('''
## 5. Your first segmentation baseline

A **baseline** is a simple solution to compare against. Here a brightness threshold separates
objects from background. The result is binary: object or background. It cannot distinguish disks
from rectangles. Intersection over union (**IoU**) measures overlap: intersection divided by
union. A score of 1 is perfect; 0 means no overlap. Lesson 06 examines the edge cases.

Predict what raising the threshold will do before running this cell. A visually pleasing result
does not guarantee accurate boundaries, so inspect the error image as well.
'''),
code('''
threshold = 0.32  # Experiment: try 0.20, 0.45, 0.60.
gray = image.mean(axis=-1)
pred = gray > threshold
truth = semantic > 0
intersection = np.logical_and(pred, truth).sum()
union = np.logical_or(pred, truth).sum()
iou = intersection / union if union else 1.0  # Both empty means a correct absence here.
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
for ax, array, title in zip(axes, [truth, pred, pred != truth],
                          ["True foreground", "Threshold prediction", "Errors (white)"]):
    ax.imshow(array, vmin=0, vmax=1)
    ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
print(f"Foreground IoU: {iou:.3f}")
'''),
md('''
## 6. Learning route

| Lesson | What you build | Main question |
|---|---|---|
| [01](01_images_masks_coordinates.ipynb) | Image and mask toolkit | What exactly is a label? |
| [02](02_thresholding.ipynb) | Global and local thresholds | Can intensity separate the objects? |
| [03](03_morphology_components.ipynb) | Mask cleanup and measurements | Which blobs should remain? |
| [04](04_edges_watershed.ipynb) | Boundary and watershed segmenters | How do we split touching objects? |
| [05](05_clustering_superpixels.ipynb) | Color clusters and superpixels | Can similar pixels form useful regions? |
| [06](06_metrics_error_analysis.ipynb) | Evaluation and error analysis | What does “good” mean? |
| [07](07_classical_machine_learning.ipynb) | Pixel classifier | Can labeled examples replace fixed rules? |
| [08](08_pytorch_segmentation_basics.ipynb) | Trainable pixel network | How does a neural network label pixels? |
| [09](09_unet_training.ipynb) | U-Net training pipeline | How do context and detail work together? |
| [10](10_real_data_transfer_learning.ipynb) | Disk data loader and optional transfer learning | How do I work with real annotations? |
| [11](11_instance_panoptic_prompting.ipynb) | Instance matching and prompted masks | Which object is which? |
| [12](12_advanced_training_deployment.ipynb) | Loss experiments, tiling, and inference | How do I make and audit a complete pipeline? |

Study at your own pace. A useful rhythm is one lesson per session, followed by another session
for its exercises. Reproduce the default first; then increase data and epochs, or try your own
images. Full real-data training takes longer than a demonstration run.
'''),
md('''
## 7. Keep an experiment record

Change one thing at a time. Record the hypothesis before you run, then compare the observation.
Never tune parameters on the final test set. Training data fits parameters, validation data guides
decisions, and an untouched test set estimates performance after decisions are frozen.
'''),
code('''
experiment_log = pd.DataFrame([{
    "lesson": "00", "seed": 7, "change": f"threshold={threshold}",
    "hypothesis": "Moderate brightness separates these synthetic objects",
    "metric": "foreground IoU", "value": round(float(iou), 4),
    "observation": "Replace this sentence with what the error image reveals",
    "next_experiment": "Increase threshold while keeping the same image",
}])
display(experiment_log)
# To save your work, uncomment these lines:
# output_dir = PROJECT_ROOT / "artifacts" / "my_experiments"
# output_dir.mkdir(parents=True, exist_ok=True)
# experiment_log.to_csv(output_dir / "lesson_00.csv", index=False)
'''),
md('''
## 8. Practice before continuing

1. Raise the threshold to 0.60. Does false foreground increase, or do true objects disappear?
2. Compare `semantic.max()` with `instances.max()`. Why are the numbers different?
3. Restart the kernel and run every cell. Explain why this is stronger than rerunning only the last cell.

<details><summary>Worked answers — open after trying</summary>

1. Raising the threshold requires brighter pixels. More true foreground is rejected (false negatives).
   Lowering it generally admits more background (false positives). The exact counts depend on noise.
2. `semantic.max()` is 2 because there are two foreground categories; `instances.max()` is 4
   because this generator draws four individually labeled objects. Class IDs and object IDs answer
   different questions; maximum ID is not a safe general object count when IDs have gaps.
3. Restarting removes hidden state. A successful clean run shows that every required variable and
   import is present in the notebook in execution order.

</details>
'''),
md('''
## Ready to move on?

- [ ] I can identify the active Python interpreter and selected compute device.
- [ ] I can explain the difference between an image, a class mask, and an instance mask.
- [ ] I changed one parameter and recorded a prediction and observation.
- [ ] I ran the notebook from a fresh kernel.

**Troubleshooting:** a busy kernel may be training; wait or interrupt before editing parameters.
For CUDA out-of-memory, reduce batch size or image size and restart the kernel. Keep
`num_workers=0` for these Windows notebooks. If `learning` cannot be imported, restore the whole
project folder. See the root README for installation and kernel instructions.

**Reference:** [JupyterLab notebook guide](https://jupyterlab.readthedocs.io/en/stable/user/notebook.html),
[PyTorch installation selector](https://pytorch.org/get-started/locally/).

**Next:** [01 · Images, masks, and coordinates](01_images_masks_coordinates.ipynb).
'''),
]
write_notebook(ROOT / "notebooks/00_start_here.ipynb", cells, "Start here")

cells = [
md('''
# 01 · Images, masks, coordinates, and annotation rules

**Level:** beginner · **Study time:** 45–60 minutes · **Prerequisite:** lesson 00.

**Build:** a small image/mask inspection and export pipeline. **Learn:** shapes, dtypes, coordinate
order, binary versus multiclass labels, correct resizing, overlays, and annotation consistency.
The default cells are fully runnable. Edit the experiment parameters, inspect the plots, then try
the exercises before opening their answers.
'''), bootstrap(),
md('''
## 1. An image is an array; a mask is a decision per pixel

We use RGB images shaped `(height, width, channels)` and integer masks shaped `(height, width)`.
Pixel values here are floating point numbers in `[0, 1]`. Other libraries may return unsigned
8-bit integers in `[0, 255]`. The scale matters to thresholds and neural network preprocessing.
A three-channel image is not three images: its last axis stores red, green, and blue.
'''),
code('''
from learning.data import make_scene, palette_mask, CLASS_NAMES
image, mask, instance_mask = make_scene(seed=12, size=128)
assert image.shape[:2] == mask.shape
assert image.shape[-1] == 3 and image.dtype == np.float32
assert set(np.unique(mask)).issubset({0, 1, 2})
print("RGB:", image.shape, image.dtype, "range:", (float(image.min()), float(image.max())))
print("Mask:", mask.shape, mask.dtype, "IDs:", np.unique(mask))
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
axes[0].imshow(image); axes[0].set_title("Image")
axes[1].imshow(palette_mask(mask)); axes[1].set_title("Semantic mask")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()
'''),
md('''
## 2. Index `[row, column]`, plot `(x, y)`

Rows move downward, columns move rightward. Array access uses `image[y, x]`, while plotting a
point uses `scatter(x, y)`. Mixing them often produces a plausible-looking annotation in the
wrong location. The slice end is excluded: rows `20:60` contain 40 pixels.
'''),
code('''
row, col = 35, 40
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
axes[0].imshow(image); axes[0].scatter([col], [row], c="lime", marker="x", s=100)
axes[0].set_title(f"Point at row={row}, col={col}")
crop = image[20:65, 15:70]
mask_crop = mask[20:65, 15:70]
axes[1].imshow(crop); axes[1].set_title(f"Paired crop: {crop.shape}")
plt.tight_layout(); plt.show()
print("RGB at point:", image[row, col], "| Class:", CLASS_NAMES[mask[row, col]])
assert crop.shape[:2] == mask_crop.shape
'''),
md('''
## 3. Label encodings are different contracts

A binary mask says whether a pixel is foreground. A semantic mask chooses one category; class 0
is real background, not “unknown.” An instance mask identifies individual objects. A one-hot
mask adds a separate channel per class; exactly one channel is 1 at each valid pixel.
PyTorch multiclass cross-entropy expects integer class IDs, so you usually do **not** one-hot
encode its targets. Binary sigmoid models and multiclass softmax models use different loss contracts.
'''),
code('''
binary_mask = mask > 0
one_hot = np.eye(len(CLASS_NAMES), dtype=np.float32)[mask]
assert one_hot.shape == (*mask.shape, 3)
assert np.all(one_hot.sum(axis=-1) == 1)
assert np.array_equal(one_hot.argmax(axis=-1), mask)
print("Binary:", binary_mask.dtype, binary_mask.shape)
print("Class IDs:", mask.dtype, mask.shape)
print("One-hot:", one_hot.dtype, one_hot.shape)
print("Instance IDs:", np.unique(instance_mask))
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for cls, ax in enumerate(axes):
    ax.imshow(one_hot[..., cls], vmin=0, vmax=1)
    ax.set_title(CLASS_NAMES[cls]); ax.axis("off")
plt.tight_layout(); plt.show()
'''),
md('''
## 4. Inspect class balance

Pixel accuracy can look good if background occupies most of an image. Count labels before
training. Also count images and objects: a million neighboring pixels are not a million
independent examples. Images from the same patient, video, or scene should share one split.
'''),
code('''
import pandas as pd
from IPython.display import display
counts = np.bincount(mask.ravel(), minlength=3)
balance = pd.DataFrame({"class": CLASS_NAMES, "pixels": counts,
                        "fraction": counts / mask.size})
display(balance)
plt.bar(CLASS_NAMES, counts / mask.size, color=["#253049", "#f4b742", "#43b9cf"])
plt.ylabel("Fraction of pixels"); plt.title("Class balance in one image"); plt.show()
'''),
md('''
## 5. Resize images smoothly and masks categorically

Bilinear interpolation blends neighboring values. That is useful for RGB images, but blending
class 0 and class 2 creates values that do not represent annotations. Casting blended values
back to integers can invent class 1. Use **nearest-neighbor** interpolation for class masks.
Keep the image and mask geometrically aligned, including during crops, flips, and rotations.
'''),
code('''
from skimage.transform import resize
target_shape = (73, 91)
resized_image = resize(image, (*target_shape, 3), order=1, anti_aliasing=True,
                       preserve_range=True).astype(np.float32)
good_mask = resize(mask, target_shape, order=0, anti_aliasing=False,
                   preserve_range=True).astype(np.uint8)
bad_mask = resize(mask.astype(float), target_shape, order=1,
                  anti_aliasing=False, preserve_range=True)
print("Nearest-neighbor IDs:", np.unique(good_mask))
print("Fractional label pixels with bilinear:", np.count_nonzero(bad_mask != np.round(bad_mask)))
assert set(np.unique(good_mask)).issubset(set(np.unique(mask)))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, arr, title in zip(axes, [resized_image, palette_mask(good_mask), bad_mask],
                          ["Resized RGB", "Correct: nearest mask", "Wrong: blended class IDs"]):
    ax.imshow(arr); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
'''),
md('''
## 6. Use overlays to audit alignment

An alpha overlay blends the image and display colors, while boundaries expose subtle shifts.
The overlay is for a person to inspect; save the original integer mask for training. Below we
deliberately shift one mask to show the kind of bug that numerical shape checks cannot catch.
'''),
code('''
from skimage.segmentation import mark_boundaries
shifted_mask = np.roll(mask, shift=8, axis=1)  # Intentional demonstration bug.
overlay = .65 * image + .35 * (palette_mask(mask) / 255.0)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, arr, title in zip(axes,
    [overlay, mark_boundaries(image, mask), mark_boundaries(image, shifted_mask)],
    ["Alpha overlay", "Aligned boundary", "Misaligned by 8 columns"]):
    ax.imshow(arr); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
'''),
md('''
## 7. Export labels without destroying them

JPEG compression changes pixels, so do not use JPEG for label masks. Store these small class IDs
as an 8-bit grayscale PNG. Store RGB previews separately. For any ID above 255, choose a suitable
16-bit format or NumPy array and verify round trips. A palette PNG can store IDs despite looking
colorful: blindly converting it to RGB can lose the original class encoding.
'''),
code('''
from PIL import Image
output_dir = PROJECT_ROOT / "artifacts" / "01_masks"
output_dir.mkdir(parents=True, exist_ok=True)
Image.fromarray((image * 255).round().astype(np.uint8)).save(output_dir / "image.png")
Image.fromarray(mask).save(output_dir / "mask_ids.png")
Image.fromarray(palette_mask(mask)).save(output_dir / "mask_preview.png")
with Image.open(output_dir / "mask_ids.png") as saved:
    loaded_mask = np.asarray(saved)
assert loaded_mask.shape == mask.shape
assert np.array_equal(loaded_mask, mask)
print("Exact label round trip passed:", output_dir / "mask_ids.png")
'''),
md('''
## 8. Annotate a polygon and define unknown pixels

Real annotations require a written labeling policy: which objects count, whether holes count as
background, how to handle occlusion, and what to do with uncertain boundaries. Here a coarse
polygon simulates one manual annotation. It is deliberately approximate. A label such as **255**
may mean “ignore during loss and metrics,” but only if the data loader, loss, and evaluator agree.
Never silently convert unknown pixels into background.
'''),
code('''
from skimage.draw import polygon
polygon_y = np.array([18, 19, 50, 53])
polygon_x = np.array([20, 54, 59, 17])
rr, cc = polygon(polygon_y, polygon_x, shape=mask.shape)
manual_mask = np.zeros_like(mask)
manual_mask[rr, cc] = 1
uncertain_mask = manual_mask.copy()
uncertain_mask[16:21, 18:58] = 255
valid = uncertain_mask != 255
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
axes[0].imshow(mark_boundaries(image, manual_mask)); axes[0].set_title("Approximate manual polygon")
axes[1].imshow(valid); axes[1].set_title("Pixels eligible for evaluation")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()
print("Ignored pixels:", (~valid).sum())
'''),
md('''
## 9. Exercises

1. Flip both image and mask horizontally and plot their boundary overlay. Then intentionally flip
   only the image. Which shape checks detect this error? Which visual check does?
2. Create a binary mask for disks only. Count its foreground pixels and verify against the table.
3. Downsample to 32×32 and upsample to the original size using nearest-neighbor masks. Calculate
   how many labels changed. Does correct interpolation guarantee no information loss?
4. Write a five-sentence annotation policy for segmenting leaves in photographs: target, background,
   occlusion, uncertain pixels, and train/validation/test grouping.

<details><summary>Worked solution code and guidance</summary>

```python
flipped_image = np.flip(image, axis=1).copy()
flipped_mask = np.flip(mask, axis=1).copy()
plt.imshow(mark_boundaries(flipped_image, flipped_mask)); plt.show()
# Shapes still match if only one is flipped. Inspect the overlay to catch the mismatch.
disk_mask = mask == 1
assert disk_mask.sum() == counts[1]
small = resize(mask, (32, 32), order=0, preserve_range=True, anti_aliasing=False)
restored = resize(small, mask.shape, order=0, preserve_range=True, anti_aliasing=False)
print("Changed labels:", np.count_nonzero(restored != mask))
```

Nearest-neighbor preserves the vocabulary of IDs, but downsampling can remove small objects and
boundary detail. A leaf policy might label visible leaf tissue, exclude stems, annotate only visible
parts under occlusion, ignore uncertain edges, and keep all pictures of a plant in one data split.
Different tasks may need different policies; consistency matters more than this particular choice.

</details>
'''),
md('''
## Checkpoint and common mistakes

- [ ] I can state the shape, dtype, scale, and label vocabulary of an image/mask pair.
- [ ] I can distinguish `[row, column]` indexing from `(x, y)` plotting.
- [ ] I resize class masks with nearest-neighbor and audit paired geometry visually.
- [ ] I save integer IDs separately from colored previews and test an exact round trip.
- [ ] I can explain background versus an ignored pixel.

**Watch for:** RGB versus OpenCV's usual BGR order; unsigned integer subtraction wrapping around;
lossy mask files; image and mask paths sorted differently; display colors mistaken for class IDs.
Use matching file stems and explicit label maps in lesson 10.

**References:** [NumPy indexing](https://numpy.org/doc/stable/user/basics.indexing.html),
[scikit-image resize](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize),
[Pillow image modes](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes).

**Next:** [02 · Thresholding](02_thresholding.ipynb).
''')]
write_notebook(ROOT / "notebooks/01_images_masks_coordinates.ipynb", cells, "Images, masks, and coordinates")
