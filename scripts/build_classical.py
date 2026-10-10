"""Build the classical segmentation lessons (02--05)."""
from pathlib import Path
from notebook_builder import md, code, write_notebook, bootstrap

ROOT = Path(__file__).resolve().parents[1]

def grid_setup(extra_imports):
    return code(extra_imports + '''

def show_grid(images, titles, columns=3, cmap="gray", limits=None):
    """A small display helper; every array is still available for inspection."""
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.4 * rows), squeeze=False)
    for ax, image, title in zip(axes.flat, images, titles):
        kwargs = {} if limits is None else {"vmin": limits[0], "vmax": limits[1]}
        ax.imshow(image, cmap=cmap, **kwargs)
        ax.set_title(title)
        ax.axis("off")
    for ax in list(axes.flat)[len(images):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
''')


thresholding = [
md('''# 02 · Thresholding: turn intensity into a mask

**Level:** beginner. **Time:** 45–60 minutes of learning; the default run is short and CPU-only.

**Prerequisites:** notebook 01, NumPy indexing, and the idea that an image is an array. You can run this notebook independently with **Restart Kernel → Run All**.

**You will:** construct a known answer; choose a threshold from a histogram; compare manual, Otsu, and local thresholding; diagnose illumination and polarity failures. All examples run offline.

**Learning habit:** predict the next plot before running each cell. Keep your predictions in a new Markdown cell. A synthetic mask is exact because we created it; it is not a human annotation of a real photograph.'''),
bootstrap(),
md('''## 1. Build an image with a known foreground

Coordinates are `(row, column)`. Our objects are bright disks on a dark background. Gaussian noise creates overlap between foreground and background intensities. A Boolean mask means `True = foreground`; it is not an RGB visualization.'''),
grid_setup('''from skimage import filters, data, util

rng = np.random.default_rng(11)
rows, cols = np.mgrid[:128, :160]
truth = np.zeros((128, 160), dtype=bool)
for row, col, radius in [(34, 35, 18), (88, 49, 23), (63, 119, 21)]:
    truth |= (rows - row) ** 2 + (cols - col) ** 2 <= radius ** 2
image = np.clip(0.20 + 0.55 * truth + rng.normal(0, 0.09, truth.shape), 0, 1)
print("Image:", image.shape, image.dtype, "range:", image.min(), image.max())
print("Mask:", truth.shape, truth.dtype, "foreground fraction:", truth.mean().round(3))'''),
code('''show_grid([image, truth], ["Noisy synthetic observation", "Known construction mask"], columns=2)
'''),
md('''## 2. Use a histogram to make a decision

A global threshold makes the same decision at every location: `image > threshold`. On the left peak most pixels are background; on the right peak most are objects. A valley is a plausible dividing point. The histogram throws away spatial arrangement, so two very different scenes can have the same histogram.

**Predict:** what happens if you move the threshold from 0.45 to 0.70?'''),
code('''manual_threshold = 0.45
manual_mask = image > manual_threshold
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].hist(image.ravel(), bins=60, range=(0, 1), color="slateblue")
axes[0].axvline(manual_threshold, color="orangered", label=f"threshold = {manual_threshold}")
axes[0].set(xlabel="Pixel intensity", ylabel="Number of pixels")
axes[0].legend()
axes[1].imshow(manual_mask, cmap="gray")
axes[1].set_title("Manual foreground mask")
axes[1].axis("off")
plt.tight_layout()
plt.show()'''),
md('''## 3. Measure overlap, not just appearance

For foreground masks, **intersection over union (IoU)** is `correct foreground / all pixels marked foreground by either mask`. False positives add unwanted pixels; false negatives miss true pixels. This introductory function scores two empty masks as 1 by explicit convention. Later evaluation lessons discuss dataset-level aggregation.

The known construction mask is used only for evaluation. We do not give it to Otsu.'''),
code('''def binary_iou(target, prediction):
    target, prediction = np.asarray(target, bool), np.asarray(prediction, bool)
    if target.shape != prediction.shape:
        raise ValueError("Masks must have matching shapes")
    intersection = np.logical_and(target, prediction).sum()
    union = np.logical_or(target, prediction).sum()
    return float(intersection / union) if union else 1.0

otsu_threshold = filters.threshold_otsu(image)
otsu_mask = image > otsu_threshold
print(f"Manual threshold {manual_threshold:.3f}: IoU = {binary_iou(truth, manual_mask):.3f}")
print(f"Otsu threshold   {otsu_threshold:.3f}: IoU = {binary_iou(truth, otsu_mask):.3f}")
error_map = np.zeros(truth.shape, dtype=np.uint8)
error_map[otsu_mask & ~truth] = 1  # false positive
error_map[truth & ~otsu_mask] = 2  # false negative
show_grid([otsu_mask, error_map], ["Otsu mask", "Errors: 0 correct, 1 extra, 2 missed"],
          columns=2, cmap="viridis", limits=(0, 2))'''),
md('''Otsu chooses a threshold that separates intensity groups according to their variance. It does **not** know what a disk is. Here the scene was designed so brightness is informative.

## 4. Parameter experiment: sweep the threshold

The curve below is a teaching diagnostic, not an unbiased test result: it uses the known answer to inspect many settings. In a real project, tune on a validation set and report performance on a separate test set.'''),
code('''thresholds = np.linspace(0.05, 0.95, 37)
scores = [binary_iou(truth, image > t) for t in thresholds]
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(thresholds, scores, "o-", markersize=3)
ax.axvline(otsu_threshold, color="orangered", linestyle="--", label="Otsu")
ax.set(xlabel="Threshold", ylabel="Foreground IoU", ylim=(0, 1.03))
ax.legend()
plt.show()
print("Broad good regions are more reassuring than a single fragile optimum.")'''),
md('''## 5. Create a failure: uneven illumination

Now the right side receives more light. Some background pixels become brighter than some object pixels. A single global threshold cannot perfectly separate the two groups. Small repeated objects make a useful local-threshold example because each neighborhood can contain both foreground and background.'''),
code('''local_truth = np.zeros_like(truth)
for row in (24, 64, 104):
    for col in (20, 60, 100, 140):
        local_truth |= (rows - row) ** 2 + (cols - col) ** 2 <= 7 ** 2
illumination = 0.12 + 0.58 * cols / (cols.shape[1] - 1)
uneven = np.clip(illumination + 0.20 * local_truth + rng.normal(0, 0.015, truth.shape), 0, 1)
global_uneven = uneven > filters.threshold_otsu(uneven)
# threshold_local returns a threshold IMAGE, not a foreground mask.
local_threshold = filters.threshold_local(uneven, block_size=35, method="gaussian", offset=-0.05)
local_mask = uneven > local_threshold
show_grid([uneven, global_uneven, local_mask],
          ["Uneven illumination", "Global Otsu", "Local threshold"])
print(f"Global IoU: {binary_iou(local_truth, global_uneven):.3f}")
print(f"Local IoU:  {binary_iou(local_truth, local_mask):.3f}")'''),
md('''## 6. Interpret local parameters

`block_size` is an odd window width in pixels. Too small: the neighborhood follows the object itself. Too large: it stops adapting to illumination. With `threshold_local`, **the offset is subtracted** from the local statistic; a negative offset therefore raises the threshold and is stricter for bright foreground.

Compare these settings; explain one missed-object case and one background-noise case. A local method is not automatically better.'''),
code('''settings = [(15, -0.02), (35, -0.05), (71, -0.08)]
trial_masks, trial_titles = [], []
for block, offset in settings:
    threshold_map = filters.threshold_local(uneven, block_size=block, offset=offset)
    prediction = uneven > threshold_map
    trial_masks.append(prediction)
    trial_titles.append(f"window={block}, offset={offset}; IoU={binary_iou(local_truth, prediction):.3f}")
show_grid(trial_masks, trial_titles)
'''),
md('''## 7. Transfer to a bundled real image

The scikit-image `coins` photograph is packaged with the library. We have **no reference mask** for this image in this lesson, so we can inspect results but cannot claim accuracy. The coin texture and uneven background violate our simple synthetic assumptions.'''),
code('''coins = util.img_as_float(data.coins())[::2, ::2]
coins_smoothed = filters.gaussian(coins, sigma=1.0)
coins_mask = coins_smoothed > filters.threshold_otsu(coins_smoothed)
show_grid([coins, coins_smoothed, coins_mask], ["Real photograph", "Gaussian smoothing", "Otsu proposal"])
print("No real-image IoU is reported: we do not have an annotated reference mask.")'''),
md('''## 8. Exercises — try these in new cells

**A. Threshold polarity.** Segment dark objects in `1 - image`. Keep the original `truth`. Which comparison operator is required?

<details><summary>Worked solution A</summary>

```python
inverted = 1 - image
dark_mask = inverted < filters.threshold_otsu(inverted)
print(binary_iou(truth, dark_mask))
```
The foreground definition did not change; its brightness did. `>` would select the background.
</details>'''),
md('''**B. Noise robustness.** Generate three new versions with noise standard deviations 0.03, 0.12, and 0.25. Compare raw Otsu with Gaussian-smoothed Otsu. State one cost of smoothing.

<details><summary>Worked solution B</summary>

```python
experiment_rng = np.random.default_rng(20)
for noise in (0.03, 0.12, 0.25):
    noisy = np.clip(0.20 + 0.55 * truth + experiment_rng.normal(0, noise, truth.shape), 0, 1)
    smooth = filters.gaussian(noisy, sigma=1)
    raw = noisy > filters.threshold_otsu(noisy)
    cleaned = smooth > filters.threshold_otsu(smooth)
    print(noise, binary_iou(truth, raw), binary_iou(truth, cleaned))
```
Smoothing can suppress isolated errors but also moves edges and can erase tiny objects. Its benefit depends on object scale.
</details>'''),
md('''**C. Make local thresholding fail.** Replace the small disks in the uneven-light example with a filled rectangle of width 80. Keep `block_size=35`. Inspect its interior and explain the result.

<details><summary>Worked solution C</summary>

```python
wide_truth = np.zeros_like(truth)
wide_truth[30:98, 40:120] = True
wide_image = np.clip(illumination + 0.20 * wide_truth, 0, 1)
wide_threshold = filters.threshold_local(wide_image, 35, offset=-0.05)
wide_mask = wide_image > wide_threshold
show_grid([wide_image, wide_truth, wide_mask], ["Observation", "Truth", "Local result"])
```
A neighborhood wholly inside a broad object has no background reference. A larger window or illumination correction may help; the current local rule is poorly suited to this case.
</details>'''),
md('''## Common mistakes and debugging

- Mixing 0–255 integers with thresholds meant for 0–1 floats. Print the dtype and range first.
- Treating a threshold map as a binary mask. Compare the image to that map.
- Using `>` for every image. Decide whether foreground is brighter or darker.
- Comparing an RGB image with a 2D mask without understanding the channel axis.
- Selecting a parameter on the test image and calling the score independent evaluation.

## Completion checklist

- [ ] I can explain the two histogram peaks and threshold polarity.
- [ ] I can calculate the foreground IoU of two tiny hand-drawn masks.
- [ ] I can describe a scene where a global threshold fails.
- [ ] I completed A–C and recorded what changed.
- [ ] I understand why a pleasing real-image mask is not evidence of measured accuracy.

**Next:** notebook 03 repairs masks and turns foreground pixels into measurable objects.'''),
md('''## Official references

- [scikit-image thresholding guide](https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding_guide.html)
- [Filter API: Otsu, local thresholds, Gaussian smoothing](https://scikit-image.org/docs/stable/api/skimage.filters.html)
- [Image dtype and intensity conventions](https://scikit-image.org/docs/stable/user_guide/data_types.html)

These are optional reading links; running this notebook does not access the internet.''')
]


morphology = [
md('''# 03 · Morphology and connected components

**Level:** beginner to intermediate. **Time:** 45–60 minutes. **Prerequisites:** Boolean masks and notebook 02 concepts. Run independently with **Restart Kernel → Run All**.

**You will:** repair a damaged mask; explain erosion, dilation, opening, and closing; label connected objects; measure their areas and boxes; see why connectivity and border policy matter. All data are generated locally with a fixed seed.

**Question to keep asking:** am I removing measurement noise, or erasing a real small object?'''),
bootstrap(),
md('''## 1. Construct damaged foreground

There are three intended objects, including one smaller disk. The noisy observation contains extra pixels and a small hole. The ideal binary mask is known only because this is a controlled simulation. In real data, a postprocessing rule needs validation against trusted annotations.'''),
grid_setup('''from scipy import ndimage as ndi
from skimage import morphology, measure, color, segmentation
import pandas as pd

yy, xx = np.mgrid[:128, :160]
truth = ((yy - 38)**2 + (xx - 38)**2 <= 21**2)
truth |= ((yy - 93)**2 + (xx - 42)**2 <= 9**2)
truth[43:93, 92:140] = True
rng = np.random.default_rng(12)
damaged = truth.copy()
damaged[rng.random(truth.shape) < 0.012] = True
damaged[32:38, 34:40] = False
damaged[61:64, 92:103] = False
print("Boolean mask:", damaged.dtype, "foreground pixels:", damaged.sum())'''),
code('''show_grid([truth, damaged, damaged ^ truth],
          ["Known ideal mask", "Damaged mask", "Changed pixels (XOR)"])
'''),
md('''## 2. A footprint defines the neighborhood

Imagine moving a small disk over the mask. **Erosion** keeps a foreground pixel only when the footprint fits inside foreground. **Dilation** adds pixels reached by the footprint. Both change object boundaries. The radius is measured in pixels, so resizing an image changes its physical meaning.

**Predict:** which operation is likely to break a narrow bridge? Which can connect nearby objects?'''),
code('''footprint = morphology.disk(2)
eroded = morphology.erosion(damaged, footprint)
dilated = morphology.dilation(damaged, footprint)
show_grid([footprint, eroded, dilated], ["Radius-2 footprint", "Erosion", "Dilation"])
print("Before / eroded / dilated pixel counts:", damaged.sum(), eroded.sum(), dilated.sum())'''),
md('''## 3. Compose operations to clean the mask

**Opening = erosion then dilation**; it suppresses features too narrow to hold the footprint. **Closing = dilation then erosion**; it can seal narrow gaps. Neither operation knows whether a thin detail is important.

`binary_fill_holes` fills every enclosed background region. A hollow object may contain a meaningful cavity, so use that step only when the object definition calls for a solid region.'''),
code('''opened = morphology.opening(damaged, footprint)
closed = morphology.closing(opened, footprint)
filled = ndi.binary_fill_holes(closed)
show_grid([opened, closed, filled], ["Opening", "Then closing", "Then fill enclosed holes"])
print("Removed by opening:", int((damaged & ~opened).sum()))
print("Added by hole filling:", int((filled & ~closed).sum()))'''),
md('''## 4. Connectivity changes the object count

In 2D, connectivity 1 uses four edge-sharing neighbors. Connectivity 2 also includes four diagonal neighbors. Two diagonal pixels are two components under the first rule, one under the second. Labels are **object IDs**: label 2 is not twice as much foreground as label 1.'''),
code('''diagonal = np.eye(5, dtype=bool)
four_connected = measure.label(diagonal, connectivity=1)
eight_connected = measure.label(diagonal, connectivity=2)
print("4-neighbor components:", four_connected.max())
print("8-neighbor components:", eight_connected.max())
show_grid([diagonal, four_connected, eight_connected],
          ["Diagonal pixels", "4-neighbor IDs", "8-neighbor IDs"], cmap="viridis")'''),
md('''## 5. Filter by area without shrinking surviving objects

Morphology changes local geometry. Component filtering instead keeps or removes an entire object. We label first, count pixels with `bincount`, then index a keep-table by the label image. Label 0 is background and must always remain excluded.

This explicit approach also shows what area filtering does internally. Current scikit-image versions offer `remove_small_objects`; its size argument differs across versions, so consult your installed version before substituting it.'''),
code('''def keep_components_at_least(mask, min_area, connectivity=2):
    labels = measure.label(np.asarray(mask, bool), connectivity=connectivity)
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False
    return keep[labels]

area_cleaned = keep_components_at_least(damaged, min_area=25)
final_mask = ndi.binary_fill_holes(area_cleaned)
object_labels = measure.label(final_mask, connectivity=2)
show_grid([damaged, area_cleaned, color.label2rgb(object_labels, bg_label=0)],
          ["Input", "Remove components <25 px", "Surviving object IDs"])
print("Detected components:", object_labels.max())'''),
md('''## 6. Measure the resulting objects

`regionprops_table` converts labeled regions to measurements. Area is in **pixels²** here; bounding boxes use `(min_row, min_col, max_row, max_col)`, with exclusive upper bounds. A centroid is an average position, not necessarily a foreground pixel. Eccentricity approaches 1 for elongated regions.

Counts and areas are measurements of the **predicted mask**, not automatically of the real world.'''),
code('''properties = measure.regionprops_table(
    object_labels,
    properties=("label", "area", "centroid", "bbox", "eccentricity", "solidity"),
)
table = pd.DataFrame(properties)
display(table.round(2))

fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(color.label2rgb(object_labels, bg_label=0))
for region in measure.regionprops(object_labels):
    row0, col0, row1, col1 = region.bbox
    ax.add_patch(plt.Rectangle((col0, row0), col1-col0, row1-row0,
                              fill=False, edgecolor="white", linewidth=1.5))
    row, col = region.centroid
    ax.text(col, row, str(region.label), color="white", ha="center", weight="bold")
ax.set_title("Labels and bounding boxes")
ax.axis("off")
plt.show()'''),
md('''## 7. Experiment: small-object removal has a cost

The smaller disk is a genuine object in our construction. Raising an area threshold can make the image look cleaner while reducing recall. Record both pixel overlap and object count; the metrics answer different questions. A merged pair can have good pixel overlap but a wrong count.'''),
code('''def overlap_iou(target, prediction):
    union = np.logical_or(target, prediction).sum()
    return np.logical_and(target, prediction).sum() / union if union else 1.0

records, masks = [], []
for minimum in (5, 25, 400):
    candidate = ndi.binary_fill_holes(keep_components_at_least(damaged, minimum))
    masks.append(candidate)
    records.append({"min_area": minimum,
                    "component_count": int(measure.label(candidate, connectivity=2).max()),
                    "foreground_IoU": overlap_iou(truth, candidate)})
display(pd.DataFrame(records).round(3))
show_grid(masks, [f"Keep area >= {m} px" for m in (5, 25, 400)])'''),
md('''## 8. Border policy and physical units

An object clipped by the image edge has an incomplete area. Some counting protocols exclude such objects; others keep them with a flag. `clear_border` implements exclusion. Set this policy before comparing experiments.

For square pixels 0.5 mm wide, an area of 100 pixels represents `100 × 0.5² = 25 mm²`. Anisotropic pixels require row-spacing × column-spacing.'''),
code('''border_demo = final_mask.copy()
border_demo[0:15, 70:85] = True
border_labels = measure.label(border_demo)
interior_labels = segmentation.clear_border(border_labels)
show_grid([border_labels, interior_labels], ["Includes a clipped object", "Border objects removed"],
          columns=2, cmap="viridis")
pixel_height_mm, pixel_width_mm = 0.5, 0.5
table["area_mm2"] = table["area"] * pixel_height_mm * pixel_width_mm
display(table[["label", "area", "area_mm2"]])'''),
md('''## 9. Exercises — predict before running

**A. Morphology scale.** Compare opening with disk radii 1, 4, and 10 on `truth`. Which real object disappears first?

<details><summary>Worked solution A</summary>

```python
results = [morphology.opening(truth, morphology.disk(r)) for r in (1, 4, 10)]
show_grid(results, ["radius 1", "radius 4", "radius 10"])
```
The radius-9 disk cannot contain a radius-10 footprint, so it disappears. Opening removes structures based on shape and width, not just total area.
</details>'''),
md('''**B. Largest component.** Keep only the largest foreground component, with an explicit empty-mask case. Why is this a questionable default for cell counting?

<details><summary>Worked solution B</summary>

```python
labels = measure.label(final_mask)
sizes = np.bincount(labels.ravel())
sizes[0] = 0
largest = labels == sizes.argmax() if sizes.max() > 0 else np.zeros_like(final_mask)
show_grid([final_mask, largest], ["All objects", "Largest only"], columns=2)
```
It deliberately discards every other object, including valid cells. It fits a one-object task only when that assumption is justified.
</details>'''),
md('''**C. Export measurements.** Save the component table under the project's `outputs` folder, including your pixel calibration. Reopen it and check its row count.

<details><summary>Worked solution C</summary>

```python
destination = PROJECT_ROOT / "outputs" / "03_component_measurements.csv"
destination.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(destination, index=False)
reloaded = pd.read_csv(destination)
assert len(reloaded) == len(table)
print(destination)
```
For a research project, record the image ID, calibration source, units, and processing parameters alongside this table.
</details>'''),
md('''## Common mistakes and completion checklist

- A Boolean mask and an integer label image have different meanings. Relabel a mask before measuring instances.
- Area filtering cannot separate touching objects; they are already one component. Notebook 04 addresses this.
- Filling all holes can erase valid anatomy or ring-shaped structures.
- Repeated cleaning can improve appearance while damaging small-object recall.
- The largest numeric label is a count only when IDs are consecutive; after filtering, use the number of unique nonzero IDs.

- [ ] I can explain all four basic morphological operations.
- [ ] I can explain 4- versus 8-neighbor connectivity.
- [ ] I can draw a bounding box using row/column conventions.
- [ ] I have completed the three exercises and saved one result.
- [ ] I can justify a minimum area and a border policy for a stated task.

**Next:** notebook 04 separates touching foreground regions with markers and watershed.'''),
md('''## Official references

- [scikit-image morphology API](https://scikit-image.org/docs/stable/api/skimage.morphology.html)
- [Measure region properties](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html)
- [Label and region measurement API](https://scikit-image.org/docs/stable/api/skimage.measure.html)
- [SciPy binary hole filling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_fill_holes.html)

The current scikit-image API (0.26) uses inclusive `max_size` for small-object/hole removal, replacing older size arguments. This notebook implements its stated **keep area ≥ minimum** rule explicitly.''')
]


watershed = [
md('''# 04 · Edges, markers, and watershed

**Level:** intermediate. **Time:** 50–70 minutes. **Prerequisites:** thresholding, connected components, and 2D indexing. This notebook runs independently and offline on the CPU.

**You will:** distinguish an edge map from an object mask; calculate distance to background; place markers; separate touching objects; diagnose undersegmentation and oversegmentation; apply intensity markers to a real image.

**Core question:** what evidence tells us that one connected region contains multiple objects?'''),
bootstrap(),
md('''## 1. Create two touching objects

Two synthetic disks overlap in the image plane, producing one connected foreground region. Their true generating centers are known to us, but the main algorithm will use the observed foreground only. The overlap has no unique pixel-level instance ownership unless we define a convention, so we avoid claiming a perfect instance ground-truth mask.'''),
grid_setup('''from scipy import ndimage as ndi
from skimage import filters, feature, measure, segmentation, color, data, util

yy, xx = np.mgrid[:128, :160]
disk_a = (yy - 64)**2 + (xx - 57)**2 <= 29**2
disk_b = (yy - 64)**2 + (xx - 101)**2 <= 29**2
foreground_truth = disk_a | disk_b
rng = np.random.default_rng(13)
observation = np.clip(0.15 + 0.65 * foreground_truth + rng.normal(0, 0.045, yy.shape), 0, 1)
smooth = filters.gaussian(observation, sigma=1)
foreground = smooth > filters.threshold_otsu(smooth)
components = measure.label(foreground, connectivity=2)
print("Generating disks: 2; thresholded connected components:", components.max())'''),
code('''show_grid([observation, foreground, color.label2rgb(components, bg_label=0)],
          ["Noisy observation", "Foreground proposal", "Connected components"])
'''),
md('''## 2. Edges are evidence, not filled objects

The Sobel filter estimates intensity change. Canny returns selected thin edges after smoothing and edge-strength decisions. Neither output directly assigns every object pixel an instance ID. Broken boundaries can leak; internal texture can create extra boundaries.

**Observe:** there is little intensity evidence at the overlap between identical disks. An edge detector cannot invent that hidden boundary.'''),
code('''gradient = filters.sobel(smooth)
edges = feature.canny(observation, sigma=1.4, low_threshold=0.08, high_threshold=0.18)
filled_edges = ndi.binary_fill_holes(edges)
show_grid([gradient, edges, filled_edges],
          ["Sobel gradient", "Canny edge pixels", "Fill closed edge loops (fragile)"])
'''),
md('''## 3. Distance transform: turn shape into a surface

For each foreground pixel, the Euclidean distance transform measures distance to the nearest background pixel. Large values often occur near object centers. Negating this map produces basins: deep near centers, shallow near boundaries. Watershed can grow an object outward from each basin's marker.

This uses a **shape assumption**. It works well for some round touching objects and can fail for branched, elongated, or heavily overlapping ones.'''),
code('''distance = ndi.distance_transform_edt(foreground)
show_grid([foreground, distance, -distance],
          ["Foreground support", "Distance to background (px)", "Negative distance: basins"], cmap="viridis")
print(f"Largest foreground distance: {distance.max():.2f} pixels")'''),
md('''## 4. Convert peaks into labeled markers

`peak_local_max` returns coordinate pairs. We put a unique positive integer at each coordinate and keep 0 elsewhere. `min_distance` limits how closely peaks may occur. A marker is a seed for one proposed object, not an object boundary.

Marker count is a modeling decision: too few seeds tend to merge objects; too many tend to split them.'''),
code('''def markers_from_distance(distance_map, mask, min_distance=15):
    coordinates = feature.peak_local_max(
        distance_map, min_distance=min_distance, labels=mask.astype(np.uint8),
        exclude_border=False,
    )
    markers = np.zeros(mask.shape, dtype=np.int32)
    for marker_id, (row, col) in enumerate(coordinates, start=1):
        markers[row, col] = marker_id
    return markers, coordinates

markers, coordinates = markers_from_distance(distance, foreground, min_distance=15)
print("Marker coordinates (row, column):", coordinates.tolist())
fig, ax = plt.subplots(figsize=(7, 4))
ax.imshow(distance, cmap="magma")
ax.scatter(coordinates[:, 1], coordinates[:, 0], marker="+", color="cyan", s=100)
ax.set_title(f"{len(coordinates)} automatically detected seeds")
ax.axis("off")
plt.show()'''),
md('''## 5. Flood the basins

The mask confines flooding to the foreground. With `watershed_line=True`, a separating line receives label 0. Those zero pixels lie **inside** the foreground support and should not silently be interpreted as missed foreground when evaluating the instance separation. Here we visualize the line explicitly.'''),
code('''instances = segmentation.watershed(-distance, markers, mask=foreground, watershed_line=True)
colored = color.label2rgb(instances, image=observation, bg_label=0, alpha=0.45)
internal_line = foreground & (instances == 0)
show_grid([color.label2rgb(components, bg_label=0), colored, internal_line],
          ["Before: one connected region", "Watershed instance proposal", "Internal watershed line"])
instance_ids = np.unique(instances)
instance_ids = instance_ids[instance_ids != 0]
print("Proposed instances:", len(instance_ids))
print("Areas:", {int(i): int((instances == i).sum()) for i in instance_ids})'''),
md('''## 6. Controlled failure experiment: marker count

We deliberately supply one, two, and three markers in the same foreground. These are manual hints chosen using the synthetic construction and are **not** an automatic detector. Compare the resulting boundaries. Watershed faithfully follows seeds even when the seeds describe the wrong number of objects.'''),
code('''seed_sets = {
    "One marker: merge": [(64, 57)],
    "Two markers: useful split": [(64, 57), (64, 101)],
    "Three markers: extra split": [(64, 57), (64, 101), (43, 57)],
}
marker_results = []
for title, seeds in seed_sets.items():
    manual_markers = np.zeros(foreground.shape, dtype=np.int32)
    for label_id, (row, col) in enumerate(seeds, 1):
        manual_markers[row, col] = label_id
    result = segmentation.watershed(-distance, manual_markers, mask=foreground)
    marker_results.append(color.label2rgb(result, bg_label=0))
show_grid(marker_results, list(seed_sets))'''),
md('''## 7. Sweep peak spacing

Increase `min_distance` until the second peak is suppressed. This parameter is in pixels, so a fixed value is not scale invariant. A stable object count across a useful range is encouraging, but a correct count alone does not validate boundaries or identity.'''),
code('''spacing_results, spacing_titles = [], []
for spacing in (5, 20, 50):
    trial_markers, trial_coordinates = markers_from_distance(distance, foreground, spacing)
    trial_instances = segmentation.watershed(-distance, trial_markers, mask=foreground)
    spacing_results.append(color.label2rgb(trial_instances, bg_label=0))
    spacing_titles.append(f"min_distance={spacing}; {len(trial_coordinates)} markers")
show_grid(spacing_results, spacing_titles)
'''),
md('''## 8. Another formulation: markers on a gradient

For a real coin photograph, assign dark pixels as likely background and bright pixels as likely foreground, leaving ambiguous pixels unmarked. Watershed on a gradient grows these two semantic regions. These seeds use brightness heuristics, not annotations; a bright background or dark coin interior can be mislabeled.

This is **foreground/background segmentation**, not a complete coin-instance detector. The packaged image has no reference annotation in this lesson.'''),
code('''coins = util.img_as_float(data.coins())[::2, ::2]
coin_gradient = filters.sobel(filters.gaussian(coins, sigma=1))
intensity_markers = np.zeros(coins.shape, dtype=np.int32)
intensity_markers[coins < 0.25] = 1  # background seed class
intensity_markers[coins > 0.65] = 2  # foreground seed class
coin_regions = segmentation.watershed(coin_gradient, intensity_markers)
show_grid([coins, intensity_markers, coin_regions == 2],
          ["Bundled real image", "0 unknown; 1 dark; 2 bright", "Foreground proposal"])
print("Inspect seed mistakes first when the final region looks wrong.")'''),
md('''## 9. Exercises

**A. Edge smoothing.** Compare Canny with sigma 0.5, 1.5, and 3.0. Which setting retains noise? Which can move or remove small edges?

<details><summary>Worked solution A</summary>

```python
edge_maps = [feature.canny(observation, sigma=s) for s in (0.5, 1.5, 3.0)]
show_grid(edge_maps, ["sigma 0.5", "sigma 1.5", "sigma 3.0"])
```
Low smoothing preserves more intensity fluctuations. High smoothing reduces noise but may weaken small features. No setting reveals a boundary absent from the measured intensities.
</details>'''),
md('''**B. Why is the mask essential?** Run distance watershed once with the foreground mask and once without it. Inspect what happens to background pixels.

<details><summary>Worked solution B</summary>

```python
bounded = segmentation.watershed(-distance, markers, mask=foreground)
unbounded = segmentation.watershed(-distance, markers)
show_grid([color.label2rgb(bounded, bg_label=0), color.label2rgb(unbounded, bg_label=0)],
          ["Confined to foreground", "Flooded background too"], columns=2)
```
Without a mask, the algorithm is asked to label the whole image. This is sometimes intended, but it does not match the instance task we defined here.
</details>'''),
md('''**C. Avoid a misleading foreground score.** Compute foreground IoU using a watershed with `watershed_line=False`. Compare it to the IoU of the original threshold mask. Does this metric tell you whether the disks were separated correctly?

<details><summary>Worked solution C</summary>

```python
no_line = segmentation.watershed(-distance, markers, mask=foreground, watershed_line=False)
def iou(a, b):
    union = (a | b).sum()
    return (a & b).sum() / union if union else 1.0
print(iou(foreground_truth, foreground))
print(iou(foreground_truth, no_line > 0))
```
With seeds covering this connected foreground, the foreground support is unchanged. Both scores are equal even if you use the wrong number of markers. Instance evaluation requires matching or comparing object partitions, which later lessons introduce.
</details>'''),
md('''## Common mistakes and completion checklist

- An edge map is not a filled semantic mask.
- Distance watershed expects `-distance` when centers should be basins.
- Coordinate order is `(row, column)`; scatter plots use `(x=column, y=row)`.
- Missing or extra markers create missing or extra instances.
- One object count does not establish segmentation accuracy.
- A real image without labels supports visual diagnosis, not a claimed accuracy score.

- [ ] I can explain why connected components merged the disks.
- [ ] I can construct markers from peak coordinates.
- [ ] I can distinguish foreground segmentation from instance separation.
- [ ] I have demonstrated both marker failure modes.
- [ ] I completed A–C and described one limitation of watershed.

**Next:** notebook 05 groups pixels by feature similarity and builds superpixels.'''),
md('''## Official references

- [Watershed example and distance-transform construction](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html)
- [Watershed parameters](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed)
- [Peak detection API](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max)
- [Canny edge detector](https://scikit-image.org/docs/stable/auto_examples/edges/plot_canny.html)
- [SciPy Euclidean distance transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html)''')
]


clustering = [
md('''# 05 · Clustering pixels and building superpixels

**Level:** intermediate. **Time:** 60–75 minutes. **Prerequisites:** RGB arrays, masks, reshape, and basic distance intuition. This notebook is independent, deterministic, offline, and CPU-friendly.

**You will:** fit K-means to color features; understand why cluster IDs are arbitrary; score partitions appropriately; add coordinates to features; generate SLIC superpixels; compress each superpixel to a region descriptor.

**Key distinction:** grouping similar pixels is not the same as recognizing semantic categories such as person, road, or tumor.'''),
bootstrap(),
md('''## 1. From an image to a feature table

K-means expects one row per sample and one column per feature. A 96 × 96 RGB image becomes a 9,216 × 3 table. We generate a scene with background, disk, and rectangle classes; its construction mask is available for a controlled evaluation. Pixels are correlated samples, so random pixel splits would not establish performance on new images.'''),
grid_setup('''from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from skimage import color, segmentation, data, util
from learning.data import make_scene, palette_mask
import pandas as pd

image, truth, instance_truth = make_scene(seed=15, size=96, noise=0.08)
height, width, channels = image.shape
rgb_features = image.reshape(-1, channels)
print("Image shape:", image.shape, "feature table:", rgb_features.shape)
print("Construction classes:", np.unique(truth))'''),
code('''show_grid([image, palette_mask(truth)], ["Synthetic observation", "Known semantic construction"], columns=2)
'''),
md('''## 2. Fit a color clustering model

K-means alternates between assigning each sample to a nearby center and updating centers. `n_clusters=3` is a deliberate hypothesis here; we know the synthetic scene contains three color groups. A fixed seed makes the demonstration repeatable. Several initializations reduce dependence on an unlucky starting arrangement.

The reconstructed image replaces every pixel with its center color. It is a color quantization, not a class prediction.'''),
code('''kmeans = KMeans(n_clusters=3, random_state=0, n_init=5)
cluster_ids = kmeans.fit_predict(rgb_features).reshape(height, width)
quantized = kmeans.cluster_centers_[cluster_ids]
show_grid([image, cluster_ids, quantized], ["Original", "Arbitrary cluster IDs", "Center-color reconstruction"])
print("RGB cluster centers:")
display(pd.DataFrame(kmeans.cluster_centers_, columns=["R", "G", "B"]).round(3))'''),
md('''## 3. Labels can be permuted without changing a partition

Cluster 0 does not mean background and cluster 1 does not mean disk. If we rename every cluster, the grouping is unchanged. Raw pixel accuracy between cluster IDs and semantic IDs is therefore invalid.

**Adjusted Rand index (ARI)** compares which pairs of pixels are grouped together. It is invariant to ID permutation: 1 means identical partitions, around 0 is chance-level agreement under its adjustment, and negative values are possible. Here it measures partition agreement with our synthetic classes; it does not prove semantic understanding.'''),
code('''permutation = np.array([2, 0, 1])
renamed_clusters = permutation[cluster_ids]
original_ari = adjusted_rand_score(truth.ravel(), cluster_ids.ravel())
renamed_ari = adjusted_rand_score(truth.ravel(), renamed_clusters.ravel())
print(f"Original ARI: {original_ari:.4f}; renamed ARI: {renamed_ari:.4f}")
assert np.isclose(original_ari, renamed_ari)
print("The groups did not change when their IDs changed.")
show_grid([cluster_ids, renamed_clusters], ["Original numbering", "Renamed numbering"], columns=2, cmap="viridis")'''),
md('''## 4. Experiment: choosing the number of groups

Try too few, plausible, and too many clusters. K-means inertia measures squared feature distance within clusters; it tends to decrease as more clusters are allowed. It is **not** semantic accuracy. ARI uses the construction mask as a diagnostic; do not tune on a real test set this way.'''),
code('''records, clusterings = [], []
for k in (2, 3, 5):
    model = KMeans(n_clusters=k, random_state=0, n_init=3)
    partition = model.fit_predict(rgb_features).reshape(height, width)
    clusterings.append(partition)
    records.append({"k": k, "inertia": model.inertia_,
                    "ARI_to_construction": adjusted_rand_score(truth.ravel(), partition.ravel())})
display(pd.DataFrame(records).round(3))
show_grid(clusterings, ["2 clusters", "3 clusters", "5 clusters"], cmap="viridis")'''),
md('''## 5. Feature engineering: color plus position

RGB distance is convenient but does not closely match perceptual color differences. We use Lab color features and divide them by 100 to keep their scale manageable. Coordinates normalized to 0–1 add a preference for nearby pixels. `spatial_weight` controls that preference; a large weight can split one uniform object by location.

These features express assumptions. There is no universally correct scale balance.'''),
code('''lab_features = color.rgb2lab(image).reshape(-1, 3) / 100.0
yy, xx = np.mgrid[:height, :width]
coordinates = np.column_stack((yy.ravel() / (height - 1), xx.ravel() / (width - 1)))
spatial_partitions = []
for spatial_weight in (0.0, 0.5, 3.0):
    features = np.column_stack((lab_features, spatial_weight * coordinates))
    partition = KMeans(n_clusters=3, random_state=0, n_init=3).fit_predict(features).reshape(height, width)
    spatial_partitions.append(partition)
show_grid(spatial_partitions, ["Lab only", "Lab + moderate position", "Lab + strong position"], cmap="viridis")'''),
md('''## 6. SLIC: intentionally oversegment the image

Superpixels are small coherent regions intended to preserve useful image boundaries. One semantic object usually contains many superpixels. SLIC combines color and spatial proximity while producing local regions. `n_segments` is an approximate requested count; connectivity cleanup can change the actual count.

We set `channel_axis=-1` for RGB and start IDs at 0. Superpixel boundaries are shown over the image; they are not annotations.'''),
code('''superpixels = segmentation.slic(image, n_segments=90, compactness=10,
                                sigma=1, start_label=0, channel_axis=-1)
boundary_overlay = segmentation.mark_boundaries(image, superpixels, color=(1, 1, 0))
superpixel_means_image = color.label2rgb(superpixels, image, kind="avg", bg_label=-1)
print("Requested 90 superpixels; actual count:", np.unique(superpixels).size)
show_grid([image, boundary_overlay, superpixel_means_image],
          ["Observation", "SLIC boundaries", "Mean color per superpixel"])'''),
md('''## 7. Compactness trades color fit for regularity

Low compactness lets regions follow color variation more strongly. High compactness emphasizes spatial regularity. Compare whether boundaries follow object edges and how much each object is subdivided. A clean grid is not necessarily a useful segmentation.'''),
code('''slic_overlays = []
for compactness in (1, 10, 50):
    labels = segmentation.slic(image, n_segments=90, compactness=compactness,
                               sigma=1, start_label=0, channel_axis=-1)
    slic_overlays.append(segmentation.mark_boundaries(image, labels, color=(1, 1, 0)))
show_grid(slic_overlays, ["compactness 1", "compactness 10", "compactness 50"])
'''),
md('''## 8. Compress pixels into region features

Represent every superpixel by its mean RGB color, then cluster those region descriptors. We use pixel-count weights so a larger region contributes more than a tiny one. Mapping region-cluster IDs back to pixels creates a coarser partition. This can stabilize noisy colors, but any superpixel that crosses a true boundary carries an irreversible mixing error in this simple pipeline.'''),
code('''region_ids, inverse = np.unique(superpixels.ravel(), return_inverse=True)
pixel_counts = np.bincount(inverse)
mean_rgb = np.column_stack([
    np.bincount(inverse, weights=image[..., channel].ravel()) / pixel_counts
    for channel in range(3)
])
region_model = KMeans(n_clusters=3, random_state=0, n_init=5)
region_model.fit(mean_rgb, sample_weight=pixel_counts)
region_clusters = region_model.labels_
region_partition = region_clusters[inverse].reshape(height, width)
print(f"Pixel rows: {height * width:,}; region rows: {len(region_ids)}")
print(f"Region pipeline ARI: {adjusted_rand_score(truth.ravel(), region_partition.ravel()):.3f}")
show_grid([cluster_ids, region_partition], ["Pixel color clusters", "Clusters of superpixel means"],
          columns=2, cmap="viridis")'''),
md('''## 9. Transfer to a real photograph

The astronaut image is bundled with scikit-image. We subsample to 128 × 128 so this is a quick local experiment. SLIC groups appearance; it does not identify the astronaut. No annotated mask is provided, so this step reports region counts and visual observations only.'''),
code('''photo = util.img_as_float(data.astronaut()[::4, ::4])
photo_regions = segmentation.slic(photo, n_segments=150, compactness=12,
                                  sigma=1, start_label=0, channel_axis=-1)
show_grid([photo, segmentation.mark_boundaries(photo, photo_regions),
           color.label2rgb(photo_regions, photo, kind="avg", bg_label=-1)],
          ["Bundled photo", "Superpixel boundaries", "Region mean colors"])
print("Real-photo superpixels:", np.unique(photo_regions).size)
print("Inspect the face, dark background, and bright suit for boundary leakage.")'''),
md('''## 10. Exercises

**A. Prove permutation invariance.** Randomly rename all SLIC region IDs and compare ARI between the original and renamed arrays. Why should it be exactly 1?

<details><summary>Worked solution A</summary>

```python
ids, dense = np.unique(superpixels.ravel(), return_inverse=True)
renaming = np.random.default_rng(9).permutation(len(ids))
renamed = renaming[dense].reshape(superpixels.shape)
print(adjusted_rand_score(superpixels.ravel(), renamed.ravel()))
```
Every pair of pixels stays together or stays apart. Only the names change.
</details>'''),
md('''**B. Superpixel scale.** Try 30, 100, and 300 requested regions on `photo`. Record actual counts and one boundary that improves or worsens.

<details><summary>Worked solution B</summary>

```python
overlays, titles = [], []
for count in (30, 100, 300):
    labels = segmentation.slic(photo, n_segments=count, compactness=12,
                               sigma=1, start_label=0, channel_axis=-1)
    overlays.append(segmentation.mark_boundaries(photo, labels))
    titles.append(f"requested={count}, actual={np.unique(labels).size}")
show_grid(overlays, titles)
```
More regions can preserve finer details at the cost of more descriptors and more fragmentation. They do not automatically create semantic labels.
</details>'''),
md('''**C. Build a region table.** Save one row per superpixel with its ID, size, and mean RGB. Confirm that all region sizes sum to the image pixel count.

<details><summary>Worked solution C</summary>

```python
region_table = pd.DataFrame({"region_id": region_ids, "pixels": pixel_counts,
                             "mean_R": mean_rgb[:, 0], "mean_G": mean_rgb[:, 1],
                             "mean_B": mean_rgb[:, 2]})
assert region_table["pixels"].sum() == height * width
destination = PROJECT_ROOT / "outputs" / "05_superpixel_features.csv"
destination.parent.mkdir(parents=True, exist_ok=True)
region_table.to_csv(destination, index=False)
display(region_table.head())
```
This table could feed a supervised region classifier when labeled training regions are available. That is a new learning task, not an automatic semantic interpretation of cluster IDs.
</details>'''),
md('''## Common mistakes and completion checklist

- Never equate K-means IDs with semantic IDs without a justified mapping learned independently of the test labels.
- A lower inertia value does not mean a better semantic segmentation.
- Unscaled coordinates can overwhelm color features.
- Superpixels are a representation; they are usually not the final objects.
- ARI evaluates the whole partition and can obscure particular rare-class failures. Inspect classes and boundaries too.
- Image-level train/validation/test separation is needed when evaluating generalization.

- [ ] I can reshape an image into samples and features and reconstruct it.
- [ ] I demonstrated why cluster ID accuracy is meaningless.
- [ ] I can explain the effect of spatial weight and compactness.
- [ ] I produced a table of region descriptors.
- [ ] I completed the three exercises and recorded one failure case.

**Next:** learn careful segmentation evaluation before training supervised models.'''),
md('''## Official references

- [scikit-learn KMeans API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Adjusted Rand score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
- [scikit-image SLIC API](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.slic)
- [Comparison of segmentation and superpixel algorithms](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html)
- [RGB to Lab conversion](https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.rgb2lab)

All links are optional reading; the executable path uses no web downloads.''')
]


def main():
    lessons = [
        ("02_thresholding.ipynb", thresholding, "Thresholding"),
        ("03_morphology_components.ipynb", morphology, "Morphology and connected components"),
        ("04_edges_watershed.ipynb", watershed, "Edges, markers, and watershed"),
        ("05_clustering_superpixels.ipynb", clustering, "Clustering and superpixels"),
    ]
    for filename, cells, title in lessons:
        write_notebook(ROOT / "notebooks" / filename, cells, title)
        print(f"Wrote {filename}: {len(cells)} cells")


if __name__ == "__main__":
    main()
