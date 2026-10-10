"""Deterministic toy images with exact visible-pixel semantic and instance labels.

These are educational data, not a proxy for real-world accuracy. Each image has
disks (class 1) and rectangles (class 2), plus background (class 0). Objects drawn
later occlude earlier objects, so all labels describe only the visible surface.
"""
import numpy as np

CLASS_NAMES = ("background", "disk", "rectangle")
PALETTE = np.array([[25, 32, 48], [244, 183, 66], [67, 185, 207]], dtype=np.uint8)


def make_scene(seed=0, size=96, noise=0.06, shift=False):
    """Return (RGB float32 HWC, semantic uint8 HW, instance int32 HW).

    `shift=True` changes colors, illumination, and noise while keeping label
    meanings fixed. It intentionally stresses models that memorize color.
    """
    if not isinstance(size, (int, np.integer)) or size < 32:
        raise ValueError("size must be an integer >= 32")
    if noise < 0:
        raise ValueError("noise must be nonnegative")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    image = np.full((size, size, 3), 0.13, dtype=np.float32)
    semantic = np.zeros((size, size), dtype=np.uint8)
    instances = np.zeros((size, size), dtype=np.int32)
    colors = np.array([[.88, .42, .20], [.20, .70, .88]])
    if shift:
        colors = colors[::-1].copy()
    # Four quadrants ensure each sample contains both classes and visible objects.
    centers = np.array([(0.29, .29), (.29, .72), (.72, .29), (.72, .72)])
    centers = centers[rng.permutation(len(centers))]  # Class must not be fixed by quadrant.
    for i, (cy, cx) in enumerate(centers, start=1):
        cy = int((cy + rng.uniform(-.06, .06)) * size)
        cx = int((cx + rng.uniform(-.06, .06)) * size)
        radius = int(rng.integers(max(3, size // 10), max(4, size // 6)))
        cls = 1 if i % 2 else 2
        if cls == 1:
            region = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        else:
            ry = max(3, int(radius * rng.uniform(.7, 1.25)))
            region = (abs(yy - cy) <= ry) & (abs(xx - cx) <= radius)
        image[region] = np.clip(colors[cls - 1] + rng.normal(0, .04, 3), 0, 1)
        semantic[region] = cls
        instances[region] = i
    illumination = (xx / max(size - 1, 1) - .5) * (.32 if shift else .12)
    image += illumination[..., None].astype(np.float32)
    image += rng.normal(0, noise * (1.5 if shift else 1), image.shape).astype(np.float32)
    return np.clip(image, 0, 1).astype(np.float32), semantic, instances


def make_dataset(n=24, size=64, seed=0, noise=0.06, shift=False):
    """Create independent scenes using consecutive seeds; split seeds by image."""
    if n < 1:
        raise ValueError("n must be positive")
    scenes = [make_scene(seed + i, size, noise, shift) for i in range(n)]
    return np.stack([s[0] for s in scenes]), np.stack([s[1] for s in scenes])


def palette_mask(mask):
    """Display class IDs without changing the stored label values."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Expected a 2-D integer semantic mask")
    if np.any((mask < 0) | (mask >= len(PALETTE))):
        raise ValueError("Expected IDs 0, 1, 2; map ignore labels separately")
    return PALETTE[mask]


def show_triplet(image, truth, prediction, title=""):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for ax, arr, label in zip(axes, [image, palette_mask(truth), palette_mask(prediction)],
                              ["Image", "Ground truth", "Prediction"]):
        ax.imshow(arr)
        ax.set_title(label)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
