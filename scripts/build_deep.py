"""Build the four deep-learning lessons. Run from the project root."""
from pathlib import Path
import sys

from notebook_builder import md, code, write_notebook, bootstrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TORCH_SETUP = '''
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from learning.data import make_scene, make_dataset, show_triplet, palette_mask
from learning.torch_models import ArrayDataset, PixelCNN, TinyUNet
from learning.torch_models import run_epoch, confusion_matrix, iou_from_confusion

SEED = 42
FAST_RUN = True  # False increases training; keep True for your first Run All.
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__, "| device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print("All DataLoaders use num_workers=0 for a simple Windows notebook workflow.")
'''


def save(name, cells, title):
    write_notebook(ROOT / "notebooks" / name, cells, title)


def lesson08():
    cells = [
md('''# 08 · Your first learned segmentation model

**Prerequisites:** you can write Python and NumPy code; complete lessons 01 and 07 first.
**Learning time:** 60–90 minutes. Default execution is a small offline run; no GPU is required.

You will turn an RGB image into a class label at every pixel, write a `Dataset`, train a small CNN, and inspect its errors. This is **semantic segmentation**: two disks share one class, even if they are different objects.

**Success check:** explain the shapes `[N,3,H,W] → [N,K,H,W] → [N,H,W]`, distinguish logits from labels, and improve held-out IoU over an all-background baseline.

Run each cell, predict its output first, then change one thing. These deliberately simple colored shapes teach the training mechanics; good scores here do not demonstrate real-world segmentation ability.'''),
bootstrap(), code(TORCH_SETUP),
md('''## 1. Make examples with known answers

The generator returns images in `[0,1]` and integer masks: `0=background`, `1=disk`, `2=rectangle`. An image contains visual evidence; its mask contains the answer supplied during supervised training. We split **whole images**, so pixels from one scene cannot leak between train and validation.'''),
code('''images, masks = make_dataset(n=48 if FAST_RUN else 240, size=48, seed=SEED)
order = np.random.default_rng(SEED).permutation(len(images))
cut = int(0.75 * len(images))
train_ids, val_ids = order[:cut], order[cut:]
assert set(train_ids).isdisjoint(val_ids)
print("images:", images.shape, images.dtype, "masks:", masks.shape, masks.dtype)
show_triplet(images[0], masks[0], masks[0], title="The supplied target mask");
plt.show()'''),
md('''## 2. Learn the tensor contract

NumPy images often put RGB last (`HWC`). PyTorch convolution layers expect channels first (`CHW`) and a batch dimension (`NCHW`). A target for multiclass cross-entropy is a `long` integer tensor, **without a channel dimension**. Do not normalize class IDs or divide masks by 255.

`permute` reorders axes; it does not rescale values. `argmax(dim=1)` will later select a class, not an x/y coordinate.'''),
code('''one_image = torch.from_numpy(images[0]).permute(2, 0, 1).float()
one_mask = torch.from_numpy(masks[0].astype(np.int64))
print("one image:", one_image.shape, one_image.dtype)
print("one mask:", one_mask.shape, one_mask.dtype)
assert one_image.shape == (3, 48, 48)
assert one_mask.shape == (48, 48) and one_mask.dtype == torch.long
assert set(one_mask.unique().tolist()) <= {0, 1, 2}'''),
md('''## 3. Separate one sample from a batch

A `Dataset` answers “what are image and target number i?” A `DataLoader` groups samples and optionally shuffles their order. Shuffle training data; keep validation stable. Use `num_workers=0` while learning on Windows to avoid notebook multiprocessing surprises.'''),
code('''class ShapeDataset(Dataset):
    def __init__(self, rgb_images, label_masks):
        self.images, self.masks = rgb_images, label_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        x = torch.from_numpy(self.images[index].transpose(2, 0, 1).copy()).float()
        y = torch.from_numpy(self.masks[index].astype(np.int64))
        return x, y

train_loader = DataLoader(ShapeDataset(images[train_ids], masks[train_ids]),
                          batch_size=6, shuffle=True, num_workers=0,
                          generator=torch.Generator().manual_seed(SEED))
val_loader = DataLoader(ShapeDataset(images[val_ids], masks[val_ids]),
                        batch_size=6, shuffle=False, num_workers=0)
xb, yb = next(iter(train_loader))
print("batch image:", xb.shape, "batch target:", yb.shape)'''),
md('''## 4. A CNN predicts one score per class per pixel

The `3×3` convolutions combine nearby RGB values. Padding keeps height and width unchanged. The final `1×1` convolution mixes features into three output scores called **logits**. We use no pooling here; lesson 09 adds an encoder, decoder, and skip connections.

This baseline has a small receptive field. Because shape colors are informative, it can succeed without truly understanding shape.'''),
code('''class FirstSegmentationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(12, 3, kernel_size=1),
        )

    def forward(self, x):
        return self.layers(x)

model = FirstSegmentationCNN().to(DEVICE)
logits = model(xb.to(DEVICE))
assert logits.shape == (len(xb), 3, 48, 48)
print("logits:", logits.shape, "parameters:", sum(p.numel() for p in model.parameters()))'''),
md('''## 5. Turn scores into a loss, probabilities, and labels

These are three different operations. Cross-entropy compares **raw logits** with integer targets and internally performs the stable log-softmax calculation. Softmax produces probabilities for inspection. Argmax produces the discrete class map. Argmax is not differentiable, so do not use an argmax mask as the training loss input.'''),
code('''criterion = nn.CrossEntropyLoss()
initial_loss = criterion(logits, yb.to(DEVICE))
probabilities = logits.softmax(dim=1)
predicted_mask = logits.argmax(dim=1)
assert torch.allclose(probabilities.sum(1), torch.ones_like(probabilities[:, 0]), atol=1e-6)
print("initial loss:", round(initial_loss.item(), 4))
print("probabilities:", probabilities.shape, "prediction:", predicted_mask.shape)
print("one pixel's probabilities:", probabilities[0, :, 20, 20].detach().cpu().numpy())'''),
md('''## 6. Train: forward → loss → zero gradients → backward → update

Adam updates parameters using gradients. Gradients accumulate unless cleared. `model.train()` activates training behavior; `model.eval()` selects inference behavior for layers such as dropout or batch normalization. This tiny model has neither, but use the correct pattern from the beginning.

We record validation loss and IoU after every epoch. Validation guides experiments; it is not a final untouched test set.'''),
code('''optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 8 if FAST_RUN else 40
history = []
for epoch in range(epochs):
    model.train()
    total, count = 0.0, 0
    for batch_images, batch_masks in train_loader:
        batch_images, batch_masks = batch_images.to(DEVICE), batch_masks.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch_images)
        loss = criterion(output, batch_masks)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch_images)
        count += len(batch_images)
    validation = run_epoch(model, val_loader, criterion, classes=3)
    history.append((total / count, validation["loss"], validation["miou"]))
    print(f"epoch {epoch+1:02d} | train CE {total/count:.3f} | val CE {validation['loss']:.3f} | val mIoU {validation['miou']:.3f}")'''),
code('''history = np.asarray(history)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(history[:, 0], label="train"); axes[0].plot(history[:, 1], label="validation")
axes[0].set(xlabel="epoch (zero based)", ylabel="cross-entropy"); axes[0].legend()
axes[1].plot(history[:, 2], color="tab:green")
axes[1].set(xlabel="epoch (zero based)", ylabel="validation macro IoU", ylim=(0, 1))
plt.tight_layout(); plt.show()'''),
md('''## 7. Inspect failures, not just a mean score

`inference_mode()` prevents gradient recording during prediction. Macro IoU gives each present class equal weight; pixel accuracy can hide small foreground failures. Our helper aggregates a confusion matrix over all validation pixels before computing per-class IoU. A class absent from truth **and** prediction is excluded as undefined.'''),
code('''model.eval()
with torch.inference_mode():
    val_tensor = torch.from_numpy(images[val_ids].transpose(0, 3, 1, 2)).to(DEVICE)
    predictions = model(val_tensor).argmax(1).cpu().numpy()
truth = torch.from_numpy(masks[val_ids].astype(np.int64))
baseline_cm = confusion_matrix(torch.zeros_like(truth), truth, 3)
print("background-only per-class IoU:", iou_from_confusion(baseline_cm).numpy().round(3))
print("trained per-class IoU:", validation["iou"].round(3))
show_triplet(images[val_ids[0]], masks[val_ids[0]], predictions[0], title="Held-out scene")
plt.show()'''),
md('''## Experiment 1 · Read a different pixel

Before running: if logits are `[N,3,H,W]`, what shape do probabilities for one pixel have? Change `(row, col)` to a background pixel and a disk pixel. Observe whether high probability corresponds to correctness.'''),
code('''row, col = 20, 20
with torch.inference_mode():
    pixel_p = model(val_tensor[:1]).softmax(1)[0, :, row, col].cpu().numpy()
print("p(background, disk, rectangle):", pixel_p.round(3))
print("true class:", masks[val_ids[0], row, col], "predicted:", pixel_p.argmax())'''),
md('''<details><summary>Worked solution 1</summary>

The shape is `(3,)`. Use `p = model(val_tensor[:1]).softmax(1)[0, :, row, col]`. The class probabilities sum to one. A confident wrong answer is possible; softmax is not a guarantee of correctness. Coordinates index `[row, column]`, not `[x, y]`.
</details>

## Experiment 2 · Remove spatial context

Replace **both** `3×3` layers with `kernel_size=1, padding=0`. Recreate the model and optimizer, then rerun training. Record per-class IoU, not only loss. Does this dataset really require spatial reasoning?'''),
code('''# Working starting point for your ablation; training is intentionally your task.
color_only = nn.Sequential(nn.Conv2d(3, 12, 1), nn.ReLU(),
                           nn.Conv2d(12, 12, 1), nn.ReLU(), nn.Conv2d(12, 3, 1)).to(DEVICE)
print("color-only output shape:", tuple(color_only(xb.to(DEVICE)).shape))'''),
md('''<details><summary>Worked solution 2</summary>

Create `optimizer_color = torch.optim.Adam(color_only.parameters(), lr=0.01)` and run `run_epoch(color_only, train_loader, criterion, optimizer_color)` for the same number of epochs. Evaluate with `run_epoch(color_only, val_loader, criterion)`. A `1×1` network classifies each pixel independently. Strong results would expose how easy color cues make this dataset; they would not prove shape understanding.
</details>

## Experiment 3 · Cause a controlled failure

Repeat the experiment with `make_dataset(..., noise=0.20)` and the same train/validation procedure. Predict what should happen to boundary errors. Keep the validation set fixed when comparing architectures.'''),
md('''<details><summary>Worked solution 3</summary>

Generate noisy scenes with a separate fixed seed, convert them to `ArrayDataset`, and evaluate the existing model. This measures robustness without retraining. Then train a fresh model with noisy training data and compare on the **same** noisy validation scenes. Do not claim the noise helped from different validation sets. Expect more ambiguity, but measure rather than assume.
</details>

## Checkpoint and common errors

- Explain why the target has no RGB channel and why it uses `torch.long`.
- If the loss reports a shape mismatch, print **both** logits and targets immediately before the loss.
- If a class is out of bounds, inspect `np.unique(masks)` and the number of output channels.
- If validation still builds graphs, use `inference_mode()` or `no_grad()`.
- If CUDA is unavailable, this lesson runs on CPU; an NVIDIA GPU also needs a CUDA-enabled PyTorch installation.
- Before moving on, explain why more training cannot fix incorrect labels or image/mask misalignment.'''),
md('''## References and next step

- [PyTorch cross-entropy contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch Dataset and DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch training loop](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

Continue with **09_unet_training.ipynb**: recover fine details using skip connections, separate train/validation/test, and save the best model.''')]
    save("08_pytorch_segmentation_basics.ipynb", cells, "Your first learned segmentation model")


def lesson09():
    # Read only our own teaching helper, so the visible architecture cannot drift.
    source = (ROOT / "learning" / "torch_models.py").read_text(encoding="utf-8")
    unet_source = source[source.index("class TinyUNet"):source.index("def confusion_matrix")].strip()
    cells = [
md('''# 09 · Build, train, and reload a small U-Net

**Prerequisite:** lesson 08. **Learning time:** 90–120 minutes. **Runs independently:** creates its own data and model.

U-Net combines low-resolution context with high-resolution detail. You will trace its tensor shapes, apply paired augmentation, train with separate validation data, restore the best epoch, and test a saved model.

**Success check:** show disjoint image splits, explain each skip connection, select a checkpoint using validation only, and reproduce its prediction after reload. Our small implementation is inspired by U-Net; it is not an exact reproduction of the original architecture.'''),
bootstrap(), code(TORCH_SETUP),
md('''## 1. Split before augmenting

Training adjusts weights. Validation selects settings or checkpoints. The test split estimates performance only after those choices are fixed. Real datasets need patient/site/video-level splits when images share a source; random image splitting alone can still leak information.'''),
code('''images, masks = make_dataset(n=60 if FAST_RUN else 300, size=48, seed=19)
ids = np.random.default_rng(SEED).permutation(len(images))
a, b = int(0.6 * len(ids)), int(0.8 * len(ids))
train_ids, val_ids, test_ids = ids[:a], ids[a:b], ids[b:]
assert set(train_ids).isdisjoint(val_ids) and set(train_ids).isdisjoint(test_ids)
assert set(val_ids).isdisjoint(test_ids)
print("train / validation / test:", len(train_ids), len(val_ids), len(test_ids))'''),
md('''## 2. Apply the same geometry to image and target

A horizontal flip must affect both arrays using the same random decision. Brightness changes belong only on the image. For resizing, images may use bilinear interpolation; masks need **nearest neighbor**, which preserves integer class IDs.

The following function demonstrates both. The training dataset uses random paired flips; the resizing example is a separate explicit inspection.'''),
code('''def paired_resize_flip(image, mask, output_size=(48, 48), flip=False):
    x = F.interpolate(image[None], size=output_size, mode="bilinear", align_corners=False)[0]
    y = F.interpolate(mask[None, None].float(), size=output_size, mode="nearest")[0, 0].long()
    if flip:
        x, y = x.flip(-1), y.flip(-1)
    return x, y

sample_x = torch.from_numpy(images[train_ids[0]].transpose(2, 0, 1))
sample_y = torch.from_numpy(masks[train_ids[0]].astype(np.int64))
aug_x, aug_y = paired_resize_flip(sample_x, sample_y, (57, 65), flip=True)
assert set(aug_y.unique().tolist()) <= set(sample_y.unique().tolist())
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].imshow(aug_x.permute(1, 2, 0)); axes[0].set_title("paired transformed image")
axes[1].imshow(palette_mask(aug_y.numpy())); axes[1].set_title("nearest-neighbor target")
for ax in axes: ax.axis("off")
plt.show()'''),
code('''train_loader = DataLoader(ArrayDataset(images[train_ids], masks[train_ids], augment=True),
                          batch_size=6, shuffle=True, num_workers=0,
                          generator=torch.Generator().manual_seed(SEED))
val_loader = DataLoader(ArrayDataset(images[val_ids], masks[val_ids]), batch_size=6, num_workers=0)
test_loader = DataLoader(ArrayDataset(images[test_ids], masks[test_ids]), batch_size=6, num_workers=0)
xb, yb = next(iter(train_loader))
assert xb.shape[1] == 3 and yb.dtype == torch.long
print("training batch:", tuple(xb.shape), tuple(yb.shape))'''),
md('''## 3. Read the architecture before running it

The encoder keeps a detailed feature map, pools to half resolution, and learns more channels. A second pooling reaches one-quarter resolution. The decoder upsamples and concatenates the corresponding encoder features along the **channel** axis.

For width 16 and a `48×48` image: `16×48×48 → 32×24×24 → 64×12×12 → 32×24×24 → 16×48×48 → 3×48×48`. The first decoder sees `64+32=96` channels. Interpolating to a skip tensor's actual size also handles odd image dimensions.'''),
code(unet_source),
code('''torch.manual_seed(SEED)
UNET_WIDTH = 16
model = TinyUNet(classes=3, width=UNET_WIDTH).to(DEVICE)
assert model(xb.to(DEVICE)).shape == (len(xb), 3, 48, 48)
assert model(torch.rand(1, 3, 49, 53, device=DEVICE)).shape == (1, 3, 49, 53)
print("parameters:", sum(p.numel() for p in model.parameters()))
print("The visible TinyUNet class matches learning/torch_models.py.")'''),
md('''## 4. Train and keep the best validation checkpoint

We optimize cross-entropy and monitor global macro IoU. Different metrics can select different epochs; state the selection rule before viewing test results. `deepcopy` matters: assigning `best_state = model.state_dict()` alone keeps references to tensors that continue changing.

Early epochs often favor background because it covers many pixels. Watch per-class predictions and the validation curve before deciding that training is broken. The quick run uses 25 short epochs to let this small encoder/decoder move beyond that initial stage; convergence still depends on the data and settings.

The helper `run_epoch` is the same forward/loss/backward loop from lesson 08. It computes a confusion matrix over the full split instead of averaging per-batch IoUs.'''),
code('''criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
epochs = 25 if FAST_RUN else 60
best_score, best_epoch, best_state = -float("inf"), -1, None
history = []
for epoch in range(epochs):
    training = run_epoch(model, train_loader, criterion, optimizer)
    validation = run_epoch(model, val_loader, criterion)
    history.append((training["loss"], validation["loss"], training["miou"], validation["miou"]))
    if validation["miou"] > best_score:
        best_score, best_epoch = validation["miou"], epoch + 1
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    print(f"epoch {epoch+1:02d} | train CE {training['loss']:.3f} | val mIoU {validation['miou']:.3f} | class IoU {validation['iou'].round(3)}")
assert best_state is not None
model.load_state_dict(best_state)
model.eval()
print("Restored best validation epoch:", best_epoch, "with mIoU", round(best_score, 3))'''),
code('''h = np.asarray(history)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for values, name in [(h[:, 0], "train"), (h[:, 1], "validation")]: axes[0].plot(values, label=name)
for values, name in [(h[:, 2], "train"), (h[:, 3], "validation")]: axes[1].plot(values, label=name)
axes[0].set(ylabel="cross-entropy", xlabel="epoch (zero based)")
axes[1].set(ylabel="macro IoU", xlabel="epoch (zero based)", ylim=(0, 1))
for ax in axes: ax.axvline(best_epoch - 1, color="gray", linestyle="--"); ax.legend()
plt.tight_layout(); plt.show()'''),
md('''## 5. Test once after choosing the checkpoint

This is a tiny illustrative test set, so treat its score as a noisy estimate. Looking at its failures is useful for a new research iteration, but repeatedly optimizing against this same test set turns it into another validation set.'''),
code('''test_result = run_epoch(model, test_loader, criterion)
print("test per-class IoU [background,disk,rectangle]:", test_result["iou"].round(3))
print("test macro IoU:", round(test_result["miou"], 3))
with torch.inference_mode():
    test_x = torch.from_numpy(images[test_ids[:2]].transpose(0, 3, 1, 2)).to(DEVICE)
    test_logits = model(test_x)
    test_prediction = test_logits.argmax(1).cpu().numpy()
for i in range(2):
    show_triplet(images[test_ids[i]], masks[test_ids[i]], test_prediction[i], title=f"Test scene {i}")
    plt.show()'''),
md('''## 6. Save a useful artifact, then prove that reload works

A weight file without architecture and preprocessing information is incomplete. Save class names, shape assumptions, selection metric, and split seed. This is an **inference checkpoint**; exact training resumption would additionally require optimizer, scheduler, and random-generator states.

Only load checkpoints you trust. The example uses your own checkpoint with `weights_only=True` and loads it onto the selected device.'''),
code('''artifact_dir = PROJECT_ROOT / "artifacts" / "09_unet"
artifact_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = artifact_dir / "tiny_unet.pt"
checkpoint = {
    "state_dict": best_state,
    "architecture": "TinyUNet", "width": UNET_WIDTH, "classes": ["background", "disk", "rectangle"],
    "input_range": [0.0, 1.0], "input_layout": "NCHW RGB", "split_seed": SEED,
    "best_epoch": best_epoch, "validation_miou": float(best_score),
    "torch_version": str(torch.__version__),
}
torch.save(checkpoint, checkpoint_path)
loaded = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
reloaded = TinyUNet(classes=len(loaded["classes"]), width=loaded["width"]).to(DEVICE)
reloaded.load_state_dict(loaded["state_dict"]); reloaded.eval()
with torch.inference_mode():
    reloaded_logits = reloaded(test_x)
torch.testing.assert_close(test_logits, reloaded_logits)
print("Reload matches original logits:", checkpoint_path)'''),
md('''## Experiment 1 · Inspect skip-connection shapes

Add temporary forward hooks to `enc1`, `enc2`, and `bridge`. Predict their output shapes before execution. Remove hooks afterward so rerunning does not accumulate print statements.'''),
code('''handles = []
def shape_hook(name):
    return lambda module, inputs, output: print(name, tuple(output.shape))
for name in ["enc1", "enc2", "bridge"]:
    handles.append(getattr(model, name).register_forward_hook(shape_hook(name)))
with torch.inference_mode():
    _ = model(test_x[:1])
for handle in handles: handle.remove()'''),
md('''<details><summary>Worked solution 1</summary>

Expected shapes are `(1,16,48,48)`, `(1,32,24,24)`, `(1,64,12,12)`. Concatenation increases channels, not spatial size. The decoder upsamples first because a `12×12` tensor cannot be concatenated with a `24×24` skip along channels.
</details>

## Experiment 2 · A fair augmentation ablation

Compare paired flips on versus off. Recreate model, optimizer, shuffled loader, and random seeds for each run. Use the same validation IDs and training budget. Leave the test split closed during this comparison.'''),
md('''<details><summary>Worked solution 2</summary>

Use `ArrayDataset(images[train_ids], masks[train_ids], augment=False)` for the second condition. Reset `torch.manual_seed(SEED)` before creating each model and initialize both loader generators with the same seed. Record best validation mIoU and per-class IoU. A single seed and this symmetric toy dataset may show little or no benefit; repeat several seeds before drawing a conclusion.
</details>

## Experiment 3 · Overfit two images as a debugging test

Build a training loader containing only two images, turn augmentation off, and train longer. The target is near-perfect **training** segmentation. Why is this useful even though it does not measure generalization?'''),
md('''<details><summary>Worked solution 3</summary>

Use `DataLoader(ArrayDataset(images[train_ids[:2]], masks[train_ids[:2]]), batch_size=2, num_workers=0)`. Create a fresh TinyUNet and optimizer, then train for 100–200 small steps. If loss cannot fall, check labels, model output channels, gradients, normalization, and whether the optimizer owns this model's parameters. Success shows the pipeline can fit these examples; it does not prove the model generalizes.
</details>'''),
md('''## Checkpoint and common errors

- Trace the 96 decoder channels back to `64 + 32`.
- Distinguish bilinear image resizing from nearest-neighbor label resizing.
- Show that train/validation/test IDs do not overlap.
- Explain why the final training epoch may differ from the saved epoch.
- A device mismatch means model and tensors must both move to `DEVICE`.
- A shape mismatch at concatenation means the spatial sizes must be aligned before `torch.cat`.
- Save after selecting the best validation state; never select with test IoU.'''),
md('''## References and next step

- [Original U-Net paper](https://arxiv.org/abs/1505.04597)
- [PyTorch interpolation API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)
- [PyTorch saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

Continue with **10_real_data_transfer_learning.ipynb** to replace arrays with image/mask folders and explore optional pretrained transfer learning.''')]
    save("09_unet_training.ipynb", cells, "Build, train, and reload a small U-Net")


def lesson10():
    cells = [
md('''# 10 · Real image/mask folders and optional transfer learning

**Prerequisites:** lessons 08–09. **Learning time:** 90–150 minutes.

The first half runs offline: it writes a small teaching dataset to disk and uses the same loader you can adapt to your own labeled images. You will verify pairing, mask IDs, resize rules, and image-level splits before training.

The second half is **optional**: Oxford-IIIT Pet images plus pretrained LRASPP/MobileNetV3. It is disabled by default, requires internet or an existing cache, and can take substantially longer. The real-data training path is supplied for you to run; it is **not validated by the offline notebook execution**.

**Success check:** catch a missing mask or unknown class ID before training; preserve class labels while resizing; explain why a pretrained head must be adapted to your class definitions.'''),
bootstrap(), code(TORCH_SETUP + '\nfrom PIL import Image\n'),
md('''## 1. The folder contract

Use `images/sample_001.png` with `masks/sample_001.png`. Pair by filename stem, not by independently sorting two arbitrary lists. Images may be RGB JPEG/PNG; labels must be lossless single-channel or palette-index PNG/TIFF. A colored preview is not necessarily a class-ID mask.

The demo intentionally stores IDs `0,10,20` to show that on-disk values need not match model output indices. We explicitly map them to contiguous `0,1,2`. Save your real data elsewhere; only the generated demo folder is rewritten by this lesson.'''),
code('''demo_root = PROJECT_ROOT / "artifacts" / "10_folder_demo"
for folder in ["images", "masks"]:
    (demo_root / folder).mkdir(parents=True, exist_ok=True)
demo_images, demo_masks = make_dataset(n=30, size=64, seed=300)
raw_ids = np.array([0, 10, 20], dtype=np.uint8)
for index, (image, mask) in enumerate(zip(demo_images, demo_masks)):
    name = f"scene_{index:03d}.png"
    Image.fromarray(np.round(image * 255).astype(np.uint8)).save(demo_root / "images" / name)
    Image.fromarray(raw_ids[mask]).save(demo_root / "masks" / name)
CLASS_MAP = {0: 0, 10: 1, 20: 2}
print("Generated demo only:", demo_root)'''),
md('''## 2. Validate the whole dataset before training

This loader rejects duplicate stems, missing pairs, size mismatches, RGB masks, unknown IDs, and empty datasets. It inspects masks **before** resizing so a tiny invalid region cannot disappear unnoticed. Palette-mode masks are read as raw indices, without converting their display colors to grayscale.

The complete implementation is visible below. For RGB-coded masks, write an explicit RGB-tuple-to-class mapping first; do not guess that grayscale conversion preserves labels.'''),
code('''class FolderSegmentationDataset(Dataset):
    def __init__(self, root, class_map, size=(48, 48), classes=3, ignore_index=255):
        self.root, self.size = Path(root), size  # size is (height, width)
        self.class_map = dict(class_map)
        self.classes, self.ignore_index = classes, ignore_index
        if not self.class_map or any(v not in {*range(classes), ignore_index} for v in self.class_map.values()):
            raise ValueError("Map values must be class IDs 0..classes-1 or ignore_index")

        def by_stem(folder, allowed):
            paths = {}
            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.suffix.lower() not in allowed:
                    continue
                key = path.stem.casefold()
                if key in paths:
                    raise ValueError(f"Duplicate stem in {folder}: {path.stem}")
                paths[key] = path
            return paths

        images = by_stem(self.root / "images", {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
        masks = by_stem(self.root / "masks", {".png", ".tif", ".tiff"})
        if not images:
            raise ValueError("No images found")
        if images.keys() != masks.keys():
            raise ValueError(f"Unpaired files: missing masks={sorted(images.keys()-masks.keys())[:5]}, "
                             f"missing images={sorted(masks.keys()-images.keys())[:5]}")
        self.pairs = [(images[key], masks[key]) for key in sorted(images)]
        for image_path, mask_path in self.pairs:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                raw = np.asarray(mask)
                if image.size != mask.size:
                    raise ValueError(f"Image/mask size mismatch: {image_path.name}")
                if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
                    raise ValueError(f"Expected 2-D integer IDs: {mask_path.name}")
                unknown = set(np.unique(raw).tolist()) - self.class_map.keys()
                if unknown:
                    raise ValueError(f"Unknown IDs {unknown} in {mask_path.name}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        pil_size = (self.size[1], self.size[0])
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize(pil_size, Image.Resampling.BILINEAR)
            x = np.array(image, dtype=np.float32) / 255.0
        with Image.open(mask_path) as mask:
            raw = np.array(mask.resize(pil_size, Image.Resampling.NEAREST))
        y = np.full(raw.shape, self.ignore_index, dtype=np.int64)
        for raw_id, class_id in self.class_map.items():
            y[raw == raw_id] = class_id
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(y)

folder_dataset = FolderSegmentationDataset(demo_root, CLASS_MAP)
x, y = folder_dataset[0]
print("samples:", len(folder_dataset), "image:", x.shape, "target IDs:", y.unique().tolist())
assert x.dtype == torch.float32 and y.dtype == torch.long
show_triplet(x.permute(1, 2, 0).numpy(), y.numpy(), y.numpy(), title="Loaded from disk")
plt.show()'''),
md('''## 3. Prove a bad annotation is caught

Validation errors should describe the problem instead of silently producing a wrong model. The next cell gives a deliberately incomplete mapping, catches the expected exception, and leaves all files unchanged.'''),
code('''try:
    FolderSegmentationDataset(demo_root, {0: 0, 10: 1})
except ValueError as error:
    print("Expected validation error:", error)
else:
    raise AssertionError("The loader should reject unknown mask ID 20")'''),
md('''## 4. Train the folder pipeline end to end

Nothing downstream should depend on whether samples came from memory, local folders, or a dataset library. Keep the tensor contract identical. For your own data, first visualize at least ten image/mask pairs and audit the class mapping.'''),
code('''permutation = np.random.default_rng(SEED).permutation(len(folder_dataset))
train_ids, val_ids, test_ids = permutation[:18], permutation[18:24], permutation[24:]
assert set(train_ids).isdisjoint(val_ids) and set(val_ids).isdisjoint(test_ids)
train_loader = DataLoader(Subset(folder_dataset, train_ids.tolist()), batch_size=6, shuffle=True, num_workers=0,
                          generator=torch.Generator().manual_seed(SEED))
val_loader = DataLoader(Subset(folder_dataset, val_ids.tolist()), batch_size=6, num_workers=0)
test_loader = DataLoader(Subset(folder_dataset, test_ids.tolist()), batch_size=6, num_workers=0)
torch.manual_seed(SEED)
model = PixelCNN(classes=3).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss(ignore_index=255)
folder_history, best_iou, best_state = [], -1.0, None
for epoch in range(8 if FAST_RUN else 40):
    train_result = run_epoch(model, train_loader, loss_function, optimizer)
    val_result = run_epoch(model, val_loader, loss_function)
    folder_history.append([train_result["loss"], val_result["loss"], val_result["miou"]])
    if val_result["miou"] > best_iou:
        best_iou = val_result["miou"]
        best_state = copy.deepcopy(model.state_dict())
model.load_state_dict(best_state)
test_result = run_epoch(model, test_loader, loss_function)
print("best validation mIoU:", round(best_iou, 3), "test mIoU:", round(test_result["miou"], 3))'''),
code('''folder_history = np.asarray(folder_history)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(folder_history[:, 0], label="train CE"); axes[0].plot(folder_history[:, 1], label="val CE")
axes[0].legend(); axes[0].set(xlabel="epoch", ylabel="cross-entropy")
axes[1].plot(folder_history[:, 2]); axes[1].set(xlabel="epoch", ylabel="validation mIoU", ylim=(0, 1))
plt.tight_layout(); plt.show()
sample_x, sample_y = folder_dataset[int(test_ids[0])]
model.eval()
with torch.inference_mode():
    sample_prediction = model(sample_x[None].to(DEVICE)).argmax(1)[0].cpu().numpy()
show_triplet(sample_x.permute(1, 2, 0).numpy(), sample_y.numpy(), sample_prediction, title="Folder pipeline test")
plt.show()'''),
md('''## 5. Optional real-data project: Oxford-IIIT Pet

**Disabled by default; this section needs an intentional opt-in.** The image/annotation archive is roughly 800 MB compressed, plus extraction space and model weights. Downloads and real-data training were not run as part of offline validation. Review the [dataset's current terms and citation](https://www.robots.ox.ac.uk/~vgg/data/pets/) before your own use.

The **original archive README** defines `1=foreground`, `2=background`, `3=not classified`. Here map `1→1` (pet), `2→0` (background), `3→255` (ignore). Ignore means those uncertain pixels contribute neither loss nor IoU; it is not a third predicted class. Avoid copying label meanings from a rendered colormap.

Set `RUN_REAL_PETS=True` and `ALLOW_DOWNLOAD=True` to fetch missing resources. With `ALLOW_DOWNLOAD=False`, this section requires an already complete dataset and cached weights. Keep the official test split untouched while developing.'''),
code('''RUN_REAL_PETS = False
ALLOW_DOWNLOAD = False
real_ready = False
PET_SIZE = (128, 128)  # Small learning experiment; square resizing distorts aspect ratio.
PET_ROOT = PROJECT_ROOT / "data" / "external"
PET_RAW_TO_CLASS = {1: 1, 2: 0, 3: 255}

if RUN_REAL_PETS:
    import torchvision
    from torchvision.datasets import OxfordIIITPet
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
    from urllib.parse import urlparse
    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    weight_cache = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    if not ALLOW_DOWNLOAD and not weight_cache.is_file():
        raise FileNotFoundError("Pretrained weights are not cached. Opt in with ALLOW_DOWNLOAD=True, then rerun.")
    pet_raw = OxfordIIITPet(PET_ROOT, split="trainval", target_types="segmentation", download=ALLOW_DOWNLOAD)
    print("torchvision:", torchvision.__version__, "trainval images:", len(pet_raw))
    real_ready = True
else:
    print("Real-data download and training skipped. The offline lesson is complete and runnable.")'''),
md('''## 6. Match the pretrained input distribution

Pretrained LRASPP expects ImageNet-style RGB normalization. We use the documented mean/std but a smaller fixed resolution for a quick training experiment; this differs from the default pretrained inference resize. Transform the mask with nearest neighbor using the **same geometric size**.

For stronger real experiments, preserve aspect ratio with paired resize/crop or padding, apply paired augmentation on training only, and retain original sizes for evaluation. The short demo below omits augmentation to make the data path easy to inspect.'''),
code('''class PetPairs(Dataset):
    def __init__(self, source, indices):
        self.source, self.indices = source, list(indices)
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, mask = self.source[int(self.indices[index])]
        raw = np.asarray(mask)
        if image.size != mask.size or raw.ndim != 2:
            raise ValueError("Pet image/mask geometry is invalid")
        if not set(np.unique(raw).tolist()) <= {1, 2, 3}:
            raise ValueError("Unexpected raw Oxford trimap IDs")
        size_wh = (PET_SIZE[1], PET_SIZE[0])
        image = np.array(image.convert("RGB").resize(size_wh, Image.Resampling.BILINEAR), dtype=np.float32) / 255
        raw = np.array(mask.resize(size_wh, Image.Resampling.NEAREST))
        target = np.full(raw.shape, 255, dtype=np.int64)
        for old_id, new_id in PET_RAW_TO_CLASS.items(): target[raw == old_id] = new_id
        x = torch.from_numpy(image.transpose(2, 0, 1).copy())
        return (x - self.mean) / self.std, torch.from_numpy(target)

if real_ready:
    pet_ids = np.random.default_rng(SEED).permutation(len(pet_raw))
    train_count, val_count = (40, 8) if FAST_RUN else (256, 64)
    pet_train_ids = pet_ids[:train_count]
    pet_val_ids = pet_ids[train_count:train_count + val_count]
    assert set(pet_train_ids).isdisjoint(pet_val_ids)
    pet_train = PetPairs(pet_raw, pet_train_ids)
    pet_val = PetPairs(pet_raw, pet_val_ids)
    pet_train_loader = DataLoader(pet_train, batch_size=2, shuffle=True, num_workers=0,
                                  generator=torch.Generator().manual_seed(SEED))
    pet_val_loader = DataLoader(pet_val, batch_size=2, num_workers=0)
    print("Optional subset train/validation:", len(pet_train), len(pet_val))
else:
    print("PetPairs defined; no real files accessed.")'''),
md('''## 7. Adapt the pretrained output head

The pretrained model predicts its original categories, not our binary pet/background labels. LRASPP combines low- and high-resolution branches, so replace **both** final classifiers. Freeze the backbone initially to train only the head; unfreezing later with a smaller learning rate is a separate experiment.

Keep frozen backbone batch-normalization statistics in evaluation mode. Freezing parameter gradients alone does not stop those running statistics from changing.'''),
code('''if real_ready:
    torch.manual_seed(SEED)
    pet_model = lraspp_mobilenet_v3_large(weights=weights).to(DEVICE)
    for parameter in pet_model.backbone.parameters(): parameter.requires_grad_(False)
    low_channels = pet_model.classifier.low_classifier.in_channels
    high_channels = pet_model.classifier.high_classifier.in_channels
    pet_model.classifier.low_classifier = nn.Conv2d(low_channels, 2, 1).to(DEVICE)
    pet_model.classifier.high_classifier = nn.Conv2d(high_channels, 2, 1).to(DEVICE)
    pet_optimizer = torch.optim.Adam([p for p in pet_model.parameters() if p.requires_grad], lr=1e-3)
    pet_criterion = nn.CrossEntropyLoss(ignore_index=255)
    pet_model.eval()
    with torch.inference_mode():
        shape_check = pet_model(pet_train[0][0][None].to(DEVICE))["out"]
    assert shape_check.shape == (1, 2, *PET_SIZE)
    print("Adapted output:", tuple(shape_check.shape))
else:
    print("Pretrained model construction skipped; no weights downloaded.")'''),
md('''## 8. Run an actual short fine-tuning/evaluation loop

The quick option trains four batches per epoch. This is a pipeline check, not a benchmark. Validation aggregates only labeled pixels, and the best validation state is restored. Increase data and training only after inspecting image/mask overlays. This path runs only when the explicit opt-in above succeeded.'''),
code('''def pet_epoch(model, loader, optimizer=None, max_batches=None):
    model.train(optimizer is not None)
    model.backbone.eval()  # Frozen features: also freeze BatchNorm running statistics.
    matrix = torch.zeros(2, 2, dtype=torch.int64)
    loss_sum, valid_pixels = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch_index, (x, y) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches: break
            x, y = x.to(DEVICE), y.to(DEVICE)
            labeled = (y != 255).sum().item()
            if labeled == 0: continue
            output = model(x)["out"]
            loss = pet_criterion(output, y)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            loss_sum += loss.item() * labeled
            valid_pixels += labeled
            matrix += confusion_matrix(output.detach().argmax(1), y, 2, ignore_index=255)
    if valid_pixels == 0: raise ValueError("No labeled pixels were evaluated")
    iou = iou_from_confusion(matrix)
    return loss_sum / valid_pixels, float(torch.nanmean(iou)), iou.numpy()

if real_ready:
    best_pet_iou, best_pet_state, pet_history = -1.0, None, []
    for epoch in range(2 if FAST_RUN else 10):
        train_stats = pet_epoch(pet_model, pet_train_loader, pet_optimizer, 4 if FAST_RUN else None)
        val_stats = pet_epoch(pet_model, pet_val_loader)
        pet_history.append([train_stats[0], val_stats[0], val_stats[1]])
        if val_stats[1] > best_pet_iou:
            best_pet_iou, best_pet_state = val_stats[1], copy.deepcopy(pet_model.state_dict())
        print(f"pet epoch {epoch+1}: train CE={train_stats[0]:.3f}, val mIoU={val_stats[1]:.3f}")
    pet_model.load_state_dict(best_pet_state); pet_model.eval()
    px, py = pet_val[0]
    with torch.inference_mode(): pp = pet_model(px[None].to(DEVICE))["out"].argmax(1)[0].cpu().numpy()
    restored_rgb = (px * pet_val.std + pet_val.mean).permute(1, 2, 0).numpy().clip(0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(restored_rgb); axes[0].set_title("Real pet")
    axes[1].imshow(np.ma.masked_equal(py.numpy(), 255), vmin=0, vmax=1); axes[1].set_title("Target: ignored pixels masked")
    axes[2].imshow(pp, vmin=0, vmax=1); axes[2].set_title("Adapted prediction")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
else:
    print("Optional real-data fine-tuning and evaluation skipped.")'''),
md('''## Experiment 1 · Adapt this to your own folders

Copy the folder convention, point a **new** `FolderSegmentationDataset` to your root, and write the raw-to-contiguous class map. Do not change the generated demo path into your real-data directory. Visualize random samples before training.

<details><summary>Worked solution 1</summary>

For masks with `0=background`, `128=road`, `255=unknown`, use `{0:0,128:1,255:255}`, `classes=2`, and a two-channel model. Keep `CrossEntropyLoss(ignore_index=255)` and exclude ignored pixels from metrics. Audit whether 255 really means unknown in **your** annotation specification; many binary masks use 255 for foreground.
</details>'''),
md('''## Experiment 2 · Test the filename contract

In a separate copy of the demo, rename one mask and explain the error. Add a same-stem `.jpg` and `.png` image to test duplicate detection. Why could independently sorted image and mask lists silently train on the wrong targets?

<details><summary>Worked solution 2</summary>

One missing file shifts list positions, so `zip(sorted(images), sorted(masks))` can pair unrelated samples and silently discard unmatched tail items. Pairing by stem and checking identical key sets prevents that. Duplicate extensions with one stem are ambiguous and should fail instead of picking one arbitrarily.
</details>'''),
md('''## Experiment 3 · Expand the real-data experiment carefully

After the optional quick run works, compare a frozen backbone with partial fine-tuning using the **same** validation images. Keep a separate official test evaluation for the end. Specify which pixels are ignored and which classes are averaged.

<details><summary>Worked solution 3</summary>

First set `FAST_RUN=False` and record a frozen-backbone baseline. In a separate run, unfreeze selected late backbone layers, use a smaller learning rate (for example `1e-4`), and decide explicitly whether BatchNorm should update. Keep preprocessing and split IDs fixed, save the best validation state, and inspect per-class IoU. A tiny subset is for debugging; a meaningful comparison needs broader data and repeated seeds.
</details>'''),
md('''## Checkpoint and common errors

- Detect unpaired files, duplicate stems, wrong spatial size, and unknown mask IDs.
- Recognize a visualization mask versus raw class IDs; never use JPEG labels.
- Avoid independent random crops/flips for images and masks.
- Expect torchvision models to return a dictionary; segmentation logits are under `["out"]`.
- An unavailable optional cache is not an offline notebook failure; keep `RUN_REAL_PETS=False` until ready.
- A model's original class names do not automatically match your annotation taxonomy.

## References

- [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) and the `annotations/README` inside its [official annotations archive](https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz) for raw ID semantics.
- [Torchvision OxfordIIITPet source and split API](https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/oxford_iiit_pet.html)
- [Torchvision LRASPP weights and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html)

Continue to lesson 11 for instance segmentation, or lesson 12 for loss choices, tiled prediction, and an end-to-end capstone.''')]
    save("10_real_data_transfer_learning.ipynb", cells, "Real image/mask folders and optional transfer learning")


def lesson12():
    cells = [
md('''# 12 · Loss experiments, large-image inference, and your capstone

**Prerequisites:** lessons 07–10. **Learning time:** 2–3 hours plus your capstone.

You will compare loss functions on a controlled imbalanced dataset, select a model on validation data, implement overlapping tiles, align test-time augmentations, inspect predictive entropy, and export labels with metadata.

**Success check:** explain what each experiment controls, prove every output pixel receives tile coverage, reverse augmentations before averaging probabilities, and distinguish uncertainty scores from calibrated error probabilities.

These are advanced workflow foundations. Expertise comes from repeated work with real annotation ambiguity, independent datasets, failure analysis, and deployment constraints; toy accuracy does not establish readiness for high-stakes use.'''),
bootstrap(), code(TORCH_SETUP + '\nimport json\nfrom PIL import Image\n'),
md('''## 1. Create a rare class deliberately

We retain normal disks but reduce rectangle regions to small patches. When most pixels are background, a model can achieve high accuracy while missing the rare class. Inspect the class histogram before choosing a loss. The editing below changes the image **and** target together.'''),
code('''images, masks = make_dataset(n=60 if FAST_RUN else 240, size=48, seed=800)
rng = np.random.default_rng(SEED)
rows, cols = np.mgrid[:48, :48]
for image, mask in zip(images, masks):
    coordinates = np.argwhere(mask == 2)
    center_row, center_col = coordinates[len(coordinates) // 4]
    keep = (abs(rows - center_row) <= 2) & (abs(cols - center_col) <= 2)
    remove = (mask == 2) & ~keep
    image[remove] = np.clip(0.13 + rng.normal(0, 0.04, (int(remove.sum()), 3)), 0, 1)
    mask[remove] = 0
ids = rng.permutation(len(images))
a, b = int(0.6 * len(ids)), int(0.8 * len(ids))
train_ids, val_ids, test_ids = ids[:a], ids[a:b], ids[b:]
assert set(train_ids).isdisjoint(val_ids) and set(train_ids).isdisjoint(test_ids) and set(val_ids).isdisjoint(test_ids)
counts = np.bincount(masks[train_ids].ravel(), minlength=3)
plt.bar(["background", "disk", "rare rectangle"], counts / counts.sum())
plt.ylabel("fraction of training pixels"); plt.show()
show_triplet(images[0], masks[0], masks[0], title="A deliberately small rare class")
plt.show()'''),
md('''## 2. Three loss choices, one prediction task

**Cross-entropy (CE)** rewards correct class probabilities per pixel. **Soft Dice** compares probabilities with one-hot targets, balancing class overlap at the batch level. We combine CE + Dice rather than using a discrete argmax inside a training loss. **Focal loss** multiplies CE by `(1-p_true)^gamma`, reducing easy-pixel contributions.

These implementations assume every pixel is labeled and use three mutually exclusive classes. They include background in soft Dice and have no ignore labels. For partial labels, mask ignored pixels from both numerator and denominator. Loss values from different formulas are not directly comparable; compare a common validation metric.'''),
code('''def soft_dice_loss(logits, target, smooth=1.0):
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)  # Sum across batch and pixels; keep each class separate.
    intersection = (probabilities * one_hot).sum(dims)
    denominator = probabilities.sum(dims) + one_hot.sum(dims)
    per_class_dice = (2 * intersection + smooth) / (denominator + smooth)
    return 1 - per_class_dice.mean()

def ce_plus_dice(logits, target):
    return F.cross_entropy(logits, target) + soft_dice_loss(logits, target)

def focal_loss(logits, target, gamma=2.0):
    ce = F.cross_entropy(logits, target, reduction="none")
    p_true = torch.exp(-ce)
    return ((1 - p_true).pow(gamma) * ce).mean()

losses = {"CE": F.cross_entropy, "CE + Dice": ce_plus_dice, "Focal gamma=2": focal_loss}
fake_logits = torch.randn(2, 3, 12, 12, requires_grad=True)
fake_target = torch.randint(0, 3, (2, 12, 12))
for name, loss_fn in losses.items():
    loss = loss_fn(fake_logits, fake_target)
    gradient = torch.autograd.grad(loss, fake_logits, retain_graph=True)[0]
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    print(name, "loss:", round(loss.item(), 3), "gradient norm:", round(gradient.norm().item(), 4))'''),
md('''## 3. A small controlled ablation

Use identical initialization, split IDs, optimizer settings, batch order, and epoch budget. Change only the loss. Each condition saves its best validation epoch; then choose the condition using validation macro IoU. This is a small single-seed teaching comparison, not evidence that one loss universally wins.

We report standard CE as a shared validation loss even when training uses another objective. Rare-class IoU is also visible because a macro mean can conceal very different class tradeoffs.'''),
code('''train_data = ArrayDataset(images[train_ids], masks[train_ids])
val_loader = DataLoader(ArrayDataset(images[val_ids], masks[val_ids]), batch_size=6, num_workers=0)
test_loader = DataLoader(ArrayDataset(images[test_ids], masks[test_ids]), batch_size=6, num_workers=0)
torch.manual_seed(SEED)
initial_state = copy.deepcopy(PixelCNN(classes=3).state_dict())
results, curves, trained_models = {}, {}, {}
epochs = 10 if FAST_RUN else 50
for name, loss_fn in losses.items():
    torch.manual_seed(SEED)
    model = PixelCNN(classes=3).to(DEVICE); model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    train_loader = DataLoader(train_data, batch_size=6, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(SEED))
    best_score, best_state, curve = -1.0, None, []
    for epoch in range(epochs):
        train_result = run_epoch(model, train_loader, loss_fn, optimizer)
        val_result = run_epoch(model, val_loader, F.cross_entropy)
        curve.append([train_result["loss"], val_result["loss"], val_result["miou"], val_result["iou"][2]])
        if val_result["miou"] > best_score:
            best_score = val_result["miou"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); model.eval()
    results[name] = run_epoch(model, val_loader, F.cross_entropy)
    curves[name], trained_models[name] = np.asarray(curve), model
    print(f"{name:15s} | validation mIoU {results[name]['miou']:.3f} | rare IoU {results[name]['iou'][2]:.3f}")'''),
code('''fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for name, curve in curves.items():
    axes[0].plot(curve[:, 2], label=name)
    axes[1].plot(curve[:, 3], label=name)
axes[0].set(ylabel="validation macro IoU", xlabel="epoch", ylim=(0, 1))
axes[1].set(ylabel="validation rare-class IoU", xlabel="epoch", ylim=(0, 1))
for ax in axes: ax.legend()
plt.tight_layout(); plt.show()
winner = max(results, key=lambda name: results[name]["miou"])
model = trained_models[winner]
test_result = run_epoch(model, test_loader, F.cross_entropy)
print("Selected using validation:", winner)
print("Untouched test per-class IoU:", test_result["iou"].round(3))
print("Test mIoU:", round(test_result["miou"], 3))'''),
md('''## 4. Predict a large image using overlapping tiles

Tiles limit memory, but convolution padding can cause seams. Average **probabilities** in overlaps with positive weights; do not average class IDs. Every pixel must have nonzero coverage. Here the window gives tile centers more weight, with a positive floor at edges.

Tiling changes available context and may disagree with full-image inference. For context-heavy architectures, use larger tiles or a halo around each tile and keep only the center. This demonstration uses a small CNN and supports images at least as large as the tile.'''),
code('''def tile_starts(length, tile, stride):
    if length < tile or not (0 < stride <= tile):
        raise ValueError("Require length >= tile and 0 < stride <= tile")
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile: starts.append(length - tile)
    return starts

@torch.inference_mode()
def tiled_probabilities(model, image, tile=48, stride=32):
    model.eval()
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must be a CHW RGB tensor")
    _, height, width = image.shape
    image = image.to(next(model.parameters()).device)
    starts_y, starts_x = tile_starts(height, tile, stride), tile_starts(width, tile, stride)
    window_1d = torch.hann_window(tile, periodic=False, device=image.device).clamp_min(0.05)
    window = window_1d[:, None] * window_1d[None, :]
    accumulated, coverage = None, torch.zeros(height, width, device=image.device)
    for top in starts_y:
        for left in starts_x:
            crop = image[:, top:top+tile, left:left+tile]
            probabilities = model(crop[None]).softmax(1)[0]
            if accumulated is None:
                accumulated = torch.zeros(probabilities.shape[0], height, width, device=image.device)
            accumulated[:, top:top+tile, left:left+tile] += probabilities * window
            coverage[top:top+tile, left:left+tile] += window
    assert (coverage > 0).all(), "Uncovered pixels would cause invalid division"
    return accumulated / coverage[None], coverage

large_image, large_mask, _ = make_scene(seed=9100, size=113)
large_x = torch.from_numpy(large_image.transpose(2, 0, 1).copy()).to(DEVICE)
tile_p, coverage = tiled_probabilities(model, large_x, tile=48, stride=32)
with torch.inference_mode(): full_p = model(large_x[None]).softmax(1)[0]
assert tile_p.shape == (3, 113, 113)
assert torch.allclose(tile_p.sum(0), torch.ones_like(tile_p[0]), atol=1e-5)
print("minimum coverage:", coverage.min().item())
print("mean |full - tiled| probability difference:", (tile_p-full_p).abs().mean().item())'''),
code('''fig, axes = plt.subplots(1, 3, figsize=(11, 3))
axes[0].imshow(coverage.cpu()); axes[0].set_title("Positive weighted coverage")
axes[1].imshow(palette_mask(tile_p.argmax(0).cpu().numpy())); axes[1].set_title("Tiled labels")
axes[2].imshow((tile_p-full_p).abs().mean(0).cpu(), cmap="magma"); axes[2].set_title("Full/tiled disagreement")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()'''),
md('''## 5. Test-time augmentation must undo geometry

Predict the original image and its horizontal reflection. Reflect the second probability map **back** into the original coordinates, then average. Without the inverse transform, corresponding pixels refer to different locations. Test-time augmentation costs extra inference and does not guarantee better accuracy.'''),
code('''@torch.inference_mode()
def flip_tta_probabilities(model, image_batch):
    model.eval()
    original = model(image_batch).softmax(1)
    flipped_back = model(image_batch.flip(-1)).softmax(1).flip(-1)
    return (original + flipped_back) / 2

tta_p = flip_tta_probabilities(model, large_x[None])[0]
assert tta_p.shape == full_p.shape
print("pixels changed by TTA:", int((tta_p.argmax(0) != full_p.argmax(0)).sum()))
assert torch.equal(large_x.flip(-1).flip(-1), large_x)
show_triplet(large_image, large_mask, tta_p.argmax(0).cpu().numpy(), title="TTA on a new, larger scene")
plt.show()'''),
md('''## 6. Entropy is a signal to investigate

Entropy `-Σ p log(p)` is high when predicted class probabilities are spread out. Divide by `log(K)` to scale it to `[0,1]` for K classes. It measures uncertainty in this model's output distribution; it is **not** the probability that a prediction is wrong.

Neural networks can be confidently wrong under a color/domain shift. Compare uncertainty with actual errors where labels exist, and use reliability/calibration analysis on independent validation data before interpreting probabilities operationally.'''),
code('''def normalized_entropy(probabilities):
    safe = probabilities.clamp_min(1e-8)
    return -(safe * safe.log()).sum(0) / np.log(probabilities.shape[0])

entropy = normalized_entropy(tta_p).cpu().numpy()
prediction = tta_p.argmax(0).cpu().numpy()
errors = prediction != large_mask
shift_image, shift_mask, _ = make_scene(seed=9100, size=113, shift=True)
shift_x = torch.from_numpy(shift_image.transpose(2, 0, 1).copy())[None].to(DEVICE)
with torch.inference_mode(): shift_p = flip_tta_probabilities(model, shift_x)[0]
shift_prediction = shift_p.argmax(0).cpu().numpy()
shift_entropy = normalized_entropy(shift_p).cpu().numpy()
print("ordinary scene error fraction:", errors.mean().round(3), "mean entropy:", entropy.mean().round(3))
print("color-shift error fraction:", (shift_prediction != shift_mask).mean().round(3), "mean entropy:", shift_entropy.mean().round(3))
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, value, title in zip(axes, [large_image, entropy, errors, shift_entropy],
                           ["Image", "Entropy", "Actual error", "Shifted-image entropy"]):
    ax.imshow(value, cmap="magma"); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()'''),
md('''## 7. Export raw labels and a complete inference contract

Save class IDs in a lossless PNG and colors in a separate preview. Export preprocessing, class names, architecture, and selection information alongside the weights. These files are local learning artifacts; production serving also needs latency/memory measurement, input validation, monitoring, and a versioned evaluation process.'''),
code('''output_dir = PROJECT_ROOT / "artifacts" / "12_capstone"
output_dir.mkdir(parents=True, exist_ok=True)
label_path = output_dir / "prediction_ids.png"
preview_path = output_dir / "prediction_preview.png"
Image.fromarray(prediction.astype(np.uint8)).save(label_path)
Image.fromarray(palette_mask(prediction)).save(preview_path)
roundtrip = np.asarray(Image.open(label_path))
assert np.array_equal(roundtrip, prediction)
metadata = {
    "architecture": "PixelCNN", "width": 12,
    "classes": ["background", "disk", "rectangle"],
    "layout": "NCHW RGB", "input_range": [0.0, 1.0],
    "normalization": "divide uint8 RGB by 255; no further normalization",
    "selected_loss": winner, "selection": "best validation macro IoU",
    "seed": SEED, "test_miou_single_pass": float(test_result["miou"]),
    "test_protocol": "single forward pass; global confusion matrix; macro IoU over present classes",
    "torch_version": str(torch.__version__), "inference": "horizontal flip TTA; argmax",
    "label_format": "uint8 PNG; IDs 0,1,2", "training_data": "synthetic rare-class demo",
}
(output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "metadata": metadata}, output_dir / "model.pt")
saved = torch.load(output_dir / "model.pt", map_location=DEVICE, weights_only=True)
restored = PixelCNN(classes=len(saved["metadata"]["classes"]), width=saved["metadata"]["width"]).to(DEVICE)
restored.load_state_dict(saved["state_dict"]); restored.eval()
with torch.inference_mode():
    torch.testing.assert_close(restored(large_x[None]), model(large_x[None]))
print("Exported labels, preview, metadata and reload-verified weights to", output_dir)'''),
md('''## 8. Measure inference latency with a clear scope

GPU kernels launch asynchronously. Synchronize before and after the timed region so the clock measures completed work. Warm up first, report repeated timings, and state what the timing includes.

This measures one resident `113×113` tensor through the model. It excludes image decoding, host-to-device transfer, preprocessing, TTA, tiling, and file export. Full application latency needs those stages measured too; a tiny synthetic input is not a deployment benchmark.'''),
code('''from time import perf_counter

def synchronize_device():
    if DEVICE.type == "cuda": torch.cuda.synchronize(DEVICE)

model.eval()
timings_ms = []
with torch.inference_mode():
    for _ in range(3): _ = model(large_x[None])
    synchronize_device()
    for _ in range(10):
        synchronize_device()
        start = perf_counter()
        _ = model(large_x[None])
        synchronize_device()
        timings_ms.append((perf_counter() - start) * 1000)
print("Model-only latency, input already on", DEVICE)
print("median ms:", round(float(np.median(timings_ms)), 3),
      "p90 ms:", round(float(np.percentile(timings_ms, 90)), 3))'''),
md('''## Experiment 1 · Which loss helps the rare class?

Change the rare patch radius from 2 to 1, then repeat the controlled ablation. Record background, disk, and rectangle IoU. Repeat at least three seeds before favoring a loss. Do not compare the numerical magnitude of focal loss with CE as if they measured the same quantity.

<details><summary>Worked solution 1</summary>

Edit the `keep` expression, regenerate the data, and rerun all dependent cells. Keep model initialization and batch order identical across conditions **within each seed**. Report the mean and spread of validation rare-class IoU. A loss can improve macro IoU while harming boundary quality; inspect errors and consider a boundary metric from lesson 06. The short optimization budget can influence the ranking.
</details>'''),
md('''## Experiment 2 · Deliberately break tiling or TTA

Try `stride=60` with `tile=48`: why must it fail? Then remove the final `.flip(-1)` in the TTA function and inspect the averaged map. Restore the correct code afterward.

<details><summary>Worked solution 2</summary>

Stride larger than the tile can leave unvisited gaps; our argument check rejects it. The last tile start must also reach the far edge when image dimensions are not multiples of the stride. Without inverse flipping, left-side class probabilities are averaged with unrelated right-side predictions, blurring or duplicating objects. Probability sums can still equal one, so that check alone cannot detect misalignment.
</details>'''),
md('''## Experiment 3 · Investigate confident mistakes

Find pixels where `shift_prediction != shift_mask` but `shift_entropy < 0.2`. Count and visualize them. What would an entropy threshold miss?

<details><summary>Worked solution 3</summary>

Use `confident_errors = (shift_prediction != shift_mask) & (shift_entropy < 0.2)` and display it over the shifted image. A low-entropy incorrect answer is a confident error. An uncertainty-only review rule will miss it. Broader out-of-distribution evaluation, calibration checks, diverse training data, and human review criteria address different parts of this problem; none is established by this toy experiment.
</details>'''),
code('''confident_errors = (shift_prediction != shift_mask) & (shift_entropy < 0.2)
print("Confidently wrong shifted pixels:", int(confident_errors.sum()))
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(shift_image); axes[0].set_title("Color-shift input")
axes[1].imshow(confident_errors, vmin=0, vmax=1); axes[1].set_title("Wrong with low entropy")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()'''),
md('''## Capstone · Replace the toy data with a real question

Choose a manageable labeled dataset, such as the optional pet dataset from lesson 10, or your own carefully annotated images. Define classes, ignore regions, unit of splitting, and intended use before training.

| Deliverable | Evidence required | Points |
|---|---|---:|
| Data audit | Pair checks, label histogram, ten overlays, annotation rules | 20 |
| Honest evaluation | Group-aware train/validation/test split and leakage checks | 20 |
| Baselines | Simple baseline plus learned baseline on identical splits | 15 |
| Controlled improvement | One-factor ablation, fixed budget, three seeds if practical | 15 |
| Error analysis | Per-class IoU/Dice, small-object and boundary failures, shift examples | 15 |
| Reproducible artifact | Reload-equal predictions, class metadata, raw-mask export | 15 |

Aim for **80/100 with no missing data-audit or split checks** before calling the project complete. Add your conclusions in Markdown cells: what failed, why you think it failed, and what evidence would change your mind. This rubric evaluates a learning project; deployment needs its own acceptance criteria.'''),
md('''## Checkpoint and common errors

- Explain why soft Dice uses probabilities but CE takes logits.
- Average probability maps, not numeric class IDs.
- Verify tile coverage and original-coordinate alignment independently.
- Keep test data out of hyperparameter and model selection.
- An all-ignored batch needs explicit handling; the toy loss implementations assume none.
- Reload on CPU or GPU using `map_location`; persist preprocessing alongside weights.
- Record limitations and next experiments instead of calling a toy model “expert-level.”

## References and further study

- [Focal Loss paper](https://arxiv.org/abs/1708.02002)
- [V-Net paper: Dice-based optimization](https://arxiv.org/abs/1606.04797)
- [Calibration of modern neural networks](https://arxiv.org/abs/1706.04599)
- [PyTorch inference_mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)

For a further project, study panoptic segmentation, promptable foundation models, active learning, or domain adaptation. Start with a precise problem and an evaluation plan, then choose the architecture.''')]
    save("12_advanced_training_deployment.ipynb", cells, "Loss experiments, large-image inference, and your capstone")


if __name__ == "__main__":
    lesson08()
    lesson09()
    lesson10()
    lesson12()
