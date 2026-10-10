"""Small, readable CPU models for the segmentation learning notebooks.

These are teaching models, not pretrained production systems. The notebooks show
the important implementations and explain the input and output contracts.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    """Convert RGB NHWC arrays and integer NHW masks into PyTorch samples."""

    def __init__(self, images, masks, augment=False):
        self.images = np.asarray(images, dtype=np.float32)
        self.masks = np.asarray(masks, dtype=np.int64)
        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError("images must have shape [N,H,W,3]")
        if self.images.shape[:3] != self.masks.shape:
            raise ValueError("images and masks must have matching N,H,W")
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].transpose(2, 0, 1).copy())
        mask = torch.from_numpy(self.masks[index].copy())
        if self.augment and torch.rand(()) < 0.5:
            image, mask = image.flip(-1), mask.flip(-1)
        return image, mask


class PixelCNN(nn.Module):
    """A small convolutional baseline: N,3,H,W -> N,classes,H,W logits."""

    def __init__(self, classes=3, width=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, classes, 1),
        )

    def forward(self, image):
        return self.net(image)


class TinyUNet(nn.Module):
    """Two-level U-Net with interpolation to support odd image dimensions."""

    def __init__(self, classes=3, width=16):
        super().__init__()

        def block(ch_in, ch_out):
            return nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 3, padding=1), nn.ReLU(),
                nn.Conv2d(ch_out, ch_out, 3, padding=1), nn.ReLU(),
            )

        self.enc1 = block(3, width)
        self.enc2 = block(width, width * 2)
        self.bridge = block(width * 2, width * 4)
        self.dec2 = block(width * 6, width * 2)
        self.dec1 = block(width * 3, width)
        self.head = nn.Conv2d(width, classes, 1)

    def forward(self, image):
        skip1 = self.enc1(image)
        skip2 = self.enc2(F.max_pool2d(skip1, 2))
        deep = self.bridge(F.max_pool2d(skip2, 2))
        up2 = F.interpolate(deep, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        decoded2 = self.dec2(torch.cat([up2, skip2], dim=1))
        up1 = F.interpolate(decoded2, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.dec1(torch.cat([up1, skip1], dim=1)))


def confusion_matrix(prediction, target, classes, ignore_index=255):
    """Rows=true, columns=predicted. Ignore target pixels are excluded."""
    prediction, target = prediction.reshape(-1), target.reshape(-1)
    valid = target != ignore_index
    prediction, target = prediction[valid].long(), target[valid].long()
    if target.numel() == 0:
        return torch.zeros(classes, classes, dtype=torch.int64)
    if not ((target >= 0) & (target < classes)).all():
        raise ValueError("target contains out-of-range class IDs")
    if not ((prediction >= 0) & (prediction < classes)).all():
        raise ValueError("prediction contains out-of-range class IDs")
    return torch.bincount(target.cpu() * classes + prediction.cpu(), minlength=classes**2).reshape(classes, classes)


def iou_from_confusion(matrix):
    """Return per-class IoU; a class absent from truth AND prediction is NaN."""
    matrix = matrix.to(torch.float64)
    intersection = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - intersection
    return torch.where(union > 0, intersection / union, torch.nan)


def run_epoch(model, loader, loss_function, optimizer=None, classes=3, ignore_index=255):
    """Train/evaluate; aggregate pixels globally and skip all-ignored batches.

    The loss must accept the labels (set its ignore_index when needed). Reported
    loss is the valid-pixel-weighted mean of batch loss values: exact for mean
    unweighted pixel CE, a descriptive training summary for batch-level Dice
    combinations. Class-weighted CE has a different normalization denominator.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, valid_pixels_seen = 0.0, 0
    matrix = torch.zeros(classes, classes, dtype=torch.int64)
    device = next(model.parameters()).device
    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            valid_pixels = int((masks != ignore_index).sum().item())
            if valid_pixels == 0:
                continue
            logits = model(images)
            loss = loss_function(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * valid_pixels
            valid_pixels_seen += valid_pixels
            matrix += confusion_matrix(logits.detach().argmax(1), masks, classes, ignore_index)
    if valid_pixels_seen == 0:
        raise ValueError("loader has no labeled pixels (empty or all ignored)")
    iou = iou_from_confusion(matrix)
    return {"loss": total_loss / valid_pixels_seen, "miou": torch.nanmean(iou).item(),
            "iou": iou.numpy(), "confusion": matrix.numpy()}
