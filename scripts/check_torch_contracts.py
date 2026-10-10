"""Offline checks for ignored labels, odd U-Net shapes, and torchvision heads.

The LRASPP check uses random weights: it verifies the installed API without
downloading a checkpoint or claiming to test real-data accuracy.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from learning.torch_models import ArrayDataset, PixelCNN, TinyUNet, run_epoch


def main():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    images = np.random.default_rng(17).random((3, 8, 8, 3), dtype=np.float32)
    masks = np.zeros((3, 8, 8), dtype=np.int64)
    masks[0] = 255  # The whole first batch is ignored; it must be skipped.
    masks[1, :6] = 255  # Unequal numbers of valid pixels per image.
    masks[2, 2:5, 2:5] = 1
    model = PixelCNN(classes=2).eval()
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    result = run_epoch(model, DataLoader(ArrayDataset(images, masks), batch_size=1),
                       criterion, classes=2)
    with torch.inference_mode():
        logits = model(torch.from_numpy(images.transpose(0, 3, 1, 2).copy()))
        expected_loss = criterion(logits, torch.from_numpy(masks)).item()
    assert abs(result["loss"] - expected_loss) < 1e-6
    assert result["confusion"].sum() == (masks != 255).sum()
    all_ignored = np.full_like(masks, 255)
    try:
        run_epoch(model, DataLoader(ArrayDataset(images, all_ignored), batch_size=1), criterion, classes=2)
    except ValueError as error:
        assert "no labeled pixels" in str(error)
    else:
        raise AssertionError("An all-ignored loader should be rejected")
    print("Ignored-label handling and global mean CE: passed")

    with torch.inference_mode():
        assert TinyUNet(classes=3)(torch.rand(1, 3, 35, 47)).shape == (1, 3, 35, 47)
    print("U-Net odd-size spatial contract: passed")

    # Mirror the optional transfer-learning classifier replacement, but offline.
    transfer = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None)
    for parameter in transfer.backbone.parameters():
        parameter.requires_grad_(False)
    head = transfer.classifier
    head.low_classifier = nn.Conv2d(head.low_classifier.in_channels, 2, 1)
    head.high_classifier = nn.Conv2d(head.high_classifier.in_channels, 2, 1)
    transfer.train()
    transfer.backbone.eval()
    x, y = torch.rand(2, 3, 64, 64), torch.zeros(2, 64, 64, dtype=torch.long)
    y[:, :8] = 255
    output = transfer(x)["out"]
    assert output.shape == (2, 2, 64, 64)
    loss = criterion(output, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert head.low_classifier.weight.grad is not None
    assert all(p.grad is None for p in transfer.backbone.parameters())
    print("Torchvision LRASPP binary heads, loss, and backward (random weights): passed")


if __name__ == "__main__":
    main()
