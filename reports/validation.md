# Notebook validation

**13 of 13 notebooks passed**, using fresh Jupyter kernels and default settings.

Deep learning ran on the NVIDIA GeForce RTX 5050 Laptop GPU. Saved notebook outputs include the checked plots and training results.

Optional Oxford Pet downloads and pretrained fine-tuning were not enabled. A separate offline check verified torchvision LRASPP classifier adaptation, forward pass, and backward pass using random weights; it does not establish pretrained or real-data accuracy.

| Notebook | Status | Seconds | Code cells | Figures |
|---|---|---:|---:|---:|
| [00_start_here.ipynb](../notebooks/00_start_here.ipynb) | passed | 5.95 | 6 | 2 |
| [01_images_masks_coordinates.ipynb](../notebooks/01_images_masks_coordinates.ipynb) | passed | 4.28 | 9 | 7 |
| [02_thresholding.ipynb](../notebooks/02_thresholding.ipynb) | passed | 3.39 | 9 | 7 |
| [03_morphology_components.ipynb](../notebooks/03_morphology_components.ipynb) | passed | 4.07 | 10 | 8 |
| [04_edges_watershed.ipynb](../notebooks/04_edges_watershed.ipynb) | passed | 3.79 | 10 | 8 |
| [05_clustering_superpixels.ipynb](../notebooks/05_clustering_superpixels.ipynb) | passed | 6.41 | 11 | 9 |
| [06_metrics_error_analysis.ipynb](../notebooks/06_metrics_error_analysis.ipynb) | passed | 4.02 | 11 | 5 |
| [07_classical_machine_learning.ipynb](../notebooks/07_classical_machine_learning.ipynb) | passed | 6.4 | 11 | 5 |
| [08_pytorch_segmentation_basics.ipynb](../notebooks/08_pytorch_segmentation_basics.ipynb) | passed | 9.16 | 12 | 3 |
| [09_unet_training.ipynb](../notebooks/09_unet_training.ipynb) | passed | 8.34 | 12 | 4 |
| [10_real_data_transfer_learning.ipynb](../notebooks/10_real_data_transfer_learning.ipynb) | passed | 9.01 | 11 | 3 |
| [11_instance_panoptic_prompting.ipynb](../notebooks/11_instance_panoptic_prompting.ipynb) | passed | 3.83 | 12 | 6 |
| [12_advanced_training_deployment.ipynb](../notebooks/12_advanced_training_deployment.ipynb) | passed | 9.39 | 13 | 7 |

## Additional checks

- Ignored-label batches and globally weighted cross-entropy: passed.
- U-Net with odd image dimensions: passed.
- Torchvision binary-head adaptation and gradient flow without downloads: passed.
- JupyterLab startup and browser route: passed; temporary server stopped.
- Representative resized masks, watershed splits, learning curves, and predictions: visually reviewed.

Execution times are observed local runs, not guaranteed runtimes or benchmark measurements. Small synthetic datasets intentionally limit what can be concluded about accuracy.

## Tested environment

```json
{
  "platform": "Windows",
  "cuda_available": true,
  "gpu": "NVIDIA GeForce RTX 5050 Laptop GPU",
  "python": "3.13.15",
  "numpy": "2.5.2",
  "scipy": "1.18.1",
  "matplotlib": "3.11.2",
  "scikit-image": "0.26.0",
  "scikit-learn": "1.9.1",
  "pandas": "3.0.6",
  "Pillow": "12.3.0",
  "torch": "2.11.0+cu128",
  "torchvision": "0.26.0+cu128",
  "jupyterlab": "4.6.4",
  "nbformat": "5.11.1",
  "nbclient": "0.11.0"
}
```
