

# Polygon Colorization UNet

*Ayna ML Assignment Solution*

## Overview

This project implements a conditional UNet in PyTorch that takes as input a polygon outline image and a target color name, and generates a colorized image of the polygon filled with the specified color.

Below are sample qualitative results:

- **Star (yellow fill)**
- **Star (yellow fill) and Triangle (green fill)**


## Dataset

- **Structure**:
    - `dataset/training/inputs/` (input images)
    - `dataset/training/outputs/` (target colorized images)
    - `dataset/training/data.json` (input/output/color mappings)
    - `dataset/validation/inputs/`, `outputs/`, `data.json` (for validation)
- **Download**:
[Google Drive Link](https://drive.google.com/open?id=1QXLgo3ZfQPorGwhYVmZUEWO_sU3i1pHM)


## Model Architecture

- **UNet**: Standard encoder-decoder with skip connections
- **Color Conditioning**: The color name is mapped through a learned embedding and RGB projection, and injected into the UNet bottleneck and decoder layers.
- **Loss**: Sum of pixelwise MSE and color embedding RGB loss (improves color accuracy).


## Dependencies

```bash
pip install torch torchvision numpy matplotlib pillow gdown tqdm wandb scikit-learn
```


## Usage

### 1. Dataset Setup

Download, extract, and ensure your dataset has the aforementioned structure.

### 2. Training

```python
# Inside train_polygon_unet.py or the corresponding notebook cell:
trainer = EnhancedPolygonTrainer(config, train_dataset, val_dataset, use_wandb=True)
trainer.train()
```

- Experiment logging to [wandb](https://wandb.ai/).


### 3. Inference

Use the notebook or this script snippet for inference:

```python
model, colors = load_enhanced_model('checkpoint_best.pth')
input_img_path = 'path/to/your/test_polygon.png'
color_name = 'yellow' # or 'green', etc.
enhanced_predict(model, colors, input_img_path, color_name)
```


### 4. Results

- The script displays the input polygon, the generated colored fill, and the predicted RGB value, as shown above.


## Key Hyperparameters

- `learning_rate = 2e-4`
- `batch_size = 8`
- `num_epochs = 40`
- `loss = reconstruction MSE + color embedding (RGB) MSE`
- **Augmentations**: Rotation, horizontal flip


## Insights \& Qualitative Review

- **Training Curve**: Validation loss improved from 0.37 to 0.011 over 40 epochs.
- **Color Conditioning**: Multi-level embedding dramatically improves color matching and spatial accuracy.
- **Augmentation**: Essential for generalization to unseen polygons/orientations.
- **wandb Logging**: All runs are tracked and visualized for analysis.

Qualitative outputs show sharp, high-fidelity fills, with predicted RGB values very close to canonical color targets (e.g., “yellow” output: [0.868, 0.877, 0.045]).

## Summary

- Built a fully working PyTorch UNet for conditional polygon colorization
- Achieved strong quantitative (loss < 0.012) and qualitative results
- All code robust, documented, ready for both training and inference

**For further questions or issues, contact [vatsa@getayna.com](mailto:vatsa@getayna.com)**

---

<div style="text-align: center">⁂</div>

[^1]: image.jpg

[^2]: image.jpg

