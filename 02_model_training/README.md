# YOLO-Seg-Root — Standalone Training Package

A self-contained package for training, evaluating, and hyperparameter-tuning the **YOLO-Seg-Root** multi-task model. The model jointly predicts bounding boxes, class labels, instance segmentation masks, and plant **root center coordinates** in a single forward pass.

---

## ✅ Getting Started Checklist

Follow these steps in order before running any script:

1. **Install dependencies** (see [Prerequisites](#prerequisites--setup))
2. **Download pretrained weights** — `yolo11m-seg.pt` (see below)
3. **Prepare your dataset** — dataset is provided in `02_model_training/data/` (or generated via `01_data_preparation/`)
4. **Edit `config.py`** if you wish to change hyperparameters or output paths
5. **Run training** — `python 02_model_training/train.py`

---

## Directory Structure

```
02_model_training/
├── data/                   # Dataset (images/, labels/, data.yaml)
│   ├── images/             # train/, val/, test/ subdirectories
│   ├── labels/             # train/, val/, test/ subdirectories
│   └── data.yaml           # Dataset configuration descriptor
├── yolov11-seg-root.yaml   # Multi-task architecture definition
├── modules.py              # CustomSegmentHead (adds root-point branch)
├── augment.py              # Mosaic, MixUp, Copy-Paste, Albumentations
├── dataset.py              # YOLOSegPointDataset (images + labels + keypoints)
├── loss.py                 # CustomLoss (Box + Seg + Cls + DFL + Pose)
├── metrics.py              # CustomSegmentMetrics + PCK evaluation
├── utils.py                # ModelBuilder, batch helpers, display utilities
├── val.py                  # CustomValidator (standalone evaluation)
├── config.py               # Config dataclass (all hyperparameters & paths)
├── train.py                # CustomTrainer (standard training pipeline)
├── tune.py                 # YOLOSegRootTuner (Bayesian hyperparameter sweep)
└── README.md
```

---

## Prerequisites & Setup

Python ≥ 3.9 (Python 3.10 recommended).

### Option A: Using Conda (Recommended)

To create and activate a fresh environment for this project:

```bash
# Create a new environment
conda create -n yolo_seg_root python=3.10 -y

# Activate the environment
conda activate yolo_seg_root

# Install dependencies from project root
pip install -r requirements.txt
```

*(Alternatively, create directly from the environment file: `conda env create -f environment.yml`)*

### Option B: Using Standard Python / Pip (Virtual Environment)

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Linux / macOS:
source venv/bin/activate

# Install dependencies:
pip install -r requirements.txt
```

> **GPU Acceleration (Recommended):** If your machine has an NVIDIA GPU, ensure you install a CUDA-enabled PyTorch build:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### Download Pretrained Weights

The model is initialized from Ultralytics YOLOv11-medium-seg pretrained weights. Download `yolo11m-seg.pt` and place it in the **project root** (`plant_detection/`):

```bash
# Using the Ultralytics CLI (recommended — downloads automatically on first use):
python -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt')"

# Or download manually from:
# https://github.com/ultralytics/assets/releases
```

Expected file location: `plant_detection/yolo11m-seg.pt`

---

## Dataset Structure

The dataset directory is located in `02_model_training/data/` (with automatic fallback to `data/exp_4class/` or `data/` if placed elsewhere).

```
02_model_training/data/     ← auto-detected by config.py
├── images/
│   ├── train/   *.jpg / *.png
│   ├── val/
│   └── test/
├── labels/
│   ├── train/   *.txt
│   ├── val/
│   └── test/
└── data.yaml             ← dataset descriptor
```

> **Automatic Path Detection:** `config.py` automatically detects the dataset if placed in `02_model_training/data/`, `../data/exp_4class`, or `../data/`.

**Label format** — each line in a `.txt` file:

```
class_id  root_x  root_y  poly_x1 poly_y1 poly_x2 poly_y2 ...
```

All coordinates are normalized to `[0, 1]`. Minimum 3 polygon points per instance.

---

## Configuration (`config.py`)

All settings live in the `Config` dataclass. Edit the file directly, or override attributes at runtime:

```python
import config as cfg
cfg.cfg.EPOCHS = 50
cfg.cfg.HYP["copy_paste"] = 0.3
```

| Parameter | Default | Description |
|---|---|---|
| `EPOCHS` | 100 | Total training epochs |
| `BATCH_SIZE` | 16 | Images per batch |
| `IMG_SIZE` | 640 | Input resolution (square) |
| `LR0` | 5e-4 | Initial learning rate |
| `LRF` | 0.001 | Final LR ratio (cosine annealing target) |
| `WARMUP_EPOCHS` | 3 | Linear LR warmup length |
| `WEIGHT_DECAY` | 1e-4 | AdamW weight decay |
| `GRAD_CLIP` | 10.0 | Gradient norm clip |
| `PATIENCE` | 10 | Early-stopping patience (epochs without improvement) |
| `CLOSE_MOSAIC_EP` | 10 | Disable mosaic for the last N epochs |
| `KPT_GAIN` | 8.0 | Root-point loss weight (`hyp.pose`) |
| `BOX_GAIN` | 7.5 | Box + segmentation loss weight |
| `CLS_GAIN` | 0.5 | Classification loss weight |
| `CLASS_WEIGHTS` | `"auto"` | `"auto"` for median-frequency weighting, or `list[float]` |
| `CONF_THRES` | 0.001 | NMS confidence threshold (validation) |
| `IOU_THRES` | 0.6 | NMS IoU threshold (validation) |
| `HYP["copy_paste"]` | 0.0 | Weed copy-paste probability (set 0.3 for imbalanced datasets) |
| `DATA_DIR` | `02_model_training/data` | Dataset root (auto-detected) |
| `OUTPUT_DIR` | `output_train/4class_original` | Training output directory |
| `WEIGHT_PATH` | `../model_inference/final_best.pt` | Fine-tuned weights for evaluation |
| `WORKERS` | `0` | DataLoader workers — **must stay 0 on Windows** (see note below) |

> **Windows note on `WORKERS`:** On Windows, PyTorch's `DataLoader` with `num_workers > 0` requires the `if __name__ == "__main__"` guard, which `train.py` does not use at the top level. Therefore `WORKERS = 0` is the safe default for Windows. On **Linux or macOS** you can set `WORKERS = 4` (or higher) in `config.py` for faster data loading.

---

## Usage

Run commands from the **project root** (`plant_detection/`) or from within `02_model_training/`.

### 1. Training

```bash
python 02_model_training/train.py
```

Outputs saved to `OUTPUT_DIR`:

| File | Content |
|---|---|
| `best.pt` | Best checkpoint by validation loss |
| `last.pt` | Latest checkpoint |
| `epoch_N.pt` | Periodic checkpoint every `SAVE_PERIOD` epochs |
| `results.csv` | Per-epoch metrics |
| `loss_curves.png` | Train vs validation loss |
| `loss_components.png` | Individual loss terms |
| `metrics_curves.png` | mAP + PCK curves |

**Tips:**
- Run with `EPOCHS = 10` first to confirm the pipeline loads without OOM errors.
- Set `HYP["copy_paste"] = 0.3` if weed classes are underrepresented.

### 2. Evaluation

```bash
python 02_model_training/val.py --weights output_train/4class_original/best.pt
```

Prints a full table of Box mAP50/50-95, Mask mAP50/50-95, and PCK@5/10/20 — both overall and per class.

### 3. Hyperparameter Tuning

```bash
python 02_model_training/tune.py
```

Configure at the bottom of `tune.py`:

| Parameter | Description |
|---|---|
| `mode="train"` | Full training sweep (searches LR, loss gains, augmentation) |
| `mode="val"` | Fast inference sweep — only `conf_thres` and `iou_thres` |
| `n_trials=40` | Number of Bayesian trials |
| `tune_epochs=50` | Epochs per trial (only for `mode="train"`) |
| `alpha=0.5` | Balance between mask mAP and PCK in composite score |

The study is saved to `optuna_seg_root.db` and resumes automatically if interrupted.

Visualize results with the interactive dashboard:

```bash
optuna-dashboard sqlite:///optuna_seg_root.db
```

---

## Mathematical Reference

### 1. Root-Point Coordinate Regression

Anchor $i$ at stride $s \in \{8, 16, 32\}$ with center $(x_i^\text{anc}, y_i^\text{anc})$:

$$\hat{x}_i = x_i^\text{anc} + dx_i \times s \qquad \hat{y}_i = y_i^\text{anc} + dy_i \times s$$

Ground-truth offset targets:

$$dx_i^\text{target} = \frac{x_i^\text{target} - x_i^\text{anc}}{s} \qquad dy_i^\text{target} = \frac{y_i^\text{target} - y_i^\text{anc}}{s}$$

### 2. Multi-Task Joint Loss

$$\mathcal{L}_\text{total} = \lambda_\text{box}\mathcal{L}_\text{box} + \lambda_\text{dfl}\mathcal{L}_\text{dfl} + \lambda_\text{cls}\mathcal{L}_\text{cls} + \lambda_\text{box}\mathcal{L}_\text{seg} + \lambda_\text{pose}\mathcal{L}_\text{root}$$

| Term | Formula | Notes |
|---|---|---|
| $\mathcal{L}_\text{box}$ | CIoU | Complete IoU bounding box loss |
| $\mathcal{L}_\text{dfl}$ | $-[(y_{i+1}-y)\log p_i + (y-y_i)\log p_{i+1}]$ | Distribution focal loss over 16 bins |
| $\mathcal{L}_\text{cls}$ | Weighted BCE per class | Weights $w_c = \min(\text{median}(N)/N_c,\ 20)$ |
| $\mathcal{L}_\text{seg}$ | BCE on prototype masks | $\hat{M}_i = \sigma(\mathbf{c}_i \cdot P)$ |
| $\mathcal{L}_\text{root}$ | Masked Smooth-L1 | Only active for instances with labeled root points |

**Root-point Smooth-L1:**

$$\mathcal{L}_\text{root} = \frac{1}{N_\text{active}} \sum_i M_i \sum_{d \in \{x,y\}} \text{SmoothL1}(e_d), \quad \text{SmoothL1}(e) = \begin{cases} 0.5\,e^2 & |e| < 1 \\ |e| - 0.5 & |e| \ge 1 \end{cases}$$

### 3. Learning Rate Schedule

Warmup ($T \le T_w$): $\eta(T) = \eta_0 \cdot T / T_w$

Cosine annealing ($T > T_w$): $\eta(T) = \eta_0 \left(\eta_f + (1-\eta_f) \cdot \tfrac{1}{2}\left(1 + \cos\tfrac{(T-T_w)\pi}{T_\text{max}-T_w}\right)\right)$

### 4. PCK (Relative) & AbsPCK (Absolute Pixel)

**Relative PCK** — distance normalized by bounding box diagonal:

$$\text{PCK}_\theta = \frac{1}{N_\text{active}} \sum_i \mathbb{I}\!\left(\|\hat{\mathbf{p}}_i - \mathbf{p}_i\|_2 \le \theta \cdot D_i\right), \quad \theta \in \{0.05, 0.10, 0.20\}$$

where $D_i = \sqrt{W_i^2 + H_i^2}$ is the bounding box diagonal.

**AbsPCK** — absolute Euclidean distance in pixels:

$$\text{AbsPCK}_\tau = \frac{1}{N_\text{active}} \sum_i \mathbb{I}\!\left(\|\hat{\mathbf{p}}_i - \mathbf{p}_i\|_2 \le \tau\right), \quad \tau \in \{5, 10, 15, 20\}\text{ px}$$

### 5. Tuning Objective

$$\text{Maximize} \quad f(\mathbf{x}) = \alpha \cdot \text{Mask mAP}_{50\text{-}95} + (1-\alpha) \cdot \text{PCK}_{10}$$
