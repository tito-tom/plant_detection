# Plant Detection & Root-Point Estimation (YOLO-Seg-Root)

This is a computer vision project for detecting plants in field images. It's not just detection — the model also draws a mask around each plant's leaf canopy and finds the exact root point (stem base) all at once.

The reason we need root points is that for robotic weeding, knowing where the leaf is isn't enough. The robot needs to know where the stem enters the ground to strike accurately. Standard YOLO bounding boxes point to the canopy center, not the root, so we added a custom keypoint branch to solve this.

---

## What the Model Actually Does

For every plant in an image, the model outputs three things in a single forward pass:

1. **Bounding box + class** — which plant it is and roughly where
2. **Instance segmentation mask** — the exact leaf shape, pixel by pixel
3. **Root point (x, y)** — the stem/root base location

There are 4 plant classes:

| Class | Name | Original CVAT label |
|---|---|---|
| 0 | `crop_small_leaf` | crop6 |
| 1 | `crop_large_leaf` | crop1 – crop5 |
| 2 | `weed_small_leaf` | weed1 – weed4 |
| 3 | `weed_large_leaf` | weed5 – weed14 |

The dataset has 2,114 annotated images split 80% train / 5% val / 15% test. The split is at the image level with a fixed seed (42) so it's reproducible.

---

## Environment Setup

Python 3.10 is recommended.

### Conda (easier)

```bash
conda create -n yolo_seg_root python=3.10 -y
conda activate yolo_seg_root
pip install -r requirements.txt
```

Or from the environment file directly:
```bash
conda env create -f environment.yml
conda activate yolo_seg_root
```

### venv (alternative)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### GPU (NVIDIA CUDA)

If you have an NVIDIA GPU, install the CUDA build of PyTorch instead:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Running the Project (New Student Start Here)

### Step 1 — Download the pretrained YOLO weights

Place `yolo11m-seg.pt` in the project root folder:
```bash
python -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt')"
```
This downloads automatically on first call. Alternatively, download manually from [Ultralytics releases](https://github.com/ultralytics/assets/releases).

### Step 2 — The dataset is already there

Everything is inside `02_model_training/data/`. You don't need to run any data preparation scripts — just go straight to training.

### Step 3 — Train

```bash
python 02_model_training/train.py
```

For a quick test to make sure it runs on your machine before committing to a full training run, open `02_model_training/config.py` and temporarily set `EPOCHS = 10`, then run.

### Step 4 — Evaluate

```bash
python 02_model_training/val.py --weights output_train/4class_original/best.pt
```

This prints a full table of box mAP, mask mAP, PCK, and AbsPCK — both overall and per class.

### Step 5 — Hyperparameter tuning (optional)

```bash
python 02_model_training/tune.py
```

Uses Optuna (Bayesian search) to find better hyperparameters. Results are saved to `optuna_seg_root.db`. You can visualize them with:
```bash
optuna-dashboard sqlite:///optuna_seg_root.db
```

---

## Repository Structure

```
plant_detection/
├── 01_data_preparation/      # Used only when annotating new images
│   ├── 01_analyze_xml.py     # Inspect a raw CVAT XML export
│   ├── 02_convert_xml_to_yolo.py  # Convert XML → YOLO label format
│   ├── 03_visualize_dataset.py    # Overlay annotations on images (sanity check)
│   ├── 04_resplit_dataset.py      # Re-split into train/val/test
│   └── README.md
├── 02_model_training/        # The main module you'll work with daily
│   ├── data/                 # Dataset packaged here (images + labels + data.yaml)
│   ├── config.py             # All hyperparameters and paths — start here
│   ├── train.py              # Training loop
│   ├── val.py                # Validation and metrics
│   ├── tune.py               # Optuna hyperparameter search
│   ├── modules.py            # Custom YOLO head with root-point branch
│   ├── augment.py            # Mosaic, MixUp, Copy-Paste augmentation
│   ├── dataset.py            # Dataset loader (images + labels + keypoints)
│   ├── loss.py               # Multi-task loss (Box + Seg + Cls + DFL + Root)
│   ├── metrics.py            # mAP + PCK + AbsPCK evaluation
│   ├── utils.py              # Model builder, calculate_pck, calculate_abspck
│   ├── yolov11-seg-root.yaml # Model architecture definition
│   └── README.md             # Detailed reference for this module
├── 04_seg_root_joint/        # Experimental joint segmentation module
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Important Note for Windows Users

In `02_model_training/config.py`, there is a setting called `WORKERS`. **Leave it at 0 on Windows.**

PyTorch's DataLoader uses multiprocessing when `num_workers > 0`, which requires a special spawn guard (`if __name__ == '__main__':`) that the training script doesn't have at the top level. On Windows this will crash immediately. On Linux or macOS you can raise it to 4 or 8 for faster data loading.

---

## Which Files to Touch Day-to-Day

Most of your work will be in these files:

- **`02_model_training/config.py`** — change epochs, batch size, learning rate, loss weights, etc.
- **`02_model_training/train.py`** — the training loop (you rarely need to edit this)
- **`02_model_training/val.py`** — run this to evaluate any checkpoint
- **`02_model_training/tune.py`** — run this to search for better hyperparameters

The files you generally don't need to touch unless you're changing the model architecture itself: `modules.py`, `loss.py`, `augment.py`, `dataset.py`, `metrics.py`, `utils.py`.

---

## Understanding the Output

Training saves everything to `output_train/4class_original/` (configurable in `config.py`):

| File | What it contains |
|---|---|
| `best.pt` | Best checkpoint by validation loss |
| `last.pt` | Most recent checkpoint |
| `results.csv` | Per-epoch metrics (loss, mAP, PCK, AbsPCK) |
| `loss_curves.png` | Train vs val loss over epochs |
| `loss_components.png` | Individual loss terms (box, seg, cls, dfl, kpt) |
| `metrics_curves.png` | mAP and PCK accuracy curves |

---

## Evaluation Metrics

Running `val.py` prints two tables:

**1. Detection & Segmentation (mAP)**

Standard COCO metrics for the bounding box and mask predictions.

**2. Root-Point Accuracy**

| Metric | How it works |
|---|---|
| `PCK@5%`, `PCK@10%`, `PCK@20%` | Fraction of root predictions within θ × bbox_diagonal of the true root (relative threshold) |
| `Abs@5px`, `Abs@10px`, `Abs@15px`, `Abs@20px` | Fraction of root predictions within τ actual pixels of the true root (absolute threshold) |

Both are reported overall and per class.

---

## If You Need to Re-Annotate or Add New Data

Only go into `01_data_preparation/` if you're starting from a new CVAT XML export. The workflow is:

```bash
# 1. Check what's in the annotation file
python 01_data_preparation/01_analyze_xml.py <your_xml_file>

# 2. Convert to YOLO format
python 01_data_preparation/02_convert_xml_to_yolo.py \
    --xml data/raw/xml_takeo-annotation-done/fifth.xml \
    --img-dir data/raw/dataset \
    --out-dir data/processed/yolo_dataset_4classes

# 3. Visualize a few samples to verify the labels look correct
python 01_data_preparation/03_visualize_dataset.py \
    --dataset data/processed/yolo_dataset_4classes \
    --output output/dataset_previews \
    --num-samples 10

# 4. Re-split into train/val/test
python 01_data_preparation/04_resplit_dataset.py \
    --src data/processed/yolo_dataset_4classes \
    --out data/processed
```

All scripts are run from the project root (`plant_detection/`), not from inside the subdirectory.

---

## More Details

- [01_data_preparation/README.md](01_data_preparation/README.md) — label format, class taxonomy, script usage
- [02_model_training/README.md](02_model_training/README.md) — config reference, architecture, loss formulas, PCK/AbsPCK math
"# plant_detection" 
