"""
config.py — Central configuration for YOLO-Seg-Root.

To change any setting, edit the values below directly, or override in code:
    import config as cfg
    cfg.cfg.MODEL_SIZE = 'm'
    cfg.cfg.EPOCHS = 50
"""

import os
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, ".."))


class Config:
    # ── 1. Model Selection ──────────────────────────────────────────
    # Options: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
    MODEL_SIZE: str = "m"

    # ── 2. Training Hyperparameters ────────────────────────────────
    EPOCHS: int          = 100
    BATCH_SIZE: int      = 16          # reduce to 8 or 4 if GPU out-of-memory
    IMG_SIZE: int        = 640         # input image resolution (square)
    LR0: float           = 5e-4        # initial learning rate
    LRF: float           = 0.001       # final LR factor (target: LR0 * LRF)
    WEIGHT_DECAY: float  = 1e-4        # AdamW weight decay
    GRAD_CLIP: float     = 10.0        # gradient norm clipping
    WARMUP_EPOCHS: int   = 3           # linear warmup length
    CLOSE_MOSAIC_EP: int = 10          # disable mosaic for the final N epochs
    PATIENCE: int        = 10          # early stopping patience (epochs)
    VAL_INTERVAL: int    = 1           # validate every N epochs
    SAVE_PERIOD: int     = 10          # save checkpoint every N epochs

    # ── 3. Multi-Task Loss Weights ─────────────────────────────────
    BOX_GAIN: float       = 7.5        # bounding box loss weight
    KPT_GAIN: float       = 8.0        # root-point keypoint loss weight
    CLS_GAIN: float       = 0.5        # classification loss weight
    CLASS_WEIGHTS: object = "auto"     # "auto" or list of floats
    WEED_CLASSES: list    = [2, 3]     # class IDs considered weeds

    # ── 4. Validation & Inference ──────────────────────────────────
    CONF_THRES: float    = 0.001       # NMS confidence threshold
    IOU_THRES: float     = 0.6         # NMS IoU threshold
    DEVICE: str          = "cuda" if torch.cuda.is_available() else "cpu"
    WORKERS: int         = 0           # MUST stay 0 on Windows to avoid crash

    # ── 5. Data Augmentation ───────────────────────────────────────
    HYP: dict = {
        "mosaic":     1.0,             # mosaic probability
        "mixup":      0.0,             # mixup probability
        "copy_paste": 0.0,             # set 0.3 for weed imbalance
        "fliplr":     0.5,             # horizontal flip probability
        "flipud":     0.0,             # vertical flip probability
        "hsv_h":      0.015,           # HSV hue fraction
        "hsv_s":      0.7,             # HSV saturation fraction
        "hsv_v":      0.4,             # HSV value fraction
        "degrees":    0.0,             # rotation degrees
        "translate":  0.1,             # translation fraction
        "scale":      0.5,             # scaling gain
    }

    # ── 6. Paths ───────────────────────────────────────────────────
    DATA_DIR: str        = os.path.join(_THIS, "data")
    OUTPUT_DIR: str      = os.path.join(_THIS, "output_train", "4class_original")
    DEFAULT_OUTPUT: str  = os.path.join(_THIS, "predictions_box")
    RESUME_WEIGHTS       = None        # checkpoint path to resume training from

    @property
    def TRAIN_IMAGES(self) -> str: return os.path.join(self.DATA_DIR, "images", "train")
    @property
    def TRAIN_LABELS(self) -> str: return os.path.join(self.DATA_DIR, "labels", "train")
    @property
    def VAL_IMAGES(self) -> str:   return os.path.join(self.DATA_DIR, "images", "val")
    @property
    def VAL_LABELS(self) -> str:   return os.path.join(self.DATA_DIR, "labels", "val")
    @property
    def TEST_IMAGES(self) -> str:  return os.path.join(self.DATA_DIR, "images", "test")
    @property
    def TEST_LABELS(self) -> str:  return os.path.join(self.DATA_DIR, "labels", "test")
    @property
    def DEFAULT_SOURCE(self) -> str: return self.TEST_IMAGES

    @property
    def PRETRAINED_WEIGHTS(self) -> str:
        name = f"yolo11{self.MODEL_SIZE.lower()}-seg.pt"
        local_pt = os.path.join(_ROOT, name)
        return local_pt if os.path.exists(local_pt) else name

    @property
    def WEIGHT_PATH(self) -> str:
        candidates = [
            os.path.join(self.OUTPUT_DIR, "best.pt"),
            os.path.join(_THIS, "output_train", "best.pt"),
            os.path.join(_ROOT, "03_model_export", "weights", "best.pt"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]


# Global singleton instance (accessible as cfg.cfg.<PARAM> or cfg.<PARAM>)
cfg = Config()
