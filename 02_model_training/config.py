import os
import torch
from dataclasses import dataclass, field


@dataclass
class Config:
    """Centralized configuration for YOLO-Seg-Root training.

    All paths are computed automatically relative to this file.
    Override any attribute after importing: ``cfg.EPOCHS = 50``.
    """

    # Training hyperparameters
    EPOCHS: int          = 100
    BATCH_SIZE: int      = 16
    IMG_SIZE: int        = 640
    LR0: float           = 5e-4
    LRF: float           = 0.001
    WEIGHT_DECAY: float  = 1e-4
    GRAD_CLIP: float     = 10.0
    WARMUP_EPOCHS: int   = 3
    CLOSE_MOSAIC_EP: int = 10
    PATIENCE: int        = 10
    VAL_INTERVAL: int    = 1
    SAVE_PERIOD: int     = 10

    # Inference settings
    CONF_THRES: float = 0.001
    IOU_THRES: float  = 0.6

    # Loss weights
    KPT_GAIN: float       = 8.0
    BOX_GAIN: float       = 7.5
    CLS_GAIN: float       = 0.5
    CLASS_WEIGHTS: object = "auto"   # "auto" or list[float] of length nc
    WEED_CLASSES: list    = field(default_factory=lambda: [2, 3])

    # Augmentation probabilities and limits
    HYP: dict = field(default_factory=lambda: {
        "mosaic":     1.0,
        "mixup":      0.0,
        "copy_paste": 0.0,   # 0.3 recommended for weed imbalance
        "fliplr":     0.5,
        "flipud":     0.0,
        "hsv_h":      0.015,
        "hsv_s":      0.7,
        "hsv_v":      0.4,
        "degrees":    0.0,
        "translate":  0.1,
        "scale":      0.5,
    })

    def __post_init__(self):
        """Compute all path-derived settings after dataclass init."""
        _this = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.normpath(os.path.join(_this, ".."))

        self.DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
        self.WORKERS = 0

        _local_data = os.path.join(_this, "data")
        
        _root_data  = os.path.normpath(os.path.join(_root, "data"))

        if os.path.isdir(_local_data) and os.path.isdir(os.path.join(_local_data, "images")):
            self.DATA_DIR = _local_data
        elif os.path.isdir(_root_data) and os.path.isdir(os.path.join(_root_data, "images")):
            self.DATA_DIR = _root_data
        else:
            self.DATA_DIR = _local_data

        self.TRAIN_IMAGES = os.path.join(self.DATA_DIR, "images", "train")
        self.TRAIN_LABELS = os.path.join(self.DATA_DIR, "labels", "train")
        self.VAL_IMAGES   = os.path.join(self.DATA_DIR, "images", "val")
        self.VAL_LABELS   = os.path.join(self.DATA_DIR, "labels", "val")
        self.TEST_IMAGES  = os.path.join(self.DATA_DIR, "images", "test")
        self.TEST_LABELS  = os.path.join(self.DATA_DIR, "labels", "test")

        self.OUTPUT_DIR     = os.path.join(_this, "output_train", "4class_original")
        self.RESUME_WEIGHTS = None

        _pt = os.path.join(_root, "yolo11m-seg.pt")
        self.PRETRAINED_WEIGHTS = _pt if os.path.exists(_pt) else "yolo11m-seg.pt"
        self.WEIGHT_PATH        = os.path.join(_root, "model_inference", "final_best.pt")

        self.DEFAULT_SOURCE = os.path.join(self.DATA_DIR, "images", "test")
        self.DEFAULT_OUTPUT = os.path.join(_this, "predictions_box")

cfg = Config()
