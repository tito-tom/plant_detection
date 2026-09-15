"""
utils.py — Model building utilities and shared helpers for YOLO-Seg-Root.
"""

import os
import sys
import colorsys as _colorsys
import yaml as _yaml
import torch
from types import SimpleNamespace
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

CLASS_NAMES = {
    0: "crop_small_leaf",
    1: "crop_large_leaf",
    2: "weed_small_leaf",
    3: "weed_large_leaf",
}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 150, 0),
    2: (0, 0, 255),
    3: (0, 0, 150),
}
DEFAULT_HYPS = {"box": 7.5, "cls": 0.5, "dfl": 1.5, "pose": 12.0}
YAML_PATH    = os.path.join(_THIS_DIR, "yolov11-seg-root.yaml")
LOSS_NAMES   = ("box", "seg", "cls", "dfl", "kpt")

ROOT_POINT_COLOR = (255, 0, 255)
DEFAULT_HYPS     = {"box": 7.5, "cls": 0.5, "dfl": 1.5, "pose": 12.0}
YAML_PATH        = os.path.join(_THIS_DIR, "yolov11-seg-root.yaml")
LOSS_NAMES       = ("box", "seg", "cls", "dfl", "kpt")

import config as cfg

def get_pretrained_weights(scale=None):
    s = (scale or getattr(cfg.cfg, "MODEL_SIZE", "s")).lower()
    name = f"yolo11{s}-seg.pt"
    local_pt = os.path.join(_ROOT_DIR, name)
    return local_pt if os.path.exists(local_pt) else name

PRETRAINED_WEIGHTS = get_pretrained_weights()


def load_yaml_config(data_dir: str) -> dict:
    """Read an experiment's data.yaml and return paths, class names, and colors.

    Returns dict with keys: nc, class_names, class_colors,
    train_images, train_labels, val_images, val_labels, exp_name.
    """
    yaml_path = os.path.join(data_dir, "data.yaml")
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")

    with open(yaml_path) as f:
        raw = _yaml.safe_load(f)

    for key in ("nc", "names"):
        if key not in raw:
            raise ValueError(f"data.yaml missing required field: '{key}'")

    nc        = int(raw["nc"])
    raw_names = raw["names"]
    if isinstance(raw_names, dict):
        class_names = {int(k): v for k, v in raw_names.items()}
    elif isinstance(raw_names, list):
        class_names = {i: v for i, v in enumerate(raw_names)}
    else:
        raise ValueError(f"Unexpected 'names' format: {type(raw_names)}")

    if nc <= 4:
        _palette    = {0: (0,255,0), 1: (0,150,0), 2: (0,0,255), 3: (0,0,150)}
        class_colors = {i: _palette[i] for i in range(nc)}
    else:
        class_colors = {}
        for i in range(nc):
            r, g, b = _colorsys.hsv_to_rgb(i / nc, 0.9, 0.9)
            class_colors[i] = (int(b*255), int(g*255), int(r*255))

    raw_train    = raw.get("train", os.path.join(data_dir, "images", "train"))
    raw_val      = raw.get("val",   os.path.join(data_dir, "images", "test"))
    train_images = str(raw_train)
    train_labels = str(raw_train).replace("images", "labels")
    val_images   = str(raw_val)
    val_labels   = str(raw_val).replace("images", "labels")
    exp_name     = os.path.basename(os.path.normpath(data_dir))

    print(f"[config] Experiment : {exp_name}")
    print(f"[config] Classes    : {nc} - {list(class_names.values())}")

    return {
        "nc": nc, "class_names": class_names, "class_colors": class_colors,
        "train_images": train_images, "train_labels": train_labels,
        "val_images": val_images, "val_labels": val_labels, "exp_name": exp_name,
    }


class ModelBuilder:
    """Builds and registers the YOLO-Seg-Root model.

    Usage::

        model = ModelBuilder.build(weights_path="best.pt", device="cuda")
    """

    _registered = False
    CURRENT_SCALE = "s"

    @classmethod
    def register(cls):
        """Register CustomSegmentHead into the Ultralytics parser namespace (idempotent)."""
        if cls._registered:
            return

        import ultralytics.nn.tasks as tasks
        from modules import CustomSegmentHead

        tasks.CustomSegmentHead = CustomSegmentHead
        _original_parse = tasks.parse_model

        def _patched_parse(d, ch, verbose=True):
            from ultralytics.nn.modules.head import Segment
            import copy

            all_layers     = d["backbone"] + d["head"]
            custom_indices = [
                (i, f, n, m, list(a))
                for i, (f, n, m, a) in enumerate(all_layers)
                if m == "CustomSegmentHead"
            ]
            if not custom_indices:
                return _original_parse(d, ch, verbose)

            d_copy = copy.deepcopy(d)

            backbone_len = len(d_copy["backbone"])
            for idx, *_ in custom_indices:
                layer_idx = idx - backbone_len
                if 0 <= layer_idx < len(d_copy["head"]):
                    d_copy["head"][layer_idx][2] = "Segment"

            model, save = _original_parse(d_copy, ch, verbose)

            for idx, f, n, m_name, orig_args in custom_indices:
                seg = model[idx]
                ch_list = [seg.cv2[i][0].conv.in_channels for i in range(len(seg.cv2))]
                head = CustomSegmentHead(
                    nc=seg.nc, nm=seg.nm, npr=seg.npr, ch=tuple(ch_list)
                )
                for attr in ("cv2", "cv3", "cv4", "proto", "dfl"):
                    if hasattr(seg, attr):
                        setattr(head, attr, getattr(seg, attr))
                head.i    = seg.i
                head.f    = seg.f
                head.type = "modules.CustomSegmentHead"
                head.np   = sum(x.numel() for x in head.parameters())
                head.stride = seg.stride
                model[idx] = head

            return model, save

        tasks.parse_model = _patched_parse
        cls._registered   = True
        print("[utils] Registered CustomSegmentHead + patched parser")

    @classmethod
    def build(cls, weights_path=None, device="cpu", model_size=None):
        """Build the YOLO-Seg-Root model.

        Args:
            weights_path: Path to fine-tuned .pt file (optional).
            device      : ``"cuda"`` or ``"cpu"``.
            model_size  : Override size: ``"n"``, ``"s"``, ``"m"``, ``"l"``, ``"x"`` (optional).

        Returns:
            YOLO model wrapper with CustomSegmentHead.
        """
        import re

        scale = model_size
        if not scale and weights_path:
            match = re.search(r"yolo11([nslmx])", str(weights_path))
            if match:
                scale = match.group(1)
        if not scale:
            scale = getattr(cfg.cfg, "MODEL_SIZE", "s")

        cls.CURRENT_SCALE = scale.lower()
        cls.register()

        print(f"[utils] Building YOLO11-{cls.CURRENT_SCALE} model from: {YAML_PATH}")
        model = YOLO(YAML_PATH, task="segment")

        pretrained = get_pretrained_weights(cls.CURRENT_SCALE)
        try:
            model.load(pretrained)
            print(f"[utils] Loaded pretrained backbone ({pretrained})")
        except Exception as e:
            print(f"[utils] Backbone loading note: {e}")

        if weights_path and os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt:
                ckpt_sd = (
                    ckpt["model"].state_dict()
                    if hasattr(ckpt["model"], "state_dict") else ckpt["model"]
                )
            else:
                ckpt_sd = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt

            model_sd = model.model.state_dict()
            ckpt_sd  = {
                k: v.float() if v.is_floating_point() else v
                for k, v in ckpt_sd.items()
                if k in model_sd and getattr(v, "shape", None) == getattr(model_sd[k], "shape", None)
            }
            result   = model.model.load_state_dict(ckpt_sd, strict=False)
            n_loaded = len(ckpt_sd) - len(result.unexpected_keys)
            print(f"[utils] Loaded fine-tuned weights: {weights_path} ({n_loaded} params)")
            if result.missing_keys:
                print(f"[utils]   missing: {result.missing_keys[:5]}...")
        elif weights_path:
            print(f"[utils] Warning: weights not found at {weights_path}")

        cls._patch_model_args(model.model)
        model.to(device)
        return model

    @staticmethod
    def _patch_model_args(inner_model):
        """Ensure model.args has required hyperparameter attributes."""
        if not hasattr(inner_model, "args"):
            inner_model.args = DEFAULT_CFG
        if isinstance(inner_model.args, dict):
            inner_model.args = SimpleNamespace(**inner_model.args)
        for key, val in DEFAULT_HYPS.items():
            if not hasattr(inner_model.args, key):
                setattr(inner_model.args, key, val)

build_model = ModelBuilder.build

def prepare_batch(targets, device):
    """Convert a flat list of per-object target dicts into a batch dict for CustomLoss."""
    if not targets:
        return None

    batch_idx, cls_list, bboxes, masks, kpts = [], [], [], [], []
    for t in targets:
        batch_idx.append(t["image_id"])
        cls_list.append(t["cls"])
        bboxes.append(torch.as_tensor(t["box"],      dtype=torch.float32))
        kpts.append(  torch.as_tensor(t["keypoint"], dtype=torch.float32))
        masks.append(t["mask"])

    if not batch_idx:
        return None

    return {
        "batch_idx": torch.tensor(batch_idx, device=device),
        "cls":       torch.tensor(cls_list,  device=device),
        "bboxes":    torch.stack(bboxes).to(device),
        "masks":     torch.stack(masks).to(device),
        "keypoints": torch.stack(kpts).to(device),
    }


def calculate_pck(pred_kpts, gt_kpts, gt_bboxes, thresholds=(0.05, 0.10, 0.20)):
    """Percentage of Correct Keypoints at relative distance thresholds."""
    if len(pred_kpts) == 0:
        return {t: 0.0 for t in thresholds}

    distances  = torch.sqrt(((pred_kpts - gt_kpts) ** 2).sum(dim=-1))
    widths     = gt_bboxes[:, 2] - gt_bboxes[:, 0]
    heights    = gt_bboxes[:, 3] - gt_bboxes[:, 1]
    diagonals  = torch.sqrt(widths**2 + heights**2).clamp(min=1.0)
    norm_errs  = distances / diagonals
    return {t: (norm_errs < t).float().mean().item() for t in thresholds}


def calculate_abspck(pred_kpts, gt_kpts, thresholds=(5, 10, 15, 20)):
    """Percentage of Correct Keypoints at absolute pixel distance thresholds.

    Args:
        pred_kpts: Tensor of shape (N, 2) with predicted (x, y) in pixel coordinates.
        gt_kpts: Tensor of shape (N, 2) with ground truth (x, y) in pixel coordinates.
        thresholds: Tuple/list of absolute distance thresholds in pixels (e.g. 5, 10, 15, 20 px).

    Returns:
        dict: Mapping each threshold px -> fraction of keypoints with L2 distance <= px.
    """
    if len(pred_kpts) == 0:
        return {t: 0.0 for t in thresholds}

    distances = torch.sqrt(((pred_kpts - gt_kpts) ** 2).sum(dim=-1))
    return {t: (distances <= t).float().mean().item() for t in thresholds}


calculate_abs_pck = calculate_abspck


def print_header():
    """Print Ultralytics-style training table header."""
    print("\n" + "".join(f"{h:>10s}" for h in
          ["Epoch", "GPU_mem", "box_loss", "seg_loss", "cls_loss", "dfl_loss", "kpt_loss",
           "Instances", "Size"]))


def print_epoch_row(epoch, total_epochs, gpu_mem, loss_items, n_instances=0, img_size=640):
    """Print one epoch row of training losses."""
    print(
        f"{epoch:>4d}/{total_epochs:<5d} {gpu_mem:>8s}"
        f" {loss_items[0]:>10.4g} {loss_items[1]:>10.4g} {loss_items[2]:>10.4g}"
        f" {loss_items[3]:>10.4g} {loss_items[4]:>10.4g}"
        f" {n_instances:>10d} {img_size:>10d}"
    )


def get_gpu_memory():
    """Return current GPU memory reservation as a string."""
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_reserved() / 1e9:.1f}G"
    return "CPU"
