"""
infer_utils.py  —  v5 Register Head Inference Utilities

Provides `build_model()` which:
  1. Registers CustomSegmentHead into ultralytics namespace.
  2. Patches parse_model to handle it like a built-in head.
  3. Builds model from YAML (parser creates CustomSegmentHead directly).
  4. Loads fine-tuned weights securely (casts to float32 and filters shape mismatches).
"""

import os
import sys
import copy
import torch
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_YAML_PATH = os.path.join(_MODEL_DIR, "yolov11-seg-root.yaml")
# yolo11m-seg.pt lives in the parent folder (03_model_export/)
_PRETRAINED = os.path.join(os.path.dirname(_MODEL_DIR), "yolo11m-seg.pt")

if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "crop_small",
    1: "crop_large",
    2: "weed_small",
    3: "weed_large",
}
_DEFAULT_HYPS = {"box": 7.5, "cls": 0.5, "dfl": 1.5, "pose": 12.0}


# ---------------------------------------------------------------------------
# Module Registration + Parser Patching
# ---------------------------------------------------------------------------
_REGISTERED = False


def register_custom_modules():
    """
    Register CustomSegmentHead so the YAML parser can find it by name
    and handle its multi-input `from: [15, 18, 21]` like built-in Segment.

    Safe and idempotent — calling multiple times is a no-op.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks
    from modules import CustomSegmentHead

    # 1. Register in namespace
    tasks.CustomSegmentHead = CustomSegmentHead

    # 2. Patch parse_model to recognize CustomSegmentHead
    _original_parse_model = tasks.parse_model

    def _patched_parse_model(d, ch, verbose=True):
        from ultralytics.nn.modules.head import Segment

        all_layers = d["backbone"] + d["head"]
        custom_indices = []
        for i, (f, n, m_name, args) in enumerate(all_layers):
            if m_name == "CustomSegmentHead":
                custom_indices.append((i, f, n, m_name, list(args)))

        if not custom_indices:
            return _original_parse_model(d, ch, verbose)

        # Temporarily swap CustomSegmentHead → Segment for parser
        d_copy = copy.deepcopy(d)
        head_layers = d_copy["head"]
        backbone_len = len(d_copy["backbone"])
        for idx, f, n, m_name, args in custom_indices:
            layer_idx = idx - backbone_len
            if 0 <= layer_idx < len(head_layers):
                head_layers[layer_idx][2] = "Segment"

        model, save = _original_parse_model(d_copy, ch, verbose)

        # Replace Segment modules with CustomSegmentHead
        for idx, f, n, m_name, orig_args in custom_indices:
            segment_module = model[idx]
            ch_list = [
                segment_module.cv2[i][0].conv.in_channels
                for i in range(len(segment_module.cv2))
            ]
            custom_head = CustomSegmentHead(
                nc=segment_module.nc,
                nm=segment_module.nm,
                npr=segment_module.npr,
                ch=tuple(ch_list),
            )
            for attr in ("cv2", "cv3", "cv4", "proto", "dfl"):
                if hasattr(segment_module, attr):
                    setattr(custom_head, attr, getattr(segment_module, attr))

            custom_head.i = segment_module.i
            custom_head.f = segment_module.f
            custom_head.type = "modules.CustomSegmentHead"
            custom_head.np = sum(x.numel() for x in custom_head.parameters())
            custom_head.stride = segment_module.stride
            model[idx] = custom_head

        return model, save

    tasks.parse_model = _patched_parse_model
    _REGISTERED = True
    print("[infer_utils] Registered CustomSegmentHead (v5) + patched parser")


# ---------------------------------------------------------------------------
# DictNamespace for ultralytics config compatibility
# ---------------------------------------------------------------------------
class DictNamespace(dict):
    """A dictionary that also supports attribute (dot) access to act as a namespace."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'DictNamespace' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'DictNamespace' object has no attribute '{name}'")


def _patch_model_args(inner_model):
    """Ensure model.args supports both dictionary brackets and attribute dot access."""
    if not hasattr(inner_model, "args") or inner_model.args is None:
        inner_model.args = DictNamespace(DEFAULT_CFG) if isinstance(DEFAULT_CFG, dict) else DictNamespace()

    if isinstance(inner_model.args, dict) and not isinstance(inner_model.args, DictNamespace):
        inner_model.args = DictNamespace(inner_model.args)
    elif hasattr(inner_model.args, "__dict__") and not isinstance(inner_model.args, DictNamespace):
        inner_model.args = DictNamespace(inner_model.args.__dict__)

    for key, default_val in _DEFAULT_HYPS.items():
        if not hasattr(inner_model.args, key):
            setattr(inner_model.args, key, default_val)


# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------
def build_model(weights_path=None, device="cpu"):
    """
    Build YOLO-Seg-Root model using v5 register head approach.

    Steps:
        1. Register CustomSegmentHead + patch parser.
        2. Build model from YAML (parser creates CustomSegmentHead).
        3. Load pretrained backbone weights.
        4. Securely load fine-tuned weights (casts to float32 & filters mismatches).
        5. Patch model.args with required hyperparameters.

    Args:
        weights_path: Path to fine-tuned .pt file (optional).
        device: "cuda" or "cpu".

    Returns:
        YOLO model wrapper with CustomSegmentHead.
    """
    # Step 1: Register
    register_custom_modules()

    # Step 2: Build from YAML
    print(f"[infer_utils] Building model from: {_YAML_PATH}")
    # Explicitly specify the task so custom models build in correct mode
    model = YOLO(_YAML_PATH, task="segment")

    # Step 3: Load pretrained backbone
    try:
        model.load(_PRETRAINED)
        print(f"[infer_utils] Loaded pretrained backbone ({_PRETRAINED})")
    except Exception as e:
        print(f"[infer_utils] Backbone loading note: {e}")

    # Step 4: Securely load fine-tuned weights (using state_dict for shape verification)
    if weights_path and os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt_sd = (
                ckpt["model"].state_dict()
                if hasattr(ckpt["model"], "state_dict")
                else ckpt["model"]
            )
        else:
            ckpt_sd = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt

        # Cast all floating weights to float32 and filter mismatched shapes
        model_sd = model.model.state_dict()
        ckpt_sd = {
            k: v.float() if v.is_floating_point() else v
            for k, v in ckpt_sd.items()
            if k in model_sd and getattr(v, "shape", None) == getattr(model_sd[k], "shape", None)
        }
        
        result = model.model.load_state_dict(ckpt_sd, strict=False)
        n_loaded = len(ckpt_sd) - len(result.unexpected_keys)
        print(f"[infer_utils] Loaded fine-tuned weights: {weights_path} ({n_loaded} params with matching shape)")
        if result.missing_keys:
            print(f"[infer_utils]   missing: {result.missing_keys[:5]}...")
    elif weights_path:
        print(f"[infer_utils] Warning: weights not found at {weights_path}")

    # Step 5: Patch args
    _patch_model_args(model.model)

    model.to(device)
    return model
