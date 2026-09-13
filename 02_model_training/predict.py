import os
import sys
import argparse
import glob

import cv2
import torch
import numpy as np

# Make sure imports work whether called from project root or from inside 02_model_training/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from utils import build_model, CLASS_NAMES, CLASS_COLORS
import config as cfg

# Root-point marker colour (magenta)
ROOT_COLOR = (255, 0, 255)
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ── Drawing helpers ──────────────────────────────────────────────────────────

def draw_mask(image: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.35) -> np.ndarray:
    """Blend a binary mask over the image with the given colour."""
    overlay          = image.copy()
    overlay[mask > 0] = np.array(color, dtype=np.uint8)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_box(image: np.ndarray, x1: int, y1: int, x2: int, y2: int,
             color: tuple, label: str) -> None:
    """Draw a bounding box with a label tag."""
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(image, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_root(image: np.ndarray, rx: int, ry: int) -> None:
    """Draw a root-point marker (circle + crosshair)."""
    cv2.circle(image, (rx, ry), 6, ROOT_COLOR, -1)
    cv2.circle(image, (rx, ry), 8, (255, 255, 255), 1)
    cv2.line(image, (rx - 12, ry), (rx + 12, ry), ROOT_COLOR, 1)
    cv2.line(image, (rx, ry - 12), (rx, ry + 12), ROOT_COLOR, 1)


# ── Inference ────────────────────────────────────────────────────────────────

def predict_single(model, image_path: str, conf_thres: float, iou_thres: float,
                   device: str) -> np.ndarray:
    """Run one image through the model and return an annotated BGR numpy array."""
    from ultralytics.utils.ops import process_mask, xyxy2xywh
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.tal import make_anchors
    from loss import CustomLoss

    raw = cv2.imread(image_path)
    if raw is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    ih, iw = raw.shape[:2]
    img_size = cfg.cfg.IMG_SIZE

    # Letterbox resize
    scale  = img_size / max(ih, iw)
    nh, nw = int(ih * scale), int(iw * scale)
    resized = cv2.resize(raw, (nw, nh))
    canvas  = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized

    tensor = torch.from_numpy(canvas).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    head = model.model.model[-1]

    # Set training=True → head returns a dict (same pattern as val.py)
    head.training = True
    with torch.no_grad():
        preds_out = model.model(tensor)
    head.training = False  # restore

    feats          = preds_out["feats"]
    pred_masks_raw = preds_out["mask_coefficient"]
    proto          = preds_out["proto"].clone()
    pred_kpts_raw  = preds_out["kpts"]

    anchor_points, stride_tensor = make_anchors(feats, head.stride, 0.5)

    # Decode keypoints
    pred_kpts_dec = pred_kpts_raw.permute(0, 2, 1).contiguous().clone()
    pred_kpts_dec[..., 0] = (pred_kpts_dec[..., 0] * 2.0 + (anchor_points[:, 0] - 0.5)) * stride_tensor[:, 0]
    pred_kpts_dec[..., 1] = (pred_kpts_dec[..., 1] * 2.0 + (anchor_points[:, 1] - 0.5)) * stride_tensor[:, 0]

    # Decode boxes using CustomLoss (same as val.py)
    pred_distri_p = preds_out["boxes"].permute(0, 2, 1).contiguous()
    criterion          = CustomLoss(model.model)
    pred_bboxes        = criterion.bbox_decode(anchor_points, pred_distri_p)
    pred_bboxes_xywh   = xyxy2xywh(pred_bboxes * stride_tensor)

    nms_input = torch.cat([
        pred_bboxes_xywh.permute(0, 2, 1),
        preds_out["scores"].sigmoid(),
        pred_masks_raw,
        pred_kpts_dec.permute(0, 2, 1),
    ], dim=1)

    detections = non_max_suppression(nms_input, conf_thres=conf_thres,
                                      iou_thres=iou_thres, nc=head.nc)

    annotated = raw.copy()

    if detections[0] is not None and len(detections[0]) > 0:
        det           = detections[0]
        p_boxes       = det[:, :4]
        p_confs       = det[:, 4]
        p_cls         = det[:, 5].int()
        p_mask_coeffs = det[:, 6: 6 + head.nm]
        p_kpts        = det[:, 6 + head.nm:]   # (N, 2)

        p_masks = process_mask(proto[0], p_mask_coeffs, p_boxes,
                               (img_size, img_size), upsample=True)
        p_masks = (p_masks > 0.5).cpu().numpy()

        # Scale boxes back to original image size
        p_boxes_np = p_boxes.cpu().numpy().copy()
        p_boxes_np[:, [0, 2]] = p_boxes_np[:, [0, 2]] / scale
        p_boxes_np[:, [1, 3]] = p_boxes_np[:, [1, 3]] / scale
        p_boxes_np = p_boxes_np.clip([0, 0, 0, 0], [iw, ih, iw, ih]).astype(int)

        p_kpts_np = p_kpts.cpu().numpy().copy()
        p_kpts_np[:, 0] /= scale
        p_kpts_np[:, 1] /= scale

        for i in range(len(det)):
            cls_id = int(p_cls[i])
            color  = CLASS_COLORS.get(cls_id, (200, 200, 200))
            name   = CLASS_NAMES.get(cls_id, str(cls_id))
            conf   = float(p_confs[i])

            # Resize mask to original image dimensions
            mask_full = cv2.resize(
                p_masks[i].astype(np.uint8),
                (img_size, img_size), interpolation=cv2.INTER_NEAREST)
            mask_orig = mask_full[:nh, :nw]
            mask_orig = cv2.resize(mask_orig, (iw, ih), interpolation=cv2.INTER_NEAREST)

            annotated = draw_mask(annotated, mask_orig, color)

            x1, y1, x2, y2 = p_boxes_np[i]
            draw_box(annotated, x1, y1, x2, y2, color, f"{name} {conf:.2f}")

            rx, ry = int(p_kpts_np[i, 0]), int(p_kpts_np[i, 1])
            if 0 <= rx < iw and 0 <= ry < ih:
                draw_root(annotated, rx, ry)

    return annotated


def collect_images(source: str) -> list:
    """Return a list of image paths from a file, directory, or glob pattern."""
    if os.path.isfile(source):
        return [source]
    if os.path.isdir(source):
        found = []
        for ext in IMG_EXTS:
            found += glob.glob(os.path.join(source, f"*{ext}"))
            found += glob.glob(os.path.join(source, f"*{ext.upper()}"))
        return sorted(set(found))
    # Try as a glob
    matches = glob.glob(source)
    return sorted(f for f in matches if os.path.splitext(f)[1].lower() in IMG_EXTS)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO-Seg-Root inference: draws boxes, masks, and root points.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to trained checkpoint (.pt). Default: cfg.WEIGHT_PATH")
    parser.add_argument("--source",  type=str, default=None,
                        help="Image file or folder to run inference on. Default: test split.")
    parser.add_argument("--output",  type=str, default=None,
                        help="Folder to save annotated images. Default: cfg.DEFAULT_OUTPUT")
    parser.add_argument("--conf",    type=float, default=cfg.cfg.CONF_THRES,
                        help=f"Confidence threshold (default: {cfg.cfg.CONF_THRES})")
    parser.add_argument("--iou",     type=float, default=cfg.cfg.IOU_THRES,
                        help=f"NMS IoU threshold (default: {cfg.cfg.IOU_THRES})")
    parser.add_argument("--device",  type=str,  default=cfg.cfg.DEVICE,
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    weights = args.weights or cfg.cfg.WEIGHT_PATH
    source  = args.source  or cfg.cfg.DEFAULT_SOURCE
    out_dir = args.output  or cfg.cfg.DEFAULT_OUTPUT

    # Resolve paths relative to project root so the script works wherever it's called from
    if not os.path.isabs(weights):
        weights = os.path.join(_ROOT_DIR, weights)
    if not os.path.isabs(source):
        source = os.path.join(_ROOT_DIR, source)
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(_ROOT_DIR, out_dir)

    if not os.path.exists(weights):
        print(f"[predict] ERROR: weights not found at: {weights}")
        print("[predict] Run train.py first, or pass --weights path/to/best.pt")
        sys.exit(1)

    images = collect_images(source)
    if not images:
        print(f"[predict] No images found in: {source}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[predict] Weights : {weights}")
    print(f"[predict] Source  : {source}  ({len(images)} images)")
    print(f"[predict] Output  : {out_dir}")
    print(f"[predict] Device  : {args.device}  |  conf={args.conf}  iou={args.iou}\n")

    model = build_model(weights_path=weights, device=args.device)
    model.model.eval()

    for i, img_path in enumerate(images, 1):
        fname = os.path.basename(img_path)
        try:
            annotated = predict_single(model, img_path, args.conf, args.iou, args.device)
            save_path = os.path.join(out_dir, fname)
            cv2.imwrite(save_path, annotated)
            print(f"  [{i:>4d}/{len(images)}]  {fname}  →  saved")
        except Exception as e:
            print(f"  [{i:>4d}/{len(images)}]  {fname}  →  ERROR: {e}")

    print(f"\n[predict] Done. Annotated images saved to: {out_dir}")


if __name__ == "__main__":
    main()

