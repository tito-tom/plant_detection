"""
metrics.py — Box + Mask mAP evaluation for YOLO-Seg-Root.

CustomSegmentMetrics accumulates per-image predictions and computes
Box/Mask mAP50/50-95 using the Ultralytics SegmentMetrics accumulator.
"""

import numpy as np
import torch
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou
class CustomSegmentMetrics:
    """Accumulates per-image predictions and ground-truths; computes Box + Mask mAP.

    Args:
        nc          : Number of classes.
        class_names : Dict mapping class id → name.

    Usage::

        eval = CustomSegmentMetrics(nc=4, class_names=CLASS_NAMES)
        for image in val_loader:
            eval.add_batch(p_boxes, p_masks, p_scores, p_cls,
                           g_boxes, g_masks, g_cls, img_idx=i)
        results = eval.compute()
    """

    def __init__(self, nc: int, class_names: dict):
        self.nc          = nc
        self.class_names = class_names
        self.iouv        = torch.linspace(0.5, 0.95, 10)
        self.niou        = self.iouv.numel()
        self.metrics     = SegmentMetrics(names=class_names)
        self.reset()

    def reset(self):
        """Clear all accumulated prediction/GT data."""
        # Key order must match Ultralytics SegmentMetrics.process()
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[], tp_m=[])

    def add_batch(self, p_boxes, p_masks, p_scores, p_cls,
                  g_boxes, g_masks, g_cls, img_idx: int = 0):
        """Register predictions and GTs for one image.

        Args:
            p_boxes  : (N, 4) predicted boxes (xyxy, pixel).
            p_masks  : (N, H, W) predicted binary masks.
            p_scores : (N,) confidence scores.
            p_cls    : (N,) predicted class IDs.
            g_boxes  : (M, 4) GT boxes (xyxy, pixel).
            g_masks  : (M, H, W) GT binary masks.
            g_cls    : (M,) GT class IDs.
            img_idx  : Image identifier for target_img accumulator.
        """
        device = p_boxes.device
        iouv   = self.iouv.to(device)
        tp_b, tp_m = self._process_image(p_boxes, p_masks, p_cls, g_boxes, g_masks, g_cls, iouv)

        self.stats["tp"].append(tp_b.cpu().numpy())
        self.stats["tp_m"].append(tp_m.cpu().numpy())
        self.stats["conf"].append(p_scores.cpu().numpy())
        self.stats["pred_cls"].append(p_cls.cpu().numpy())
        self.stats["target_cls"].append(g_cls.cpu().numpy())
        self.stats["target_img"].append(np.full(len(g_cls), img_idx))

    def _process_image(self, p_boxes, p_masks, p_cls, g_boxes, g_masks, g_cls, iouv):
        """Compute TP matrices for one image at COCO IoU thresholds.

        Returns:
            tp_b : (N, 10) bool — box true-positives.
            tp_m : (N, 10) bool — mask true-positives.
        """
        n_pred = len(p_boxes)
        tp_b   = torch.zeros((n_pred, len(iouv)), dtype=torch.bool, device=p_boxes.device)
        tp_m   = torch.zeros((n_pred, len(iouv)), dtype=torch.bool, device=p_boxes.device)

        if n_pred == 0 or len(g_boxes) == 0:
            return tp_b, tp_m

        iou_box = box_iou(g_boxes, p_boxes)

        if len(p_masks) > 0 and len(g_masks) > 0:
            g_flat  = (g_masks.view(len(g_masks), -1) > 0.5).float()
            p_flat  = (p_masks.view(len(p_masks), -1) > 0.5).float()
            iou_msk = mask_iou(g_flat, p_flat)
        else:
            iou_msk = torch.zeros_like(iou_box)

        tp_b = self._match(p_cls, g_cls, iou_box, iouv)
        tp_m = self._match(p_cls, g_cls, iou_msk, iouv)
        return tp_b, tp_m

    @staticmethod
    def _match(pred_cls, gt_cls, iou, iouv):
        """Greedy bipartite matching of predictions → GTs at each IoU threshold.

        Returns (N_pred, N_thresh) bool tensor.
        """
        correct    = torch.zeros(len(pred_cls), len(iouv), dtype=torch.bool, device=iouv.device)
        same_class = gt_cls[:, None] == pred_cls[None, :]
        iou        = iou * same_class

        for j, thr in enumerate(iouv):
            matches = torch.nonzero(iou >= thr)
            if matches.shape[0] == 0:
                continue
            gi, pi  = matches[:, 0], matches[:, 1]
            order   = iou[gi, pi].argsort(descending=True)
            gi, pi  = gi[order], pi[order]
            seen_g, seen_p = set(), set()
            for g, p in zip(gi.tolist(), pi.tolist()):
                if g not in seen_g and p not in seen_p:
                    seen_g.add(g); seen_p.add(p)
                    correct[p, j] = True
        return correct

    def compute(self) -> dict:
        """Compute and return aggregate + per-class metrics dict."""
        empty = {
            "mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0,
            "mask_mAP50": 0.0, "mask_mAP50_95": 0.0, "mask_precision": 0.0, "mask_recall": 0.0,
            **{f"per_class_{k}": {i: 0.0 for i in range(self.nc)}
               for k in ["ap50", "ap50_95", "mask_ap50", "mask_ap50_95",
                         "precision", "recall", "mask_precision", "mask_recall"]},
        }
        if not self.stats["tp"]:
            return empty

        tp         = np.concatenate(self.stats["tp"])
        tp_m       = np.concatenate(self.stats["tp_m"])
        conf       = np.concatenate(self.stats["conf"])
        pred_cls   = np.concatenate(self.stats["pred_cls"])
        target_cls = np.concatenate(self.stats["target_cls"])

        import inspect
        sig = inspect.signature(self.metrics.process)
        if "tp" in sig.parameters or "tp_m" in sig.parameters:
            self.metrics.process(tp=tp, tp_m=tp_m, conf=conf,
                                 pred_cls=pred_cls, target_cls=target_cls)
        else:
            for k in ["tp", "conf", "pred_cls", "target_cls", "target_img", "tp_m"]:
                if k in self.stats:
                    self.metrics.stats[k] = self.stats[k]
            self.metrics.process()

        rd    = self.metrics.results_dict
        box_m = getattr(self.metrics, "box", self.metrics)
        seg_m = getattr(self.metrics, "seg", None)

        instances_per_class = {i: int((target_cls == i).sum()) for i in range(self.nc)}

        return {
            "instances_per_class":      instances_per_class,
            "mAP50":                    rd.get("metrics/mAP50(B)",     0.0),
            "mAP50_95":                 rd.get("metrics/mAP50-95(B)",  0.0),
            "precision":                rd.get("metrics/precision(B)",  0.0),
            "recall":                   rd.get("metrics/recall(B)",    0.0),
            "mask_mAP50":               rd.get("metrics/mAP50(M)",     0.0),
            "mask_mAP50_95":            rd.get("metrics/mAP50-95(M)",  0.0),
            "mask_precision":           rd.get("metrics/precision(M)",  0.0),
            "mask_recall":              rd.get("metrics/recall(M)",    0.0),
            "per_class_ap50":           _extract_ap(box_m, self.nc, is95=False),
            "per_class_ap50_95":        _extract_ap(box_m, self.nc, is95=True),
            "per_class_mask_ap50":      _extract_ap(seg_m, self.nc, is95=False) if seg_m else empty["per_class_mask_ap50"],
            "per_class_mask_ap50_95":   _extract_ap(seg_m, self.nc, is95=True)  if seg_m else empty["per_class_mask_ap50_95"],
            "per_class_precision":      _extract_val(box_m, "p", self.nc),
            "per_class_recall":         _extract_val(box_m, "r", self.nc),
            "per_class_mask_precision": _extract_val(seg_m, "p", self.nc) if seg_m else empty["per_class_mask_precision"],
            "per_class_mask_recall":    _extract_val(seg_m, "r", self.nc) if seg_m else empty["per_class_mask_recall"],
        }


def _extract_ap(m_obj, nc: int, is95: bool) -> dict:
    """Extract per-class AP50 or AP50-95 from an Ultralytics metric object."""
    ap_dict = {i: 0.0 for i in range(nc)}
    attr    = "ap" if is95 else "ap50"
    if m_obj is None:
        return ap_dict
    if hasattr(m_obj, "ap_class_index") and hasattr(m_obj, attr):
        for i, cls_idx in enumerate(m_obj.ap_class_index):
            if cls_idx < nc:
                vals = np.asarray(getattr(m_obj, attr)[i])
                ap_dict[int(cls_idx)] = float(vals.flat[0]) if vals.size > 0 else 0.0
    return ap_dict


def _extract_val(m_obj, attr: str, nc: int) -> dict:
    """Extract per-class precision or recall from an Ultralytics metric object."""
    res = {i: 0.0 for i in range(nc)}
    if m_obj is None:
        return res
    if hasattr(m_obj, "ap_class_index") and hasattr(m_obj, attr):
        for i, cls_idx in enumerate(m_obj.ap_class_index):
            if cls_idx < nc:
                res[int(cls_idx)] = float(getattr(m_obj, attr)[i])
    return res
