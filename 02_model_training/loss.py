"""
loss.py — Multi-task loss for YOLO-Seg-Root.

Combines five components: box (CIoU), seg (BCE masks), cls (weighted BCE),
dfl (Distribution Focal Loss), and kpt (Smooth-L1 root-point regression).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.ops import crop_mask


class CustomLoss(v8SegmentationLoss):
    """Joint detection + segmentation + root-point loss.

    Args:
        model        : Inner DetectionModel (not the YOLO wrapper).
        class_weights: List[float] of length nc, or None for uniform weights.
    """

    def __init__(self, model, class_weights=None):
        super().__init__(model)

        if class_weights is not None:
            pw = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
            print(f"[Loss] Class weights applied: {pw.tolist()}")
        else:
            print("[Loss] No class weights (uniform BCE)")

        m = model.model[-1]
        if not hasattr(self, "no"):
            self.no = m.nc + m.reg_max * 4
        if not hasattr(self, "reg_max"):
            self.reg_max = m.reg_max

        self.kpt_shape = [1, 2]
        self.overlap   = False

    def __call__(self, preds, batch):
        """Compute total loss and return (total, items).

        Args:
            preds : dict with keys boxes, scores, feats, mask_coefficient, proto, kpts.
            batch : dict with keys batch_idx, cls, bboxes, masks, keypoints.

        Returns:
            total_loss  : scalar tensor (backward-able).
            loss_items  : detached (5,) tensor [box, seg, cls, dfl, kpt].
        """
        loss = torch.zeros(5, device=self.device)

        if not isinstance(preds, dict):
            return super().__call__(preds, batch)

        feats      = preds["feats"]
        pred_masks = preds["mask_coefficient"].permute(0, 2, 1).contiguous()
        proto      = preds["proto"]
        pred_kpts  = preds["kpts"].permute(0, 2, 1).contiguous()
        bs         = proto.shape[0]
        _, _, mask_h, mask_w = proto.shape

        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()

        dtype       = pred_scores.dtype
        imgsz       = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        batch_idx  = batch["batch_idx"].view(-1, 1)
        targets    = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets    = self.preprocess(targets, bs, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt    = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes   = self.bbox_decode(anchor_points, pred_distri)

        pred_kpts_dec = pred_kpts.clone()
        pred_kpts_dec[..., 0] = (pred_kpts_dec[..., 0] * 2.0 + (anchor_points[:, 0] - 0.5)) * stride_tensor[:, 0]
        pred_kpts_dec[..., 1] = (pred_kpts_dec[..., 1] * 2.0 + (anchor_points[:, 1] - 0.5)) * stride_tensor[:, 0]

        # NaN guard — prevents CUDA crashes from aggressive hyperparameter choices
        if not torch.isfinite(pred_scores).all() or not torch.isfinite(pred_bboxes).all():
            pred_scores = torch.nan_to_num(pred_scores, nan=0.0, posinf=0.0, neginf=0.0)
            pred_bboxes = torch.nan_to_num(pred_bboxes, nan=0.0, posinf=0.0, neginf=0.0)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[3] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor,
                target_scores, target_scores_sum, fg_mask,
                imgsz, stride_tensor,
            )

            gt_masks = batch["masks"].to(self.device).float()
            if tuple(gt_masks.shape[-2:]) != (mask_h, mask_w):
                gt_masks = F.interpolate(gt_masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, gt_masks, target_gt_idx, target_bboxes,
                batch_idx, proto, pred_masks, imgsz,
            )

            if "keypoints" in batch:
                gt_kpts = batch["keypoints"].to(self.device).float().clone()
                gt_kpts[..., 0] *= imgsz[1]
                gt_kpts[..., 1] *= imgsz[0]
                loss[4] = self._kpt_loss(fg_mask, target_gt_idx, gt_kpts, batch_idx, pred_kpts_dec, imgsz)
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()
            loss[4] += (pred_kpts * 0).sum()

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.box
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl
        loss[4] *= self.hyp.pose

        return loss.sum() * bs, loss.detach()

    def _kpt_loss(self, fg_mask, target_gt_idx, keypoints, batch_idx, pred_kpts, imgsz):
        """Smooth-L1 root-point loss normalized by image resolution."""
        total, n = 0.0, 0
        for i, (fg, gt_idx) in enumerate(zip(fg_mask, target_gt_idx)):
            if not fg.any():
                continue
            img_kpts   = keypoints[batch_idx.view(-1) == i]
            matched_gt = img_kpts[gt_idx[fg]]
            matched_pr = pred_kpts[i][fg]
            l1_loss    = F.smooth_l1_loss(matched_pr, matched_gt, reduction="none").sum(-1)
            total     += (l1_loss / imgsz[0]).mean()
            n         += 1
        return total / max(n, 1)
