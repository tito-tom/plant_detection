"""
val.py — Standalone validator for YOLO-Seg-Root.

Runs a full evaluation loop computing multi-task loss, Box mAP50/50-95,
Mask mAP50/50-95, and PCK@5/10/20 — both aggregate and per class.

Can be called from a trainer during training, or run standalone:
    python val.py --weights path/to/best.pt

All dataset paths and thresholds are configured in config.py.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh, process_mask
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.tal import make_anchors

from utils import build_model, prepare_batch, calculate_pck, calculate_abspck, CLASS_NAMES, LOSS_NAMES
from metrics import CustomSegmentMetrics
from dataset import YOLOSegPointDataset
from loss import CustomLoss

import config as cfg

DEFAULT_WEIGHTS = cfg.cfg.WEIGHT_PATH


class CustomValidator(SegmentationValidator):
    """Standalone validator for YOLO-Seg-Root.

    Runs a full validation loop computing loss, Box/Mask mAP, and PCK
    for bounding boxes, segmentation masks, and root keypoints.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        if args is None:
            args = {}
        if isinstance(args, dict):
            args.setdefault("conf",   cfg.cfg.CONF_THRES)
            args.setdefault("iou",    cfg.cfg.IOU_THRES)
            args.setdefault("imgsz",  cfg.cfg.IMG_SIZE)
            args.setdefault("batch",  cfg.cfg.BATCH_SIZE)
            args.setdefault("device", cfg.cfg.DEVICE)
            args.setdefault("split",  "val")
        else:
            if getattr(args, "conf",   None) is None: args.conf   = cfg.cfg.CONF_THRES
            if getattr(args, "iou",    None) is None: args.iou    = cfg.cfg.IOU_THRES
            if getattr(args, "imgsz",  None) is None: args.imgsz  = cfg.cfg.IMG_SIZE
            if getattr(args, "batch",  None) is None: args.batch  = cfg.cfg.BATCH_SIZE
            if getattr(args, "device", None) is None: args.device = cfg.cfg.DEVICE
            if getattr(args, "split",  None) is None: args.split  = "val"
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self.nc        = len(CLASS_NAMES)
        self.criterion = None

    def build_dataset(self, img_path, mode="val", batch=None):
        return YOLOSegPointDataset(cfg.cfg.VAL_IMAGES, cfg.cfg.VAL_LABELS,
                                   img_size=self.args.imgsz, augment=False)

    def get_dataloader(self, dataset_path, batch_size):
        ds = self.build_dataset(dataset_path)
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          collate_fn=YOLOSegPointDataset.collate_fn)

    def init_metrics(self, model):
        self.map_eval = CustomSegmentMetrics(nc=self.nc, class_names=CLASS_NAMES)
        self.all_pred_kpts  = []
        self.all_gt_kpts    = []
        self.all_gt_bboxes  = []
        self.per_class_gt_total   = {i: 0 for i in CLASS_NAMES}
        self.per_class_counts     = {i: 0 for i in CLASS_NAMES}
        self.per_class_pred_kpts  = {i: [] for i in CLASS_NAMES}
        self.per_class_gt_kpts    = {i: [] for i in CLASS_NAMES}
        self.per_class_gt_bboxes  = {i: [] for i in CLASS_NAMES}
        self.n_batches = 0
        self.criterion = CustomLoss(model)

    def preprocess(self, batch):
        if isinstance(batch, tuple) and len(batch) == 2:
            imgs, targets = batch
            prep = prepare_batch(targets, self.device)
            if prep is not None:
                prep["img"] = imgs.to(self.device)
            return prep
        return batch

    def update_metrics(self, preds, batch):
        pass

    def __call__(self, trainer=None, model=None):
        """Run the full validation loop.

        Can be called from a trainer (training mode) or standalone (inference mode).

        Returns:
            (avg_loss, avg_items, map_res, pck, per_class_counts)
        """
        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            model       = trainer.model
        else:
            if model is None:
                model = build_model(weights_path=self.args.model, device=self.args.device)
            self.device = getattr(model, "device", self.args.device)

        if hasattr(model, "model"):
            model = model.model

        model.float()
        model.eval()
        head = model.model[-1]

        self.dataloader = self.dataloader or self.get_dataloader(None, self.args.batch)
        self.init_metrics(model)

        total_loss = 0.0
        loss_sums  = torch.zeros(5)
        head.training = True   # keep raw dict output for loss computation

        from tqdm import tqdm
        for imgs, targets in tqdm(self.dataloader, desc="Evaluating"):
            batch = prepare_batch(targets, self.device)
            if batch is None:
                continue

            with torch.no_grad():
                preds_out = model(imgs.to(self.device))

                feats          = preds_out["feats"]
                pred_masks_raw = preds_out["mask_coefficient"]
                proto          = preds_out["proto"].clone()
                pred_kpts_raw  = preds_out["kpts"]
                bs             = proto.shape[0]

                loss, items = self.criterion(preds_out, batch)
                total_loss += loss.item()
                loss_sums  += items.cpu()
                self.n_batches += 1

                pred_kpts_perm = pred_kpts_raw.permute(0, 2, 1).contiguous()
                anchor_points, stride_tensor = make_anchors(feats, head.stride, 0.5)
                pred_kpts_dec  = pred_kpts_perm.clone()
                pred_kpts_dec[..., 0] = (pred_kpts_dec[..., 0] * 2.0 + (anchor_points[:, 0] - 0.5)) * stride_tensor[:, 0]
                pred_kpts_dec[..., 1] = (pred_kpts_dec[..., 1] * 2.0 + (anchor_points[:, 1] - 0.5)) * stride_tensor[:, 0]

                pred_distri = preds_out["boxes"]
                pred_scores = preds_out["scores"]
                pred_distri_p = pred_distri.permute(0, 2, 1).contiguous()
                pred_scores_p = pred_scores.permute(0, 2, 1).contiguous()

                dtype  = pred_scores_p.dtype
                imgsz  = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * head.stride[0]

                pred_bboxes        = self.criterion.bbox_decode(anchor_points, pred_distri_p)
                pred_bboxes_scaled = pred_bboxes * stride_tensor
                pred_bboxes_xywh   = xyxy2xywh(pred_bboxes_scaled)

                nms_input = torch.cat([
                    pred_bboxes_xywh.permute(0, 2, 1),
                    pred_scores.sigmoid(),
                    pred_masks_raw,
                    pred_kpts_dec.permute(0, 2, 1),
                ], dim=1)
                nms_preds = non_max_suppression(nms_input,
                                                conf_thres=self.args.conf,
                                                iou_thres=self.args.iou,
                                                nc=head.nc)

                batch_idx_t   = batch["batch_idx"].view(-1, 1)
                gt_bboxes_abs = batch["bboxes"].clone()
                gt_bboxes_abs[:, 0] *= imgsz[1]; gt_bboxes_abs[:, 1] *= imgsz[0]
                gt_bboxes_abs[:, 2] *= imgsz[1]; gt_bboxes_abs[:, 3] *= imgsz[0]
                gt_bboxes_xyxy = xywh2xyxy(gt_bboxes_abs)

                for i in range(bs):
                    img_gt_mask  = batch_idx_t.view(-1) == i
                    img_gt_boxes = gt_bboxes_xyxy[img_gt_mask]
                    img_gt_cls   = batch["cls"][img_gt_mask]
                    img_gt_masks = batch["masks"][img_gt_mask]
                    det          = nms_preds[i]
                    ih, iw       = int(imgsz[0]), int(imgsz[1])

                    if len(det) > 0:
                        p_boxes       = det[:, :4]
                        p_scores_det  = det[:, 4]
                        p_cls         = det[:, 5].int()
                        p_mask_coeffs = det[:, 6: 6 + head.nm]
                        p_masks       = process_mask(proto[i], p_mask_coeffs, p_boxes, (ih, iw), upsample=True)
                        p_masks       = (p_masks > 0.5).float()
                        self.map_eval.add_batch(p_boxes, p_masks, p_scores_det, p_cls,
                                                img_gt_boxes, img_gt_masks, img_gt_cls.int(),
                                                img_idx=(self.n_batches * bs + i))
                    else:
                        self.map_eval.add_batch(
                            torch.zeros(0, 4), torch.zeros(0, ih, iw),
                            torch.zeros(0), torch.zeros(0, dtype=torch.int),
                            img_gt_boxes, img_gt_masks, img_gt_cls.int(),
                            img_idx=(self.n_batches * bs + i),
                        )

                    for gt_cls_id in img_gt_cls.cpu().int().tolist():
                        if gt_cls_id in self.per_class_gt_total:
                            self.per_class_gt_total[gt_cls_id] += 1

                # PCK matching
                gt_targets   = torch.cat((batch_idx_t, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
                gt_targets   = self.criterion.preprocess(gt_targets, bs, scale_tensor=imgsz[[1, 0, 1, 0]])
                gt_labels_t, gt_bboxes_a = gt_targets.split((1, 4), 2)
                mask_gt_t    = gt_bboxes_a.sum(2, keepdim=True).gt_(0.0)

                _ps = pred_scores_p.detach().sigmoid()
                _pb = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes_a.dtype)
                if not torch.isfinite(_ps).all() or not torch.isfinite(_pb).all():
                    _ps = torch.nan_to_num(_ps, nan=0.0, posinf=0.0, neginf=0.0)
                    _pb = torch.nan_to_num(_pb, nan=0.0, posinf=0.0, neginf=0.0)

                _, _, _, fg_mask, target_gt_idx = self.criterion.assigner(
                    _ps, _pb,
                    anchor_points * stride_tensor, gt_labels_t, gt_bboxes_a, mask_gt_t,
                )

                gt_kpts_all = batch["keypoints"].float().clone()
                gt_kpts_all[..., 0] *= imgsz[1]
                gt_kpts_all[..., 1] *= imgsz[0]

                for i in range(bs):
                    if not fg_mask[i].any():
                        continue
                    img_mask      = batch_idx_t.view(-1) == i
                    img_kpts      = gt_kpts_all[img_mask]
                    img_cls       = batch["cls"][img_mask]
                    matched_gt    = img_kpts[target_gt_idx[i][fg_mask[i]]]
                    matched_pred  = pred_kpts_dec[i][fg_mask[i]]
                    matched_boxes = gt_bboxes_a[i][target_gt_idx[i][fg_mask[i]]]

                    self.all_pred_kpts.append(matched_pred.cpu())
                    self.all_gt_kpts.append(matched_gt.cpu())
                    self.all_gt_bboxes.append(matched_boxes.cpu())

                    matched_cls = img_cls[target_gt_idx[i][fg_mask[i]]]
                    for j, cls_id in enumerate(matched_cls.cpu().int().tolist()):
                        if cls_id in self.per_class_counts:
                            self.per_class_counts[cls_id] += 1
                            self.per_class_pred_kpts[cls_id].append(matched_pred[j:j+1].cpu())
                            self.per_class_gt_kpts[cls_id].append(matched_gt[j:j+1].cpu())
                            self.per_class_gt_bboxes[cls_id].append(matched_boxes[j:j+1].cpu())

        head.training = False

        self.avg_loss  = total_loss / max(self.n_batches, 1)
        self.avg_items = loss_sums  / max(self.n_batches, 1)

        self.pck = (
            calculate_pck(torch.cat(self.all_pred_kpts),
                          torch.cat(self.all_gt_kpts),
                          torch.cat(self.all_gt_bboxes))
            if self.all_pred_kpts else {0.05: 0.0, 0.10: 0.0, 0.20: 0.0}
        )

        self.abs_pck = (
            calculate_abspck(torch.cat(self.all_pred_kpts),
                             torch.cat(self.all_gt_kpts))
            if self.all_pred_kpts else {5: 0.0, 10: 0.0, 15: 0.0, 20: 0.0}
        )

        self.per_class_pck = {}
        self.per_class_abs_pck = {}
        for cls_id in CLASS_NAMES:
            if self.per_class_pred_kpts[cls_id]:
                self.per_class_pck[cls_id] = calculate_pck(
                    torch.cat(self.per_class_pred_kpts[cls_id]),
                    torch.cat(self.per_class_gt_kpts[cls_id]),
                    torch.cat(self.per_class_gt_bboxes[cls_id]),
                )
                self.per_class_abs_pck[cls_id] = calculate_abspck(
                    torch.cat(self.per_class_pred_kpts[cls_id]),
                    torch.cat(self.per_class_gt_kpts[cls_id]),
                )
            else:
                self.per_class_pck[cls_id] = {0.05: 0.0, 0.10: 0.0, 0.20: 0.0}
                self.per_class_abs_pck[cls_id] = {5: 0.0, 10: 0.0, 15: 0.0, 20: 0.0}

        self.map_res = self.map_eval.compute()

        if not self.training:
            self.print_results()

        return self.avg_loss, self.avg_items, self.map_res, self.pck, self.per_class_counts

    def _count_gt_instances(self):
        """Count GT instances directly from label .txt files."""
        import glob
        counts = {i: 0 for i in CLASS_NAMES}
        for fpath in glob.glob(os.path.join(cfg.cfg.VAL_LABELS, "*.txt")):
            with open(fpath) as fp:
                for line in fp:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(float(parts[0]))
                        if cls_id in counts:
                            counts[cls_id] += 1
        return counts

    def print_results(self):
        """Print formatted validation results to stdout."""
        print("\n" + "=" * 90)
        print("  LOSS BREAKDOWN")
        print("=" * 90)
        print(f"  Total Loss: {self.avg_loss:.4f}")
        for i, name in enumerate(LOSS_NAMES):
            sym = "└─" if i == len(LOSS_NAMES) - 1 else "├─"
            print(f"  {sym} {name:>3s} Loss:  {self.avg_items[i]:.4f}")

        print("\n" + "=" * 90)
        print("  DETECTION & SEGMENTATION ACCURACY (mAP)")
        print("=" * 90)

        n_images = len(self.dataloader.dataset)
        n_total  = sum(self.map_res["instances_per_class"].values())
        header   = (
            f"{'Class':>14s}{'Images':>8s}{'Instances':>10s}"
            f"  {'Box(P':>7s}{'R':>8s}{'mAP50':>8s}{'mAP50-95)':>10s}"
            f"  {'Mask(P':>7s}{'R':>8s}{'mAP50':>8s}{'mAP50-95)':>10s}"
            f"  {'PCK@5':>7s}{'PCK@10':>7s}{'PCK@20':>7s}"
        )
        print(header)
        print("-" * len(header))
        print(
            f"{'all':>14s}{n_images:>8d}{n_total:>10d}"
            f"  {self.map_res['precision']:>7.3f}{self.map_res['recall']:>8.3f}"
            f"{self.map_res['mAP50']:>8.3f}{self.map_res['mAP50_95']:>10.3f}"
            f"  {self.map_res['mask_precision']:>7.3f}{self.map_res['mask_recall']:>8.3f}"
            f"{self.map_res['mask_mAP50']:>8.3f}{self.map_res['mask_mAP50_95']:>10.3f}"
            f"  {self.pck[0.05]:>7.3f}{self.pck[0.10]:>7.3f}{self.pck[0.20]:>7.3f}"
        )
        for cls_id, name in CLASS_NAMES.items():
            cnt   = self.map_res["instances_per_class"].get(cls_id, 0)
            cp    = self.per_class_pck.get(cls_id, {0.05: 0, 0.10: 0, 0.20: 0})
            print(
                f"{name:>14s}{n_images:>8d}{cnt:>10d}"
                f"  {self.map_res['per_class_precision'].get(cls_id,0):>7.3f}"
                f"{self.map_res['per_class_recall'].get(cls_id,0):>8.3f}"
                f"{self.map_res['per_class_ap50'].get(cls_id,0):>8.3f}"
                f"{self.map_res['per_class_ap50_95'].get(cls_id,0):>10.3f}"
                f"  {self.map_res['per_class_mask_precision'].get(cls_id,0):>7.3f}"
                f"{self.map_res['per_class_mask_recall'].get(cls_id,0):>8.3f}"
                f"{self.map_res['per_class_mask_ap50'].get(cls_id,0):>8.3f}"
                f"{self.map_res['per_class_mask_ap50_95'].get(cls_id,0):>10.3f}"
                f"  {cp[0.05]:>7.3f}{cp[0.10]:>7.3f}{cp[0.20]:>7.3f}"
            )

        print("\n" + "=" * 90)
        print("  ROOT-POINT ACCURACY (Relative PCK & Absolute Pixel PCK)")
        print("=" * 90)
        root_hdr = (
            f"{'Class':>16s}{'Instances':>10s}"
            f"  {'PCK@5%':>8s}{'PCK@10%':>9s}{'PCK@20%':>9s}"
            f"  {'Abs@5px':>8s}{'Abs@10px':>9s}{'Abs@15px':>9s}{'Abs@20px':>9s}"
        )
        print(root_hdr)
        print("-" * len(root_hdr))
        print(
            f"{'all':>16s}{n_total:>10d}"
            f"  {self.pck[0.05]:>8.3f}{self.pck[0.10]:>9.3f}{self.pck[0.20]:>9.3f}"
            f"  {self.abs_pck[5]:>8.3f}{self.abs_pck[10]:>9.3f}{self.abs_pck[15]:>9.3f}{self.abs_pck[20]:>9.3f}"
        )
        for cls_id, name in CLASS_NAMES.items():
            cnt = self.map_res["instances_per_class"].get(cls_id, 0)
            cp  = self.per_class_pck.get(cls_id, {0.05: 0, 0.10: 0, 0.20: 0})
            ap  = self.per_class_abs_pck.get(cls_id, {5: 0, 10: 0, 15: 0, 20: 0})
            print(
                f"{name:>16s}{cnt:>10d}"
                f"  {cp[0.05]:>8.3f}{cp[0.10]:>9.3f}{cp[0.20]:>9.3f}"
                f"  {ap[5]:>8.3f}{ap[10]:>9.3f}{ap[15]:>9.3f}{ap[20]:>9.3f}"
            )
        print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO-Seg-Root model")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    validator = CustomValidator(args={"model": args.weights})
    validator()


if __name__ == "__main__":
    main()
