"""
dataset.py — Dataset loader for YOLO-Seg-Root.

Label format (each .txt line):
    class_id  root_x  root_y  poly_x1 poly_y1 poly_x2 poly_y2 ...  (normalized 0–1)
"""

import os
import glob
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A

from augment import InstanceCopyPaste, Mosaic, MixUp, YOLOAlbumentations
from config import cfg


class YOLOSegPointDataset(Dataset):
    """Loads images and YOLO segmentation labels with root keypoints.

    Args:
        images_dir  : Path to image folder (.jpg / .png).
        labels_dir  : Path to label folder (.txt).
        img_size    : Square output resolution (default 640).
        hyp         : Augmentation hyperparameter dict.
        augment     : True for training, False for validation/inference.
        image_files : Optional explicit file list (for k-fold splits).
    """

    def __init__(self, images_dir, labels_dir, img_size=640, hyp=None,
                 augment=False, image_files=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size   = img_size
        self.hyp        = hyp or {}
        self.augment    = augment

        self.mosaic     = Mosaic(img_size=self.img_size,
                                 p=float(self.hyp.get("mosaic", 0.0)) if augment else 0.0)
        self.mixup      = MixUp(img_size=self.img_size,
                                p=float(self.hyp.get("mixup",  0.0)) if augment else 0.0)
        self.copy_paste = InstanceCopyPaste(
            p=float(self.hyp.get("copy_paste", 0.0)) if augment else 0.0,
            paste_classes=cfg.WEED_CLASSES,
        )
        self.albu = YOLOAlbumentations(img_size=self.img_size, hyp=self.hyp, augment=augment)

        if image_files is not None:
            self.image_files = sorted(image_files)
        else:
            self.image_files = sorted(
                glob.glob(os.path.join(images_dir, "*.jpg")) +
                glob.glob(os.path.join(images_dir, "*.png"))
            )
        if not self.image_files:
            raise FileNotFoundError(f"No images found in: {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def _load_image(self, idx):
        img = cv2.imread(self.image_files[idx])
        if img is None:
            raise ValueError(f"Cannot read: {self.image_files[idx]}")
        return img

    def _load_labels(self, idx, img):
        """Return list of dicts {cls, pt, poly} in pixel coordinates."""
        h, w   = img.shape[:2]
        stem   = os.path.splitext(os.path.basename(self.image_files[idx]))[0]
        lpath  = os.path.join(self.labels_dir, stem + ".txt")
        labels = []
        if not os.path.exists(lpath):
            return labels
        with open(lpath) as f:
            for line in f:
                try:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) < 9:   # min: cls(1) + root(2) + 3×poly(6)
                        continue
                    cls  = int(parts[0])
                    pt   = np.array(parts[1:3]) * [w, h]
                    poly = np.array(parts[3:]).reshape(-1, 2) * [w, h]
                    labels.append({"cls": cls, "pt": pt, "poly": poly})
                except (ValueError, IndexError):
                    pass
        return labels

    def _build_targets(self, img, labels):
        """Normalize coordinates, compute boxes from polygons, rasterize masks."""
        h, w    = img.shape[:2]
        targets = []
        for lab in labels:
            poly = np.array(lab["poly"], dtype=float)
            if poly.shape[0] < 3:
                continue
            xs, ys = poly[:, 0], poly[:, 1]
            cx = float(np.clip((xs.min() + xs.max()) / 2 / w, 0, 1))
            cy = float(np.clip((ys.min() + ys.max()) / 2 / h, 0, 1))
            bw = float(np.clip((xs.max() - xs.min()) / w,     0, 1))
            bh = float(np.clip((ys.max() - ys.min()) / h,     0, 1))
            if bw < 0.005 or bh < 0.005:
                continue

            kx = float(np.clip(lab["pt"][0] / w, 0, 1))
            ky = float(np.clip(lab["pt"][1] / h, 0, 1))

            poly_n = poly.copy()
            poly_n[:, 0] /= w
            poly_n[:, 1] /= h

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

            targets.append({
                "cls":       lab["cls"],
                "box":       [cx, cy, bw, bh],
                "mask_poly": poly_n.flatten().tolist(),
                "keypoint":  [kx, ky],
                "mask":      torch.from_numpy(mask).float(),
            })
        return targets

    def __getitem__(self, idx):
        # Mosaic or plain load
        if self.augment and random.random() < self.mosaic.p:
            img, labels = self.mosaic(dataset=self, idx=idx)
        else:
            img    = self._load_image(idx)
            labels = self._load_labels(idx, img)

        # MixUp
        if self.augment and random.random() < self.mixup.p:
            img, labels = self.mixup(dataset=self, img1=img, labs1=labels)

        # Instance Copy-Paste
        if self.augment:
            img, labels = self.copy_paste(img_dst=img, labels_dst=labels, dataset=self)

        # Albumentations (geometry + color + resize + pad)
        img, labels = self.albu(img=img, labels=labels)

        # Build normalized targets
        targets = self._build_targets(img, labels)

        # Convert to CHW float tensor
        img = np.ascontiguousarray(img[:, :, ::-1])   # BGR → RGB
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0, targets

    @staticmethod
    def collate_fn(batch):
        """Stack images; annotate targets with batch index."""
        images, targets = [], []
        for b_idx, (img, tlist) in enumerate(batch):
            images.append(img)
            for t in tlist:
                t["image_id"] = b_idx
                targets.append(t)
        return torch.stack(images, 0), targets
