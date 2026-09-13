"""
augment.py — Augmentation pipeline for YOLO-Seg-Root.

Provides Mosaic, MixUp, Instance Copy-Paste, and Albumentations wrappers.
All transforms synchronize bounding boxes, segmentation polygons, and root keypoints.
"""

import random
import cv2
import numpy as np
import albumentations as A


class InstanceCopyPaste:
    """Copy-Paste augmentation that pastes instances of target classes onto the
    destination image with full polygon + keypoint synchronization.

    Args:
        p            : Probability of triggering per image.
        paste_classes: Class IDs to sample (default [2, 3] = weed classes).
        scale_range  : Min/max scale multiplier for pasted instances.
        max_paste    : Maximum instances to paste per trigger.
    """

    def __init__(self, p=0.3, paste_classes=None, scale_range=(0.8, 1.2), max_paste=3):
        self.p = p
        self.paste_classes = paste_classes if paste_classes is not None else [2, 3]
        self.scale_range   = scale_range
        self.max_paste     = max_paste

    def __call__(self, img_dst, labels_dst, dataset=None):
        """Apply copy-paste augmentation.

        Args:
            img_dst    : Destination image (H, W, 3).
            labels_dst : List of label dicts {cls, pt, poly}.
            dataset    : Source dataset to sample from.

        Returns:
            (img_dst, labels_dst) with new instances pasted in.
        """
        if random.random() > self.p or dataset is None:
            return img_dst, labels_dst

        src_idx    = random.randint(0, len(dataset) - 1)
        img_src    = dataset._load_image(src_idx)
        labels_src = dataset._load_labels(src_idx, img_src)

        instances = [lab for lab in labels_src if lab["cls"] in self.paste_classes]
        if not instances:
            return img_dst, labels_dst

        n_paste  = min(random.randint(1, self.max_paste), len(instances))
        selected = random.sample(instances, n_paste)
        for inst in selected:
            img_dst, labels_dst = self._paste_one(img_dst, labels_dst, img_src, inst)
        return img_dst, labels_dst

    def _paste_one(self, img_dst, labels_dst, img_src, inst):
        """Paste a single instance using the affine transform: X_dst = scale*(X_src - C_src) + C_dst."""
        poly_src = inst["poly"].copy()
        root_src = inst["pt"].copy()

        xs, ys = poly_src[:, 0], poly_src[:, 1]
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        y_min, y_max = float(np.min(ys)), float(np.max(ys))
        src_w, src_h = x_max - x_min, y_max - y_min
        if src_w < 2 or src_h < 2:
            return img_dst, labels_dst

        cx_src, cy_src = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0

        scale          = random.uniform(*self.scale_range)
        dst_w, dst_h   = src_w * scale, src_h * scale
        h_dst, w_dst   = img_dst.shape[:2]
        margin_x, margin_y = dst_w / 2.0, dst_h / 2.0

        if w_dst <= 2 * margin_x or h_dst <= 2 * margin_y:
            return img_dst, labels_dst

        cx_dst = random.uniform(margin_x, w_dst - margin_x)
        cy_dst = random.uniform(margin_y, h_dst - margin_y)

        h_src, w_src = img_src.shape[:2]
        mask_src = np.zeros((h_src, w_src), dtype=np.uint8)
        cv2.fillPoly(mask_src, [poly_src.astype(np.int32)], 255)

        tx = cx_dst - scale * cx_src
        ty = cy_dst - scale * cy_src
        M  = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)

        warped_img  = cv2.warpAffine(img_src,  M, (w_dst, h_dst),
                                     flags=cv2.INTER_LINEAR,  borderMode=cv2.BORDER_CONSTANT)
        warped_mask = cv2.warpAffine(mask_src, M, (w_dst, h_dst),
                                     flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        bool_mask = warped_mask > 0
        img_dst[bool_mask] = warped_img[bool_mask]

        c_src   = np.array([cx_src, cy_src])
        c_dst   = np.array([cx_dst, cy_dst])
        poly_dst = scale * (poly_src - c_src) + c_dst
        root_dst = scale * (root_src - c_src) + c_dst

        labels_dst.append({"cls": inst["cls"], "pt": root_dst, "poly": poly_dst})
        return img_dst, labels_dst


class Mosaic:
    """Stitches 4 random images into a 2×2 grid and crops a central (img_size × img_size) window."""

    def __init__(self, img_size=640, p=1.0):
        self.img_size = img_size
        self.p        = p

    def __call__(self, dataset, idx):
        s  = self.img_size
        cs = s * 2
        cx = random.randint(s // 2, 3 * s // 2)
        cy = random.randint(s // 2, 3 * s // 2)

        canvas = np.full((cs, cs, 3), 114, dtype=np.uint8)
        quads  = [(cx-s, cy-s, cx, cy), (cx, cy-s, cx+s, cy),
                  (cx-s, cy,   cx, cy+s), (cx, cy,   cx+s, cy+s)]

        indices    = [idx] + random.choices(range(len(dataset.image_files)), k=3)
        all_labels = []

        for i, (x1, y1, x2, y2) in zip(indices, quads):
            img_i  = dataset._load_image(i)
            labs_i = dataset._load_labels(i, img_i)
            ih, iw = img_i.shape[:2]
            qw, qh = x2 - x1, y2 - y1
            img_i  = cv2.resize(img_i, (qw, qh))
            sx, sy = qw / iw, qh / ih

            px1, py1 = max(x1, 0), max(y1, 0)
            px2, py2 = min(x2, cs), min(y2, cs)
            lx1, ly1 = px1 - x1, py1 - y1
            canvas[py1:py2, px1:px2] = img_i[ly1:ly1+(py2-py1), lx1:lx1+(px2-px1)]

            for lab in labs_i:
                pt, poly = lab["pt"].copy(), lab["poly"].copy()
                pt[0] = pt[0] * sx + x1
                pt[1] = pt[1] * sy + y1
                poly[:, 0] = poly[:, 0] * sx + x1
                poly[:, 1] = poly[:, 1] * sy + y1
                all_labels.append({"cls": lab["cls"], "pt": pt, "poly": poly})

        cx1, cy1 = max(cx - s // 2, 0), max(cy - s // 2, 0)
        cx2, cy2 = min(cx1 + s, cs), min(cy1 + s, cs)
        img_m    = canvas[cy1:cy2, cx1:cx2]

        for lab in all_labels:
            lab["pt"][0]     -= cx1
            lab["pt"][1]     -= cy1
            lab["poly"][:, 0] -= cx1
            lab["poly"][:, 1] -= cy1

        aw, ah = cx2 - cx1, cy2 - cy1
        img_m  = cv2.resize(img_m, (s, s))
        rx, ry = s / aw, s / ah
        for lab in all_labels:
            lab["pt"][0]     *= rx
            lab["pt"][1]     *= ry
            lab["poly"][:, 0] *= rx
            lab["poly"][:, 1] *= ry

        return img_m, all_labels


class MixUp:
    """Blends two images at 50% alpha and merges their label lists."""

    def __init__(self, img_size=640, p=0.0):
        self.img_size = img_size
        self.p        = p

    def __call__(self, dataset, img1, labs1):
        idx2 = random.randint(0, len(dataset.image_files) - 1)
        if hasattr(dataset, "mosaic") and random.random() < dataset.mosaic.p:
            img2, labs2 = dataset.mosaic(dataset, idx2)
        else:
            img2       = dataset._load_image(idx2)
            labs2      = dataset._load_labels(idx2, img2)
            oh, ow     = img2.shape[:2]
            img2       = cv2.resize(img2, (self.img_size, self.img_size))
            rx, ry     = self.img_size / ow, self.img_size / oh
            for lab in labs2:
                lab["pt"][0]     *= rx
                lab["pt"][1]     *= ry
                lab["poly"][:, 0] *= rx
                lab["poly"][:, 1] *= ry

        if img1.shape[:2] != (self.img_size, self.img_size):
            oh, ow = img1.shape[:2]
            img1   = cv2.resize(img1, (self.img_size, self.img_size))
            rx, ry = self.img_size / ow, self.img_size / oh
            for lab in labs1:
                lab["pt"][0]     *= rx
                lab["pt"][1]     *= ry
                lab["poly"][:, 0] *= rx
                lab["poly"][:, 1] *= ry

        blended = cv2.addWeighted(
            img1.astype(np.float32), 0.5,
            img2.astype(np.float32), 0.5, 0.0,
        ).astype(np.uint8)
        return blended, labs1 + labs2


class YOLOAlbumentations:
    """Albumentations pipeline: geometry, color, noise, resize, and letterbox padding.

    Synchronizes all keypoints (root point + polygon vertices) through every transform.
    """

    def __init__(self, img_size=640, hyp=None, augment=False):
        self.img_size  = img_size
        self.hyp       = hyp or {}
        self.augment   = augment
        self.transform = self._build_transform()

    def _build_transform(self):
        resize = [
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size,
                          border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114)),
        ]
        kp_params = A.KeypointParams(format="xy", remove_invisible=False)

        if not self.augment:
            return A.Compose(resize, keypoint_params=kp_params)

        h   = self.hyp
        aug = [
            A.HorizontalFlip(p=h.get("fliplr", 0.5)),
            A.VerticalFlip(p=h.get("flipud", 0.0)),
            A.Affine(
                translate_percent=h.get("translate", 0.1),
                scale=(1 - h.get("scale", 0.3), 1 + h.get("scale", 0.3)),
                rotate=h.get("degrees", 0.0),
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
                p=0.5,
            ),
            A.Perspective(scale=(0.02, 0.08), border_mode=cv2.BORDER_CONSTANT,
                          fill=(114, 114, 114), p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=int(h.get("hsv_h", 0.015) * 180),
                sat_shift_limit=int(h.get("hsv_s", 0.7)   * 255),
                val_shift_limit=int(h.get("hsv_v", 0.4)   * 255),
                p=0.5,
            ),
            A.CLAHE(p=0.2),
            A.RandomShadow(p=0.1),
            A.GaussianBlur(p=0.2),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, 32),
                            hole_width_range=(1, 32), fill=114, p=0.3),
        ]
        return A.Compose(aug + resize, keypoint_params=kp_params)

    def __call__(self, img, labels):
        kpts   = []
        for lab in labels:
            kpts.append((float(lab["pt"][0]), float(lab["pt"][1])))
            for pt in lab["poly"]:
                kpts.append((float(pt[0]), float(pt[1])))

        out   = self.transform(image=img, keypoints=kpts)
        img   = out["image"]
        tkpts = out["keypoints"]

        i = 0
        for lab in labels:
            if i < len(tkpts):
                lab["pt"] = np.array(tkpts[i], dtype=float)
                i += 1
            n = len(lab["poly"])
            if i + n <= len(tkpts):
                lab["poly"] = np.array(tkpts[i:i+n], dtype=float).reshape(-1, 2)
                i += n
            else:
                lab["poly"] = np.zeros((0, 2), dtype=float)
        return img, labels
