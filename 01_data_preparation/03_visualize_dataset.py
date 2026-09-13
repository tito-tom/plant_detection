#!/usr/bin/env python3
"""
03_visualize_dataset.py — Preview YOLO Seg+Root Dataset Labels
================================================================
Draws bounding boxes, segmentation masks, class labels, and root points
on images from the converted YOLO dataset for verification.
"""

import os
import glob
import cv2
import numpy as np
import yaml

COLORS = [
    (0, 255, 0),    # Class 0: crop_small_leaf (Green)
    (255, 165, 0),  # Class 1: crop_large_leaf (Orange)
    (0, 0, 255),    # Class 2: weed_small_leaf (Red)
    (255, 0, 255),  # Class 3: weed_large_leaf (Magenta)
]


def visualize_yolo_labels(dataset_dir, output_dir="output/dataset_visualizations", num_samples=10):
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            data_cfg = yaml.safe_load(f)
            class_names = data_cfg.get("names", {})
    else:
        class_names = {0: "crop_small", 1: "crop_large", 2: "weed_small", 3: "weed_large"}

    subsets = ["train", "val"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Visualizing images from {dataset_dir} ...")

    for subset in subsets:
        img_dir = os.path.join(dataset_dir, "images", subset)
        lbl_dir = os.path.join(dataset_dir, "labels", subset)

        if not os.path.exists(img_dir):
            continue

        images = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
        for img_path in images[:num_samples]:
            img_name = os.path.basename(img_path)
            txt_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")

            img = cv2.imread(img_path)
            if img is None or not os.path.exists(txt_path):
                continue

            h, w = img.shape[:2]
            overlay = img.copy()

            with open(txt_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue

                cls_id = int(parts[0])
                rx_norm, ry_norm = float(parts[1]), float(parts[2])
                poly_coords = [float(p) for p in parts[3:]]

                color = COLORS[cls_id % len(COLORS)]
                rx, ry = int(rx_norm * w), int(ry_norm * h)

                # Draw Polygon Mask
                pts = np.array(poly_coords).reshape(-1, 2)
                pts[:, 0] *= w
                pts[:, 1] *= h
                pts_int = pts.astype(np.int32)

                cv2.fillPoly(overlay, [pts_int], color)
                cv2.polylines(img, [pts_int], isClosed=True, color=color, thickness=2)

                # Draw Root Keypoint
                cv2.circle(img, (rx, ry), 6, (255, 255, 255), -1)
                cv2.circle(img, (rx, ry), 4, (0, 0, 0), -1)
                cv2.circle(img, (rx, ry), 2, (0, 255, 255), -1)

                # Draw Class Tag
                cls_name = class_names.get(cls_id, f"cls_{cls_id}")
                cv2.putText(img, cls_name, (pts_int[0][0], max(15, pts_int[0][1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Blend mask overlay
            blended = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
            save_path = os.path.join(output_dir, f"{subset}_{img_name}")
            cv2.imwrite(save_path, blended)

    print(f"✅ Saved dataset preview visualizations to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/yolo_dataset_4classes")
    parser.add_argument("--output", default="output/dataset_previews")
    parser = argparse.ArgumentParser(
        description="Visualize YOLO segmentation + root-point labels on dataset images."
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/yolo_dataset_4classes",
        default="02_model_training/data",
        help="Path to the YOLO dataset directory (must contain images/ and data.yaml). Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        default="output/dataset_previews",
        help="Directory to save annotated preview images. Default: %(default)s",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Max number of images to preview per split (train/val). Default: %(default)s",
    )
    args = parser.parse_args()

    visualize_yolo_labels(args.dataset, args.output)
    visualize_yolo_labels(args.dataset, args.output, num_samples=args.num_samples)
