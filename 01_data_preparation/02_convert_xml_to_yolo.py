#!/usr/bin/env python3
"""
02_convert_xml_to_yolo.py — Convert CVAT XML Annotations to YOLO Seg+Root Format
=================================================================================
Converts CVAT XML export (polygons & keypoint root points) into standard YOLO formatted
label files containing:
  class_id  root_x  root_y  x1 y1 x2 y2 ... xN yN

Classes (4 target classes):
  0: crop_small_leaf
  1: crop_large_leaf
  2: weed_small_leaf
  3: weed_large_leaf
"""

import os
import xml.etree.ElementTree as ET
import collections
import cv2
import numpy as np
import shutil
import random
import yaml
from pathlib import Path

# Taxonomy Mapping from CVAT raw labels to 4 unified target classes
CLASS_MAPPING = {
    "crop6": "crop_small_leaf",
    "crop1": "crop_large_leaf",
    "crop2": "crop_large_leaf",
    "crop3": "crop_large_leaf",
    "crop4": "crop_large_leaf",
    "crop5": "crop_large_leaf",
    "weed1": "weed_small_leaf",
    "weed2": "weed_small_leaf",
    "weed3": "weed_small_leaf",
    "weed4": "weed_small_leaf",
    "weed5": "weed_large_leaf",
    "weed6": "weed_large_leaf",
    "weed7": "weed_large_leaf",
    "weed8": "weed_large_leaf",
    "weed9": "weed_large_leaf",
    "weed10": "weed_large_leaf",
    "weed11": "weed_large_leaf",
    "weed12": "weed_large_leaf",
    "weed13": "weed_large_leaf",
    "weed14": "weed_large_leaf",
}

TARGET_CLASSES = [
    "crop_small_leaf",
    "crop_large_leaf",
    "weed_small_leaf",
    "weed_large_leaf",
]
TARGET_CLASS_MAP = {name: i for i, name in enumerate(TARGET_CLASSES)}


def normalize_coords(coords, width, height):
    """Normalize (x, y) coordinates to [0, 1] relative to width and height."""
    normalized = []
    for x, y in coords:
        x_norm = max(0.0, min(float(x) / width, 1.0))
        y_norm = max(0.0, min(float(y) / height, 1.0))
        normalized.extend([x_norm, y_norm])
    return normalized


def point_inside_polygon(pt, poly_pts):
    """Check if point (x,y) is strictly inside or on edge of polygon."""
    contours = np.array(poly_pts, dtype=np.float32)
    return cv2.pointPolygonTest(contours, (float(pt[0]), float(pt[1])), False) >= 0


def convert_xml_to_yolo(xml_file, image_dir, output_dir, val_split=0.15):
    """Parse CVAT XML and convert to YOLO segmentation + root point dataset."""
    print(f"Parsing XML file: {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    images_dir_train = Path(output_dir) / "images" / "train"
    labels_dir_train = Path(output_dir) / "labels" / "train"
    images_dir_val = Path(output_dir) / "images" / "val"
    labels_dir_val = Path(output_dir) / "labels" / "val"

    for d in [images_dir_train, labels_dir_train, images_dir_val, labels_dir_val]:
        d.mkdir(parents=True, exist_ok=True)

    image_nodes = root.findall("image")
    random.seed(42)
    random.shuffle(image_nodes)

    split_idx = int(len(image_nodes) * (1 - val_split))
    train_nodes = image_nodes[:split_idx]
    val_nodes = image_nodes[split_idx:]

    processed_count = 0

    for nodes, split_name in [(train_nodes, "train"), (val_nodes, "val")]:
        img_out_dir = Path(output_dir) / "images" / split_name
        lbl_out_dir = Path(output_dir) / "labels" / split_name

        for img_node in nodes:
            img_name = img_node.get("name")
            width = float(img_node.get("width", 1500))
            height = float(img_node.get("height", 1500))

            src_img_path = Path(image_dir) / img_name
            if not src_img_path.exists():
                continue

            points_list = []
            polygons_list = []

            for annot in img_node:
                raw_label = annot.get("label")
                if raw_label not in CLASS_MAPPING:
                    continue
                target_label = CLASS_MAPPING[raw_label]
                class_id = TARGET_CLASS_MAP[target_label]

                points_str = annot.get("points")
                if not points_str:
                    continue

                pairs = [
                    tuple(map(float, p.split(",")))
                    for p in points_str.split(";")
                    if "," in p
                ]

                if annot.tag == "points":
                    for pt in pairs:
                        points_list.append(
                            {"class_id": class_id, "pt": pt, "used": False}
                        )
                elif annot.tag in ["polygon", "polyline"]:
                    polygons_list.append(
                        {"class_id": class_id, "pts": pairs, "matched_pt": None}
                    )

            # Match keypoint root points to enclosing instance polygons
            label_lines = []
            for poly in polygons_list:
                cid = poly["class_id"]
                pts = poly["pts"]

                # Find candidate keypoint inside polygon
                matched_root = None
                for pt_info in points_list:
                    if not pt_info["used"] and pt_info["class_id"] == cid:
                        if point_inside_polygon(pt_info["pt"], pts):
                            matched_root = pt_info["pt"]
                            pt_info["used"] = True
                            break

                # If no explicit point inside, calculate polygon centroid as fallback root
                if matched_root is None:
                    pts_np = np.array(pts, dtype=np.float32)
                    matched_root = (float(np.mean(pts_np[:, 0])), float(np.mean(pts_np[:, 1])))

                rx_norm = max(0.0, min(matched_root[0] / width, 1.0))
                ry_norm = max(0.0, min(matched_root[1] / height, 1.0))

                poly_norm = normalize_coords(pts, width, height)
                poly_str = " ".join(f"{v:.6f}" for v in poly_norm)

                label_lines.append(f"{cid} {rx_norm:.6f} {ry_norm:.6f} {poly_str}")

            if label_lines:
                # Copy image and write label txt file
                dst_img_path = img_out_dir / img_name
                shutil.copy2(src_img_path, dst_img_path)

                lbl_name = Path(img_name).stem + ".txt"
                with open(lbl_out_dir / lbl_name, "w") as f:
                    f.write("\n".join(label_lines) + "\n")

                processed_count += 1

    # Write data.yaml dataset descriptor
    yaml_content = {
        "nc":    4,
        "path":  os.path.abspath(output_dir),
        "train": "images/train",
        "val":   "images/val",
        "names": {i: name for i, name in enumerate(TARGET_CLASSES)},
    }
    with open(Path(output_dir) / "data.yaml", "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"✅ Dataset converted successfully: {processed_count} images saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert CVAT XML to YOLO dataset")
    parser.add_argument("--xml", default="data/raw/xml_takeo-annotation-done/fifth.xml")
    parser.add_argument("--img-dir", default="data/raw/dataset")
    parser.add_argument("--out-dir", default="data/processed/yolo_dataset_4classes")
    args = parser.parse_args()

    convert_xml_to_yolo(args.xml, args.img_dir, args.out_dir)
