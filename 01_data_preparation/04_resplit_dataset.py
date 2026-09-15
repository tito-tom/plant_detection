#!/usr/bin/env python3
"""
04_resplit_dataset.py — Train/Val/Test Dataset Splitter
======================================================
Splits dataset into 80% Train, 5% Validation, 15% Test with customizable class mappings
(e.g., 4-class, 3-class merged weed, 2-class crop/weed).

Expected input (output of 02_convert_xml_to_yolo.py):
    source_dir/
    ├── images/
    │   ├── train/   *.jpg / *.png
    │   └── val/     *.jpg / *.png
    └── labels/
        ├── train/   *.txt
        └── val/     *.txt

Output (written to output_root/yolo_dataset_split/):
    yolo_dataset_split/
    ├── images/  train/ val/ test/
    ├── labels/  train/ val/ test/
    └── data.yaml
"""

import os
import shutil
import random
import yaml
from pathlib import Path


def resplit_dataset(source_dir, output_root, seed=42, train_ratio=0.80, val_ratio=0.05, test_ratio=0.15):
    """Gather all image-label pairs from source_dir (including any sub-splits),
    shuffle them, and write clean train/val/test splits to output_root.

    Args:
        source_dir   : Root of the converted YOLO dataset (output of step 2).
        output_root  : Parent directory to write 'yolo_dataset_split/' into.
        seed         : Random seed for reproducibility (default 42).
        train_ratio  : Fraction of data for training (default 0.80).
        val_ratio    : Fraction of data for validation (default 0.05).
        test_ratio   : Fraction of data for testing (default 0.15).
                       Remaining data after train+val goes to test automatically.
    """
    source_dir = Path(source_dir)
    output_root = Path(output_root)

    img_root = source_dir / "images"
    lbl_root = source_dir / "labels"

    # Collect images from ALL subdirectories (handles both flat and nested layouts)
    all_images = sorted(
        list(img_root.rglob("*.jpg")) + list(img_root.rglob("*.png"))
    )

    if not all_images:
        print(f"[Warning] No images found under: {img_root}")
        print("    Make sure you ran 02_convert_xml_to_yolo.py first, or check --src path.")
        return

    pairs = []
    for img_path in all_images:
        # Mirror the images/ → labels/ path to find the matching .txt label
        rel = img_path.relative_to(img_root)
        lbl_path = lbl_root / rel.with_suffix(".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            print(f"  ⚠️  No label found for {img_path.name} — skipping.")

    if not pairs:
        print("⚠️  No valid image-label pairs found. Aborting.")
        return

    random.seed(seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }

    out_dataset = output_root / "yolo_dataset_split"
    for split_name, split_pairs in splits.items():
        dst_img_dir = out_dataset / "images" / split_name
        dst_lbl_dir = out_dataset / "labels" / split_name
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_p, lbl_p in split_pairs:
            shutil.copy2(img_p, dst_img_dir / img_p.name)
            shutil.copy2(lbl_p, dst_lbl_dir / lbl_p.name)

    # Generate data.yaml (includes nc so utils.py can load it without error)
    yaml_data = {
        "nc":    4,
        "path":  os.path.abspath(out_dataset),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "names": {
            0: "crop_small_leaf",
            1: "crop_large_leaf",
            2: "weed_small_leaf",
            3: "weed_large_leaf",
        },
    }
    with open(out_dataset / "data.yaml", "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"[OK] Successfully created split at {out_dataset}:")
    print(f"   - Train : {len(splits['train'])} samples")
    print(f"   - Val   : {len(splits['val'])} samples")
    print(f"   - Test  : {len(splits['test'])} samples")
    print()
    print("   Next step: rename / copy the output folder so it matches the")
    print("   DATA_DIR in 02_model_training/config.py  (default: data/exp_4class)")
    print(f"   e.g.:  mv {out_dataset}  data/exp_4class")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Re-split a YOLO dataset into Train / Val / Test."
    )
    parser.add_argument(
        "--src",
        default="data/processed/yolo_dataset_4classes",
        help="Source YOLO dataset directory (output of step 2). Default: %(default)s",
    )
    parser.add_argument(
        "--out",
        default="data/processed",
        help="Output parent directory. Split written to <out>/yolo_dataset_split/. Default: %(default)s",
    )
    args = parser.parse_args()
    resplit_dataset(args.src, args.out)

