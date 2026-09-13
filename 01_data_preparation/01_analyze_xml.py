#!/usr/bin/env python3
"""
01_analyze_xml.py — Analyze CVAT XML Annotations
=================================================
Parses raw CVAT XML annotation files to inspect label distribution, annotation types
(points, polygon, polyline), image dimensions, and out-of-bounds coordinates.
"""

import os
import sys
import collections
import xml.etree.ElementTree as ET


def analyze_xml(xml_path):
    if not os.path.exists(xml_path):
        print(f"Error: File not found at {xml_path}")
        return

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return

    labels = collections.defaultdict(int)
    types = collections.defaultdict(int)
    out_of_bounds = []

    images = root.findall("image")
    print(f"\n==========================================")
    print(f" XML Annotation Analysis: {os.path.basename(xml_path)}")
    print(f"==========================================")
    print(f"Total Images: {len(images)}")

    for image in images:
        img_id = image.get("id")
        img_name = image.get("name")
        w = float(image.get("width", 1500))
        h = float(image.get("height", 1500))

        for annot in image:
            tag = annot.tag
            label = annot.get("label")
            types[tag] += 1
            labels[label] += 1

            points_str = annot.get("points")
            if points_str:
                pairs = points_str.split(";")
                for pair in pairs:
                    try:
                        x, y = map(float, pair.split(","))
                        if not (0 <= x <= w and 0 <= y <= h):
                            out_of_bounds.append(
                                f"Img {img_id} ({img_name}): {tag} '{label}' point ({x:.1f}, {y:.1f}) bounds ({w}x{h})"
                            )
                    except ValueError:
                        pass

    print("-" * 42)
    print("Annotation Types Breakdown:")
    for t, count in types.items():
        print(f"  - {t:<15}: {count}")

    print("-" * 42)
    print("Label Class Distribution:")
    for l, count in sorted(labels.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {l:<20}: {count}")

    print("-" * 42)
    if out_of_bounds:
        print(f"⚠️ Warning: Found {len(out_of_bounds)} out-of-bounds point(s):")
        for err in out_of_bounds[:5]:
            print(f"    {err}")
    else:
        print("✓ All coordinates are inside image boundaries.")
    print("==========================================\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_xml = sys.argv[1]
    else:
        target_xml = r"data/raw/xml_takeo-annotation-done/fifth.xml"
    analyze_xml(target_xml)
