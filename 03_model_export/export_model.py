"""
export_model.py  —  Export trained YOLO-Seg-Root model to ONNX.

Run this on your training machine (Windows) before copying to Jetson.

Usage:
    python export_model.py
"""

import os
import sys

# ============================================================
# CONFIGURATION — edit these paths before running
# ============================================================

# Path to your trained weights (best.pt)
WEIGHTS_PATH = r"C:\Users\TITO TOM\Downloads\plant_detection\03_model_export\weights\best.pt"

# Where to save the exported .onnx file
OUTPUT_DIR = r"C:\Users\TITO TOM\Downloads\plant_detection\03_model_export\exported"

# Input image size (must match what you trained with, default 640)
IMG_SIZE = 640

# ============================================================

# Ensure the model directory is in path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from model.infer_utils import build_model


class ONNXExporter:
    """Export YOLO-Seg-Root model to ONNX format."""

    def __init__(self, weights_path: str, output_dir: str, img_size: int = 640):
        self.weights_path = os.path.abspath(weights_path)
        self.output_dir = output_dir
        self.img_size = img_size

    def validate_paths(self) -> bool:
        """Check input file exists and create output directory."""
        os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(self.weights_path):
            print(f"[Error] Weights file not found: {self.weights_path}")
            return False
        return True

    def export_model(self) -> str:
        """Run the ONNX export."""
        print("=" * 60)
        print("  YOLO-Seg-Root Model Export (ONNX)")
        print(f"  Weights : {self.weights_path}")
        print(f"  Output  : {self.output_dir}")
        print(f"  Size    : {self.img_size}")
        print("=" * 60)

        model = build_model(weights_path=self.weights_path, device="cpu")

        print("\n[1/1] Exporting to ONNX...")
        onnx_path = model.export(
            format="onnx",
            imgsz=self.img_size,
            opset=13,
            simplify=True,
            dynamic=False,  # Fixed size is required for optimal TensorRT speed
        )

        # ultralytics saves next to the YAML (e.g. "yolov11m-seg-root.onnx" in the
        # current directory) since the model was built from a YAML, not a .pt file —
        # move it into OUTPUT_DIR/best.onnx to match this repo's documented layout.
        final_path = os.path.join(self.output_dir, "best.onnx")
        if os.path.abspath(onnx_path) != os.path.abspath(final_path):
            os.replace(onnx_path, final_path)
        return final_path

    def print_post_export_instructions(self, onnx_path: str):
        """Show next steps for Jetson deployment."""
        print("\n" + "=" * 60)
        print("  TensorRT Conversion (run ON THE JETSON):")
        print("=" * 60)
        print(f"""
  trtexec \\
    --onnx={os.path.basename(onnx_path)} \\
    --saveEngine=yolo_seg_root_fp16.engine \\
    --fp16 \\
    --workspace=4096
""")
        print("✓ Export complete!")
        print(f"  Copy the .onnx file to your Jetson.")

    def run(self):
        """Run the full export pipeline."""
        if not self.validate_paths():
            return

        onnx_path = self.export_model()
        if onnx_path:
            self.print_post_export_instructions(onnx_path)


if __name__ == "__main__":
    exporter = ONNXExporter(
        weights_path=WEIGHTS_PATH,
        output_dir=OUTPUT_DIR,
        img_size=IMG_SIZE,
    )
    exporter.run()
