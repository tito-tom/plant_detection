# 03 — Model Export

This folder exports your trained model to ONNX format and lets you test it locally before deploying to the Jetson.

## Folder Structure

```
03_model_export/
├── export_model.py           # Step 1 — Export best.pt → best.onnx
├── predict.py                # Step 2 — Run inference on a test image
├── yolo11m-seg.pt            # Pretrained backbone (required for export)
├── weights/
│   └── best.pt               # Your trained weights (place it here)
├── exported/
│   └── best.onnx             # Output after running export_model.py
├── test_image/
│   └── BMC_19.jpg            # Sample test image
├── test_images/
│   └── test_output.jpg       # Output after running predict.py
└── model/                    # Custom model architecture (do not edit)
    ├── modules.py
    ├── infer_utils.py
    └── yolov11-seg-root.yaml
```

---

## Step 1 — Export to ONNX

Open [`export_model.py`](export_model.py) and check the paths at the top:

```python
WEIGHTS_PATH = r"C:\...\03_model_export\weights\best.pt"  # path to best.pt
OUTPUT_DIR   = r"C:\...\03_model_export\exported"          # output folder
IMG_SIZE     = 640
```

Then run:

```bash
cd 03_model_export
python export_model.py
```

This creates `exported/best.onnx`.

---

## Step 2 — Test the ONNX Model

Open [`predict.py`](predict.py) and check the paths at the top:

```python
ONNX_MODEL_PATH   = r"C:\...\03_model_export\exported\best.onnx"
TEST_IMAGE_PATH   = r"C:\...\03_model_export\test_image\BMC_19.jpg"
OUTPUT_IMAGE_PATH = r"C:\...\03_model_export\test_images\test_output.jpg"
CONF_THRES        = 0.25
```

Then run:

```bash
python predict.py
```

Check `test_images/test_output.jpg` — you should see:
- **Green boxes** = crop (small / large leaf)
- **Red boxes** = weed (small / large leaf)
- **Blue dot** = predicted root point
- **Colored overlay** = segmentation mask

---

## Step 3 — Deploy to Jetson (TensorRT)

After confirming the ONNX model works locally, convert it to a TensorRT engine on the Jetson for real-time speed:

```bash
trtexec \
  --onnx=best.onnx \
  --saveEngine=best_fp16.engine \
  --fp16 \
  --workspace=4096
```
