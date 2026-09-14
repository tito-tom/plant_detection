import os
import cv2
import time
import numpy as np
import torch
import onnxruntime as ort
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import process_mask

# ============================================================
# CONFIGURATION — edit these paths before running
# ============================================================

# Path to your exported ONNX model
ONNX_MODEL_PATH = r"C:\Users\TITO TOM\Downloads\plant_detection\03_model_export\exported\best.onnx"

# Path to the image you want to test
TEST_IMAGE_PATH = r"C:\Users\TITO TOM\Downloads\plant_detection\03_model_export\test_image\BMC_19.jpg"

# Where to save the output image
OUTPUT_IMAGE_PATH = r"C:\Users\TITO TOM\Downloads\plant_detection\03_model_export\test_images\test_output.jpg"

# Confidence threshold for detections (this model's calibrated scores are ~0.0005 - 0.005)
CONF_THRES = 0.0005

# ============================================================


class CustomONNXPredictor:
    """Predictor for custom YOLO11-seg with Root-Point regression."""

    def __init__(self, onnx_path: str, conf_thres: float = 0.0005, img_size: int = 640):
        self.onnx_path = onnx_path
        self.conf_thres = conf_thres
        self.img_size = img_size
        self.class_names = {
            0: "crop_small_leaf",
            1: "crop_large_leaf",
            2: "weed_small_leaf",
            3: "weed_large_leaf",
        }

        # Architecture constants
        self.nm = 32  # mask coefficient channels
        self.nk = 2   # keypoint channels (x, y)

        print(f"[Info] Loading ONNX model from: {self.onnx_path}")
        self.session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        # Auto-detect image size from ONNX graph
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4 and isinstance(input_shape[2], int):
            self.img_size = input_shape[2]
            print(f"[Info] Auto-detected input size: {self.img_size}x{self.img_size}")

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """Aspect-ratio preserving resize with padding."""
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def predict(self, image_path: str, output_path: str = "test_output.jpg"):
        """Run inference on a single image and save the result."""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[Error] Failed to load image: {image_path}")
            return

        original_frame = frame.copy()

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # 1. Preprocess
        img_lb, ratio, (pad_w, pad_h) = self._letterbox(frame, new_shape=(self.img_size, self.img_size))
        rgb = img_lb[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        tensor_np = np.ascontiguousarray(rgb, dtype=np.float32)[np.newaxis] / 255.0

        # 2. Run Inference
        t0 = time.time()
        output_names = [o.name for o in self.session.get_outputs()]
        outputs_list = self.session.run(None, {self.input_name: tensor_np})
        outputs_dict = dict(zip(output_names, outputs_list))

        latency_ms = (time.time() - t0) * 1000
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        print(f"[Info] Inference: {latency_ms:.1f} ms ({fps:.1f} FPS)")

        # Ultralytics names segmentation ONNX outputs "output0" (predictions) / "output1" (proto)
        raw = torch.from_numpy(outputs_dict["output0"].astype(np.float32))
        proto = torch.from_numpy(outputs_dict["output1"].astype(np.float32)) if "output1" in outputs_dict else None

        # 3. NMS
        nms_out = non_max_suppression(raw, conf_thres=self.conf_thres, iou_thres=0.45,
                                      nc=len(self.class_names))
        det = nms_out[0]

        if det is None or len(det) == 0:
            print("[Info] No detections found.")
            cv2.imwrite(output_path, original_frame)
            return

        print(f"[Info] {len(det)} detection(s) found.")

        masks_img = None
        if proto is not None:
            masks_img = process_mask(proto[0], det[:, 6:6 + self.nm], det[:, :4],
                                     (self.img_size, self.img_size), upsample=True)

        # 4. Draw results
        for i, row in enumerate(det):
            row_np = row.cpu().numpy()
            x1, y1, x2, y2 = row_np[:4]
            conf = float(row_np[4])
            cls_id = int(row_np[5])

            h_orig, w_orig = original_frame.shape[:2]

            # Rescale bounding box (clamped to image bounds)
            x1 = max(0, min(w_orig - 1, int((x1 - pad_w) / ratio)))
            y1 = max(0, min(h_orig - 1, int((y1 - pad_h) / ratio)))
            x2 = max(0, min(w_orig - 1, int((x2 - pad_w) / ratio)))
            y2 = max(0, min(h_orig - 1, int((y2 - pad_h) / ratio)))

            # Rescale root point (clamped to image bounds)
            root_x = max(0, min(w_orig - 1, int((row_np[6 + self.nm] - pad_w) / ratio)))
            root_y = max(0, min(h_orig - 1, int((row_np[6 + self.nm + 1] - pad_h) / ratio)))

            color = (0, 255, 0) if cls_id in [0, 1] else (0, 0, 255)

            # Bounding box
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(original_frame, f"{self.class_names.get(cls_id)} {conf:.4f}",
                        (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Root point (blue dot)
            cv2.circle(original_frame, (root_x, root_y), 5, (255, 0, 0), -1)

            # Segmentation mask
            if masks_img is not None:
                mask = masks_img[i].cpu().numpy()
                mask = cv2.resize(mask, (img_lb.shape[1], img_lb.shape[0]))

                h, w = original_frame.shape[:2]
                mask_unpad = mask[int(pad_h):int(img_lb.shape[0] - pad_h),
                                  int(pad_w):int(img_lb.shape[1] - pad_w)]
                mask_resized = cv2.resize(mask_unpad, (w, h))

                colored_mask = np.zeros_like(original_frame)
                colored_mask[mask_resized > 0.5] = color
                cv2.addWeighted(colored_mask, 0.4, original_frame, 1.0, 0, original_frame)

        # FPS overlay
        cv2.putText(original_frame, f"FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imwrite(output_path, original_frame)
        print(f"[Success] Result saved to: {output_path}")


if __name__ == "__main__":
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"[Error] ONNX model not found: {ONNX_MODEL_PATH}")
    elif not os.path.exists(TEST_IMAGE_PATH):
        print(f"[Error] Test image not found: {TEST_IMAGE_PATH}")
    else:
        predictor = CustomONNXPredictor(onnx_path=ONNX_MODEL_PATH, conf_thres=CONF_THRES)
        predictor.predict(image_path=TEST_IMAGE_PATH, output_path=OUTPUT_IMAGE_PATH)
