# @deployment: yolo-seg-root v5 (register head)
# CustomSegmentHead — Instance Segmentation + Root Point Regression.

"""
Custom YOLO detection head: Instance Segmentation + Root Point Regression.

Extends the standard Ultralytics Segment head by adding a parallel keypoint
regression branch (cv5) for predicting a single root-point (x, y) per object.

Architecture::

    Neck Features [P3, P4, P5]
            │
    ┌───────┼───────────┐
    │       │           │
    cv2   cv3   cv4   cv5       ← Root-point branch
  (boxes) (cls) (masks) (root pts)
    │       │     │     │
    └───────┴─────┴─────┘
            │
      Combined Output

This module is registered into the ultralytics namespace so the YAML parser
can instantiate it directly - no post-load head swapping required.

Compatible with Ultralytics >= 8.4.x dict-based forward API.
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Segment, Detect
from ultralytics.nn.modules.conv import Conv


class CustomSegmentHead(Segment):
    """
    Segmentation head with an extra Root-Point Regression branch (cv5).

    Architecture (per FPN level):
        cv2  →  box distribution (DFL)
        cv3  →  class scores
        cv4  →  mask coefficients (nm channels)
        cv5  →  root-point [x, y]   ← NEW

    Args:
        nc        : number of object classes.
        nm        : mask prototype channels (default 32).
        npr       : proto network channels  (default 256).
        ch        : input channel tuple from neck (P3, P4, P5).
        kpt_shape : (n_keypoints, dims) — (1, 2) for one root-point.
    """

    def __init__(self, nc=80, nm=32, npr=256, ch=(), kpt_shape=(1, 2)):
        super().__init__(nc=nc, nm=nm, npr=npr, ch=ch)
        self._end2end = False  # We don't use NMS-free end2end detection

        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]      # = 2  (x, y)

        # Root-point branch  — lightweight 3-layer conv per FPN level
        c5 = max(ch[0] // 4, self.nk)
        self.cv5 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c5, 3),
                Conv(c5, c5, 3),
                nn.Conv2d(c5, self.nk, 1),
            )
            for x in ch
        )

    # --- Properties for the new dict-based forward API -------------------

    @property
    def one2many(self):
        """Returns the one-to-many head components including the kpt branch."""
        return dict(
            box_head=self.cv2, cls_head=self.cv3,
            mask_head=self.cv4, kpt_head=self.cv5,
        )


    # --- Core forward methods --------------------------------------------

    def forward_head(self, x, box_head, cls_head, mask_head, kpt_head=None):
        """Add keypoint predictions to the Segment forward_head output."""
        preds = Segment.forward_head(self, x, box_head, cls_head, mask_head)
        if kpt_head is not None:
            bs = x[0].shape[0]
            preds["kpts"] = torch.cat(
                [kpt_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2
            )
        return preds

    def forward(self, x):
        """
        Forward pass compatible with Ultralytics 8.4.x dict-based API.

        Training returns:
            dict with keys: boxes, scores, feats, mask_coefficient, proto, kpts

        Inference returns:
            ((combined_tensor, proto), preds_dict)
        """
        # Detect.forward handles one2many/one2one routing and calls forward_head
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs

        # Mask prototypes from P3
        proto = self.proto(x[0])

        if isinstance(preds, dict):
            preds["proto"] = proto

        if self.training:
            return preds

        # Inference / Export
        if self.export:
            return (outputs, proto)
        return ((outputs[0], proto), preds)

    def _inference(self, x):
        """Decode boxes + scores + mask_coeff + keypoints for inference."""
        preds = Segment._inference(self, x)
        if "kpts" in x:
            return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)
        return preds

    # ------------------------------------------------------------------
    def kpts_decode(self, kpts):
        """
        Decode raw keypoint predictions → absolute pixel coordinates.

        Formula:  decoded = (raw * 2 + anchor - 0.5) * stride

        Args:
            kpts : (B, nk, A) raw keypoint tensor.

        Returns:
            (B, nk, A) decoded keypoints in pixel space.
        """
        ndim = self.kpt_shape[1]   # 2
        y = kpts.clone()
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y
