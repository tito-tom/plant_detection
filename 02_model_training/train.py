"""
train.py — Custom training loop for YOLO-Seg-Root.

Implements CustomTrainer (extends Ultralytics SegmentationTrainer) with a
hand-rolled PyTorch loop giving full control over loss weighting, augmentation
scheduling (mosaic close-out), early stopping, and metric logging.

Run directly:
    python train.py

All hyperparameters and dataset paths are configured in config.py.
"""

import os
import csv
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics.models.yolo.segment.train import SegmentationTrainer

from utils import build_model, prepare_batch, print_header, print_epoch_row, get_gpu_memory, CLASS_NAMES
from dataset import YOLOSegPointDataset
from loss import CustomLoss
from val import CustomValidator

import config as cfg


def compute_class_weights(labels_dir: str, nc: int) -> list:
    """Compute median-frequency class weights from label files."""
    import glob
    import statistics

    counts = [0] * nc
    for lpath in glob.glob(os.path.join(labels_dir, "*.txt")):
        with open(lpath) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(float(parts[0]))
                    if 0 <= cls_id < nc:
                        counts[cls_id] += 1

    total = sum(counts)
    if total == 0:
        print("[Warning] No labels found — using uniform class weights.")
        return [1.0] * nc

    median_count = statistics.median(c for c in counts if c > 0) or 1
    weights = [
        20.0 if c == 0 else min(median_count / c, 20.0)
        for c in counts
    ]
    print(f"[AutoWeights] Instance counts : {counts}")
    print(f"[AutoWeights] Class weights   : {[round(w, 4) for w in weights]}")
    return weights


def get_lr(epoch, warmup, lr0, lrf, total):
    """Cosine annealing with linear warmup."""
    if epoch <= warmup:
        return lr0 * epoch / max(warmup, 1)
    prog = (epoch - warmup) / max(total - warmup, 1)
    return lr0 * (lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * prog)))


class CustomTrainer(SegmentationTrainer):
    """Custom training loop for YOLO-Seg-Root multi-task model.

    Inherits from Ultralytics SegmentationTrainer but overrides ``train()``
    with a hand-rolled PyTorch loop for full control over loss, metrics, and
    augmentation scheduling.
    """

    def __init__(self, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides.setdefault("model",   cfg.cfg.RESUME_WEIGHTS or cfg.cfg.PRETRAINED_WEIGHTS)
        overrides.setdefault("epochs",  cfg.cfg.EPOCHS)
        overrides.setdefault("batch",   cfg.cfg.BATCH_SIZE)
        overrides.setdefault("imgsz",   cfg.cfg.IMG_SIZE)
        overrides.setdefault("device",  cfg.cfg.DEVICE)
        overrides.setdefault("project", os.path.dirname(cfg.cfg.OUTPUT_DIR))
        overrides.setdefault("name",    os.path.basename(cfg.cfg.OUTPUT_DIR))
        overrides.setdefault("data",    os.path.join(cfg.cfg.DATA_DIR, "data.yaml"))
        super().__init__(overrides=overrides, _callbacks=_callbacks)

        self.nc      = len(CLASS_NAMES)
        self.history = {k: [] for k in [
            "epoch", "train_loss", "val_loss",
            "box_loss", "seg_loss", "cls_loss", "dfl_loss", "kpt_loss",
            "box_mAP50", "box_mAP5095", "mask_mAP50", "mask_mAP5095",
            "PCK5", "PCK10", "PCK20",
        ]}

    def get_model(self, cfg=None, weights=None, verbose=True):
        return build_model(weights_path=weights, device=self.device)

    def get_validator(self):
        from pathlib import Path
        self.validator = CustomValidator(args=self.args, save_dir=Path(self.save_dir))
        return self.validator

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode == "train":
            return YOLOSegPointDataset(cfg.cfg.TRAIN_IMAGES, cfg.cfg.TRAIN_LABELS,
                                       cfg.cfg.IMG_SIZE, cfg.cfg.HYP, augment=True)
        return YOLOSegPointDataset(cfg.cfg.VAL_IMAGES, cfg.cfg.VAL_LABELS,
                                   cfg.cfg.IMG_SIZE, cfg.cfg.HYP, augment=False)

    def _plot(self):
        """Generate dark-themed training plots to the output directory."""
        if len(self.history["epoch"]) < 2:
            return

        plt.rcParams.update({
            "figure.facecolor": "#1e1e2e", "axes.facecolor": "#1e1e2e",
            "axes.edgecolor":   "#444",    "axes.labelcolor": "#ccc",
            "text.color":       "#ccc",    "grid.color":      "#333",
            "legend.facecolor": "#2a2a3e", "font.size":       11,
        })

        eps     = self.history["epoch"]
        out_dir = str(self.save_dir)

        # Loss curves
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eps, self.history["train_loss"], "o-", color="#4fc3f7", lw=2, ms=3, label="Train")
        ve = [e for e, v in zip(eps, self.history["val_loss"]) if v is not None]
        vv = [v for v in self.history["val_loss"] if v is not None]
        if vv:
            ax.plot(ve, vv, "s-", color="#f06292", lw=2, ms=4, label="Val")
        ax.set(xlabel="Epoch", ylabel="Loss", title="Train vs Val Loss")
        ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
        plt.close(fig)

        # Loss components
        fig, ax = plt.subplots(figsize=(10, 5))
        colors  = {"box":"#4fc3f7","seg":"#81c784","cls":"#ffb74d","dfl":"#ce93d8","kpt":"#f06292"}
        for k, c in colors.items():
            ax.plot(eps, self.history[f"{k}_loss"], "o-", color=c, lw=1.8, ms=3, label=k)
        ax.set(xlabel="Epoch", ylabel="Loss", title="Loss Components")
        ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "loss_components.png"), dpi=150)
        plt.close(fig)

        # Metrics curves
        me = [e for e, v in zip(eps, self.history["box_mAP50"]) if v is not None]
        if not me:
            return

        def _c(k):
            return [v for v in self.history[k] if v is not None]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(me, _c("box_mAP50"),    "o-", color="#4fc3f7", lw=2, ms=4, label="Box mAP50")
        ax1.plot(me, _c("box_mAP5095"),  "s-", color="#81c784", lw=2, ms=4, label="Box mAP50-95")
        ax1.plot(me, _c("mask_mAP50"),   "^-", color="#ffb74d", lw=2, ms=4, label="Mask mAP50")
        ax1.plot(me, _c("mask_mAP5095"), "x-", color="#ce93d8", lw=2, ms=4, label="Mask mAP50-95")
        ax1.set(xlabel="Epoch", ylabel="mAP", title="Detection & Segmentation", ylim=(0, 1))
        ax1.legend(); ax1.grid(alpha=.3)

        ax2.plot(me, _c("PCK5"),  "o-", color="#ce93d8", lw=2, ms=4, label="PCK@5")
        ax2.plot(me, _c("PCK10"), "s-", color="#f06292", lw=2, ms=4, label="PCK@10")
        ax2.plot(me, _c("PCK20"), "^-", color="#ffb74d", lw=2, ms=4, label="PCK@20")
        ax2.set(xlabel="Epoch", ylabel="PCK", title="Root-Point Accuracy", ylim=(0, 1))
        ax2.legend(); ax2.grid(alpha=.3)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "metrics_curves.png"), dpi=150)
        plt.close(fig)

    def train(self):
        """Run the full custom training loop."""
        os.makedirs(self.save_dir, exist_ok=True)

        print("\n" + "=" * 65)
        print("  YOLO-Seg-Root — Single-Phase Joint Training")
        print("=" * 65)
        print(f"  Classes:       {self.nc} ({', '.join(CLASS_NAMES.values())})")
        print(f"  Dataset:       {cfg.cfg.DATA_DIR}")
        print(f"  Epochs:        {self.args.epochs}  (patience={cfg.cfg.PATIENCE})")
        print(f"  Image size:    {self.args.imgsz}  |  Batch: {self.args.batch}")
        print(f"  LR:            {cfg.cfg.LR0} → {cfg.cfg.LR0 * cfg.cfg.LRF:.2e}  (warmup={cfg.cfg.WARMUP_EPOCHS})")
        print(f"  KPT gain:      {cfg.cfg.KPT_GAIN}")
        print(f"  Close mosaic:  last {cfg.cfg.CLOSE_MOSAIC_EP} epochs")
        print(f"  Class weights: {'auto' if cfg.cfg.CLASS_WEIGHTS == 'auto' else cfg.cfg.CLASS_WEIGHTS}")
        print(f"  Device:        {self.device}")
        print(f"  Output:        {self.save_dir}")
        if self.args.model:
            print(f"  Model:         {self.args.model}")
        print("=" * 65 + "\n")

        train_ds = self.build_dataset(cfg.cfg.TRAIN_IMAGES, mode="train")
        val_ds   = self.build_dataset(cfg.cfg.VAL_IMAGES,   mode="val")
        train_loader = DataLoader(train_ds, self.args.batch, shuffle=True,
                                  num_workers=cfg.cfg.WORKERS,
                                  collate_fn=YOLOSegPointDataset.collate_fn)
        val_loader   = DataLoader(val_ds,   self.args.batch, shuffle=False,
                                  num_workers=cfg.cfg.WORKERS,
                                  collate_fn=YOLOSegPointDataset.collate_fn)
        print(f"  Train: {len(train_ds)} images | Val: {len(val_ds)} images\n")

        resolved_weights = (
            compute_class_weights(cfg.cfg.TRAIN_LABELS, self.nc)
            if cfg.cfg.CLASS_WEIGHTS == "auto" else cfg.cfg.CLASS_WEIGHTS
        )

        self.model = self.get_model(weights=self.args.model if str(self.args.model).endswith(".pt") else None)
        criterion  = CustomLoss(self.model.model, class_weights=resolved_weights)
        criterion.hyp.pose = cfg.cfg.KPT_GAIN
        criterion.hyp.box  = cfg.cfg.BOX_GAIN
        criterion.hyp.cls  = cfg.cfg.CLS_GAIN
        optimizer  = optim.AdamW(self.model.model.parameters(),
                                 lr=cfg.cfg.LR0, weight_decay=cfg.cfg.WEIGHT_DECAY)

        csv_path = os.path.join(self.save_dir, "results.csv")
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "lr", "train_loss",
                "box_loss", "seg_loss", "cls_loss", "dfl_loss", "kpt_loss",
                "val_loss", "box_mAP50", "box_mAP50-95",
                "mask_mAP50", "mask_mAP50-95", "PCK@5", "PCK@10", "PCK@20",
            ])

        best_val_loss  = float("inf")
        no_improve_cnt = 0
        validator      = self.get_validator()
        validator.dataloader = val_loader

        print_header()

        for epoch in range(1, self.args.epochs + 1):
            if epoch == self.args.epochs - cfg.cfg.CLOSE_MOSAIC_EP + 1:
                train_ds.augment = False
                if hasattr(train_ds, "mosaic"):
                    train_ds.mosaic.p = 0.0
                print(f"\n  [Epoch {epoch}] Mosaic disabled — final {cfg.cfg.CLOSE_MOSAIC_EP} stable epochs.\n")

            self.model.model.train()
            lr = get_lr(epoch, cfg.cfg.WARMUP_EPOCHS, cfg.cfg.LR0, cfg.cfg.LRF, self.args.epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            epoch_loss  = 0.0
            epoch_items = torch.zeros(5)
            n_batches   = 0
            n_inst      = 0

            pbar = tqdm(train_loader, desc=f"  {epoch}/{self.args.epochs}", leave=False, ncols=110)
            for imgs, targets in pbar:
                batch = prepare_batch(targets, self.device)
                if batch is None:
                    continue
                optimizer.zero_grad()
                preds = self.model.model(imgs.to(self.device))
                loss, items = criterion(preds, batch)

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Training diverged: loss={loss.item()} at epoch {epoch}."
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), cfg.cfg.GRAD_CLIP)
                optimizer.step()

                epoch_loss  += loss.item()
                epoch_items += items.cpu()
                n_batches   += 1
                n_inst       = len(batch["cls"])
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train = epoch_loss  / max(n_batches, 1)
            avg_items = epoch_items / max(n_batches, 1)
            print_epoch_row(epoch, self.args.epochs, get_gpu_memory(), avg_items, n_inst, self.args.imgsz)

            v_loss = v_map50 = v_map5095 = v_mask50 = v_mask5095 = 0.0
            pck05 = pck10 = pck20 = 0.0
            ran_val = False

            if epoch % cfg.cfg.VAL_INTERVAL == 0 or epoch == self.args.epochs:
                ran_val = True
                self.model.model.eval()
                v_loss, v_items, map_res, pck, cls_cnts = validator(trainer=self)

                v_map50    = map_res.get("mAP50",         0.0)
                v_map5095  = map_res.get("mAP50_95",      0.0)
                v_mask50   = map_res.get("mask_mAP50",    0.0)
                v_mask5095 = map_res.get("mask_mAP50_95", 0.0)
                pck05      = pck.get(0.05, 0.0)
                pck10      = pck.get(0.10, 0.0)
                pck20      = pck.get(0.20, 0.0)

                if v_loss < best_val_loss:
                    best_val_loss  = v_loss
                    no_improve_cnt = 0
                    self.model.save(os.path.join(self.save_dir, "best.pt"))
                    print(f"  ★ New best!  val_loss={v_loss:.4f}  (saved best.pt)")
                else:
                    no_improve_cnt += 1
                    print(f"  → No improvement  ({no_improve_cnt}/{cfg.cfg.PATIENCE})")
                    if no_improve_cnt >= cfg.cfg.PATIENCE:
                        print(f"\n  ⏹  Early stopping at epoch {epoch}.")
                        break

            if epoch % cfg.cfg.SAVE_PERIOD == 0:
                self.model.save(os.path.join(self.save_dir, f"epoch_{epoch}.pt"))
            self.model.save(os.path.join(self.save_dir, "last.pt"))

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(avg_train)
            self.history["val_loss"].append(v_loss if ran_val else None)
            for j, k in enumerate(["box_loss", "seg_loss", "cls_loss", "dfl_loss", "kpt_loss"]):
                self.history[k].append(avg_items[j].item())
            self.history["box_mAP50"].append(v_map50   if ran_val else None)
            self.history["box_mAP5095"].append(v_map5095 if ran_val else None)
            self.history["mask_mAP50"].append(v_mask50  if ran_val else None)
            self.history["mask_mAP5095"].append(v_mask5095 if ran_val else None)
            self.history["PCK5"].append(pck05  if ran_val else None)
            self.history["PCK10"].append(pck10 if ran_val else None)
            self.history["PCK20"].append(pck20 if ran_val else None)

            def _f(v, r): return f"{v:.4f}" if r else ""
            def _m(v, r): return f"{v:.3f}"  if r else ""
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, f"{lr:.6f}", f"{avg_train:.4f}",
                    f"{avg_items[0]:.4f}", f"{avg_items[1]:.4f}", f"{avg_items[2]:.4f}",
                    f"{avg_items[3]:.4f}", f"{avg_items[4]:.4f}",
                    _f(v_loss, ran_val), _m(v_map50, ran_val), _m(v_map5095, ran_val),
                    _m(v_mask50, ran_val), _m(v_mask5095, ran_val),
                    _m(pck05, ran_val), _m(pck10, ran_val), _m(pck20, ran_val),
                ])

            self._plot()

        print("\n" + "=" * 65)
        print("  Training complete!")
        print(f"  Best val loss : {best_val_loss:.4f}")
        print(f"  Best model    : {os.path.join(self.save_dir, 'best.pt')}")
        print(f"  CSV           : {csv_path}")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    trainer = CustomTrainer()
    trainer.train()
