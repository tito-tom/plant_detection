"""
tune.py — Bayesian hyperparameter tuning for YOLO-Seg-Root.

Uses Optuna to maximise: score = alpha * mask_mAP50-95 + (1 - alpha) * PCK@10.
Two modes: "train" (full short run per trial) or "val" (inference params only).
Resume-safe: trials are saved to a SQLite database.
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # synchronous CUDA errors for safe trial recovery

import sys
import shutil
import warnings
import math
from contextlib import contextmanager

import torch
import optuna

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import config as cfg

warnings.filterwarnings("ignore", category=UserWarning)


class YOLOSegRootTuner:
    """Optuna-based Bayesian hyperparameter tuner for YOLO-Seg-Root.

    Args:
        mode       : ``"train"`` (full training sweep) or ``"val"`` (inference params only).
        n_trials   : Total Optuna trials to run.
        tune_epochs: Epochs per trial when ``mode="train"``.
        alpha      : Weight for mask mAP in composite score (PCK weight = 1 - alpha).
        study_name : Optuna study name.
        db_name    : SQLite database filename (stored next to this script).
        val_weights: Path to pretrained weights for ``mode="val"``.
    """

    def __init__(
        self,
        mode: str        = "train",
        n_trials: int    = 20,
        tune_epochs: int = 10,
        alpha: float     = 0.5,
        study_name: str  = "yolo_seg_root_tuning",
        db_name: str     = "optuna_seg_root.db",
        val_weights: str = None,
    ):
        self.mode        = mode
        self.n_trials    = n_trials
        self.tune_epochs = tune_epochs
        self.alpha       = alpha
        self.study_name  = study_name
        self.db_path     = os.path.join(_THIS_DIR, db_name)
        self.db_url      = f"sqlite:///{self.db_path}"
        self.val_weights = val_weights or cfg.cfg.WEIGHT_PATH

    @staticmethod
    @contextmanager
    def patched_config(overrides: dict):
        """Temporarily override ``cfg`` attributes and restore on exit."""
        originals = {k: getattr(cfg.cfg, k) for k in overrides}
        for k, v in overrides.items():
            setattr(cfg.cfg, k, v)
        try:
            yield
        finally:
            for k, v in originals.items():
                setattr(cfg.cfg, k, v)

    def suggest_train_params(self, trial: optuna.Trial) -> dict:
        """Define the training hyperparameter search space."""
        return {
            "lr0":           trial.suggest_float("lr0",          1e-4, 5e-3, log=True),
            "lrf":           trial.suggest_float("lrf",          0.001, 0.01),
            "weight_decay":  trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "betas":         trial.suggest_float("betas",        0.85, 0.999, log=True),
            "warmup_epochs": trial.suggest_int(  "warmup_epochs", 1, 5),
            "grad_clip":     trial.suggest_float("grad_clip",    1.0, 50.0, log=True),
            "box":           trial.suggest_float("box",          5.0, 10.0),
            "cls":           trial.suggest_float("cls",          0.3, 1.0),
            "kpt_gain":      trial.suggest_float("kpt_gain",     8.0, 15.0),
            "mosaic":        trial.suggest_float("mosaic",       0.0, 1.0),
            "copy_paste":    trial.suggest_float("copy_paste",   0.1, 0.5),
            "scale":         trial.suggest_float("scale",        0.1, 0.6),
            "close_mosaic":  trial.suggest_int(  "close_mosaic", 5, 15),
            "hsv_h":         trial.suggest_float("hsv_h",        0.0, 0.5),
            "hsv_s":         trial.suggest_float("hsv_s",        0.0, 1.0),
            "hsv_v":         trial.suggest_float("hsv_v",        0.0, 1.0),
            "degrees":       trial.suggest_float("degrees",      0.0, 360.0),
            "translate":     trial.suggest_float("translate",    0.0, 0.5),
            "flipud":        trial.suggest_float("flipud",       0.0, 1.0),
            "fliplr":        trial.suggest_float("fliplr",       0.0, 1.0),
        }

    def suggest_val_params(self, trial: optuna.Trial) -> dict:
        """Define the inference hyperparameter search space."""
        return {
            "conf_thres": trial.suggest_float("conf_thres", 0.001, 0.5, log=True),
            "iou_thres":  trial.suggest_float("iou_thres",  0.3, 0.8),
        }

    def objective_train(self, trial: optuna.Trial) -> float:
        """Train one short run and return the composite score."""
        params     = self.suggest_train_params(trial)
        trial_name = f"optuna_trial_{trial.number}"

        print(f"\n{'-' * 60}")
        print(f"  Trial {trial.number}  |  kpt_gain={params['kpt_gain']:.2f}"
              f"  lr0={params['lr0']:.2e}  wd={params['weight_decay']:.2e}")
        print(f"{'-' * 60}")

        trial_hyp = cfg.cfg.HYP.copy()
        for aug_key in ("mosaic", "copy_paste", "scale", "hsv_h", "hsv_s",
                        "hsv_v", "degrees", "translate", "flipud", "fliplr"):
            if aug_key in params:
                trial_hyp[aug_key] = params[aug_key]

        trial_output_dir = os.path.normpath(
            os.path.join(cfg.cfg.OUTPUT_DIR, "..", "optuna", trial_name)
        )

        config_overrides = {
            "LR0":            params["lr0"],
            "LRF":            params["lrf"],
            "WEIGHT_DECAY":   params["weight_decay"],
            "KPT_GAIN":       params["kpt_gain"],
            "BOX_GAIN":       params["box"],
            "CLS_GAIN":       params["cls"],
            "WARMUP_EPOCHS":  params["warmup_epochs"],
            "GRAD_CLIP":      params["grad_clip"],
            "CLOSE_MOSAIC_EP": params["close_mosaic"],
            "EPOCHS":         self.tune_epochs,
            "PATIENCE":       0,
            "OUTPUT_DIR":     trial_output_dir,
            "HYP":            trial_hyp,
            "SAVE_PERIOD":    999,
        }

        try:
            with self.patched_config(config_overrides):
                from train import CustomTrainer
                from pathlib import Path

                trainer = CustomTrainer(overrides={
                    "epochs": self.tune_epochs,
                    "batch":  cfg.cfg.BATCH_SIZE,
                    "imgsz":  cfg.cfg.IMG_SIZE,
                    "device": cfg.cfg.DEVICE,
                    "name":   trial_name,
                })
                trainer.save_dir = Path(trial_output_dir)
                os.makedirs(trial_output_dir, exist_ok=True)
                trainer.train()

            mask50_scores   = [v for v in trainer.history["mask_mAP50"]   if v is not None]
            mask5095_scores = [v for v in trainer.history["mask_mAP5095"] if v is not None]
            pck_scores      = [v for v in trainer.history["PCK10"]        if v is not None]
            best_mask50     = max(mask50_scores)   if mask50_scores   else 0.0
            best_mask5095   = max(mask5095_scores) if mask5095_scores else 0.0
            best_pck        = max(pck_scores)      if pck_scores      else 0.0
            score           = self.alpha * best_mask5095 + (1.0 - self.alpha) * best_pck

            trial.set_user_attr("mask_mAP50",   best_mask50)
            trial.set_user_attr("mask_mAP5095", best_mask5095)
            trial.set_user_attr("mask_mAP",     best_mask5095)
            trial.set_user_attr("PCK",          best_pck)

            print(f"  [OK] Trial {trial.number}  mask_mAP50-95={best_mask5095:.4f}"
                  f"  PCK@10={best_pck:.4f}  composite={score:.4f}")
            return score

        except Exception as e:
            import traceback
            print(f"  [FAIL] Trial {trial.number}: {e}")
            traceback.print_exc()
            return 0.0

        finally:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
            if os.path.isdir(trial_output_dir):
                try:
                    shutil.rmtree(trial_output_dir)
                except OSError:
                    pass

    def objective_val(self, trial: optuna.Trial) -> float:
        """Evaluate a pre-trained model with suggested inference hyperparameters."""
        params = self.suggest_val_params(trial)
        print(f"\n{'-' * 60}")
        print(f"  Trial {trial.number}  |  conf={params['conf_thres']:.4f}"
              f"  iou={params['iou_thres']:.3f}")
        print(f"{'-' * 60}")

        try:
            with self.patched_config({
                "CONF_THRES": params["conf_thres"],
                "IOU_THRES":  params["iou_thres"],
            }):
                from val import CustomValidator
                from train import CustomTrainer
                from torch.utils.data import DataLoader
                from dataset import YOLOSegPointDataset
                from utils import build_model
                from pathlib import Path

                model  = build_model(weights_path=self.val_weights, device=cfg.cfg.DEVICE)
                val_ds = YOLOSegPointDataset(cfg.cfg.VAL_IMAGES, cfg.cfg.VAL_LABELS,
                                             cfg.cfg.IMG_SIZE, cfg.cfg.HYP, augment=False)
                val_loader = DataLoader(val_ds, cfg.cfg.BATCH_SIZE, shuffle=False,
                                        num_workers=cfg.cfg.WORKERS,
                                        collate_fn=YOLOSegPointDataset.collate_fn)

                trainer       = CustomTrainer.__new__(CustomTrainer)
                trainer.model  = model
                trainer.device = cfg.cfg.DEVICE
                trainer.args   = type("Args", (), {
                    "imgsz": cfg.cfg.IMG_SIZE,
                    "batch": cfg.cfg.BATCH_SIZE,
                })()

                validator = CustomValidator(
                    args=trainer.args,
                    save_dir=Path(os.path.join(_THIS_DIR, "optuna_val_tmp")),
                )
                validator.dataloader = val_loader
                model.model.eval()
                v_loss, v_items, map_res, pck, cls_cnts = validator(trainer=trainer)

                mask_mAP50   = map_res.get("mask_mAP50",    0.0)
                mask_mAP5095 = map_res.get("mask_mAP50_95", 0.0)
                pck10        = pck.get(0.10, 0.0)
                score        = self.alpha * mask_mAP5095 + (1.0 - self.alpha) * pck10

                trial.set_user_attr("mask_mAP50",   mask_mAP50)
                trial.set_user_attr("mask_mAP5095", mask_mAP5095)
                trial.set_user_attr("mask_mAP",     mask_mAP5095)
                trial.set_user_attr("PCK",          pck10)

                print(f"  [OK] Trial {trial.number}  mask_mAP50-95={mask_mAP5095:.4f}"
                      f"  PCK@10={pck10:.4f}  composite={score:.4f}")
                return score

        except Exception as e:
            import traceback
            print(f"  [FAIL] Trial {trial.number}: {e}")
            traceback.print_exc()
            return 0.0

        finally:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
            tmp = os.path.join(_THIS_DIR, "optuna_val_tmp")
            if os.path.isdir(tmp):
                try:
                    shutil.rmtree(tmp)
                except OSError:
                    pass

    def print_results(self, study: optuna.Study):
        """Print best trial summary and ready-to-paste config block."""
        print("\n" + "=" * 60)
        print("  OPTUNA STUDY COMPLETE — YOLO-Seg-Root")
        print("=" * 60)
        print(f"  Study name  : {study.study_name}")
        print(f"  Mode        : {self.mode}")
        print(f"  Trials      : {len(study.trials)}")
        print(f"  Best trial  : #{study.best_trial.number}")
        print(f"  Best score  : {study.best_value:.4f}  (alpha={self.alpha})")
        print()

        bp = study.best_params
        print("  +--- Best Hyperparameters ---+")
        for k, v in bp.items():
            print(f"  |  {k:<20s} = {v:.6f}" if isinstance(v, float) else f"  |  {k:<20s} = {v}")
        print("  +----------------------------+")

        print("\n  Paste into config.py:")
        print("  " + "-" * 50)
        if self.mode == "train":
            print(f'  LR0              = {bp["lr0"]:.6e}')
            print(f'  LRF              = {bp["lrf"]:.6f}')
            print(f'  WEIGHT_DECAY     = {bp["weight_decay"]:.6e}')
            print(f'  KPT_GAIN         = {bp["kpt_gain"]:.4f}')
            print(f'  BOX_GAIN         = {bp["box"]:.4f}')
            print(f'  CLS_GAIN         = {bp["cls"]:.4f}')
            print(f'  WARMUP_EPOCHS    = {bp["warmup_epochs"]}')
            print(f'  GRAD_CLIP        = {bp["grad_clip"]:.4f}')
            print(f'  CLOSE_MOSAIC_EP  = {bp["close_mosaic"]}')
            print()
            print(f'  HYP = {{')
            for k in ("mosaic","copy_paste","scale","hsv_h","hsv_s","hsv_v","degrees","translate","flipud","fliplr"):
                print(f'      "{k}": {bp[k]:.4f},')
            print(f'  }}')
        else:
            print(f'  CONF_THRES = {bp["conf_thres"]:.6f}')
            print(f'  IOU_THRES  = {bp["iou_thres"]:.4f}')
        print("  " + "-" * 50)
        print(f"\n  Visualise: optuna-dashboard sqlite:///{os.path.basename(self.db_path)}")
        print("=" * 60 + "\n")

    def run(self):
        """Create the Optuna study and run optimization."""
        print("\n" + "=" * 60)
        print("  Optuna Hyperparameter Tuning — YOLO-Seg-Root")
        print("=" * 60)
        print(f"  Mode     : {self.mode}")
        print(f"  Study    : {self.study_name}")
        print(f"  Trials   : {self.n_trials}")
        if self.mode == "train":
            print(f"  Epochs   : {self.tune_epochs} per trial")
            print(f"  Alpha    : {self.alpha}  (mask*a + PCK*(1-a))")
        else:
            print(f"  Weights  : {self.val_weights}")
        print(f"  Dataset  : {cfg.cfg.DATA_DIR}")
        print(f"  Database : {self.db_path}")
        print("=" * 60 + "\n")

        if self.mode == "val" and not os.path.isfile(self.val_weights):
            print(f"  ERROR: weights not found: {self.val_weights}")
            print("  Train a model first, or update val_weights path.")
            return

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.db_url,
            load_if_exists=True,
            direction="maximize",
        )
        objective_fn = self.objective_train if self.mode == "train" else self.objective_val
        study.optimize(objective_fn, n_trials=self.n_trials)
        self.print_results(study)


if __name__ == "__main__":
    tuner = YOLOSegRootTuner(
        mode="train",
        n_trials=40,
        tune_epochs=50,
        alpha=0.5,
        study_name="yolo_seg_root_opt",
        db_name="optuna_seg_root.db",
    )
    tuner.run()
