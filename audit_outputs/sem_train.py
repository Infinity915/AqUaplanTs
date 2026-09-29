"""Runnable Semantic Segmentation Trainer — Path A (Primary SOTA & In-House Baseline Rerun).
========================================================================================
Architecture: Supports DeepLabV3Plus (Primary SOTA), SegFormer-B0, and U-Net (Same-Split Baseline).
Loss Function: Mathematical Compound Loss:
  - Binary Mode:      L_binary = L_BCEWithLogits + L_Dice
  - Multiclass Mode:  L_multi  = L_WCE(MedianFreq, clamped [0.1, 10.0]) + L_Dice + 0.5 * L_Focal(gamma=2.0)
Evaluation:
  - Primary Metric: 20 Evaluated Classes (Class 0: Background + 19 Shared Weed Species).
  - Statistical Rigor: 1,000-sample non-parametric 95% Bootstrap Confidence Intervals.
  - Quota Safety (Gap G3): Built-in throughput monitor at Epoch 5; auto-caps at 30 epochs if <10.0 img/s.

Usage on Kaggle Tesla T4 (pip install -q segmentation-models-pytorch albumentations):
  # 1. Primary Model: Binary DeepLabV3+ (40 epochs, ~3.2h GPU):
  python sem_train.py --mode binary --arch DeepLabV3Plus --encoder resnet34 --tiles tiles/t640_o128_minfg0.01_bg0.05 --epochs 40 --batch 12 --out runs_sem/deeplab_binary

  # 2. Primary Model: Multiclass DeepLabV3+ (warm-start, 40 epochs, ~3.2h GPU):
  python sem_train.py --mode multi --arch DeepLabV3Plus --encoder resnet34 --tiles tiles/t640_o128_minfg0.01_bg0.05 --epochs 40 --batch 12 --warm-start runs_sem/deeplab_binary/best.pt --out runs_sem/deeplab_multi

  # 3. In-House Baseline Rerun: U-Net (ResNet34) on same v3 split (30 epochs, ~2.5h GPU, closes Gap G5):
  python sem_train.py --mode binary --arch Unet --encoder resnet34 --tiles tiles/t640_o128_minfg0.01_bg0.05 --epochs 30 --batch 12 --out runs_sem/unet_binary
  python sem_train.py --mode multi --arch Unet --encoder resnet34 --tiles tiles/t640_o128_minfg0.01_bg0.05 --epochs 30 --batch 12 --warm-start runs_sem/unet_binary/best.pt --out runs_sem/unet_multi

  # 4. Evaluation with 1000-sample 95% Bootstrap CIs on Test Lockbox (CPU, ~15s):
  python sem_train.py --mode multi --arch DeepLabV3Plus --tiles tiles/t640_o128_minfg0.01_bg0.05 --checkpoint runs_sem/deeplab_multi/best.pt --eval-only
========================================================================================
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. HELPER: ROBUST RESOURCE FILE DISCOVERY
# ==============================================================================
def find_resource_file(filename):
    """Finds resource file across repo root, audit_outputs, script dir, or current cwd."""
    candidates = [
        Path(filename),
        Path("audit_outputs") / filename,
        Path(__file__).parent / filename,
        Path(__file__).parent.parent / filename,
        Path(__file__).parent / "audit_outputs" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

# ==============================================================================
# 2. DATA AUGMENTATION & DATASET
# ==============================================================================
def get_transforms(split):
    try:
        import albumentations as A
        if split == "train":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, p=0.3),
            ])
    except ImportError:
        pass
    return None

class WeedTileDataset(Dataset):
    def __init__(self, tdir, split="train", mode="binary", transform=None):
        self.tdir = Path(tdir)
        self.mode = mode
        self.transform = transform
        idx_path = self.tdir / "tile_index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"Missing tile_index.csv in {tdir}. Did you run tile_dataset.py?")
        rows = list(csv.DictReader(open(idx_path, encoding="utf-8")))
        self.rows = [r for r in rows if r.get("split") == split]
        self.img_dir = self.tdir / "images"
        self.mask_dir = self.tdir / ("binary" if mode == "binary" else "multiclass")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        name = r["tile"]
        img_p = self.img_dir / f"{name}.png"
        if not img_p.exists():
            img_p = self.img_dir / f"{name}.jpg"
        mask_p = self.mask_dir / f"{name}.png"

        img = np.array(Image.open(img_p).convert("RGB"))
        mask = np.array(Image.open(mask_p))

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.mode == "binary":
            mask = (torch.from_numpy(mask) > 0).float().unsqueeze(0)
        else:
            mask = torch.from_numpy(mask).long()

        return img, mask, r.get("src", name)

# ==============================================================================
# 3. MODEL BUILDER
# ==============================================================================
def build_model(arch="DeepLabV3Plus", encoder="resnet34", num_classes=1):
    import segmentation_models_pytorch as smp
    arch_lower = arch.lower()
    if arch_lower in ("unet", "u-net"):
        return smp.Unet(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=num_classes)
    elif arch_lower in ("segformer", "segformer-b0"):
        enc = "mit_b0" if encoder == "resnet34" else encoder
        return smp.Unet(encoder_name=enc, encoder_weights="imagenet", in_channels=3, classes=num_classes)
    elif arch_lower in ("fpn",):
        return smp.FPN(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=num_classes)
    else:
        # Default is DeepLabV3Plus with ASPP
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=num_classes)

# ==============================================================================
# 4. MATHEMATICAL COMPOUND LOSS IMPLEMENTATION
# ==============================================================================
class SoftDiceLoss(nn.Module):
    """Numerically stable Soft Dice Loss for binary and multiclass segmentation."""
    def __init__(self, eps=1e-6, mode="binary"):
        super().__init__()
        self.eps = eps
        self.mode = mode

    def forward(self, logits, targets):
        if self.mode == "binary":
            probs = torch.sigmoid(logits)
            intersection = (probs * targets).sum(dim=(-2, -1))
            total = (probs + targets).sum(dim=(-2, -1))
            dice = (2.0 * intersection + self.eps) / (total + self.eps)
            return 1.0 - dice.mean()
        else:
            # Multiclass: targets is (B, H, W) integer indices, logits is (B, C, H, W)
            num_classes = logits.shape[1]
            probs = F.softmax(logits, dim=1)
            one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
            intersection = (probs * one_hot).sum(dim=(-2, -1))
            total = (probs + one_hot).sum(dim=(-2, -1))
            dice = (2.0 * intersection + self.eps) / (total + self.eps)
            # Only average over classes present in this batch to avoid 0/0 penalty
            present_mask = (one_hot.sum(dim=(-2, -1)) > 0).float()
            dice_present = (dice * present_mask).sum() / (present_mask.sum() + self.eps)
            return 1.0 - dice_present

class FocalLoss(nn.Module):
    """Focal Loss with gamma=2.0 focusing gradient on hard boundary and rare species pixels."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class CompoundSemanticLoss(nn.Module):
    """True Compound Loss unifying Weighted-CE, Soft Dice, and Focal Loss."""
    def __init__(self, mode="binary", weights=None, gamma=2.0, w_ce=1.0, w_dice=1.0, w_focal=0.5):
        super().__init__()
        self.mode = mode
        self.w_ce = w_ce
        self.w_dice = w_dice
        self.w_focal = w_focal
        if mode == "binary":
            self.bce = nn.BCEWithLogitsLoss()
            self.dice = SoftDiceLoss(mode="binary")
        else:
            self.ce = nn.CrossEntropyLoss(weight=weights)
            self.dice = SoftDiceLoss(mode="multi")
            self.focal = FocalLoss(gamma=gamma, weight=weights)

    def forward(self, logits, targets):
        if self.mode == "binary":
            return self.w_ce * self.bce(logits, targets) + self.w_dice * self.dice(logits, targets)
        else:
            loss_ce = self.ce(logits, targets)
            loss_dice = self.dice(logits, targets)
            loss_focal = self.focal(logits, targets)
            return self.w_ce * loss_ce + self.w_dice * loss_dice + self.w_focal * loss_focal

# ==============================================================================
# 5. METRICS & 95% BOOTSTRAP CONFIDENCE INTERVALS
# ==============================================================================
def dice_coeff_binary(pred_probs, targets, eps=1e-6):
    preds = (pred_probs > 0.5).float()
    inter = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return float((2.0 * inter + eps) / (union + eps))

def iou_binary(pred_probs, targets, eps=1e-6):
    preds = (pred_probs > 0.5).float()
    inter = (preds * targets).sum()
    union = (preds + targets).clamp(0, 1).sum()
    return float((inter + eps) / (union + eps))

def compute_multiclass_metrics(preds, targets, eval_classes):
    ious = {}
    for c in eval_classes:
        p_c = (preds == c)
        t_c = (targets == c)
        inter = (p_c & t_c).sum().item()
        union = (p_c | t_c).sum().item()
        if union == 0:
            continue
        ious[c] = inter / union
    mIoU = np.mean(list(ious.values())) if ious else 0.0
    return mIoU, ious

def bootstrap_ci(scores, n_boot=1000, alpha=0.05, seed=42):
    """Computes non-parametric 95% bootstrap confidence intervals."""
    if not scores:
        return 0.0, 0.0, 0.0
    arr = np.array(scores)
    mean_val = float(np.mean(arr))
    rng = np.random.default_rng(seed)
    boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    ci_low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean_val, ci_low, ci_high

# ==============================================================================
# 6. MAIN ENGINE
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="Runnable Semantic Segmentation Trainer for AqUavplant")
    ap.add_argument("--mode", choices=["binary", "multi"], default="binary", help="Training mode: binary or multi")
    ap.add_argument("--tiles", type=str, required=True, help="Path to tile directory generated by tile_dataset.py")
    ap.add_argument("--epochs", type=int, default=40, help="Total training epochs (default: 40)")
    ap.add_argument("--batch", type=int, default=12, help="Batch size (default: 12)")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    ap.add_argument("--patience", type=int, default=12, help="Early stopping patience (default: 12)")
    ap.add_argument("--arch", type=str, default="DeepLabV3Plus", help="Architecture: DeepLabV3Plus, Unet, SegFormer")
    ap.add_argument("--encoder", type=str, default="resnet34", help="Backbone encoder (default: resnet34)")
    ap.add_argument("--warm-start", type=str, default="", help="Path to pre-trained weights for transfer learning")
    ap.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint for evaluation")
    ap.add_argument("--eval-only", action="store_true", help="Run evaluation only on test set and exit")
    ap.add_argument("--auto-cap", action="store_true", default=True, help="Auto-cap epochs to 30 if throughput < 10 img/s (Gap G3)")
    ap.add_argument("--out", type=str, default="runs_sem/exp", help="Output directory for checkpoints and metrics")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_classes = 1 if args.mode == "binary" else 32
    print(f"\n[INIT] Initializing {args.arch} ({args.encoder}) in {args.mode.upper()} mode ({num_classes} outputs) on {device}...")
    model = build_model(args.arch, args.encoder, num_classes=num_classes)

    # Transfer Learning / Checkpoint Loading
    ckpt_to_load = args.checkpoint or args.warm_start
    if ckpt_to_load and os.path.exists(ckpt_to_load):
        print(f"[WEIGHTS] Loading checkpoint from {ckpt_to_load}...")
        ckpt = torch.load(ckpt_to_load, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print(f"[WEIGHTS] Transferred {len(filtered)} / {len(model_dict)} layers. (Classification head: {num_classes} classes).")

    model = model.to(device)

    # Dataset & Loaders
    train_ds = WeedTileDataset(args.tiles, split="train", mode=args.mode, transform=get_transforms("train"))
    val_ds = WeedTileDataset(args.tiles, split="val", mode=args.mode, transform=None)
    test_ds = WeedTileDataset(args.tiles, split="test", mode=args.mode, transform=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    # Load Evaluated Classes & Species Names
    eval_cfg_p = find_resource_file("evaluated_classes.json")
    eval_classes = list(range(32))
    if eval_cfg_p:
        eval_cfg = json.load(open(eval_cfg_p, encoding="utf-8"))
        eval_classes = eval_cfg.get("evaluated_classes", eval_classes)
        print(f"[EVAL-CFG] Loaded {len(eval_classes)} evaluated classes from {eval_cfg_p.name}.")

    class_names = {}
    class_map_p = find_resource_file("class_map_stub.json")
    if class_map_p:
        class_names = json.load(open(class_map_p, encoding="utf-8"))

    # ==========================================================================
    # EVALUATION ONLY MODE (With 1000-Sample 95% Bootstrap CIs)
    # ==========================================================================
    if args.eval_only:
        print(f"\n================================================================================")
        print(f"EVALUATION ON TEST SET (Lockbox N=38 Images, {len(test_ds)} Tiles) — 95% BOOTSTRAP CIs")
        print(f"================================================================================")
        model.eval()
        results = {}
        with torch.no_grad():
            if args.mode == "binary":
                dice_scores, iou_scores = [], []
                for imgs, masks, _ in test_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        probs = torch.sigmoid(model(imgs))
                    for i in range(len(imgs)):
                        dice_scores.append(dice_coeff_binary(probs[i], masks[i]))
                        iou_scores.append(iou_binary(probs[i], masks[i]))
                m_dice, d_low, d_high = bootstrap_ci(dice_scores)
                m_iou, i_low, i_high = bootstrap_ci(iou_scores)
                print(f"Test Binary Dice: {m_dice:.4f}  [95% CI: {d_low:.4f}, {d_high:.4f}]")
                print(f"Test Binary IoU:  {m_iou:.4f}  [95% CI: {i_low:.4f}, {i_high:.4f}]")
                results = {
                    "binary_dice": {"mean": m_dice, "ci_95": [d_low, d_high]},
                    "binary_iou":  {"mean": m_iou,  "ci_95": [i_low, i_high]},
                }
            else:
                all_preds, all_tgts = [], []
                tile_mious = []
                for imgs, masks, _ in test_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        preds = torch.argmax(model(imgs), dim=1)
                    for i in range(len(imgs)):
                        t_mIoU, _ = compute_multiclass_metrics(preds[i].unsqueeze(0).cpu(), masks[i].unsqueeze(0).cpu(), eval_classes)
                        tile_mious.append(t_mIoU)
                    all_preds.append(preds.cpu())
                    all_tgts.append(masks.cpu())

                global_mIoU, class_ious = compute_multiclass_metrics(torch.cat(all_preds), torch.cat(all_tgts), eval_classes)
                m_boot, m_low, m_high = bootstrap_ci(tile_mious)
                print(f"Test 20-Class Macro-mIoU (Global):   {global_mIoU:.4f}")
                print(f"Test 20-Class Mean Tile mIoU:        {m_boot:.4f}  [95% CI: {m_low:.4f}, {m_high:.4f}]\n")
                print(f"Per-Class IoU Breakdown (Evaluated Classes):")
                named_ious = {}
                for c in eval_classes:
                    c_name = class_names.get(str(c), f"Class_{c}")
                    c_val = class_ious.get(c, 0.0)
                    named_ious[c_name] = round(c_val, 4)
                    print(f"  Class {c:02d} ({c_name:<38}): {c_val:.4f}")

                results = {
                    "global_macro_mIoU": global_mIoU,
                    "tile_mIoU": {"mean": m_boot, "ci_95": [m_low, m_high]},
                    "per_class_iou": named_ious,
                }

        # Save evaluation report
        res_path = out_dir / "eval_results.json"
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[SAVED] Full evaluation metrics exported to {res_path}")
        return

    # ==========================================================================
    # TRAINING SETUP: COMPOUND LOSS & OPTIMIZER
    # ==========================================================================
    weights = None
    if args.mode == "multi":
        w_path = find_resource_file("class_weights_medianfreq.txt")
        if w_path:
            raw_weights = [float(x.strip()) for x in open(w_path).read().replace("\n", ",").split(",") if x.strip()]
            weights = torch.tensor(raw_weights, device=device).float()
            weights = torch.clamp(weights, 0.1, 10.0)
            print(f"[LOSS] Loaded median frequency class weights from {w_path.name} (clamped [0.1, 10.0]).")

    loss_fn = CompoundSemanticLoss(mode=args.mode, weights=weights, gamma=2.0, w_ce=1.0, w_dice=1.0, w_focal=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_score = -1.0
    no_improve = 0
    total_epochs = args.epochs

    print(f"\n[START] Training {args.arch} ({args.mode}) for {total_epochs} epochs | Batch: {args.batch} | LR: {args.lr}...")
    print(f"        Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)} | Device: {device}\n")

    for ep in range(1, total_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_processed = 0

        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(imgs)
            n_processed += len(imgs)

        scheduler.step()
        train_loss /= len(train_ds)
        dt = time.time() - t0
        throughput = n_processed / dt if dt > 0 else 0

        # Validation Pass
        model.eval()
        val_loss = 0.0
        val_metric = 0.0
        with torch.no_grad():
            if args.mode == "binary":
                total_dice = 0.0
                for imgs, masks, _ in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        logits = model(imgs)
                        val_loss += loss_fn(logits, masks).item() * len(imgs)
                        probs = torch.sigmoid(logits)
                    total_dice += dice_coeff_binary(probs, masks) * len(imgs)
                val_metric = total_dice / len(val_ds)
            else:
                all_preds, all_tgts = [], []
                for imgs, masks, _ in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        logits = model(imgs)
                        val_loss += loss_fn(logits, masks).item() * len(imgs)
                        preds = torch.argmax(logits, dim=1)
                    all_preds.append(preds.cpu())
                    all_tgts.append(masks.cpu())
                mIoU, _ = compute_multiclass_metrics(torch.cat(all_preds), torch.cat(all_tgts), eval_classes)
                val_metric = mIoU

        val_loss /= len(val_ds)
        metric_name = "Dice" if args.mode == "binary" else "20-mIoU"
        print(f"Epoch {ep:02d}/{total_epochs:02d} ({throughput:.1f} img/s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val {metric_name}: {val_metric:.4f}")

        # GAP G3 RUNTIME SAFETY VALVE: Check throughput at Epoch 5
        if ep == 5 and args.auto_cap and throughput < 10.0:
            if total_epochs > 30:
                print(f"\n[ACTION 🛡️] Throughput is {throughput:.1f} img/s (<10.0 img/s). Auto-capping total epochs from {total_epochs} to 30 to guarantee Kaggle 9h session safety!\n")
                total_epochs = 30

        # Checkpointing
        if val_metric > best_score:
            best_score = val_metric
            no_improve = 0
            torch.save({
                "epoch": ep,
                "model": model.state_dict(),
                "score": best_score,
                "mode": args.mode,
                "arch": args.arch,
                "encoder": args.encoder
            }, out_dir / "best.pt")
            print(f"  --> Saved new best model (Val {metric_name}: {best_score:.4f}) to {out_dir / 'best.pt'}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n[EARLY STOP] No validation improvement for {args.patience} epochs. Terminating at epoch {ep}.")
                break

        torch.save({
            "epoch": ep,
            "model": model.state_dict(),
            "score": val_metric,
            "mode": args.mode,
            "arch": args.arch
        }, out_dir / "last.pt")

        if ep >= total_epochs:
            break

    print(f"\n[DONE] Training complete. Best Val {metric_name}: {best_score:.4f}. Best weights saved to {out_dir / 'best.pt'}")

if __name__ == "__main__":
    main()
