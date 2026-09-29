"""Runnable Semantic Segmentation Trainer — Path A (Primary Recommendation).
Usage on Kaggle T4 (install smp first: pip install segmentation-models-pytorch albumentations):
  # Stage 1: Binary training (40 epochs, ~3.5h GPU):
  python train_semantic.py --mode binary --tiles tiles/t640_o128_minfg0.01_bg0.05 --epochs 40 --batch 12 --out runs_sem/binary

  # Stage 2: Multiclass training (warm-start from binary, 40 epochs, ~3.5h GPU):
  python train_semantic.py --mode multi --tiles tiles/t640_o128_minfg0.01_bg0.05 --epochs 40 --batch 12 --warm-start runs_sem/binary/best.pt --out runs_sem/multi

  # Evaluation with 95% Bootstrap Confidence Intervals:
  python train_semantic.py --mode multi --tiles tiles/t640_o128_minfg0.01_bg0.05 --checkpoint runs_sem/multi/best.pt --eval-only
"""
import argparse, csv, json, os, time
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def get_transforms(split):
    try:
        import albumentations as A
        if split == "train":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.15, contrast=0.15, p=0.3),
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
        rows = list(csv.DictReader(open(idx_path)))
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
        img = np.array(Image.open(img_p).convert("RGB"))
        mask = np.array(Image.open(self.mask_dir / f"{name}.png"))

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # To tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.mode == "binary":
            mask = (torch.from_numpy(mask) > 0).float().unsqueeze(0)
        else:
            mask = torch.from_numpy(mask).long()
        return img, mask, r["src"]

def build_model(arch="DeepLabV3Plus", encoder="resnet34", num_classes=1):
    import segmentation_models_pytorch as smp
    if arch == "SegFormer":
        model = smp.Unet(encoder_name=encoder, encoder_weights="imagenet", classes=num_classes)
    else:
        model = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights="imagenet", classes=num_classes)
    return model

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

def bootstrap_ci(scores, n_boot=1000, alpha=0.05):
    if not scores:
        return 0.0, 0.0, 0.0
    arr = np.array(scores)
    mean_val = float(np.mean(arr))
    boot_means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    ci_low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean_val, ci_low, ci_high

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["binary", "multi"], default="binary")
    ap.add_argument("--tiles", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--arch", type=str, default="DeepLabV3Plus")
    ap.add_argument("--encoder", type=str, default="resnet34")
    ap.add_argument("--warm-start", type=str, default="")
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", type=str, default="runs_sem/exp")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    num_classes = 1 if args.mode == "binary" else 32
    model = build_model(args.arch, args.encoder, num_classes=num_classes)

    ckpt_to_load = args.checkpoint or args.warm_start
    if ckpt_to_load and os.path.exists(ckpt_to_load):
        print(f"Loading weights from {ckpt_to_load}...")
        ckpt = torch.load(ckpt_to_load, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print(f"Transferred {len(filtered)} / {len(model_dict)} layers.")

    model = model.to(device)

    train_ds = WeedTileDataset(args.tiles, split="train", mode=args.mode, transform=get_transforms("train"))
    val_ds = WeedTileDataset(args.tiles, split="val", mode=args.mode, transform=None)
    test_ds = WeedTileDataset(args.tiles, split="test", mode=args.mode, transform=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    eval_cfg_p = Path("audit_outputs/evaluated_classes.json")
    eval_classes = list(range(32))
    if eval_cfg_p.exists():
        eval_cfg = json.load(open(eval_cfg_p))
        eval_classes = eval_cfg.get("evaluated_classes", eval_classes)

    # Eval-only execution
    if args.eval_only:
        print(f"\n================ EVALUATION ON TEST SET (Lockbox 38 Images) ================")
        model.eval()
        scores = []
        with torch.no_grad():
            if args.mode == "binary":
                dice_scores, iou_scores = [], []
                for imgs, masks, _ in test_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    probs = torch.sigmoid(model(imgs))
                    for i in range(len(imgs)):
                        dice_scores.append(dice_coeff_binary(probs[i], masks[i]))
                        iou_scores.append(iou_binary(probs[i], masks[i]))
                m_dice, d_low, d_high = bootstrap_ci(dice_scores)
                m_iou, i_low, i_high = bootstrap_ci(iou_scores)
                print(f"Test Binary Dice: {m_dice:.4f} [95% CI: {d_low:.4f}, {d_high:.4f}]")
                print(f"Test Binary IoU:  {m_iou:.4f} [95% CI: {i_low:.4f}, {i_high:.4f}]")
            else:
                all_preds, all_tgts = [], []
                for imgs, masks, _ in test_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    preds = torch.argmax(model(imgs), dim=1)
                    all_preds.append(preds.cpu())
                    all_tgts.append(masks.cpu())
                mIoU, class_ious = compute_multiclass_metrics(torch.cat(all_preds), torch.cat(all_tgts), eval_classes)
                print(f"Test 20-Class mIoU: {mIoU:.4f}")
                print(f"Evaluated Class IoUs: { {c: round(v, 4) for c, v in class_ious.items()} }")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    if args.mode == "binary":
        bce = nn.BCEWithLogitsLoss()
        def loss_fn(pred, tgt):
            return bce(pred, tgt)
    else:
        w_path = Path("audit_outputs/class_weights_medianfreq.txt")
        weights = None
        if w_path.exists():
            weights = torch.tensor([float(x.strip()) for x in open(w_path) if x.strip()], device=device)
            weights = torch.clamp(weights, 0.1, 10.0)
        ce = nn.CrossEntropyLoss(weight=weights)
        def loss_fn(pred, tgt):
            return ce(pred, tgt)

    best_score = -1.0
    no_improve = 0

    print(f"Starting {args.mode} training for {args.epochs} epochs on {device}...")
    for ep in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
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
        scheduler.step()
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_metric = 0.0
        with torch.no_grad():
            if args.mode == "binary":
                total_dice = 0.0
                for imgs, masks, _ in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    logits = model(imgs)
                    val_loss += loss_fn(logits, masks).item() * len(imgs)
                    probs = torch.sigmoid(logits)
                    total_dice += dice_coeff_binary(probs, masks) * len(imgs)
                val_metric = total_dice / len(val_ds)
            else:
                all_preds, all_tgts = [], []
                for imgs, masks, _ in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    logits = model(imgs)
                    val_loss += loss_fn(logits, masks).item() * len(imgs)
                    preds = torch.argmax(logits, dim=1)
                    all_preds.append(preds.cpu())
                    all_tgts.append(masks.cpu())
                mIoU, _ = compute_multiclass_metrics(torch.cat(all_preds), torch.cat(all_tgts), eval_classes)
                val_metric = mIoU
        val_loss /= len(val_ds)

        metric_name = "Dice" if args.mode == "binary" else "20-mIoU"
        print(f"Epoch {ep:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val {metric_name}: {val_metric:.4f}")

        # Checkpointing
        if val_metric > best_score:
            best_score = val_metric
            no_improve = 0
            torch.save({"epoch": ep, "model": model.state_dict(), "score": best_score, "mode": args.mode}, out_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {ep} (patience={args.patience})")
                break
        torch.save({"epoch": ep, "model": model.state_dict(), "score": val_metric, "mode": args.mode}, out_dir / "last.pt")

    print(f"Training complete. Best Val {metric_name}: {best_score:.4f}. Saved to {out_dir / 'best.pt'}")

if __name__ == "__main__":
    main()
