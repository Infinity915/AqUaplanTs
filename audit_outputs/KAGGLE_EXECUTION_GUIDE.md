# AqUavplant — Kaggle Execution Guide (What / Where / Time)

Source: `audit_outputs/research_paper_plan.txt v3.2` | Split: `133 Train / 26 Val / 38 Test` | Audit: `9.6/10 GO`
Dataset: `D:\AqUavplant` 197 triplets (474 MB) + `audit_outputs/` as Kaggle Dataset.

## Platform map — what to use where

| Stage | Where | Runtime | Time |
|---|---|---|---|
| NB0 tiling + verify | Kaggle | CPU, GPU OFF | ~20 min |
| NB1 binary train | Kaggle (primary) | T4 GPU | ~3.5h |
| NB1 U-Net baseline | Colab (parallel) | T4 GPU | ~2.5h |
| NB2 multi train | Kaggle | T4 GPU | ~3.5h |
| NB3 eval + CIs + LOLO | Kaggle | CPU (or T4 <1h) | ~15 sec – 1h |
| Day 3 tables/figs | Local | CPU | 0 GPU burn |

> Do NOT use TPU (v5e-1/v6e-1) — `sem_train.py` + YOLO-seg need CUDA.
> Total: ~7h GPU + ~1h CPU. Fits Kaggle `9h/session, 30h/week`.

---

## NB0 — Preprocessing [Kaggle CPU, ~20 min]

```bash
pip install -q segmentation-models-pytorch albumentations opencv-python-headless

python audit_outputs/tile_dataset.py \
  --tile 640 --overlap 128 \
  --min-fg 0.01 --bg-keep 0.05 \
  --split-csv audit_outputs/per_image_inventory_with_split.csv
```

Verify (must pass before training):

```python
import json
meta = json.load(open("tiles/t640_o128_minfg0.01_bg0.05/tile_meta.json"))
assert meta["leakage_ok"] == True
print(meta["by_split"], meta["final"])
# expect ~3-4k tiles, leakage_ok=True
```

Gap G1 check:

```bash
python audit_outputs/export_yolo_labels.py \
  --tiles tiles/t640_o128_minfg0.01_bg0.05 --active binary
# inspect 3x Zinda_Park_1*.txt: <100KB each, no polygon explosion
```

## NB1 — Binary [Kaggle T4, ~3.5h]

```bash
# Primary
python audit_outputs/sem_train.py \
  --mode binary --arch DeepLabV3Plus --encoder resnet34 \
  --tiles tiles/t640_o128_minfg0.01_bg0.05 \
  --epochs 40 --batch 12 --lr 3e-4 \
  --out runs_sem/binary

# Decision rule at epoch 5 (auto in sem_train.py):
# >=12 img/s -> full 40ep (~2.8h) | 10-12 -> full 40ep (~3.4h) | <10 -> cap to 30ep
```

Parallel on Colab T4:

```bash
python audit_outputs/sem_train.py \
  --mode binary --arch Unet --encoder resnet34 \
  --tiles tiles/t640_o128_minfg0.01_bg0.05 \
  --epochs 30 --batch 12 --lr 3e-4 \
  --out runs_sem/unet_binary
```

Optional edge binary:

```bash
yolo segment train model=yolo11s-seg.pt data=audit_outputs/aqua.yaml \
  imgsz=640 batch=8 epochs=45 patience=12 optimizer=AdamW amp=True \
  mosaic=0.8 close_mosaic=10 copy_paste=0.3 overlap_mask=False \
  project=runs_yolo name=binary
```

## NB2 — Multiclass [Kaggle T4, ~3.5h]

```bash
# Primary (warm-started)
python audit_outputs/sem_train.py \
  --mode multi --arch DeepLabV3Plus --encoder resnet34 \
  --tiles tiles/t640_o128_minfg0.01_bg0.05 \
  --epochs 40 --batch 12 --lr 1e-4 \
  --warm-start runs_sem/binary/best.pt \
  --out runs_sem/multi

# Baseline
python audit_outputs/sem_train.py \
  --mode multi --arch Unet --encoder resnet34 \
  --tiles tiles/t640_o128_minfg0.01_bg0.05 \
  --epochs 30 --batch 12 --lr 1e-4 \
  --warm-start runs_sem/unet_binary/best.pt \
  --out runs_sem/unet_multi
```

Optional Path-B multi + rasterize:

```bash
python audit_outputs/export_yolo_labels.py --tiles tiles/t640_o128_minfg0.01_bg0.05 --active multi
yolo segment train model=runs_yolo/binary/weights/best.pt data=audit_outputs/aqua32.yaml \
  imgsz=640 batch=8 epochs=40 freeze=10 patience=12 optimizer=AdamW amp=True \
  copy_paste=0.5 project=runs_yolo name=multi
yolo segment predict model=runs_yolo/multi/weights/best.pt \
  source=tiles/t640_o128_minfg0.01_bg0.05/images \
  imgsz=640 conf=0.25 iou=0.85 agnostic_nms=True save_txt=True save_conf=False \
  project=runs_yolo name=predict
python audit_outputs/rasterize_yolo_to_semantic.py \
  --pred runs_yolo/predict/labels \
  --tiles tiles/t640_o128_minfg0.01_bg0.05 \
  --out runs_yolo/pred_semantic
```

## NB3 — Eval [CPU, ~15 sec]

```bash
python audit_outputs/sem_train.py \
  --mode multi --arch DeepLabV3Plus \
  --tiles tiles/t640_o128_minfg0.01_bg0.05 \
  --checkpoint runs_sem/multi/best.pt \
  --eval-only

python audit_outputs/sem_train.py \
  --mode multi --arch Unet \
  --tiles tiles/t640_o128_minfg0.01_bg0.05 \
  --checkpoint runs_sem/unet_multi/best.pt \
  --eval-only
```

You get: `Binary Dice [95% CI], Binary IoU [95% CI], 20-mIoU [95% CI], per-class IoU -> eval_results.json`
Targets: `Dice >0.88 vs 0.692, IoU >0.80 vs 0.561, mIoU >0.65 vs 0.441, LOLO F1 >52% vs 36.89%`.

## Day 3 — Paper [0 GPU]

Fill Table 1 (lockbox N=38), Table 2 (LOLO BAU Museum), Table 3 (A1-A4 ablation), Table 4 (FPS: DeepLab 42 / SegFormer 58.5 / YOLO 86 on T4) + Fig 2/4.
