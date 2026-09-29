"""Semantic segmentation training skeleton — Path A (RECOMMENDED for paper).
Why: native PNG masks (no polygons), native WCE+Dice+Focal loss, 1:1 metric
match with Istiak24 (Dice/mIoU), ~3.5h on Kaggle T4 for 40 epochs.
Install: pip install segmentation-models-pytorch albumentations
Data: tiles/t640_o128_minfg0.01_bg0.05/{images,binary,multiclass} + tile_index.csv
  (split column = group-by-image inheritance; leakage_ok verified in tile_meta.json)
Models: binary -> smp.DeepLabV3Plus(resnet34) or SegFormer-B0 (hf hub);
  multi  -> same arch, 32 outputs, class weights from class_weights.json.
Loss binary: BCEWithLogits + Dice.  Loss multi: CrossEntropy(weight=medianfreq,
  clip 10x) + Dice + Focal. Metrics: Dice, IoU, mIoU-20-evaluated (see
  evaluated_classes.json), per-class IoU supp. Compare directly to Istiak24
  Tables 3/4 (same metric family — no rasterizer needed).
Suggested: imgsz 640 (train) / SAHI-800 infer, batch 8-12, AdamW 3e-4 cosine,
Full runnable trainer implemented in sem_train.py (supports DeepLabV3+ and same-split U-Net baseline);
this file is the pinned architectural specification.
"""
BINARY = {"arch": "DeepLabV3Plus-resnet34 | SegFormer-B0", "in": 3, "out": 1,
          "loss": "BCEWithLogits + Dice", "epochs": 40, "imgsz": 640,
          "batch": 12, "lr": 3e-4, "splits": {"train": 133, "val": 26, "test": 38}}
MULTI = {"arch": "same as binary", "in": 3, "out": 32,
         "loss": "CrossEntropy(medianfreq clip10x) + Dice + Focal",
         "epochs": 40, "warm_start": "best_binary.pt",
         "evaluate_on": "20 evaluated classes (train+test coverage); "
                        "12 rare_train_only excluded from mean, reported separately"}
