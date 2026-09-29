"""Tile 4K triplets into patches for training — v2 (council-patched).
Upgrades vs v1:
  - fg filtering + stratified bg keep (Kaggle survival: 11.8k -> ~3-4k tiles)
  - group-by-image split inheritance (leakage fix: tiles inherit image split)
  - hashes + per-split lists for reproducibility
  - SAHI inference note: train 640, infer 800/ov240-320

Usage (Kaggle NB0, GPU OFF):
  python tile_dataset.py --tile 640 --overlap 128 --min-fg 0.01 --bg-keep 0.05 --split-csv per_image_inventory_with_split.csv
  python tile_dataset.py --tile 640 --overlap 128 --limit 2   (dry-run)
Inference (SAHI, train 640):
  sliding window 800 ov 240-320 + multiscale 0.75/1.0/1.25, NMS agnostic IoU 0.85
Masks: PNG nearest-neighbor ONLY. Never JPEG-compress masks.
"""
import argparse, csv, json, hashlib, random
from pathlib import Path
from PIL import Image

ROOT = Path(r"D:\AqUavplant")
AUDIT = ROOT / "audit_outputs"
OUT_ROOT_DEFAULT = AUDIT / "tiles"

def sha1_of_file(p, n=65536):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()[:12]

def tile_one(w, h, tile, overlap):
    stride = tile - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            yield (x, y, min(x + tile, w), min(y + tile, h))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--min-fg", type=float, default=0.01,
                    help="min binary fg fraction to auto-keep (0-1). Below this -> bg pool.")
    ap.add_argument("--bg-keep", type=float, default=0.05,
                    help="fraction of bg-pool tiles to keep (stratified, default 5%%).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-csv", type=str,
                    default="per_image_inventory_with_split.csv",
                    help="CSV with image_id->split column (group-by-image, leakage fix).")
    ap.add_argument("--out", type=str, default=str(OUT_ROOT_DEFAULT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--format", type=str, default="png", choices=["png", "jpg"])
    args = ap.parse_args()
    random.seed(args.seed)

    inv_path = AUDIT / args.split_csv
    if not inv_path.exists():  # fallback to base inventory (split=unknown)
        inv_path = AUDIT / "per_image_inventory.csv"
    inv = list(csv.DictReader(open(inv_path)))
    if args.limit:
        inv = inv[:args.limit]
    id2split = {r["image_id"]: r.get("split", "unknown") for r in inv}

    out = Path(args.out) / f"t{args.tile}_o{args.overlap}_minfg{args.min_fg}_bg{args.bg_keep}"
    for d in ("images", "binary", "multiclass"):
        (out / d).mkdir(parents=True, exist_ok=True)

    import numpy as np
    kept_rows, bg_pool = [], []
    n_tiles_total = 0
    for r in inv:
        jpg = ROOT / r["jpg"]; b = ROOT / r["binaryMask"]; m = ROOT / r["multiclassMask"]
        im_j = Image.open(jpg); im_b = Image.open(b); im_m = Image.open(m)
        W, H = im_j.size
        base = r["image_id"].replace("/", "_").replace(" ", "_")
        split = id2split.get(r["image_id"], "unknown")
        for (x1, y1, x2, y2) in tile_one(W, H, args.tile, args.overlap):
            n_tiles_total += 1
            pj = im_j.crop((x1, y1, x2, y2))
            pb = im_b.crop((x1, y1, x2, y2))
            pm = im_m.crop((x1, y1, x2, y2))
            if pj.size != (args.tile, args.tile):  # pad edge tiles
                c = Image.new("RGB", (args.tile, args.tile)); c.paste(pj, (0, 0)); pj = c
                c = Image.new("L", (args.tile, args.tile)); c.paste(pb, (0, 0)); pb = c
                c = Image.new("L", (args.tile, args.tile)); c.paste(pm, (0, 0)); pm = c
            fg = float((np.array(pb) > 0).mean())
            name = f"{base}_x{x1}_y{y1}"
            row = {"tile": name, "src": r["image_id"], "location": r.get("location", ""),
                   "split": split, "x": x1, "y": y1, "fg_frac": round(fg, 5)}
            if fg >= args.min_fg:
                kept_rows.append((row, pj, pb, pm))
            else:
                bg_pool.append((row, pj, pb, pm))

    # stratified bg keep: deterministic sample
    n_bg_keep = int(len(bg_pool) * args.bg_keep)
    bg_keep_rows = random.sample(bg_pool, n_bg_keep) if bg_pool and n_bg_keep else []
    final = kept_rows + bg_keep_rows
    random.shuffle(final)

    for (row, pj, pb, pm) in final:
        name = row["tile"]
        if args.format == "jpg":
            pj.save(out / "images" / f"{name}.jpg", quality=95)
        else:
            pj.save(out / "images" / f"{name}.png")
        pb.save(out / "binary" / f"{name}.png")
        pm.save(out / "multiclass" / f"{name}.png")

    with open(out / "tile_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tile", "src", "location", "split", "x", "y", "fg_frac"])
        w.writeheader()
        for (row, _, _, _) in sorted(final, key=lambda t: t[0]["tile"]):
            w.writerow(row)
    # per-split lists + leakage check
    from collections import Counter
    by_split = Counter(r["split"] for (r, _, _, _) in final)
    # verify zero image overlap across splits
    split_images = {}
    for (r, _, _, _) in final:
        split_images.setdefault(r["split"], set()).add(r["src"])
    overlap = set()
    keys = list(split_images)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap |= split_images[keys[i]] & split_images[keys[j]]
    meta = {"tile": args.tile, "overlap": args.overlap, "min_fg": args.min_fg,
            "bg_keep": args.bg_keep, "seed": args.seed,
            "tiles_total_scanned": n_tiles_total, "fg_kept": len(kept_rows),
            "bg_pool": len(bg_pool), "bg_kept": len(bg_keep_rows),
            "final": len(final), "by_split": dict(by_split),
            "image_overlap_across_splits": sorted(overlap),
            "leakage_ok": len(overlap) == 0}
    with open(out / "tile_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"scanned={n_tiles_total} fg_kept={len(kept_rows)} bg_pool={len(bg_pool)} "
          f"bg_kept={len(bg_keep_rows)} final={len(final)} by_split={dict(by_split)} "
          f"leakage_ok={meta['leakage_ok']} out={out}")

if __name__ == "__main__":
    main()
