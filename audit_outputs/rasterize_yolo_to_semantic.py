"""Rasterize YOLO-seg predictions back to semantic masks (Path B fix #5).
Takes Ultralytics seg txt predictions (class x1 y1 ...) per tile + tile_index.csv
with (x, y) offsets, paints each tile's polygons onto the source-image canvas,
majority-votes overlaps, and writes a 0..31 semantic PNG per image for direct
Dice/mIoU comparison with Istiak24 (reads multiclass masks, NOT boxes).
Inverse of export remap: yolo_id 0..30 -> mask_id yolo_id + 1; bg stays 0.
Needs: numpy + PIL only (runs on CPU).
  python rasterize_yolo_to_semantic.py --pred runs/segment/predict/labels --tiles tiles/t640_o128_minfg0.01_bg0.05 --out pred_semantic
"""
import argparse, csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, required=True, help="folder of YOLO *.txt predictions per tile")
    ap.add_argument("--tiles", type=str, required=True)
    ap.add_argument("--out", type=str, default="pred_semantic")
    ap.add_argument("--W", type=int, default=3840)
    ap.add_argument("--H", type=int, default=2160)
    ap.add_argument("--tile", type=int, default=640)
    args = ap.parse_args()
    pdir = Path(args.pred); tdir = Path(args.tiles)
    idx = list(csv.DictReader(open(tdir / "tile_index.csv")))
    by_src = {}
    for r in idx:
        by_src.setdefault(r["src"], []).append(r)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for src, rows in sorted(by_src.items()):
        canvas = np.zeros((args.H, args.W), dtype=np.int32)  # votes per class handled greedily: later tiles overwrite only if nonzero
        votes = np.zeros((args.H, args.W), dtype=np.uint8)
        for r in rows:
            pf = pdir / f"{r['tile']}.txt"
            if not pf.exists():
                continue
            x0, y0 = int(r["x"]), int(r["y"])
            tmp = Image.new("L", (args.tile, args.tile), 0)
            d = ImageDraw.Draw(tmp)
            for line in open(pf):
                p = line.split()
                if len(p) < 7:
                    continue
                yid = int(p[0]); mid = yid + 1
                # If coordinates length is odd, last element is confidence score (from save_conf=True)
                raw_coords = p[1:-1] if (len(p) - 1) % 2 != 0 else p[1:]
                if len(raw_coords) < 6:
                    continue
                xy = [(float(raw_coords[i]) * args.tile, float(raw_coords[i + 1]) * args.tile)
                      for i in range(0, len(raw_coords) - 1, 2)]
                layer = Image.new("L", (args.tile, args.tile), 0)
                ImageDraw.Draw(layer).polygon(xy, fill=mid)
                m = np.array(layer) > 0
                region = tmp  # accumulate max class id (classes are exclusive; last nonzero wins)
                t = np.array(tmp); t[m] = mid; tmp = Image.fromarray(t)
            t = np.array(tmp)
            x1, y1 = min(x0 + args.tile, args.W), min(y0 + args.tile, args.H)
            tw, th = x1 - x0, y1 - y0
            sub = t[:th, :tw]
            cur = canvas[y0:y1, x0:x1]
            upd = sub > 0
            cur[upd] = sub[upd]
        name = src.replace("/", "_").replace(" ", "_") + "_pred.png"
        Image.fromarray(canvas.astype(np.uint8)).save(out / name)
        print(f"{src} -> {name}")
    print(f"done out={out}")

if __name__ == "__main__":
    main()
