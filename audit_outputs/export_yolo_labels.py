"""Export YOLO-seg polygon labels from tiled PNG masks (Path B fix #1).
Reads:  tiles/<run>/{images,binary,multiclass,tile_index.csv}
Writes: tiles/<run>/labels_binary/*.txt, labels_multi/*.txt + tile_lists/{train,val,test}.txt
Remap (YOLO has NO background class):
  binary mask {0,1}      -> YOLO class {0: weed} (mask 0 = unannotated negative)
  multiclass mask 1..31  -> YOLO class 0..30 (yolo_id = mask_id - 1; mask 0 skipped)
Format per line: <cls> x1 y1 x2 y2 ... (normalized 0-1).
Contour: cv2.RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, eps=1.0, min-area=16px (P2-safe).
Empty tiles (no contour) -> empty .txt (valid YOLO negative) + logged.
Run on Kaggle NB0 AFTER tile_dataset.py (cv2 preinstalled with ultralytics).
  python export_yolo_labels.py --tiles tiles/t640_o128_minfg0.01_bg0.05
"""
import argparse, csv
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(r"D:\AqUavplant")

def mask_to_polys(mask_arr, min_area=16, eps=1.0):
    try:
        import cv2
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless", "--quiet"])
        import cv2
    polys = []
    for v in sorted(int(x) for x in np.unique(mask_arr) if int(x) != 0):
        binm = ((mask_arr == v).astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            approx = cv2.approxPolyDP(c, eps, True)
            if len(approx) < 3:
                continue
            polys.append((v, approx.reshape(-1, 2)))
    return polys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=str, required=True)
    ap.add_argument("--min-area", type=int, default=16)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--active", type=str, choices=["binary", "multi", "none"], default="binary",
                    help="which label set to link/copy to 'labels/' for immediate Ultralytics training")
    args = ap.parse_args()
    tdir = Path(args.tiles)
    idx = list(csv.DictReader(open(tdir / "tile_index.csv")))
    lb = tdir / "labels_binary"; lm = tdir / "labels_multi"; tl = tdir / "tile_lists"
    lb.mkdir(exist_ok=True); lm.mkdir(exist_ok=True); tl.mkdir(exist_ok=True)
    n_poly_b = n_poly_m = n_empty = 0
    lists = {"train": [], "val": [], "test": [], "unknown": []}
    for r in idx:
        tile = r["tile"]; split = r.get("split", "unknown")
        ext = ".jpg" if (tdir / "images" / f"{tile}.jpg").exists() else ".png"
        lists.setdefault(split, []).append(f"images/{tile}{ext}")
        for kind, mdir, ldir, remap in (
            ("b", tdir / "binary", lb, lambda v: 0),
            ("m", tdir / "multiclass", lm, lambda v: v - 1),
        ):
            mp = mdir / f"{tile}.png"
            a = np.array(Image.open(mp))
            H, W = a.shape[:2]
            polys = mask_to_polys(a, args.min_area, args.eps)
            lines = []
            for (v, pts) in polys:
                norm = " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in pts)
                lines.append(f"{remap(v)} {norm}")
            with open(ldir / f"{tile}.txt", "w") as f:
                f.write("\n".join(lines))
            if kind == "b":
                n_poly_b += len(lines)
            else:
                n_poly_m += len(lines)
            if not lines:
                n_empty += 1
    for s, paths in lists.items():
        if paths and s != "unknown":
            with open(tl / f"{s}.txt", "w") as f:
                f.write("\n".join(sorted(paths)))

    # Ultralytics strictly looks for a directory named 'labels'. Link or copy active set:
    if args.active != "none":
        target = lb if args.active == "binary" else lm
        labels_dir = tdir / "labels"
        if labels_dir.is_symlink() or labels_dir.is_file():
            labels_dir.unlink()
        elif labels_dir.exists():
            import shutil
            shutil.rmtree(labels_dir)
        try:
            labels_dir.symlink_to(target.name, target_is_directory=True)
        except Exception:
            import shutil
            shutil.copytree(target, labels_dir, dirs_exist_ok=True)
        print(f"Active Ultralytics labels linked to: {target.name}")

    print(f"tiles={len(idx)} poly_binary={n_poly_b} poly_multi={n_poly_m} "
          f"empty_tiles={n_empty} lists={ {s: len(p) for s, p in lists.items()} }")

if __name__ == "__main__":
    main()
