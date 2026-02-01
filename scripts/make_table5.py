import glob
import os

import pandas as pd
from ultralytics import YOLO

# Settings (paper uses imgsz=320)
IMGSZ = 320
DEVICE = "0"     # "0" = first GPU, or "cpu"
WARMUP = 5
LIMIT = None     # set to 200 if you want a quick test

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

experiments = [
    # model_family, trained_on, weights_path, val_images_folder
    ("YOLOv8x-seg",  "GI",          "runs/segment/yolo8x_GI/weights/best.pt",          "data/datasets/GI/val/images"),
    ("YOLOv8x-seg",  "AI5",         "runs/segment/yolo8x_AI5/weights/best.pt",         "data/datasets/AI5/val/images"),
    ("YOLOv8x-seg",  "AI10",        "runs/segment/yolo8x_AI10/weights/best.pt",        "data/datasets/AI10/val/images"),
    ("YOLOv8x-seg",  "GI+AI5+AI10", "runs/segment/yolo8x_GI_AI5_AI10/weights/best.pt", "data/datasets/GI_AI5_AI10/val/images"),

    ("YOLOv11x-seg", "GI",          "runs/segment/yolo11x_GI/weights/best.pt",          "data/datasets/GI/val/images"),
    ("YOLOv11x-seg", "AI5",         "runs/segment/yolo11x_AI5/weights/best.pt",         "data/datasets/AI5/val/images"),
    ("YOLOv11x-seg", "AI10",        "runs/segment/yolo11x_AI10/weights/best.pt",        "data/datasets/AI10/val/images"),
    ("YOLOv11x-seg", "GI+AI5+AI10", "runs/segment/yolo11x_GI_AI5_AI10/weights/best.pt", "data/datasets/GI_AI5_AI10/val/images"),
]

rows = []

for model_family, trained_on, weights_path, val_dir in experiments:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    # collect images
    images = sorted(glob.glob(os.path.join(val_dir, "**", "*.*"), recursive=True))
    images = [p for p in images if p.lower().endswith(IMG_EXTS)]
    if not images:
        raise FileNotFoundError(f"No images found in: {val_dir}")

    model = YOLO(weights_path)

    # warmup (stabilizes timing)
    for _ in range(WARMUP):
        model.predict(images[0], imgsz=IMGSZ, device=DEVICE, verbose=False, save=False)

    pre = 0.0
    inf = 0.0
    post = 0.0
    n = 0

    for r in model.predict(images, imgsz=IMGSZ, device=DEVICE, stream=True, verbose=False, save=False):
        s = r.speed or {}
        pre += float(s.get("preprocess", 0.0))
        inf += float(s.get("inference", 0.0))
        post += float(s.get("postprocess", 0.0))
        n += 1

    rows.append({
        "pretrained_model": model_family,
        "trained_on": trained_on,
        "preprocess_ms": pre / n,
        "inference_ms": inf / n,
        "postprocess_ms": post / n,
        "total_ms": (pre + inf + post) / n,
        "n_images": n,
    })

df = pd.DataFrame(rows)
os.makedirs("results/tables", exist_ok=True)
df.to_csv("results/tables/table5_inference_time.csv", index=False)

table = df.pivot(index="pretrained_model", columns="trained_on", values="total_ms").round(2)
print("\nTable 5: total_ms per image (preprocess + inference + postprocess)\n")
print(table.to_markdown())
