import os
import pandas as pd

RUN_DIRS = {
    "YOLOv8x-seg": {
        "GI":  "runs/segment/yolov8x_GI",
        "AI5": "runs/segment/yolov8x_AI5",
        "AI10":"runs/segment/yolov8x_AI10",
    },
    "YOLOv11x-seg": {
        "GI":  "runs/segment/yolo11x_GI",
        "AI5": "runs/segment/yolo11x_AI5",
        "AI10":"runs/segment/yolo11x_AI10",
    },
}

def f1(p, r):
    return 0.0 if (p + r) == 0 else (2 * p * r / (p + r))

def best_row(csv_path):
    df = pd.read_csv(csv_path)
    df["fitness"] = df["metrics/mAP50-95(B)"] + df["metrics/mAP50-95(M)"]
    return df.loc[df["fitness"].idxmax()]

datasets = ["GI", "AI5", "AI10"]
rows = []

for model_family in ["YOLOv8x-seg", "YOLOv11x-seg"]:
    for task in ["Detection", "Segmentation"]:
        row = {"Pre-trained Model": model_family, "Task": task}

        for ds in datasets:
            csv_path = os.path.join(RUN_DIRS[model_family][ds], "results.csv")
            b = best_row(csv_path)

            if task == "Detection":
                p = float(b["metrics/precision(B)"])
                r = float(b["metrics/recall(B)"])
                ap50 = float(b["metrics/mAP50(B)"])
            else:
                p = float(b["metrics/precision(M)"])
                r = float(b["metrics/recall(M)"])
                ap50 = float(b["metrics/mAP50(M)"])

            row[f"{ds}_F1"] = f1(p, r)
            row[f"{ds}_P"] = p
            row[f"{ds}_R"] = r
            row[f"{ds}_AP50"] = ap50

        rows.append(row)

df = pd.DataFrame(rows)
cols = ["Pre-trained Model", "Task"] + [f"{ds}_{m}" for ds in datasets for m in ["F1", "P", "R", "AP50"]]
df = df[cols].round(2)

os.makedirs("results/tables", exist_ok=True)
df.to_csv("results/tables/table4.csv", index=False)
print(df.to_string(index=False))
