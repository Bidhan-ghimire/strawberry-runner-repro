import os
import pandas as pd

RUNS_DIR = "runs/segment"
MAIN_SEED = 42  # choose the seed you want to present as the "paper-like" Table 4

def f1(p, r):
    return 0.0 if (p + r) == 0 else (2 * p * r / (p + r))

def parse_run_name(name):
    # expected patterns:
    # yolo8x_GI
    # yolo8x_GI_seed42
    parts = name.split("_")
    model = parts[0]  # yolo8x or yolo11x

    seed = 0
    if parts[-1].startswith("seed"):
        seed = int(parts[-1].replace("seed", ""))
        dataset = "_".join(parts[1:-1])
    else:
        dataset = "_".join(parts[1:])

    dataset = dataset.replace(",", "_")  # normalize yolo8x_GI_AI5,AI10 -> GI_AI5_AI10
    return model, dataset, seed

def best_metrics_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Ultralytics-like fitness: 0.1*mAP50 + 0.9*mAP50-95, summed over Box and Mask
    fitness = (
        0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"] +
        0.1 * df["metrics/mAP50(M)"] + 0.9 * df["metrics/mAP50-95(M)"]
    )
    i = fitness.idxmax()
    best = df.loc[i]

    det_p = float(best["metrics/precision(B)"])
    det_r = float(best["metrics/recall(B)"])
    det_ap50 = float(best["metrics/mAP50(B)"])

    seg_p = float(best["metrics/precision(M)"])
    seg_r = float(best["metrics/recall(M)"])
    seg_ap50 = float(best["metrics/mAP50(M)"])

    return {
        "epochs_completed": len(df),
        "best_epoch": int(best["epoch"]),
        "det_F1": f1(det_p, det_r),
        "det_P": det_p,
        "det_R": det_r,
        "det_AP50": det_ap50,
        "seg_F1": f1(seg_p, seg_r),
        "seg_P": seg_p,
        "seg_R": seg_r,
        "seg_AP50": seg_ap50,
    }

# 1) Collect all runs
rows = []
for run_name in sorted(os.listdir(RUNS_DIR)):
    run_dir = os.path.join(RUNS_DIR, run_name)
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.isdir(run_dir):
        continue
    if not os.path.exists(csv_path):
        continue

    model, dataset, seed = parse_run_name(run_name)
    m = best_metrics_from_csv(csv_path)

    rows.append({
        "run": run_name,
        "model": model,
        "dataset": dataset,
        "seed": seed,
        **m
    })

df_runs = pd.DataFrame(rows)

os.makedirs("results/tables", exist_ok=True)
df_runs.to_csv("results/tables/table4_seed_runs.csv", index=False)

# 2) Summary: mean/std across seeds for each model+dataset
summary = (
    df_runs
    .groupby(["model", "dataset"])
    .agg({
        "det_F1": ["mean", "std"],
        "det_AP50": ["mean", "std"],
        "seg_F1": ["mean", "std"],
        "seg_AP50": ["mean", "std"],
        "epochs_completed": ["mean", "min", "max"],
        "best_epoch": ["mean", "min", "max"],
    })
)

summary.columns = ["_".join(c).strip() for c in summary.columns.values]
summary = summary.reset_index().round(3)
summary.to_csv("results/tables/table4_seed_summary.csv", index=False)

# 3) Build a paper-like Table 4 for one chosen seed (MAIN_SEED)
df_main = df_runs[(df_runs["seed"] == MAIN_SEED)].copy()
# if MAIN_SEED does not exist for some combos, you can change MAIN_SEED to 0 or 42
# or filter manually; keep it simple here.

# Convert model tags to paper-like names
model_name_map = {"yolo8x": "YOLOv8x-seg", "yolo11x": "YOLOv11x-seg"}
df_main["Pre-trained Model"] = df_main["model"].map(model_name_map).fillna(df_main["model"])

# Make Table 4 rows: Detection and Segmentation per model
datasets_order = ["GI", "AI5", "AI10", "GI_AI5_AI10"]

table_rows = []
for mname in ["YOLOv8x-seg", "YOLOv11x-seg"]:
    for task in ["Detection", "Segmentation"]:
        row = {"Pre-trained Model": mname, "Task": task}
        for ds in datasets_order:
            sub = df_main[(df_main["Pre-trained Model"] == mname) & (df_main["dataset"] == ds)]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            if task == "Detection":
                row[f"{ds}_F1"] = r["det_F1"]
                row[f"{ds}_P"] = r["det_P"]
                row[f"{ds}_R"] = r["det_R"]
                row[f"{ds}_AP50"] = r["det_AP50"]
            else:
                row[f"{ds}_F1"] = r["seg_F1"]
                row[f"{ds}_P"] = r["seg_P"]
                row[f"{ds}_R"] = r["seg_R"]
                row[f"{ds}_AP50"] = r["seg_AP50"]
        table_rows.append(row)

df_table4 = pd.DataFrame(table_rows).round(3)
df_table4.to_csv("results/tables/table4_main.csv", index=False)

print("Saved:")
print("  results/tables/table4_seed_runs.csv")
print("  results/tables/table4_seed_summary.csv")
print("  results/tables/table4_main.csv")
print()
print(df_table4.to_string(index=False))
