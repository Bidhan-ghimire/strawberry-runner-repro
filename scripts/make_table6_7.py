import multiprocessing as mp
import os
import pandas as pd
from ultralytics import YOLO

IMGSZ = 320
DEVICE = "0"  # GPU 0; use "cpu" if needed

VAL_YAML = {
    "GI":  r"data\fixed_yamls\GI.yaml",
    "AI5": r"data\fixed_yamls\AI5.yaml",
    "AI10": r"data\fixed_yamls\AI10.yaml",
}

TRAINSETS = ["GI", "AI5", "AI10", "GI_AI5_AI10"]
TRAINSETS_DISPLAY = {
    "GI": "GI",
    "AI5": "AI5",
    "AI10": "AI10",
    "GI_AI5_AI10": "GI+AI5+AI10",
}

WEIGHTS = {
    "YOLOv8x-seg": {
        "GI": r"runs\segment\yolo8x_GI_seed42\weights\best.pt",
        "AI5": r"runs\segment\yolo8x_AI5_seed42\weights\best.pt",
        "AI10": r"runs\segment\yolo8x_AI10_seed42\weights\best.pt",
        "GI_AI5_AI10": r"runs\segment\yolo8x_GI_AI5_AI10_seed42\weights\best.pt",
    },
    "YOLOv11x-seg": {
        "GI": r"runs\segment\yolo11x_GI_seed42\weights\best.pt",
        "AI5": r"runs\segment\yolo11x_AI5_seed42\weights\best.pt",
        "AI10": r"runs\segment\yolo11x_AI10_seed42\weights\best.pt",
        "GI_AI5_AI10": r"runs\segment\yolo11x_GI_AI5_AI10_seed42\weights\best.pt",
    },
}

def F1(p, r):
    if (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)

def make_table(df_long, metric_prefix):
    out = []

    for model_name in ["YOLOv8x-seg", "YOLOv11x-seg"]:
        avg_rows = []
        for trained_on in TRAINSETS:
            sub = df_long[(df_long["model"] == model_name) & (df_long["trained_on"] == trained_on)]
            avg_rows.append({
                "model": model_name,
                "trained_on": trained_on,
                "validated_on": "Average",
                metric_prefix + "F1": sub[metric_prefix + "F1"].mean(),
                metric_prefix + "P": sub[metric_prefix + "P"].mean(),
                metric_prefix + "R": sub[metric_prefix + "R"].mean(),
                metric_prefix + "AP50": sub[metric_prefix + "AP50"].mean(),
            })

        df_all = pd.concat([df_long, pd.DataFrame(avg_rows)], ignore_index=True)

        for valset in ["GI", "AI5", "AI10", "Average"]:
            row = {"Pre-trained Model": model_name, "Validated on": valset}

            for trained_on in TRAINSETS:
                disp = TRAINSETS_DISPLAY[trained_on]
                r = df_all[
                    (df_all["model"] == model_name) &
                    (df_all["trained_on"] == trained_on) &
                    (df_all["validated_on"] == valset)
                ].iloc[0]

                row[f"{disp}_F1"] = r[metric_prefix + "F1"]
                row[f"{disp}_P"] = r[metric_prefix + "P"]
                row[f"{disp}_R"] = r[metric_prefix + "R"]
                row[f"{disp}_AP50"] = r[metric_prefix + "AP50"]

            out.append(row)

    return pd.DataFrame(out).round(3)

def main():
    rows = []
    for model_name in ["YOLOv8x-seg", "YOLOv11x-seg"]:
        for trained_on in TRAINSETS:
            w = WEIGHTS[model_name][trained_on]
            y = YOLO(w)

            for valset in ["GI", "AI5", "AI10"]:
                v = y.val(
                    data=VAL_YAML[valset],
                    imgsz=IMGSZ,
                    split="val",
                    device=DEVICE,
                    verbose=False
                )
                d = v.results_dict

                box_p = float(d["metrics/precision(B)"])
                box_r = float(d["metrics/recall(B)"])
                box_ap50 = float(d["metrics/mAP50(B)"])

                mask_p = float(d["metrics/precision(M)"])
                mask_r = float(d["metrics/recall(M)"])
                mask_ap50 = float(d["metrics/mAP50(M)"])

                rows.append({
                    "model": model_name,
                    "trained_on": trained_on,
                    "validated_on": valset,

                    "box_F1": F1(box_p, box_r),
                    "box_P": box_p,
                    "box_R": box_r,
                    "box_AP50": box_ap50,

                    "mask_F1": F1(mask_p, mask_r),
                    "mask_P": mask_p,
                    "mask_R": mask_r,
                    "mask_AP50": mask_ap50,
                })

    df_long = pd.DataFrame(rows)

    os.makedirs(r"results\tables", exist_ok=True)
    df_long.to_csv(r"results\tables\table6_7_long_seed42.csv", index=False)

    table6 = make_table(df_long, "box_")
    table7 = make_table(df_long, "mask_")

    table6.to_csv(r"results\tables\table6_detection_seed42.csv", index=False)
    table7.to_csv(r"results\tables\table7_segmentation_seed42.csv", index=False)

    print("\nTable 6 saved: results/tables/table6_detection_seed42.csv\n")
    print(table6.to_string(index=False))

    print("\nTable 7 saved: results/tables/table7_segmentation_seed42.csv\n")
    print(table7.to_string(index=False))

if __name__ == "__main__":
    mp.freeze_support()
    main()
