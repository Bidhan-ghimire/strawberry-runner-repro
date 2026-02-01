import pandas as pd

runs = {
    "yolov8x_GI":  r"runs\segment\yolo8x_GI\results.csv",
    "yolov8x_AI5": r"runs\segment\yolo8x_AI5\results.csv",
    "yolov8x_AI10":r"runs\segment\yolovx_AI10\results.csv",

    "yolo11x_GI":  r"runs\segment\yolo11x_GI\results.csv",
    "yolo11x_AI5": r"runs\segment\yolo11x_AI5\results.csv",
    "yolo11x_AI10":r"runs\segment\yolo11x_AI10\results.csv",
}

for name, csv in runs.items():
    df = pd.read_csv(csv)

    # Use Ultralytics-like fitness (0.1*mAP50 + 0.9*mAP50-95) for both Box and Mask
    fitness = (
        0.1 * df["metrics/mAP50(B)"]     + 0.9 * df["metrics/mAP50-95(B)"] +
        0.1 * df["metrics/mAP50(M)"]     + 0.9 * df["metrics/mAP50-95(M)"]
    )

    best_i = fitness.idxmax()
    best = df.loc[best_i]
    last = df.iloc[-1]

    print("\n", name)
    print("epochs_completed:", len(df), " best_epoch:", int(best["epoch"]), " last_epoch:", int(last["epoch"]))
    print("best box AP50:", round(float(best["metrics/mAP50(B)"]), 3),
          " best mask AP50:", round(float(best["metrics/mAP50(M)"]), 3))
    print("last box AP50:", round(float(last["metrics/mAP50(B)"]), 3),
          " last mask AP50:", round(float(last["metrics/mAP50(M)"]), 3))
