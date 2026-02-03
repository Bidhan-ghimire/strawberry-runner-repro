# Table 6 and Table 7 Reproduction Report (Cross-Dataset Evaluation)

## What I reproduced
- Table 6: Cross-dataset detection results using box metrics (P, R, AP50, F1).
- Table 7: Cross-dataset segmentation results using mask metrics (P, R, AP50, F1).

All results in this document use seed = 42 to match the main Table 4 configuration.

## Environment
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU (16303 MiB)
- OS: Windows 11
- Python: 3.10.19
- PyTorch: 2.10.0+cu128
- Ultralytics: 8.4.9

## Datasets
Dataset source:
- Dryad dataset DOI: 10.5061/dryad.bzkh189nw

Training datasets (table columns):
- GI
- AI5
- AI10
- GI+AI5+AI10 (integrated dataset)

Validation datasets (table rows):
- GI
- AI5
- AI10

## Protocol (how the tables are computed)
For each pretrained model family:
- YOLOv8x-seg
- YOLOv11x-seg

For each training dataset (GI, AI5, AI10, GI+AI5+AI10):
1) Load the trained segmentation checkpoint:
   runs/segment/<model>_<trainset>_seed42/weights/best.pt
2) Validate the same checkpoint on each validation dataset (GI, AI5, AI10) using Ultralytics validation.
3) Record metrics:
   - Table 6 uses box metrics (B): precision(B), recall(B), mAP50(B)
   - Table 7 uses mask metrics (M): precision(M), recall(M), mAP50(M)
4) Compute F1 as:
   F1 = 2 * P * R / (P + R)
5) Compute the "Average" row as the mean across the three validation datasets (GI, AI5, AI10) for each training column.

Important note:
- Even though task=segment is used, the model outputs both boxes and masks. Therefore one best.pt provides both Table 6 (box) and Table 7 (mask) metrics.

## Script and outputs
Script:
- scripts/make_table6_7.py

Outputs:
- results/tables/table6_detection_seed42.csv
- results/tables/table7_segmentation_seed42.csv
- results/tables/table6_7_long_seed42.csv  (all cells in long format)

## How to run
From repo root:
python scripts/make_table6_7.py
