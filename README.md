# Strawberry Runner Reproduction (YOLOv8x-seg, YOLOv11x-seg)

This repository reproduces key results from the strawberry runner detection and segmentation study using Ultralytics YOLO segmentation models.

Main deliverables:
- Table 4 reproduction (in-domain performance on each dataset’s own validation split)
- Table 6 reproduction (cross-dataset detection performance using box metrics)
- Table 7 reproduction (cross-dataset segmentation performance using mask metrics)
- Seed sweep documentation for Table 4 to show run-to-run variability

Key reproduction outcomes (seed = 42)
- Integrated multi-platform training (GI+AI5+AI10) provides the most robust cross-dataset performance for both detection (box metrics) and segmentation (mask metrics).
- Integrated-column results are close to the paper overall, especially for segmentation.
- YOLOv11x can be sensitive to training dynamics (random seed and early stopping behavior) on single datasets; this can shrink the YOLOv11x advantage unless training is stabilized. A seed sweep is included.

Important note about metrics
- For segmentation training (task=segment), a single model checkpoint (best.pt) produces both:
  - Box metrics (B): detection-style metrics on predicted bounding boxes
  - Mask metrics (M): segmentation metrics on predicted instance masks


 Extension: SAM-assisted pseudo-labeling (Option B, no official-val leakage)

In addition to reproducing the paper tables, I implemented a SAM-assisted pseudo-labeling workflow to explore the paper’s suggested future direction of using foundation segmentation models to expand training masks with less manual labeling.

Definitions:
- L = labeled subset of the original training split (example: 20%).
- U = unlabeled subset of the original training split (remaining 80%).
- Official validation split is used only for final student evaluation, not for teacher training or pseudo-label creation (Option B).

Runs (seed = 42):
- Run 1 (L-only baseline): Train YOLO segmentation using only L, evaluate on the official validation split.
- Run 2 (fair pseudo-labeling): Train a teacher detector only on L (with an internal teacher-val split taken from L), generate boxes on U, use SAM (sam2_t) to convert boxes→masks, then train a student segmentation model on L + pseudo(U).
- Run 3 (oracle teacher upper bound, diagnostic): Train a stronger teacher detector using full training labels (internal train/val split taken from the training split), then pseudo-label U and train the same student pipeline. This is NOT a label-efficiency setting; it is used to estimate an upper bound and diagnose whether box quality is the main bottleneck.

Key findings (L=20%, seed=42):
- Run 2 performed worse than the L-only baseline, indicating that naive SAM pseudo-labeling can introduce label noise when the teacher detector is trained on a small labeled subset.
- Run 3 improved substantially over Run 2, suggesting the bottleneck is largely teacher box quality; however, mask mAP50-95 did not improve versus the L-only baseline, indicating pseudo-mask boundary quality remains a limitation.

All run summaries are extracted from each run’s results.csv:
- results/tables/sam_runs_run1_run2_run3_summary.csv


## Repository layout

- configs/
  - training configs used in runs
- data/
  - fixed_yamls/          dataset YAMLs (GI.yaml, AI5.yaml, AI10.yaml, GI_AI5_AI10.yaml)
               
- scripts/
  - make_table4_seed_report.py
  - make_table6_7.py
- results/tables/
  - table4_main.csv
  - table4_seed_runs.csv
  - table4_seed_summary.csv
  - table6_detection_seed42.csv
  - table7_segmentation_seed42.csv
  - table6_7_long_seed42.csv
- docs/
  - table4_reproduction.md
  - table6_7_reproduction.md

## Environment

Tested environment
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU (16303 MiB)
- OS: Windows 11
- Python: 3.10.19
- PyTorch: 2.10.0+cu128
- Ultralytics: 8.4.9

## Data

Dataset source:
- Dryad dataset DOI: 10.5061/dryad.bzkh189nw

Expected dataset structure after extraction:
- data/datasets/GI/train/images, data/datasets/GI/train/labels
- data/datasets/GI/val/images,   data/datasets/GI/val/labels
(and similarly for AI5, AI10, GI_AI5_AI10)

## Installation (typical)

Create an environment and install dependencies:
- Python 3.10
- pip install ultralytics pandas numpy

Verify GPU:
- nvidia-smi
- python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

## Reproducing Table 4 (in-domain)

1) Train (paper-style settings; example for YOLOv11x on GI):

Windows cmd:
yolo segment train ^
  model=yolo11x-seg.pt ^
  data=data/fixed_yamls/GI.yaml ^
  imgsz=320 ^
  batch=32 ^
  optimizer=auto ^
  lr0=0.01 ^
  cos_lr=False ^
  weight_decay=0.0005 ^
  warmup_epochs=3 ^
  epochs=1000 ^
  patience=20 ^
  seed=42 ^
  deterministic=True ^
  project=runs/segment ^
  name=yolo11x_GI_seed42

Repeat for each model (yolo8x, yolo11x) and each training dataset (GI, AI5, AI10, GI_AI5_AI10).

2) Generate Table 4 and seed-sweep CSVs:
python scripts/make_table4_seed_report.py

Outputs:
- results/tables/table4_main.csv
- results/tables/table4_seed_runs.csv
- results/tables/table4_seed_summary.csv

Documentation:
- docs/table4_reproduction.md

## Reproducing Tables 6 and 7 (cross-dataset)

Tables 6 and 7 are computed by validating each trained model (trained on GI, AI5, AI10, GI+AI5+AI10) on each validation dataset (GI, AI5, AI10).

Seed policy:
- Table 6 and Table 7 are reported using seed=42 to match the main Table 4 configuration.

Generate Tables 6 and 7:
python scripts/make_table6_7.py

Outputs:
- results/tables/table6_detection_seed42.csv
- results/tables/table7_segmentation_seed42.csv
- results/tables/table6_7_long_seed42.csv

Documentation:
- docs/table6_7_reproduction.md

## Notes on reproducibility and early stopping

- The paper specifies early stopping patience=20.
- In practice, YOLOv11x can be seed-sensitive under early stopping on smaller single datasets.
- This repo includes a Table 4 seed sweep (0, 7, 17, 37, 42) to document variability under identical hyperparameters.

## AI assistance

This repository was created with assistance from an AI tool (ChatGPT) for drafting code and proofreading documentation. All outputs were reviewed and edited by the author.


