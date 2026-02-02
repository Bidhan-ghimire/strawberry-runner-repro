# Table 4 Reproduction Report

## What I reproduced
- Table 4 (Detection and Segmentation metrics) using Ultralytics YOLO segmentation training.

## Environment
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU, 16303MiB)
- Python: 3.10.19
- Torch: 2.10.0+cu128
- Ultralytics: 8.4.9

- OS: Windows 11


## Dataset splits used
- GI: train/val from Dryad release
- AI5: train/val from Dryad release
- AI10: train/val from Dryad release
- GI_AI5_AI10: train/val from Dryad release
- 
## Training settings (paper-stated)
- imgsz=320
- batch=32
- optimizer=auto
- lr0=0.01
- linear LR decay (cos_lr=false)
- weight_decay=0.0005
- warmup_epochs=3
- epochs up to 1000
- early stopping patience=20

## Seed runs
Seeds tested: 0(default), 7, 17, 37, 42

All per-seed results:
- results/tables/table4_seed_runs.csv

Summary (mean/std):
- results/tables/table4_seed_summary.csv

## Main Table 4 (single seed for paper-like comparison)
I report Table 4 using seed=42
- results/tables/table4_main.csv

Rationale:
- I present Table 4 using a single fixed seed (seed = 42) so that the main comparison is deterministic and easy for others to reproduce exactly. In addition, I report a seed sweep (0, 7, 17, 37, 42) and summary statistics to show how sensitive the results are to stochastic training effects when all other hyperparameters are held constant.
