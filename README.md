# Strawberry Runner Reproduction (YOLOv8x-seg, YOLOv11x-seg)

This repository reproduces key results from a strawberry runner detection and segmentation study using Ultralytics YOLO segmentation models, and includes an extension using SAM-assisted pseudo-labeling.

## Main deliverables

- **Table 4** (in-domain performance on each dataset’s own validation split): `results/tables/table4_main.csv`
- **Table 6** (cross-dataset detection performance, box metrics): `results/tables/table6_detection_seed42.csv`
- **Table 7** (cross-dataset segmentation performance, mask metrics): `results/tables/table7_segmentation_seed42.csv`
- **Table 4 seed sweep summary** (run-to-run variability): `results/tables/table4_seed_summary.csv`

## Key reproduction outcomes (seed = 42)

- Integrated multi-platform training (**GI+AI5+AI10**) provides the most robust cross-dataset performance for both detection (box metrics) and segmentation (mask metrics).
- Integrated-column results are generally consistent with the paper’s trends, especially for segmentation.
- YOLOv11x can be sensitive to training dynamics (random seed and early stopping behavior) on smaller single-dataset runs; this can shrink the YOLOv11x advantage unless training is stabilized. A seed sweep is included.

---

## Extension: SAM-assisted pseudo-labeling 

I implemented a SAM-assisted pseudo-labeling workflow to test whether segmentation can be improved with less manual labeling.

### Definitions

- **L** = labeled subset of the original training split (example: 20%).
- **U** = unlabeled subset of the original training split (remaining images).
- The **official validation split** is used only for final evaluation of the segmentation student. It is **not** used for teacher training or pseudo-label creation (Option B).

### Runs (seed = 42, L = 20%)

- **Run 1 (L-only baseline)**: train YOLO segmentation on L only; evaluate on the official validation split.
- **Run 2 (fair pseudo-labeling)**: train a detector on L only (with an internal detector-val split taken from L), run the detector on U to get boxes, use SAM (`sam2_t`) to convert boxes to masks, then train a segmentation student on L + pseudo-labeled U.
- **Run 3 (full-data detector for diagnosis)**: train a stronger detector using all training labels (internal split from the training set), then pseudo-label U with the same SAM step and retrain the student. This run is for comparison/diagnosis and is not a reduced-label setting.

### Key findings (seed = 42)

- Run 2 performed worse than Run 1, suggesting pseudo-label noise can hurt when the detector is trained on a small labeled subset.
- Run 3 improved over Run 2, suggesting detector box quality is a major bottleneck; however, mask boundary quality remains a limitation.

Run summaries :
- `results/tables/sam_runs_run1_run2_run3_summary.csv`

---

## Environment

Tested environment:
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU (16303 MiB)
- OS: Windows 11
- Python: 3.10.19
- PyTorch: 2.10.0+cu128
- Ultralytics: 8.4.9

## Data

Dataset source:
- Dryad dataset DOI: `10.5061/dryad.bzkh189nw`

Expected dataset structure after extraction:
- `data/datasets/GI/train/images`, `data/datasets/GI/train/labels`
- `data/datasets/GI/val/images`,   `data/datasets/GI/val/labels`
- similarly for `AI5`, `AI10`, `GI_AI5_AI10`

## Installation (typical)

Create an environment and install dependencies:

```bash
pip install -U pip
pip install ultralytics pandas numpy pyyaml
```

## AI assistance
This repository was created with assistance from an AI tool (ChatGPT) for drafting code and proofreading documentation. All outputs were reviewed and edited by the author.




