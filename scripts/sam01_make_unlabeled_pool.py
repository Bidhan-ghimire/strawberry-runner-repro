import random
import shutil
from pathlib import Path

# EDIT THESE
SRC = Path(r"data\datasets\GI_AI5_AI10")          # your existing extracted dataset
OUT = Path(r"data\ssl\GI_AI5_AI10_20pct_seed42")  # new experiment folder
LABELED_FRACTION = 0.20
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

random.seed(SEED)

# Source folders
src_train_img = SRC / "train" / "images"
src_train_lbl = SRC / "train" / "labels"
src_val_img   = SRC / "val" / "images"
src_val_lbl   = SRC / "val" / "labels"

# Output folders
labeled_train_img = OUT / "labeled" / "train" / "images"
labeled_train_lbl = OUT / "labeled" / "train" / "labels"
unlab_img         = OUT / "unlabeled" / "images"
val_img           = OUT / "val" / "images"
val_lbl           = OUT / "val" / "labels"

for p in [labeled_train_img, labeled_train_lbl, unlab_img, val_img, val_lbl]:
    p.mkdir(parents=True, exist_ok=True)

# List train images
train_images = [p for p in src_train_img.iterdir() if p.suffix.lower() in IMG_EXTS]
train_images.sort()
random.shuffle(train_images)

n_labeled = int(len(train_images) * LABELED_FRACTION)
labeled_set = set(train_images[:n_labeled])

# Copy train images into labeled vs unlabeled
for img_path in train_images:
    if img_path in labeled_set:
        shutil.copy2(img_path, labeled_train_img / img_path.name)
        lbl_path = src_train_lbl / (img_path.stem + ".txt")
        shutil.copy2(lbl_path, labeled_train_lbl / lbl_path.name)
    else:
        shutil.copy2(img_path, unlab_img / img_path.name)

# Copy full validation split unchanged
val_images = [p for p in src_val_img.iterdir() if p.suffix.lower() in IMG_EXTS]
for img_path in val_images:
    shutil.copy2(img_path, val_img / img_path.name)
    lbl_path = src_val_lbl / (img_path.stem + ".txt")
    shutil.copy2(lbl_path, val_lbl / lbl_path.name)

print("DONE")
print("Labeled train images:", n_labeled)
print("Unlabeled train images:", len(train_images) - n_labeled)
print("Val images:", len(val_images))
print("Output:", OUT)
