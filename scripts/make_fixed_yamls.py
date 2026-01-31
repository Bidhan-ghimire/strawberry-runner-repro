from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "data" / "datasets"
OUT_DIR = ROOT / "data" / "fixed_yamls"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mapping = {
    "GI": "GI",
    "AI5": "AI5",
    "AI10": "AI10",
    "GI_AI5_AI10": "GI_AI5_AI10",
}

for name, folder in mapping.items():
    ds_root = (DATASETS_ROOT / folder).resolve()
    src_yaml = ds_root / "data.yaml"
    if not src_yaml.exists():
        raise FileNotFoundError(f"Missing {src_yaml}. Check your unzip folder names.")

    d = yaml.safe_load(src_yaml.read_text())

    # Force absolute dataset root so Ultralytics always finds images/labels
    d["path"] = str(ds_root)
    d["train"] = "train/images"
    d["val"] = "val/images"

    out = OUT_DIR / f"{name}.yaml"
    out.write_text(yaml.safe_dump(d, sort_keys=False))
    print("Wrote", out)
