$ErrorActionPreference = "Stop"

$baseCfg = "configs/base_train.yaml"

$models = @(
  @{ tag = "yolov8x-seg"; weights = "yolov8x-seg.pt" },
  @{ tag = "yolo11x-seg"; weights = "yolo11x-seg.pt" }
)

$datasets = @(
  @{ tag = "GI";          yaml = "data/fixed_yamls/GI.yaml" },
  @{ tag = "AI5";         yaml = "data/fixed_yamls/AI5.yaml" },
  @{ tag = "AI10";        yaml = "data/fixed_yamls/AI10.yaml" },
  @{ tag = "GI_AI5_AI10"; yaml = "data/fixed_yamls/GI_AI5_AI10.yaml" }  

# Pick any three seeds you want (excluding 0 and 42)
$seeds = @(7, 19, 37)

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

foreach ($m in $models) {
  foreach ($d in $datasets) {
    foreach ($s in $seeds) {
      $name = "{0}_{1}_seed{2}" -f $m.tag, $d.tag, $s
      Write-Host "Running: $name"

      # Run and also save a log file per experiment
      yolo cfg=$baseCfg model=$($m.weights) data=$($d.yaml) seed=$s name=$name 2>&1 |
        Tee-Object -FilePath ("logs\{0}.txt" -f $name)
    }
  }
}
