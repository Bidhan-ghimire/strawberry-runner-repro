@echo off
setlocal EnableExtensions EnableDelayedExpansion

set SEEDS=7 17 37

REM Models
set MODELS=yolov8x-seg.pt yolo11x-seg.pt

REM Dataset YAML names
set DATASETS=GI AI5 AI10 GI_AI5_AI10

REM Shared training args
set IMGSZ=320
set BATCH=32
set LR0=0.01
set WD=0.0005
set WARMUP=3
set EPOCHS=1000
set PATIENCE=20

for %%S in (%SEEDS%) do (
  for %%M in (%MODELS%) do (
    for %%D in (%DATASETS%) do (

      REM Short tag for naming
      if "%%M"=="yolov8x-seg.pt" (set TAG=yolov8x) else (set TAG=yolo11x)

      echo.
      echo Running: !TAG!  dataset=%%D  seed=%%S

      yolo segment train ^
        model=%%M ^
        data=data/fixed_yamls/%%D.yaml ^
        imgsz=%IMGSZ% ^
        batch=%BATCH% ^
        optimizer=auto ^
        lr0=%LR0% ^
        lrf=%LRF% ^
        cos_lr=False ^
        weight_decay=%WD% ^
        warmup_epochs=%WARMUP% ^
        epochs=%EPOCHS% ^
        patience=%PATIENCE% ^
        seed=%%S ^
        project=runs/segment ^
        name=!TAG!_%%D_seed%%S

    )
  )
)

endlocal
