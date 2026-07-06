# 8 样本过拟合冒烟测试（Windows PowerShell）
# 用法: .\scripts\overfit_single.ps1
# 或先设置数据路径:
#   $env:TRAIN_CSV = ".\data_splits\train_split.csv"
#   $env:STL_ROOT_DIR = "D:\path\to\stl"
#   .\scripts\overfit_single.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

New-Item -ItemType Directory -Force -Path ".\logs\overfit_8", ".\checkpoints\overfit_8" | Out-Null

$env:PYTHONUNBUFFERED = "1"
$env:BACKBONE_TYPE = if ($env:BACKBONE_TYPE) { $env:BACKBONE_TYPE } else { "transolver" }
$env:OVERFIT_MODE = "1"
if (-not $env:OVERFIT_SUBSET_SIZE) { $env:OVERFIT_SUBSET_SIZE = "8" }
$env:NUM_WORKERS = if ($env:NUM_WORKERS) { $env:NUM_WORKERS } else { "0" }
$env:BATCH_SIZE = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { "8" }
$env:LEARNING_RATE = if ($env:LEARNING_RATE) { $env:LEARNING_RATE } else { "1e-4" }
$env:NUM_EPOCHS = if ($env:NUM_EPOCHS) { $env:NUM_EPOCHS } else { "500" }
$env:NUM_POINTS = if ($env:NUM_POINTS) { $env:NUM_POINTS } else { "2048" }
$env:USE_AMP = if ($env:USE_AMP) { $env:USE_AMP } else { "0" }
$env:ENABLE_POINT_CACHE = if ($env:ENABLE_POINT_CACHE) { $env:ENABLE_POINT_CACHE } else { "0" }
$env:CHECKPOINT_DIR = if ($env:CHECKPOINT_DIR) { $env:CHECKPOINT_DIR } else { ".\checkpoints\overfit_8" }
$env:LOG_DIR = if ($env:LOG_DIR) { $env:LOG_DIR } else { ".\logs\overfit_8" }
$env:VAL_CSV = ""
$env:LOG_INTERVAL = if ($env:LOG_INTERVAL) { $env:LOG_INTERVAL } else { "1" }

if (-not $env:TRAIN_CSV) {
    if (Test-Path ".\data_splits\train_split.csv") {
        $env:TRAIN_CSV = ".\data_splits\train_split.csv"
    } else {
        Write-Error "请设置 `$env:TRAIN_CSV 指向训练 CSV（需包含 Design 与 Average Cd 列）"
    }
}

Write-Host "[INFO] OVERFIT_MODE=1 过拟合测试 ($($env:OVERFIT_SUBSET_SIZE) 样本)"
Write-Host "[INFO] TRAIN_CSV=$($env:TRAIN_CSV)"
Write-Host "[INFO] BACKBONE_TYPE=$($env:BACKBONE_TYPE)"
Write-Host "[INFO] OVERFIT_SUBSET_SIZE=$($env:OVERFIT_SUBSET_SIZE)"
Write-Host "[INFO] LEARNING_RATE=$($env:LEARNING_RATE)"

python "$ProjectRoot\train.py" @args
