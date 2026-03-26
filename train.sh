#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=36G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../slurm/%A-%a.out

set -euo pipefail

export LD_LIBRARY_PATH="/home1/xiaoyuto/.conda/envs/limip_py37/lib:$LD_LIBRARY_PATH"
PYTHON=/home1/xiaoyuto/.conda/envs/limip_py37/bin/python

echo "========== BASIC JOB INFO =========="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "PWD: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<not set>}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-<not set>}"
echo "num epoch = 50, epoch size = 312, validation batch size = 16"
echo

echo "========== PYTHON ENV =========="
$PYTHON --version
$PYTHON -c "import sys; print('Python executable:', sys.executable)"
$PYTHON -c "import os; print('PATH:', os.environ.get('PATH', ''))"
echo

echo "========== NVIDIA-SMI =========="
nvidia-smi || echo "nvidia-smi not available"
echo

echo "========== PACKAGE CHECKS =========="
$PYTHON - <<'PY'
import importlib

pkgs = [
    "numpy",
    "torch",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
    "torch_cluster",
]
for pkg in pkgs:
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg}: OK | version={getattr(m, '__version__', 'unknown')}")
    except Exception as e:
        print(f"{pkg}: FAILED | {type(e).__name__}: {e}")
PY
echo

echo "========== TORCH / CUDA CHECK =========="
$PYTHON - <<'PY'
import os
import torch

print("torch version:", torch.__version__)
print("torch file:", torch.__file__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
print("torch.backends.cudnn.enabled:", torch.backends.cudnn.enabled)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name:", torch.cuda.get_device_name(i))
    x = torch.tensor([1.0, 2.0]).cuda()
    print("CUDA tensor test: OK | device =", x.device)
else:
    print("WARNING: CUDA is not available to PyTorch.")
PY
echo

echo "========== TRAIN.PY PRESENCE CHECK =========="
ls -l train.py
echo

echo "========== START TRAINING =========="
#$PYTHON -u train.py --g 0 --prob_seq setcover_densize_0.15 --data_path data/samples_2.0 
$PYTHON -u train.py \
  --g 0 --data_path data/samples_2.0 --number_of_epochs 50 --lambda_l 10\
  --prob_seq setcover_densize_0.05-setcover_densize_0.075-setcover_densize_0.1-setcover_densize_0.125-setcover_densize_0.15-setcover_densize_0.2