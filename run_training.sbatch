#!/bin/bash
#SBATCH --job-name=cifar-all
#SBATCH --output=cifar-all-%j.out
#SBATCH --error=cifar-all-%j.err
#SBATCH -o out4c.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:30:00
#SBATCH --mail-type=FAIL,END,BEGIN
#SBATCH --mail-user=tcarroll@uh.edu
# SBATCH --account=YOUR_ALLOCATION
 
set -euo pipefail
 
module load Miniforge3/py3.10
module load CUDA/12.4.0

#conda init
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/amcheng/Tcarroll/Athan_testing/VENV4
set -u


export TORCH_HOME="/project/amcheng/Tcarroll/Athan_testing/ThresholdOptimizerperScene/Cacheing/torch/"
export HF_HOME="/project/amcheng/Tcarroll/Athan_testing/ThresholdOptimizerperScene/Cacheing/huggingFace/"
mkdir -p "$TORCH_HOME" "$HF_HOME"
 
nvidia-smi

python -c "import torch, torchvision, timm; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('timm:', timm.__version__); assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name())"
 
# Download CIFAR-100 and create the fixed split.
python -m experiments.cifar100.prepare_data --download
 
# Train the initializer and all 24 cascade candidates.
python -m experiments.cifar100.train \
  --candidate all \
  --device cuda
 
# Train the ConvNeXt V2-L CIFAR-100 head.
python -m experiments.cifar100.train_detector \
  --device cuda
 
# Confirm that the final manifest includes every model.
python -m experiments.cifar100.build_training_manifest \
  --output-dir checkpoints/cifar100/training \
  --device-description "Sabine single-GPU training"