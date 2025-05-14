#!/bin/bash
#SBATCH --job-name=blip2_ft_s2_xray  
#SBATCH --partition=48-4,48-6
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/ft_s2_xray_%A.out   
#SBATCH --error=logs/ft_s2_xray_%A.err

# --- Environment Setup ---
echo "Loading modules and activating Conda environment..."
. /usr/share/Modules/init/profile.sh
module purge
module load cuda/11.8 # Loads CUDA Toolkit (adds /apps/cuda/cuda-11.8/lib64 to LD_LIBRARY_PATH)
echo "LD_LIBRARY_PATH after module load: $LD_LIBRARY_PATH"

# Add system driver path (for libcuda.so) - this should come AFTER PyTorch's cuDNN path if both have cuDNN
# but libcuda.so is usually not in the same place as libcudnn.so from PyTorch
# Let's keep it here for now, but the PyTorch cuDNN path will be prepended next.
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH after adding system driver path: $LD_LIBRARY_PATH"

source ~/miniconda3/etc/profile.d/conda.sh
CONDA_ENV_NAME="blip2_lavis_env"
conda activate $CONDA_ENV_NAME
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX: $CONDA_PREFIX"

# *** MOST IMPORTANT PART: Prepend PyTorch's own cuDNN library path ***
# Path identified from ldd output. Assuming Python 3.10.
# This path should contain the cuDNN 8.7.0 that PyTorch expects.
PYTORCH_BUNDLED_CUDNN_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib"

if [ -d "$PYTORCH_BUNDLED_CUDNN_PATH" ]; then
  export LD_LIBRARY_PATH="$PYTORCH_BUNDLED_CUDNN_PATH:$LD_LIBRARY_PATH"
  echo "LD_LIBRARY_PATH after prepending PyTorch's bundled cuDNN path: $LD_LIBRARY_PATH"
else
  echo "WARNING: PyTorch bundled cuDNN path $PYTORCH_BUNDLED_CUDNN_PATH not found!"
  # Fallback to general Conda lib, though we know it doesn't directly contain cuDNN from previous 'ls'
  if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    echo "LD_LIBRARY_PATH after prepending Conda env lib (fallback): $LD_LIBRARY_PATH"
  fi
fi

unset BNB_CUDA_VERSION

# --- Define Paths ---
PROJECT_ROOT=$HOME/OmniMed
FINETUNE_SCRIPT_PATH=$PROJECT_ROOT/scripts/fine-tuning/finetune_blip2.py
PYTHON_EXEC=$(which python) # Get python from activated env

# Data paths for Stage 1 (Fundus)
# TRAIN_JSON=$PROJECT_ROOT/finetuning_datasets/OmniMedVQA_Splits_ImgLevel/Fundus_Photography/train.json
# VAL_JSON=$PROJECT_ROOT/finetuning_datasets/OmniMedVQA_Splits_ImgLevel/Fundus_Photography/val.json
# Data paths for Stage 2 (X-ray)
TRAIN_JSON_STAGE2=$PROJECT_ROOT/finetuning_datasets/OmniMedVQA_Splits_ImgLevel/X-Ray/train.json
VAL_JSON_STAGE2=$PROJECT_ROOT/finetuning_datasets/OmniMedVQA_Splits_ImgLevel/X-Ray/val.json


IMAGE_BASE_PATH=$PROJECT_ROOT/OmniMedVQA/OmniMedVQA/ # Path to the top 'Images' dir

# Output directory for this stage's checkpoints
OUTPUT_DIR_STAGE1=$PROJECT_ROOT/finetuned_checkpoints/stage1_fundus_full_ft

STAGE1_MODEL_CHECKPOINT=$OUTPUT_DIR_STAGE1/final_model.pth # Using final_model.pth as requested

# Output directory for Stage 2 checkpoints (X-ray model, from Fundus)
OUTPUT_DIR_STAGE2=$PROJECT_ROOT/finetuned_checkpoints/stage2_xray_from_fundus_full_ft


# --- Training Parameters ---
EPOCHS=5
BATCH_SIZE=1  # This is now per GPU
LEARNING_RATE=1e-7    # Adjusted for OPT model
NUM_GPUS=1 
MASTER_PORT=29501

# echo "===================================================================="
# echo "Starting DDP Stage 1: Full Fine-Tuning on Fundus Photography"
# echo "Python Script: $FINETUNE_SCRIPT_PATH"
# echo "Train JSON: $TRAIN_JSON"
# echo "Val JSON: $VAL_JSON"
# echo "Image Base: $IMAGE_BASE_PATH"
# echo "Output Dir: $OUTPUT_DIR_STAGE1"
# echo "Epochs: $EPOCHS, Per-GPU Batch Size: $BATCH_SIZE, Global Batch Size: $(($BATCH_SIZE * $NUM_GPUS)), LR: $LEARNING_RATE"
# echo "Number of GPUs: $NUM_GPUS"
# echo "===================================================================="

echo "===================================================================="
echo "Starting DDP Stage 2: Full Fine-Tuning on X-Ray (from Fundus model)"
echo "Python Script: $FINETUNE_SCRIPT_PATH"
echo "Loading Stage 1 Model from: $STAGE1_MODEL_CHECKPOINT"
echo "Train JSON (Stage 2): $TRAIN_JSON_STAGE2"
echo "Val JSON (Stage 2): $VAL_JSON_STAGE2"
echo "Image Base: $IMAGE_BASE_PATH"
echo "Output Dir (Stage 2): $OUTPUT_DIR_STAGE2"
echo "Epochs: $EPOCHS, Per-GPU Batch Size: $BATCH_SIZE, Global Batch Size: $(($BATCH_SIZE * $NUM_GPUS * 16)), LR: $LEARNING_RATE" # Assuming grad_accum_steps=16
echo "Number of GPUs: $NUM_GPUS"
echo "Using MASTER_PORT: $MASTER_PORT" 
echo "===================================================================="

export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The LD_LIBRARY_PATH is set up earlier in the script.
# No need to modify it again here.

# --- Execute Fine-Tuning Script ---
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT "$FINETUNE_SCRIPT_PATH" \
    --base_model_name "blip2_t5" \
    --base_model_type "pretrain_flant5xl" \
    --load_checkpoint "$STAGE1_MODEL_CHECKPOINT" \
    --unfreeze_all \
    --train_json_path "$TRAIN_JSON_STAGE2" \
    --val_json_path "$VAL_JSON_STAGE2" \
    --image_base_path "$IMAGE_BASE_PATH" \
    --output_dir "$OUTPUT_DIR_STAGE2" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 16 \
    --learning_rate $LEARNING_RATE \
    --max_grad_norm 1.0 \
    --log_interval 10 \
    --num_workers 4
    # --no_amp

EXIT_CODE=$?
echo "Fine-tuning script finished with exit code $EXIT_CODE"
# ... (conda deactivate) ...