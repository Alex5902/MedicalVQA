#!/bin/bash
#SBATCH --job-name=blip2_lora_seq_ft  # Updated job name
#SBATCH --partition=48-4,48-6
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00 # Adjust time per stage if needed
#SBATCH --output=logs/lora_ft_seq_%A_stage%x.out # %x for array job stage index if used
#SBATCH --error=logs/lora_ft_seq_%A_stage%x.err

# --- Environment Setup ---
echo "Loading modules and activating Conda environment..."
. /usr/share/Modules/init/profile.sh
module purge
module load cuda/11.8 
echo "LD_LIBRARY_PATH after module load: $LD_LIBRARY_PATH"

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH after adding system driver path: $LD_LIBRARY_PATH"

source ~/miniconda3/etc/profile.d/conda.sh
CONDA_ENV_NAME="blip2_lavis_env" # Make sure this is your correct PEFT environment
conda activate $CONDA_ENV_NAME
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX: $CONDA_PREFIX"

# echo "--- Environment Check inside SLURM Job ---"
# echo "Which Python: $(which python)"
# echo "Python Version: $(python --version)"
# echo "Which pip: $(which pip)"
# echo "pip list output:"
# pip list
# echo "Attempting to import peft in a simple Python command:"
# python -c "import peft; print('PEFT imported successfully in test command. Version:', peft.__version__)"
# echo "--- End of Environment Check ---"

PYTORCH_BUNDLED_CUDNN_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib"
if [ -d "$PYTORCH_BUNDLED_CUDNN_PATH" ]; then
  export LD_LIBRARY_PATH="$PYTORCH_BUNDLED_CUDNN_PATH:$LD_LIBRARY_PATH"
  echo "LD_LIBRARY_PATH after prepending PyTorch's bundled cuDNN path: $LD_LIBRARY_PATH"
else
  echo "WARNING: PyTorch bundled cuDNN path $PYTORCH_BUNDLED_CUDNN_PATH not found!"
  if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    echo "LD_LIBRARY_PATH after prepending Conda env lib (fallback): $LD_LIBRARY_PATH"
  fi
fi
unset BNB_CUDA_VERSION

# --- Define Paths ---
PROJECT_ROOT=$HOME/OmniMed
# IMPORTANT: Update to your new LoRA fine-tuning script
FINETUNE_SCRIPT_PATH=$PROJECT_ROOT/scripts/fine-tuning/lora/lora_finetune.py 
PYTHON_EXEC=$(which python) 

# Data paths
IMAGE_BASE_PATH=$PROJECT_ROOT/OmniMedVQA/OmniMedVQA/ 

# Stage 1 (Fundus) Data
TRAIN_JSON_S1=$PROJECT_ROOT/finetuning_datasets/OmniMedVQA_Splits_ImgLevel/Fundus_Photography/train.json
VAL_JSON_S1=$PROJECT_ROOT/finetuning_datasets/OmniMedVQA_Splits_ImgLevel/Fundus_Photography/val.json

# Stage 2 (X-ray) Data
TRAIN_JSON_S2=$PROJECT_ROOT/finetuning_datasets/OmniMedVQA_Splits_ImgLevel/X-Ray/train.json
VAL_JSON_S2=$PROJECT_ROOT/finetuning_datasets/OmniMedVQA_Splits_ImgLevel/X-Ray/val.json

# Stage 3 (Microscopy) Data
TRAIN_JSON_S3=$PROJECT_ROOT/finetuning_datasets/OmniMedVQA_Splits_ImgLevel/Microscopy_Images/train.json
VAL_JSON_S3=$PROJECT_ROOT/finetuning_datasets/OmniMedVQA_Splits_ImgLevel/Microscopy_Images/val.json

# --- LoRA Output Directories ---
# Base directory for all LoRA outputs for this sequential experiment
LORA_EXP_OUTPUT_BASE=$PROJECT_ROOT/lora_finetuned_adapters_seq
OUTPUT_DIR_S1_LORA=$LORA_EXP_OUTPUT_BASE/stage1_fundus_lora
OUTPUT_DIR_S2_LORA=$LORA_EXP_OUTPUT_BASE/stage2_xray_lora
OUTPUT_DIR_S3_LORA=$LORA_EXP_OUTPUT_BASE/stage3_microscopy_lora

# --- LoRA Adapter Configuration JSON files ---
# These files will tell Stage 2 and Stage 3 which previous adapters to load.
# You need to create these JSON files manually before running Stage 2 or 3.

# Example: previous_adapters_s2.json (for starting Stage 2)
# Create this file at $PROJECT_ROOT/scripts/fine-tuning/lora_configs/previous_adapters_s2.json
# with content like:
# [
#   {"path": "$LORA_EXP_OUTPUT_BASE/stage1_fundus_lora/lora_adapters/fundus_t5_final", "name": "fundus_t5", "component": "t5"},
#   {"path": "$LORA_EXP_OUTPUT_BASE/stage1_fundus_lora/lora_adapters/fundus_qformer_final", "name": "fundus_qformer", "component": "qformer"}
# ]
PREVIOUS_ADAPTERS_CONFIG_S2=$PROJECT_ROOT/scripts/fine-tuning/lora_configs/previous_adapters_s2.json

# Example: previous_adapters_s3.json (for starting Stage 3)
# Create this file at $PROJECT_ROOT/scripts/fine-tuning/lora_configs/previous_adapters_s3.json
# with content like:
# [
#   {"path": "$LORA_EXP_OUTPUT_BASE/stage1_fundus_lora/lora_adapters/fundus_t5_final", "name": "fundus_t5", "component": "t5"},
#   {"path": "$LORA_EXP_OUTPUT_BASE/stage1_fundus_lora/lora_adapters/fundus_qformer_final", "name": "fundus_qformer", "component": "qformer"},
#   {"path": "$LORA_EXP_OUTPUT_BASE/stage2_xray_lora/lora_adapters/xray_t5_final", "name": "xray_t5", "component": "t5"},
#   {"path": "$LORA_EXP_OUTPUT_BASE/stage2_xray_lora/lora_adapters/xray_qformer_final", "name": "xray_qformer", "component": "qformer"}
# ]
PREVIOUS_ADAPTERS_CONFIG_S3=$PROJECT_ROOT/scripts/fine-tuning/lora_configs/previous_adapters_s3.json


# --- LoRA Training Parameters ---
EPOCHS=5 # LoRA might need more epochs or different LR
BATCH_SIZE=1  # Per GPU
# LoRA often uses a higher learning rate
LEARNING_RATE_LORA=1e-4 
GRAD_ACCUM_STEPS=16 # Keep your gradient accumulation

# LoRA specific hyperparams (adjust these based on your findings/experiments)
LORA_T5_TARGET_MODULES="\
encoder.block.0.layer.0.SelfAttention.q \
encoder.block.0.layer.0.SelfAttention.v \
encoder.block.1.layer.0.SelfAttention.q \
encoder.block.1.layer.0.SelfAttention.v \
encoder.block.2.layer.0.SelfAttention.q \
encoder.block.2.layer.0.SelfAttention.v \
encoder.block.3.layer.0.SelfAttention.q \
encoder.block.3.layer.0.SelfAttention.v \
encoder.block.4.layer.0.SelfAttention.q \
encoder.block.4.layer.0.SelfAttention.v \
encoder.block.5.layer.0.SelfAttention.q \
encoder.block.5.layer.0.SelfAttention.v \
encoder.block.6.layer.0.SelfAttention.q \
encoder.block.6.layer.0.SelfAttention.v \
encoder.block.7.layer.0.SelfAttention.q \
encoder.block.7.layer.0.SelfAttention.v \
encoder.block.8.layer.0.SelfAttention.q \
encoder.block.8.layer.0.SelfAttention.v \
encoder.block.9.layer.0.SelfAttention.q \
encoder.block.9.layer.0.SelfAttention.v \
encoder.block.10.layer.0.SelfAttention.q \
encoder.block.10.layer.0.SelfAttention.v \
encoder.block.11.layer.0.SelfAttention.q \
encoder.block.11.layer.0.SelfAttention.v \
encoder.block.12.layer.0.SelfAttention.q \
encoder.block.12.layer.0.SelfAttention.v \
encoder.block.13.layer.0.SelfAttention.q \
encoder.block.13.layer.0.SelfAttention.v \
encoder.block.14.layer.0.SelfAttention.q \
encoder.block.14.layer.0.SelfAttention.v \
encoder.block.15.layer.0.SelfAttention.q \
encoder.block.15.layer.0.SelfAttention.v \
encoder.block.16.layer.0.SelfAttention.q \
encoder.block.16.layer.0.SelfAttention.v \
encoder.block.17.layer.0.SelfAttention.q \
encoder.block.17.layer.0.SelfAttention.v \
encoder.block.18.layer.0.SelfAttention.q \
encoder.block.18.layer.0.SelfAttention.v \
encoder.block.19.layer.0.SelfAttention.q \
encoder.block.19.layer.0.SelfAttention.v \
encoder.block.20.layer.0.SelfAttention.q \
encoder.block.20.layer.0.SelfAttention.v \
encoder.block.21.layer.0.SelfAttention.q \
encoder.block.21.layer.0.SelfAttention.v \
encoder.block.22.layer.0.SelfAttention.q \
encoder.block.22.layer.0.SelfAttention.v \
encoder.block.23.layer.0.SelfAttention.q \
encoder.block.23.layer.0.SelfAttention.v \
decoder.block.0.layer.0.SelfAttention.q \
decoder.block.0.layer.0.SelfAttention.v \
decoder.block.0.layer.1.EncDecAttention.q \
decoder.block.0.layer.1.EncDecAttention.v \
decoder.block.1.layer.0.SelfAttention.q \
decoder.block.1.layer.0.SelfAttention.v \
decoder.block.1.layer.1.EncDecAttention.q \
decoder.block.1.layer.1.EncDecAttention.v \
decoder.block.2.layer.0.SelfAttention.q \
decoder.block.2.layer.0.SelfAttention.v \
decoder.block.2.layer.1.EncDecAttention.q \
decoder.block.2.layer.1.EncDecAttention.v \
decoder.block.3.layer.0.SelfAttention.q \
decoder.block.3.layer.0.SelfAttention.v \
decoder.block.3.layer.1.EncDecAttention.q \
decoder.block.3.layer.1.EncDecAttention.v \
decoder.block.4.layer.0.SelfAttention.q \
decoder.block.4.layer.0.SelfAttention.v \
decoder.block.4.layer.1.EncDecAttention.q \
decoder.block.4.layer.1.EncDecAttention.v \
decoder.block.5.layer.0.SelfAttention.q \
decoder.block.5.layer.0.SelfAttention.v \
decoder.block.5.layer.1.EncDecAttention.q \
decoder.block.5.layer.1.EncDecAttention.v \
decoder.block.6.layer.0.SelfAttention.q \
decoder.block.6.layer.0.SelfAttention.v \
decoder.block.6.layer.1.EncDecAttention.q \
decoder.block.6.layer.1.EncDecAttention.v \
decoder.block.7.layer.0.SelfAttention.q \
decoder.block.7.layer.0.SelfAttention.v \
decoder.block.7.layer.1.EncDecAttention.q \
decoder.block.7.layer.1.EncDecAttention.v \
decoder.block.8.layer.0.SelfAttention.q \
decoder.block.8.layer.0.SelfAttention.v \
decoder.block.8.layer.1.EncDecAttention.q \
decoder.block.8.layer.1.EncDecAttention.v \
decoder.block.9.layer.0.SelfAttention.q \
decoder.block.9.layer.0.SelfAttention.v \
decoder.block.9.layer.1.EncDecAttention.q \
decoder.block.9.layer.1.EncDecAttention.v \
decoder.block.10.layer.0.SelfAttention.q \
decoder.block.10.layer.0.SelfAttention.v \
decoder.block.10.layer.1.EncDecAttention.q \
decoder.block.10.layer.1.EncDecAttention.v \
decoder.block.11.layer.0.SelfAttention.q \
decoder.block.11.layer.0.SelfAttention.v \
decoder.block.11.layer.1.EncDecAttention.q \
decoder.block.11.layer.1.EncDecAttention.v \
decoder.block.12.layer.0.SelfAttention.q \
decoder.block.12.layer.0.SelfAttention.v \
decoder.block.12.layer.1.EncDecAttention.q \
decoder.block.12.layer.1.EncDecAttention.v \
decoder.block.13.layer.0.SelfAttention.q \
decoder.block.13.layer.0.SelfAttention.v \
decoder.block.13.layer.1.EncDecAttention.q \
decoder.block.13.layer.1.EncDecAttention.v \
decoder.block.14.layer.0.SelfAttention.q \
decoder.block.14.layer.0.SelfAttention.v \
decoder.block.14.layer.1.EncDecAttention.q \
decoder.block.14.layer.1.EncDecAttention.v \
decoder.block.15.layer.0.SelfAttention.q \
decoder.block.15.layer.0.SelfAttention.v \
decoder.block.15.layer.1.EncDecAttention.q \
decoder.block.15.layer.1.EncDecAttention.v \
decoder.block.16.layer.0.SelfAttention.q \
decoder.block.16.layer.0.SelfAttention.v \
decoder.block.16.layer.1.EncDecAttention.q \
decoder.block.16.layer.1.EncDecAttention.v \
decoder.block.17.layer.0.SelfAttention.q \
decoder.block.17.layer.0.SelfAttention.v \
decoder.block.17.layer.1.EncDecAttention.q \
decoder.block.17.layer.1.EncDecAttention.v \
decoder.block.18.layer.0.SelfAttention.q \
decoder.block.18.layer.0.SelfAttention.v \
decoder.block.18.layer.1.EncDecAttention.q \
decoder.block.18.layer.1.EncDecAttention.v \
decoder.block.19.layer.0.SelfAttention.q \
decoder.block.19.layer.0.SelfAttention.v \
decoder.block.19.layer.1.EncDecAttention.q \
decoder.block.19.layer.1.EncDecAttention.v \
decoder.block.20.layer.0.SelfAttention.q \
decoder.block.20.layer.0.SelfAttention.v \
decoder.block.20.layer.1.EncDecAttention.q \
decoder.block.20.layer.1.EncDecAttention.v \
decoder.block.21.layer.0.SelfAttention.q \
decoder.block.21.layer.0.SelfAttention.v \
decoder.block.21.layer.1.EncDecAttention.q \
decoder.block.21.layer.1.EncDecAttention.v \
decoder.block.22.layer.0.SelfAttention.q \
decoder.block.22.layer.0.SelfAttention.v \
decoder.block.22.layer.1.EncDecAttention.q \
decoder.block.22.layer.1.EncDecAttention.v \
decoder.block.23.layer.0.SelfAttention.q \
decoder.block.23.layer.0.SelfAttention.v \
decoder.block.23.layer.1.EncDecAttention.q \
decoder.block.23.layer.1.EncDecAttention.v \
"
LORA_R_T5=16
LORA_ALPHA_T5=32
LORA_DROPOUT_T5=0.05
LORA_BIAS_T5="none"

LORA_QFORMER_TARGET_MODULES="\
bert.encoder.layer.0.attention.self.query \
bert.encoder.layer.0.attention.self.value \
bert.encoder.layer.0.crossattention.self.query \
bert.encoder.layer.0.crossattention.self.value \
bert.encoder.layer.1.attention.self.query \
bert.encoder.layer.1.attention.self.value \
bert.encoder.layer.2.attention.self.query \
bert.encoder.layer.2.attention.self.value \
bert.encoder.layer.2.crossattention.self.query \
bert.encoder.layer.2.crossattention.self.value \
bert.encoder.layer.3.attention.self.query \
bert.encoder.layer.3.attention.self.value \
bert.encoder.layer.4.attention.self.query \
bert.encoder.layer.4.attention.self.value \
bert.encoder.layer.4.crossattention.self.query \
bert.encoder.layer.4.crossattention.self.value \
bert.encoder.layer.5.attention.self.query \
bert.encoder.layer.5.attention.self.value \
bert.encoder.layer.6.attention.self.query \
bert.encoder.layer.6.attention.self.value \
bert.encoder.layer.6.crossattention.self.query \
bert.encoder.layer.6.crossattention.self.value \
bert.encoder.layer.7.attention.self.query \
bert.encoder.layer.7.attention.self.value \
bert.encoder.layer.8.attention.self.query \
bert.encoder.layer.8.attention.self.value \
bert.encoder.layer.8.crossattention.self.query \
bert.encoder.layer.8.crossattention.self.value \
bert.encoder.layer.9.attention.self.query \
bert.encoder.layer.9.attention.self.value \
bert.encoder.layer.10.attention.self.query \
bert.encoder.layer.10.attention.self.value \
bert.encoder.layer.10.crossattention.self.query \
bert.encoder.layer.10.crossattention.self.value \
bert.encoder.layer.11.attention.self.query \
bert.encoder.layer.11.attention.self.value \
"
LORA_R_QFORMER=8
LORA_ALPHA_QFORMER=16
LORA_DROPOUT_QFORMER=0.05
LORA_BIAS_QFORMER="none"


NUM_GPUS=1 
MASTER_PORT=29503 # Changed port slightly to avoid conflict if other jobs running

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Determine which stage to run (e.g., by commenting out others)
RUN_STAGE1=true
RUN_STAGE2=false
RUN_STAGE3=false

# ====================================================================
# STAGE 1: LoRA Fine-Tuning on Fundus Photography
# ====================================================================
if [ "$RUN_STAGE1" = true ]; then
    echo "===================================================================="
    echo "Starting LoRA Stage 1: Fine-Tuning on Fundus Photography"
    echo "Python Script: $FINETUNE_SCRIPT_PATH"
    echo "Train JSON: $TRAIN_JSON_S1"
    echo "Val JSON: $VAL_JSON_S1"
    echo "Output Dir: $OUTPUT_DIR_S1_LORA"
    echo "Epochs: $EPOCHS, Per-GPU Batch Size: $BATCH_SIZE, LR: $LEARNING_RATE_LORA"
    echo "LoRA T5 Targets: $LORA_T5_TARGET_MODULES, r=$LORA_R_T5, alpha=$LORA_ALPHA_T5"
    echo "LoRA QF Targets: $LORA_QFORMER_TARGET_MODULES, r=$LORA_R_QFORMER, alpha=$LORA_ALPHA_QFORMER"
    echo "===================================================================="

    mkdir -p "$OUTPUT_DIR_S1_LORA"
    # Ensure the lora_configs directory exists for later stages
    mkdir -p "$PROJECT_ROOT/scripts/fine-tuning/lora_configs/"


    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT "$FINETUNE_SCRIPT_PATH" \
        --base_model_name "blip2_t5" \
        --base_model_type "pretrain_flant5xl" \
        --train_json_path "$TRAIN_JSON_S1" \
        --val_json_path "$VAL_JSON_S1" \
        --image_base_path "$IMAGE_BASE_PATH" \
        --lora_t5_modules_to_save "lm_head" \
        --output_dir "$OUTPUT_DIR_S1_LORA" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
        --learning_rate $LEARNING_RATE_LORA \
        --max_grad_norm 1.0 \
        --log_interval 50 \
        --num_workers 4 \
        --ddp_find_unused_parameters \
        --lora_train_current_adapter_name "fundus" \
        --lora_t5_target_modules $LORA_T5_TARGET_MODULES \
        --lora_r_t5 $LORA_R_T5 \
        --lora_alpha_t5 $LORA_ALPHA_T5 \
        --lora_dropout_t5 $LORA_DROPOUT_T5 \
        --lora_bias_t5 $LORA_BIAS_T5 \
        --lora_qformer_target_modules $LORA_QFORMER_TARGET_MODULES \
        --lora_r_qformer $LORA_R_QFORMER \
        --lora_alpha_qformer $LORA_ALPHA_QFORMER \
        --lora_dropout_qformer $LORA_DROPOUT_QFORMER \
        --lora_bias_qformer $LORA_BIAS_QFORMER \
        --enable_nan_debugging_hooks \
        --print_model_structure
        # Add --no_amp if you want to disable AMP

    EXIT_CODE_S1=$?
    echo "Stage 1 LoRA fine-tuning script finished with exit code $EXIT_CODE_S1"
    if [ $EXIT_CODE_S1 -ne 0 ]; then
        echo "Error in Stage 1, exiting."
        exit $EXIT_CODE_S1
    fi
    # IMPORTANT: After Stage 1, manually create/update $PREVIOUS_ADAPTERS_CONFIG_S2
    # to point to the saved adapters from $OUTPUT_DIR_S1_LORA/lora_adapters/fundus_t5_final (and _qformer_final)
    echo "ACTION REQUIRED: Update $PREVIOUS_ADAPTERS_CONFIG_S2 with paths from $OUTPUT_DIR_S1_LORA"
fi

# ====================================================================
# STAGE 2: LoRA Fine-Tuning on X-Ray (from base + Fundus LoRA)
# ====================================================================
if [ "$RUN_STAGE2" = true ]; then
    echo "===================================================================="
    echo "Starting LoRA Stage 2: Fine-Tuning on X-Ray"
    echo "Loading Previous Adapters from: $PREVIOUS_ADAPTERS_CONFIG_S2"
    echo "Train JSON: $TRAIN_JSON_S2"
    echo "Val JSON: $VAL_JSON_S2"
    echo "Output Dir: $OUTPUT_DIR_S2_LORA"
    echo "===================================================================="
    
    if [ ! -f "$PREVIOUS_ADAPTERS_CONFIG_S2" ]; then
        echo "ERROR: Previous adapters config file for Stage 2 not found: $PREVIOUS_ADAPTERS_CONFIG_S2"
        exit 1
    fi
    mkdir -p "$OUTPUT_DIR_S2_LORA"

    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT "$FINETUNE_SCRIPT_PATH" \
        --base_model_name "blip2_t5" \
        --base_model_type "pretrain_flant5xl" \
        --train_json_path "$TRAIN_JSON_S2" \
        --val_json_path "$VAL_JSON_S2" \
        --image_base_path "$IMAGE_BASE_PATH" \
        --output_dir "$OUTPUT_DIR_S2_LORA" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
        --learning_rate $LEARNING_RATE_LORA \
        --max_grad_norm 1.0 \
        --log_interval 50 \
        --num_workers 4 \
        --ddp_find_unused_parameters \
        --lora_load_previous_adapters_config "$PREVIOUS_ADAPTERS_CONFIG_S2" \
        --lora_train_current_adapter_name "xray" \
        --lora_t5_target_modules $LORA_T5_TARGET_MODULES \
        --lora_r_t5 $LORA_R_T5 \
        --lora_alpha_t5 $LORA_ALPHA_T5 \
        --lora_dropout_t5 $LORA_DROPOUT_T5 \
        --lora_bias_t5 $LORA_BIAS_T5 \
        --lora_qformer_target_modules $LORA_QFORMER_TARGET_MODULES \
        --lora_r_qformer $LORA_R_QFORMER \
        --lora_alpha_qformer $LORA_ALPHA_QFORMER \
        --lora_dropout_qformer $LORA_DROPOUT_QFORMER \
        --lora_bias_qformer $LORA_BIAS_QFORMER \
        --enable_nan_debugging_hooks

    EXIT_CODE_S2=$?
    echo "Stage 2 LoRA fine-tuning script finished with exit code $EXIT_CODE_S2"
    if [ $EXIT_CODE_S2 -ne 0 ]; then
        echo "Error in Stage 2, exiting."
        exit $EXIT_CODE_S2
    fi
    # IMPORTANT: After Stage 2, manually create/update $PREVIOUS_ADAPTERS_CONFIG_S3
    echo "ACTION REQUIRED: Update $PREVIOUS_ADAPTERS_CONFIG_S3 with paths from $OUTPUT_DIR_S1_LORA and $OUTPUT_DIR_S2_LORA"
fi

# ====================================================================
# STAGE 3: LoRA Fine-Tuning on Microscopy (from base + Fundus & X-ray LoRAs)
# ====================================================================
if [ "$RUN_STAGE3" = true ]; then
    echo "===================================================================="
    echo "Starting LoRA Stage 3: Fine-Tuning on Microscopy"
    echo "Loading Previous Adapters from: $PREVIOUS_ADAPTERS_CONFIG_S3"
    echo "Train JSON: $TRAIN_JSON_S3"
    echo "Val JSON: $VAL_JSON_S3"
    echo "Output Dir: $OUTPUT_DIR_S3_LORA"
    echo "===================================================================="

    if [ ! -f "$PREVIOUS_ADAPTERS_CONFIG_S3" ]; then
        echo "ERROR: Previous adapters config file for Stage 3 not found: $PREVIOUS_ADAPTERS_CONFIG_S3"
        exit 1
    fi
    mkdir -p "$OUTPUT_DIR_S3_LORA"

    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT "$FINETUNE_SCRIPT_PATH" \
        --base_model_name "blip2_t5" \
        --base_model_type "pretrain_flant5xl" \
        --train_json_path "$TRAIN_JSON_S3" \
        --val_json_path "$VAL_JSON_S3" \
        --image_base_path "$IMAGE_BASE_PATH" \
        --output_dir "$OUTPUT_DIR_S3_LORA" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
        --learning_rate $LEARNING_RATE_LORA \
        --max_grad_norm 1.0 \
        --log_interval 50 \
        --num_workers 4 \
        --ddp_find_unused_parameters \
        --lora_load_previous_adapters_config "$PREVIOUS_ADAPTERS_CONFIG_S3" \
        --lora_train_current_adapter_name "microscopy" \
        --lora_t5_target_modules $LORA_T5_TARGET_MODULES \
        --lora_r_t5 $LORA_R_T5 \
        --lora_alpha_t5 $LORA_ALPHA_T5 \
        --lora_dropout_t5 $LORA_DROPOUT_T5 \
        --lora_bias_t5 $LORA_BIAS_T5 \
        --lora_qformer_target_modules $LORA_QFORMER_TARGET_MODULES \
        --lora_r_qformer $LORA_R_QFORMER \
        --lora_alpha_qformer $LORA_ALPHA_QFORMER \
        --lora_dropout_qformer $LORA_DROPOUT_QFORMER \
        --lora_bias_qformer $LORA_BIAS_QFORMER \
        --enable_nan_debugging_hooks

    EXIT_CODE_S3=$?
    echo "Stage 3 LoRA fine-tuning script finished with exit code $EXIT_CODE_S3"
    if [ $EXIT_CODE_S3 -ne 0 ]; then
        echo "Error in Stage 3, exiting."
        exit $EXIT_CODE_S3
    fi
fi

echo "All selected LoRA fine-tuning stages complete."
conda deactivate
echo "Conda environment deactivated."