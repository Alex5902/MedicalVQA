#!/bin/bash
#SBATCH --job-name=blip2_s3_micro_eval_array
#SBATCH --partition=48-4,48-6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G                 
#SBATCH --cpus-per-task=8         
#SBATCH --time=24:00:00 # Adjusted time, prefix scoring can be lengthy per modality
#SBATCH --output=logs/blip2_s3_micro_eval_%A_%a.out 
#SBATCH --error=logs/blip2_s3_micro_eval_%A_%a.err 
#SBATCH --array=2,4,6 

# --- Environment Setup ---
echo "Loading modules and activating Conda environment..."
. /usr/share/Modules/init/profile.sh
module purge
module load cuda/11.8 # Match your training environment
echo "LD_LIBRARY_PATH after module load: $LD_LIBRARY_PATH"

# Add system driver path (for libcuda.so)
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH after adding system driver path: $LD_LIBRARY_PATH"

source ~/miniconda3/etc/profile.d/conda.sh 

CONDA_ENV_NAME="blip2_lavis_env" # Same as your training env
conda activate $CONDA_ENV_NAME
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "ERROR: Failed to activate conda environment '$CONDA_ENV_NAME'"
  exit $EXIT_CODE
fi
echo "Successfully activated Conda environment: '$CONDA_ENV_NAME'"
echo "CONDA_PREFIX: $CONDA_PREFIX"

# Prepend PyTorch's own cuDNN library path (crucial!)
PYTHON_VERSION_DIR="python3.10" # Make sure this matches your Conda env's Python
PYTORCH_BUNDLED_CUDNN_PATH="$CONDA_PREFIX/lib/$PYTHON_VERSION_DIR/site-packages/nvidia/cudnn/lib"

if [ -d "$PYTORCH_BUNDLED_CUDNN_PATH" ]; then
  export LD_LIBRARY_PATH="$PYTORCH_BUNDLED_CUDNN_PATH:$LD_LIBRARY_PATH"
  echo "LD_LIBRARY_PATH after prepending PyTorch's bundled cuDNN path: $LD_LIBRARY_PATH"
else
  echo "WARNING: PyTorch bundled cuDNN path $PYTORCH_BUNDLED_CUDNN_PATH not found!"
  # Fallback, though less ideal
  if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    echo "LD_LIBRARY_PATH after prepending Conda env lib (fallback): $LD_LIBRARY_PATH"
  fi
fi
echo "Final LD_LIBRARY_PATH for Python script: $LD_LIBRARY_PATH"
echo "Python executable for script: $(which python)"

# --- Define Paths ---
PROJECT_ROOT=$HOME/OmniMed 
OMNI_ROOT=$PROJECT_ROOT/OmniMedVQA # Contains the combined_test.json
OMNI_IMAGES_ROOT=$PROJECT_ROOT/OmniMedVQA/OmniMedVQA # Contains the Images/ folder
ARENA_ROOT=$PROJECT_ROOT/Multi-Modality-Arena/MedicalEval 

PYTHON_EXEC=$(which python) # Use python from the activated Conda environment

# --- Configuration for Fine-Tuned BLIP-2 Run ---
# MODEL_TAG="blip2-flant5xl-ft-fundus-ep5" 
MODEL_TAG="blip2-flant5xl-ft-s2-xray"

# Path to your best fine-tuned model checkpoint
# FT_CHECKPOINT_PATH="$PROJECT_ROOT/finetuned_checkpoints/stage1_fundus_full_ft/best_model_epoch_5.pth" 
# FT_CHECKPOINT_PATH="$PROJECT_ROOT/finetuned_checkpoints/stage2_xray_from_fundus_full_ft/final_model.pth"
FT_CHECKPOINT_PATH="$PROJECT_ROOT/finetuned_checkpoints/stage3_microscopy_from_xray_full_ft/final_model.pth"

# INPUT_JSON_NAME: Use your test set JSON
# INPUT_JSON_NAME="../finetuning_datasets/OmniMedVQA_CombinedTest/combined_test.json" # For REAL images on test split
INPUT_JSON_NAME="../finetuning_datasets/OmniMedVQA_CombinedTest/combined_test_dummy.json" # For DUMMY images on test split

# RESULTS_DIR: Base directory for results. Orchestrator will make subdirs.
RESULTS_DIR_BASE="$PROJECT_ROOT/results_eval_s3_ft/dummy" # New base for fine-tuned eval results

# --- Driver Script ---
# Point to your NEW Python orchestrator script for evaluating fine-tuned models
PY="$PROJECT_ROOT/scripts/fine-tuning/eval_finetuned_blip2.py"

# --- Modality Setup ---
ALL_MODALITIES=(
    "CT(Computed Tomography)"
    "Dermoscopy"
    "Fundus Photography"
    "MR (Mag-netic Resonance Imaging)" 
    "Microscopy Images"
    "OCT (Optical Coherence Tomography" 
    "X-Ray"
    "ultrasound"
    "unknown"
)

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set. This script must be run as a job array."
  exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge ${#ALL_MODALITIES[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID is out of bounds for ALL_MODALITIES array (size ${#ALL_MODALITIES[@]})."
    exit 1
fi
CURRENT_MODALITY=${ALL_MODALITIES[$SLURM_ARRAY_TASK_ID]}

echo "===================================================================="
echo "Starting SLURM Job: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running evaluation for Fine-Tuned BLIP-2 Model Tag: $MODEL_TAG"
echo "Fine-tuned Checkpoint: $FT_CHECKPOINT_PATH"
echo "Processing Modality: '$CURRENT_MODALITY'"
echo "Using Input JSON (relative to OMNI_ROOT): $INPUT_JSON_NAME"
echo "Results Base Directory: $RESULTS_DIR_BASE"
echo "Python Orchestrator Script: $PY"
echo "Start Time: $(date)"
echo "===================================================================="

# --- Execute Orchestrator Script ---
"$PYTHON_EXEC" "$PY" \
     --omni_root       "$OMNI_ROOT" \
     --arena_root      "$ARENA_ROOT" \
     --model_tag       "$MODEL_TAG" \
     --results_dir     "$RESULTS_DIR_BASE" \
     --modality        "$CURRENT_MODALITY" \
     --input_json      "$INPUT_JSON_NAME" \
     --image_base_path "$OMNI_IMAGES_ROOT" \
     --python_executable "$PYTHON_EXEC" \
     --ft_checkpoint_path "$FT_CHECKPOINT_PATH" \
     --batch-size 8 \
     --eval-type "prefix" \
     --prompt_idx 4 # Default prompt index, can be changed

EXIT_CODE=$? 

echo "===================================================================="
echo "Finished modality: '$CURRENT_MODALITY'"
echo "End Time: $(date)"
echo "Python script exit code: $EXIT_CODE"
echo "===================================================================="

conda deactivate
exit $EXIT_CODE