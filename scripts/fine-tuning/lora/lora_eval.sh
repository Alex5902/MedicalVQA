#!/bin/bash
#SBATCH --job-name=blip2_lora_s1_fundus_eval # MODIFIED: Job name
#SBATCH --partition=48-4,48-6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G                 
#SBATCH --cpus-per-task=8         
#SBATCH --time=12:00:00 # MODIFIED: Might be shorter for one modality eval
#SBATCH --output=logs/eval_lora_s1_fundus_%A.out # MODIFIED: Simplified output name
#SBATCH --error=logs/eval_lora_s1_fundus_%A.err  # MODIFIED: Simplified error name
# SBATCH --array=2,4,6 # MODIFIED: Commented out if only evaluating Fundus on Fundus

# --- Environment Setup ---
echo "Loading modules and activating Conda environment..."
. /usr/share/Modules/init/profile.sh
module purge
module load cuda/11.8 
echo "LD_LIBRARY_PATH after module load: $LD_LIBRARY_PATH"

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH after adding system driver path: $LD_LIBRARY_PATH"

source ~/miniconda3/etc/profile.d/conda.sh 
CONDA_ENV_NAME="blip2_lavis_env" 
conda activate $CONDA_ENV_NAME
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "ERROR: Failed to activate conda environment '$CONDA_ENV_NAME'"
  exit $EXIT_CODE
fi
echo "Successfully activated Conda environment: '$CONDA_ENV_NAME'"
echo "CONDA_PREFIX: $CONDA_PREFIX"

PYTHON_VERSION_DIR="python3.10" 
PYTORCH_BUNDLED_CUDNN_PATH="$CONDA_PREFIX/lib/$PYTHON_VERSION_DIR/site-packages/nvidia/cudnn/lib"
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
echo "Final LD_LIBRARY_PATH for Python script: $LD_LIBRARY_PATH"
echo "Python executable for script: $(which python)"

# --- Define Paths ---
PROJECT_ROOT=$HOME/OmniMed 
# OMNI_ROOT points to where the modality-specific test JSONs are, or the combined one
# For evaluating Stage 1 (Fundus) on Fundus test data:
OMNI_ROOT_MODALITY_DATA=$PROJECT_ROOT/finetuning_datasets/OmniMedVQA_Splits_ImgLevel/Fundus_Photography 
OMNI_IMAGES_ROOT=$PROJECT_ROOT/OmniMedVQA/OmniMedVQA # Contains the Images/ folder
# ARENA_ROOT points to where your lora_medical_eval.py (or medical_blip2_lora.py) is located
ARENA_ROOT=$PROJECT_ROOT/Multi-Modality-Arena/MedicalEval 

PYTHON_EXEC=$(which python) 

# --- Configuration for LoRA Stage 1 Evaluation ---
MODEL_TAG="blip2-flant5xl-lora-s1-fundus" # MODIFIED: Tag for this specific LoRA model

# --- MODIFIED: Paths to your saved Stage 1 LoRA adapters ---
LORA_S1_BASE_DIR="$PROJECT_ROOT/lora_finetuned_adapters_seq/stage1_fundus_lora/lora_adapters"
LORA_T5_ADAPTER_PATH="$LORA_S1_BASE_DIR/fundus_t5_final"
LORA_QFORMER_ADAPTER_PATH="$LORA_S1_BASE_DIR/fundus_qformer_final"

# INPUT_JSON_NAME: Use the test set JSON for the Fundus modality
INPUT_JSON_NAME="test.json" # Assumes test.json is inside OMNI_ROOT_MODALITY_DATA

# RESULTS_DIR: Base directory for results. Orchestrator will make subdirs.
RESULTS_DIR_BASE="$PROJECT_ROOT/results_eval_lora_s1/fundus_on_fundus" # MODIFIED: Specific results dir

# --- Driver Script ---
# MODIFIED: Point to your NEW Python orchestrator script for LoRA evaluation
PY="$PROJECT_ROOT/scripts/fine-tuning/lora/lora_eval.py" # Make sure this path is correct

# --- Modality Setup ---
# MODIFIED: For evaluating Stage 1 Fundus model on Fundus test set, we only need Fundus Photography
CURRENT_MODALITY="Fundus Photography"

# If you want to use the array job to evaluate this S1 Fundus model on MULTIPLE modalities:
# You would keep the --array and ALL_MODALITIES setup.
# The OMNI_ROOT would then point to the directory holding the combined_test.json,
# and INPUT_JSON_NAME would be "combined_test.json".
# For now, let's assume you're just evaluating Fundus on Fundus.

echo "===================================================================="
echo "Starting SLURM Job: $SLURM_JOB_ID" # Removed Task ID if not using array
echo "Running evaluation for LoRA Stage 1 Model Tag: $MODEL_TAG"
echo "T5 LoRA Adapter: $LORA_T5_ADAPTER_PATH"
echo "Q-Former LoRA Adapter: $LORA_QFORMER_ADAPTER_PATH"
echo "Processing Modality: '$CURRENT_MODALITY'"
echo "Using Input JSON (relative to OMNI_ROOT_MODALITY_DATA): $INPUT_JSON_NAME"
echo "Results Base Directory: $RESULTS_DIR_BASE"
echo "Python Orchestrator Script: $PY"
echo "Start Time: $(date)"
echo "===================================================================="

# --- Execute Orchestrator Script ---
"$PYTHON_EXEC" "$PY" \
     --omni_root       "$OMNI_ROOT_MODALITY_DATA" \
     --arena_root      "$ARENA_ROOT" \
     --model_tag       "$MODEL_TAG" \
     --results_dir     "$RESULTS_DIR_BASE" \
     --modality        "$CURRENT_MODALITY" \
     --input_json      "$INPUT_JSON_NAME" \
     --image_base_path "$OMNI_IMAGES_ROOT" \
     --python_executable "$PYTHON_EXEC" \
     --lora_t5_adapter_path "$LORA_T5_ADAPTER_PATH" \
     --lora_qformer_adapter_path "$LORA_QFORMER_ADAPTER_PATH" \
     --batch-size 8 \
     --eval-type "prefix" \
     --prompt_idx 4 
     # REMOVED: --ft_checkpoint_path (not used for LoRA eval)

EXIT_CODE=$? 

echo "===================================================================="
echo "Finished modality: '$CURRENT_MODALITY'"
echo "End Time: $(date)"
echo "Python script exit code: $EXIT_CODE"
echo "===================================================================="

conda deactivate
exit $EXIT_CODE