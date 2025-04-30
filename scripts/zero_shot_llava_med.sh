#!/bin/bash
#SBATCH --job-name=omni_eval_array
#SBATCH --partition=48-3,48-4,48-6,48-2 # Use multiple partitions
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --time=72:00:00
#SBATCH --output=logs/omni_%A_%a.out # %A=job ID, %a=array index
#SBATCH --error=logs/omni_%A_%a.err
#SBATCH --array=0-8              # 0 to N-1 modalities

# load your venv, etc.
. /usr/share/Modules/init/profile.sh
module purge
module load cuda/12.1 python/3.9.5
source ~/OmniMed/venv/bin/activate

# --- Define Paths ---
PROJECT_ROOT=$HOME/OmniMed
MODEL_CHECKPOINT_PATH="$HOME/OmniMed/llava-v1.5-7b" # Base LLM path
MM_PROJECTOR_PATH="$MODEL_CHECKPOINT_PATH/mm_projector.bin" # Projector path
VISION_TOWER_PATH="openai/clip-vit-large-patch14-336" # Vision tower ID (336 version)

OMNI_ROOT=$PROJECT_ROOT/OmniMedVQA
ARENA_ROOT=$PROJECT_ROOT/Multi-Modality-Arena
RESULTS_DIR=$PROJECT_ROOT/results
MODEL_TAG=llava-med # Tag used for output filenames in prefix script

# --- Modality Setup ---
# Ensure this list matches your dataset exactly, in order
ALL_MODALITIES=("CT(Computed Tomography)" "Dermoscopy" "Fundus Photography" "MR (Mag-netic Resonance Imaging)" "Microscopy Images" "OCT (Optical Coherence Tomography" "X-Ray" "ultrasound" "unknown")

# Get the modality for the current array task ID
CURRENT_MODALITY=${ALL_MODALITIES[$SLURM_ARRAY_TASK_ID]}

echo "Starting evaluation for modality: $CURRENT_MODALITY (Task ID: $SLURM_ARRAY_TASK_ID) on $(date)"

# --- Driver Script ---
# Point to the *modified* driver script
PY="$PROJECT_ROOT/scripts/zero_shot_llava_med.py"

# --- Execute Driver Script with ALL necessary arguments ---
python "$PY" \
     --omni_root       "$OMNI_ROOT" \
     --arena_root      "$ARENA_ROOT" \
     --model           "$MODEL_TAG" \
     --results_dir     "$RESULTS_DIR" \
     --modality        "$CURRENT_MODALITY" \
     --model_path      "$MODEL_CHECKPOINT_PATH" \
     --vision_tower    "$VISION_TOWER_PATH" \
     --mm_projector    "$MM_PROJECTOR_PATH"
     # Add --conv_mode if you want to change it from default
     # Add --answers_base_path if you want to change it from default

echo "Finished modality: $CURRENT_MODALITY at $(date) with exit code $EXIT_CODE"

deactivate
