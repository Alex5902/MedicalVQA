#!/bin/bash
#SBATCH --job-name=blip2_eval_array # Changed job name
#SBATCH --partition=48-3,48-4,48-6,48-2 # Use appropriate partitions
#SBATCH --gres=gpu:1
#SBATCH --mem=48G                 # Increased memory slightly for FlanT5-XL, adjust if needed
#SBATCH --cpus-per-task=8         # Increased CPUs slightly, adjust if needed
#SBATCH --time=72:00:00
#SBATCH --output=logs/blip2_%A_%a.out # Changed log prefix, %A=job ID, %a=array index
#SBATCH --error=logs/blip2_%A_%a.err
#SBATCH --array=2,4,6             # Example: Run for ALL modalities (0 to 8 for 9 items)
                                # Or set specific indices like: 2,4,6

# --- Environment Setup ---
echo "Loading modules and activating Conda environment..."
. /usr/share/Modules/init/profile.sh
module purge
module load cuda/12.1 python/3.9.5 # Or your required versions
source ~/miniconda3/etc/profile.d/conda.sh # Adjust path if needed
# Or sometimes: . ~/miniconda3/bin/activate

# --- Activate the NEW Conda environment for BLIP-2/LAVIS ---
CONDA_ENV_NAME="blip2_lavis_env"
conda activate $CONDA_ENV_NAME
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "ERROR: Failed to activate conda environment '$CONDA_ENV_NAME'"
  exit $EXIT_CODE
fi
echo "Successfully activated Conda environment: '$CONDA_ENV_NAME'"
echo "Python executable: $(which python)" # Verify python path

# --- Define Paths ---
PROJECT_ROOT=$HOME/OmniMed # Adjust if your project root is different
OMNI_ROOT=$PROJECT_ROOT/OmniMedVQA
OMNI_IMAGES_ROOT=$PROJECT_ROOT/OmniMedVQA/OmniMedVQA
ARENA_ROOT=$PROJECT_ROOT/Multi-Modality-Arena/MedicalEval # Path where medical_blip2.py resides (or its parent)

PYTHON_EXEC=/home/alex.ia/miniconda3/envs/blip2_lavis_env/bin/python

# --- Configuration for BLIP-2 Run ---
# MODEL_TAG corresponds to --model_tag argument in run_blip2_modality.py
MODEL_TAG="blip2-flant5xl"

# INPUT_JSON_NAME corresponds to --input_json argument.
# Choose ONE of the following:
INPUT_JSON_NAME="qa_items.json"       # For REAL images
# INPUT_JSON_NAME="combined_dummy.json" # For DUMMY images

# RESULTS_DIR corresponds to --results_dir argument.
# The orchestrator script will create subdirs based on model_tag and input type (real/dummy)
RESULTS_DIR=$PROJECT_ROOT/results/blip2_normal_qa # Base directory for BLIP-2 results

# --- Driver Script ---
# Point to the NEW Python orchestrator script for BLIP-2
PY="$PROJECT_ROOT/scripts/zero_shot_blip2.py" # Adjust path if needed

# --- Modality Setup ---
# Ensure this list matches your dataset exactly, in order
# Double-check spacing and exact names from your dataset JSON file
ALL_MODALITIES=(
    "CT(Computed Tomography)"
    "Dermoscopy"
    "Fundus Photography"
    "MR (Mag-netic Resonance Imaging)" # Check spacing around MR
    "Microscopy Images"
    "OCT (Optical Coherence Tomography" # Check if trailing parenthesis is missing
    "X-Ray"
    "ultrasound"
    "unknown"
)

# Get the modality for the current array task ID
# Check if SLURM_ARRAY_TASK_ID is set (it should be in an array job)
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set. This script must be run as a job array."
  exit 1
fi
# Check if the index is valid
if [ "$SLURM_ARRAY_TASK_ID" -ge ${#ALL_MODALITIES[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID is out of bounds for ALL_MODALITIES array (size ${#ALL_MODALITIES[@]})."
    exit 1
fi
CURRENT_MODALITY=${ALL_MODALITIES[$SLURM_ARRAY_TASK_ID]}

echo "===================================================================="
echo "Starting SLURM Job: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running evaluation for BLIP-2 Model Tag: $MODEL_TAG"
echo "Processing Modality: '$CURRENT_MODALITY'"
echo "Using Input JSON: $INPUT_JSON_NAME"
echo "Results Base Directory: $RESULTS_DIR"
echo "Python Orchestrator Script: $PY"
echo "Start Time: $(date)"
echo "===================================================================="

# --- Execute Driver Script with BLIP-2 arguments ---
python "$PY" \
     --omni_root       "$OMNI_ROOT" \
     --arena_root      "$ARENA_ROOT" \
     --model_tag       "$MODEL_TAG" \
     --results_dir     "$RESULTS_DIR" \
     --modality        "$CURRENT_MODALITY" \
     --input_json      "$INPUT_JSON_NAME" \
     --image_base_path "$OMNI_IMAGES_ROOT" \
     --python_executable "$PYTHON_EXEC" \
     --batch-size 8 \
     --eval-type "qa"  

EXIT_CODE=$? # Capture the exit code of the python script

echo "===================================================================="
echo "Finished modality: '$CURRENT_MODALITY'"
echo "End Time: $(date)"
echo "Python script exit code: $EXIT_CODE"
echo "===================================================================="

# Deactivate conda environment (optional, good practice)
conda deactivate

exit $EXIT_CODE