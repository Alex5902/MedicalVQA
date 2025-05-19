#!/bin/bash
#SBATCH --job-name=merge_qf_knots_baseline
#SBATCH --partition=48-6,48-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8          # Adjust based on expected CPU load
#SBATCH --mem=64G                  # Adjust based on expected memory load (SVD can be memory hungry)
#SBATCH --time=24:00:00            # Adjust based on how long you expect it to run (e.g., 2 hours)
#SBATCH --output=logs/merge_qf_knots_baseline_%A.out
#SBATCH --error=logs/merge_qf_knots_baseline_%A.err

# --- Environment Setup ---
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Starting time: $(date)"
echo "Loading modules and activating Conda environment..."

. /usr/share/Modules/init/profile.sh # Or your system's equivalent
module purge
module load cuda/11.8 # Load CUDA if using GPU for PyTorch device, even if SVD is CPU-bound
# Add other modules if your environment requires them (e.g., gcc, anaconda)

# Activate your Conda environment
source ~/miniconda3/etc/profile.d/conda.sh # Or your Anaconda/Miniconda path
CONDA_ENV_NAME="blip2_lavis_env" # Or your environment name
conda activate $CONDA_ENV_NAME
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX: $CONDA_PREFIX"

# Optional: Set LD_LIBRARY_PATH for PyTorch's bundled cuDNN if needed (usually for training)
# PYTORCH_BUNDLED_CUDNN_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib"
# if [ -d "$PYTORCH_BUNDLED_CUDNN_PATH" ]; then
#   export LD_LIBRARY_PATH="$PYTORCH_BUNDLED_CUDNN_PATH:$LD_LIBRARY_PATH"
#   echo "LD_LIBRARY_PATH after prepending PyTorch's bundled cuDNN path: $LD_LIBRARY_PATH"
# fi

# --- Define Paths ---
PROJECT_ROOT=$HOME/OmniMed # Or your project root
MERGE_SCRIPT_PATH=$PROJECT_ROOT/scripts/fine-tuning/merging/merge_lora_adapters.py
PYTHON_EXEC=$(which python)

KNOTS_REPO_ACTUAL_PATH="$PROJECT_ROOT/KnOTS" 
if [ -d "$KNOTS_REPO_ACTUAL_PATH" ]; then
    export PYTHONPATH="$KNOTS_REPO_ACTUAL_PATH:$PYTHONPATH"
    echo "PYTHONPATH set to include KnOTS repo: $PYTHONPATH"
else
    echo "ERROR: KnOTS repository directory not found at $KNOTS_REPO_ACTUAL_PATH"
    exit 1
fi

# Paths to your Q-Former Only LoRA adapters (from Track C)
ADAPTER_FUNDUS_PATH="$PROJECT_ROOT/lora_finetuned_adapters_seq/stage1_fundus_qformer_only_lora/lora_adapters/fundus_qf_only_qformer_final/fundus_qf_only_qformer"
ADAPTER_XRAY_PATH="$PROJECT_ROOT/lora_finetuned_adapters_seq/stage2_xray_qformer_only_lora/lora_adapters/xray_qf_only_qformer_final/xray_qf_only_qformer"
ADAPTER_MICROSCOPY_PATH="$PROJECT_ROOT/lora_finetuned_adapters_seq/stage3_microscopy_qformer_only_lora/lora_adapters/microscopy_qf_only_qformer_final/microscopy_qf_only_qformer"

# Output path for the merged adapter
OUTPUT_MERGED_ADAPTER_DIR="$PROJECT_ROOT/merged_qformer_adapters/knots_baseline_merged_qf_adapter"

# LoRA Rank used for Q-Former during training (must match)
QFORMER_LORA_RANK=8

# KnOTS Baseline Merging Parameters
# For "pure KnOTS" (SVD alignment + default averaging of SVD components):
# - mask_method: 'tv' (which should map to tv_masking in KnOTS utils)
# - topK: 100.0 (to ensure tv_masking effectively creates an all-ones mask, i.e., no pruning)
# - merging_type: 'mean' (for the final aggregation of (masked) SVD components)
KNOTS_MASK_METHOD="tv"
KNOTS_TOPK=100.0 
KNOTS_MERGING_TYPE="mean"

# Device for PyTorch operations (can be "cuda" or "cpu")
# SVD might be CPU-bound anyway, but PyTorch parts might use GPU.
# If your node has a GPU and you want PyTorch to use it for tensor ops before/after SVD:
DEVICE="cuda"
# If you want to force CPU for everything (e.g., if SVD is very memory intensive on GPU):
# DEVICE="cpu"

# --- Create Output Directory ---
mkdir -p "$(dirname "$OUTPUT_MERGED_ADAPTER_DIR")" # Create parent of the adapter dir
# The script itself will create the final adapter directory if `save_pretrained` is used correctly

# --- Run the Merging Script ---
echo "Starting adapter merging script..."
echo "Python executable: $PYTHON_EXEC"
echo "Merge script: $MERGE_SCRIPT_PATH"
echo "Output directory for merged adapter: $OUTPUT_MERGED_ADAPTER_DIR"

srun $PYTHON_EXEC "$MERGE_SCRIPT_PATH" \
    --base_model_name "blip2_t5" \
    --base_model_type "pretrain_flant5xl" \
    --task_adapter_paths "$ADAPTER_FUNDUS_PATH" "$ADAPTER_XRAY_PATH" "$ADAPTER_MICROSCOPY_PATH" \
    --qformer_lora_r $QFORMER_LORA_RANK \
    --output_merged_adapter_path "$OUTPUT_MERGED_ADAPTER_DIR" \
    --knots_mask_method "$KNOTS_MASK_METHOD" \
    --knots_topK $KNOTS_TOPK \
    --knots_merging_type "$KNOTS_MERGING_TYPE" \
    --device "$DEVICE"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Merging script failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "Merging script completed successfully."
echo "Merged adapter saved to: $OUTPUT_MERGED_ADAPTER_DIR"
echo "Ending time: $(date)"