#!/bin/bash

# This script preprocesses downloaded SVS slides using CLAM.
# Assumes CLAM (https://github.com/mahmoodlab/CLAM) is installed and its
# scripts are accessible or their path is provided.

# --- Configuration ---
# !!! SET THESE PATHS BEFORE RUNNING !!!
SVS_DIR="./finetuning_datasets/WSI_VQA/svs"                 # Directory containing downloaded .svs files
PATCHES_DIR="./finetuning_datasets/WSI_VQA/patches"         # Temporary directory for patches
FEATURES_DIR="./finetuning_datasets/WSI_VQA/WSI_features"   # Final directory for extracted features (.pt files)
CLAM_CREATE_PATCHES_SCRIPT="create_patches_fp.py"           # Name/path of CLAM's patch creation script
CLAM_EXTRACT_FEATURES_SCRIPT="extract_features_fp.py"       # Name/path of CLAM's feature extraction script
FEATURE_EXTRACT_MODEL="resnet50"                            # Feature extractor model (e.g., resnet50, resnet50_trunc)
PATCH_SIZE=256                                              # Patch size used by CLAM
STEP_SIZE=$PATCH_SIZE                                       # Step size (set equal to patch_size for no overlap)
BATCH_SIZE=64                                               # Batch size for feature extraction (adjust based on GPU memory)
NUM_WORKERS=8                                               # Number of workers for feature extraction

# Create output directories
mkdir -p "$PATCHES_DIR"
mkdir -p "$FEATURES_DIR"

echo "Starting WSI preprocessing using CLAM..."
echo "Source SVS directory: $SVS_DIR"
echo "Output feature directory: $FEATURES_DIR"
echo "Using feature extractor: $FEATURE_EXTRACT_MODEL"

# Check if SVS directory exists and is not empty
if [ ! -d "$SVS_DIR" ] || [ -z "$(ls -A $SVS_DIR/*.svs 2>/dev/null)" ]; then
    echo "Error: SVS directory '$SVS_DIR' not found or contains no .svs files."
    echo "Please ensure slides are downloaded using the GDC client and the manifest file first."
    exit 1
fi

# Find CLAM scripts (add flexibility if CLAM path is needed)
# If CLAM scripts are not in PATH, you might need to specify full paths:
# CLAM_DIR="/path/to/clam"
# CLAM_CREATE_PATCHES_SCRIPT="$CLAM_DIR/create_patches_fp.py"
# CLAM_EXTRACT_FEATURES_SCRIPT="$CLAM_DIR/extract_features_fp.py"

if ! command -v "$CLAM_CREATE_PATCHES_SCRIPT" &> /dev/null; then
    echo "Error: CLAM script '$CLAM_CREATE_PATCHES_SCRIPT' not found. Is CLAM installed and in your PATH?"
    # exit 1 # Commented out to allow manual path setting above
fi
if ! command -v "$CLAM_EXTRACT_FEATURES_SCRIPT" &> /dev/null; then
    echo "Error: CLAM script '$CLAM_EXTRACT_FEATURES_SCRIPT' not found. Is CLAM installed and in your PATH?"
    # exit 1 # Commented out to allow manual path setting above
fi


# Process each SVS file
for SVS_FILE in "$SVS_DIR"/*.svs; do
    if [ -f "$SVS_FILE" ]; then
        FILENAME=$(basename "$SVS_FILE")
        SLIDE_ID="${FILENAME%.*}" # Remove .svs extension
        CURRENT_PATCH_DIR="$PATCHES_DIR/$SLIDE_ID"
        CURRENT_FEATURE_FILE="$FEATURES_DIR/${SLIDE_ID}.pt"

        echo "--------------------------------------------------"
        echo "Processing: $FILENAME (Slide ID: $SLIDE_ID)"
        echo "--------------------------------------------------"

        # Check if features already exist
        if [ -f "$CURRENT_FEATURE_FILE" ]; then
            echo "Features already exist: $CURRENT_FEATURE_FILE. Skipping."
            continue
        fi

        # 1. Create Patches
        echo "Step 1: Creating patches..."
        python "$CLAM_CREATE_PATCHES_SCRIPT" --source "$SVS_DIR" --save_dir "$PATCHES_DIR" \
               --patch_size "$PATCH_SIZE" --step_size "$STEP_SIZE" --patch --seg --stitch \
               --slide_list "$FILENAME" # Process only the current slide specified by filename

        # Check if patch creation was successful (basic check: directory exists)
        if [ ! -d "$CURRENT_PATCH_DIR/patches" ]; then
            echo "Error: Patch directory '$CURRENT_PATCH_DIR/patches' not created for $FILENAME. Skipping feature extraction."
            continue # Skip to next slide
        fi

        # 2. Extract Features
        echo "Step 2: Extracting features..."
        python "$CLAM_EXTRACT_FEATURES_SCRIPT" --data_h5_dir "$CURRENT_PATCH_DIR" --csv_path NONE \
               --feat_dir "$FEATURES_DIR" --batch_size "$BATCH_SIZE" --slide_list "$FILENAME" \
               --model_type "$FEATURE_EXTRACT_MODEL" --no_auto_skip --num_workers "$NUM_WORKERS"

        # Check if feature extraction was successful
        if [ -f "$CURRENT_FEATURE_FILE" ]; then
            echo "Features extracted successfully: $CURRENT_FEATURE_FILE"

            # 3. Delete Patches (as requested)
            echo "Step 3: Deleting temporary patches for $SLIDE_ID..."
            rm -rf "$CURRENT_PATCH_DIR"
            if [ $? -eq 0 ]; then
                echo "Deleted patch directory: $CURRENT_PATCH_DIR"
            else
                echo "Warning: Failed to delete patch directory $CURRENT_PATCH_DIR"
            fi
        else
            echo "Error: Feature file '$CURRENT_FEATURE_FILE' not created for $FILENAME. Patches not deleted."
        fi

        echo "Finished processing $FILENAME."

    fi
done

echo "--------------------------------------------------"
echo "WSI Preprocessing Script Finished."
echo "Extracted features are in: $FEATURES_DIR"
echo "--------------------------------------------------"