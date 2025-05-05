#!/bin/bash

# Use environment variable MODEL, default to LLaMA-Adapter-v2 if not set
MODEL=${MODEL:-LLaMA-Adapter-v2}
# Use environment variable TEST_PATH, error if not set
DATA=${TEST_PATH:?"TEST_PATH environment variable not set"}

echo "QA Script Using MODEL=$MODEL and DATA=$DATA"

# Set output directory relative to the script's location -> ../output/MODEL_NAME
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OUTPUT_DIR="$SCRIPT_DIR/../output/$MODEL"
mkdir -p "$OUTPUT_DIR"

# Logic based on MODEL env var
if [ "$MODEL" == 'llava-med' ]; then
    echo "Evaluating LLaVA-Med QA"
    # Adjust path and arguments for the specific LLaVA-Med QA script
    python_script="../LLaVA-Med/llava/eval/model_med_eval.py" # Relative path
    if [ ! -f "$SCRIPT_DIR/$python_script" ]; then
        echo "ERROR: Python script not found at $SCRIPT_DIR/$python_script"
        exit 1
    fi
    # Check llava-med script args - it uses --question-file and --answers-base-path
    python "$SCRIPT_DIR/$python_script" \
        --question-file "$DATA" \
        --answers-base-path "$OUTPUT_DIR" \
        # Add other necessary args like --model-name, --vision-tower, --mm-projector if needed by this script
else
    echo "Evaluating $MODEL QA using eval_medical.py"
    # Adjust path for the generic eval_medical.py script
    python_script="../eval_medical.py" # Relative path
    if [ ! -f "$SCRIPT_DIR/$python_script" ]; then
        echo "ERROR: Python script not found at $SCRIPT_DIR/$python_script"
        exit 1
    fi
    # Check eval_medical.py args - it uses --model_name and --dataset_path
    python "$SCRIPT_DIR/$python_script" \
        --model_name "$MODEL" \
        --dataset_path "$DATA" \
        --answer_path "$OUTPUT_DIR" # Pass the output directory

fi

exit $? # Exit with the python script's exit code