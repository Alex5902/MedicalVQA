#!/usr/bin/env bash
# ------------------------------------------------------------------
# Run PREFIX-BASED score for a single model.
# This script is called by the Python driver which exports:
#   MODEL     – short tag of the model to test  (e.g. llava-med)
#   TEST_PATH – absolute path of the filtered OmniMedVQA JSON
# ------------------------------------------------------------------

set -e

: "${MODEL:?need MODEL env var}"          # fail early if missing
: "${TEST_PATH:?need TEST_PATH env var}"

# ---------- paths ----------
THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MED_EVAL_ROOT=$( dirname "$THIS_DIR" )               # …/Prefix_based_Score
ARENA_ROOT=$( dirname "$MED_EVAL_ROOT" )             # …/MedicalEval
# ↓ the runner is **inside the metric folder**
LAVAMED_RUNNER="$MED_EVAL_ROOT/LLaVA-Med/llava/eval/model_med_eval_sp.py"
OUT_DIR="$ARENA_ROOT/results"
mkdir -p "$OUT_DIR"

# ---------- pick the runner ----------
case "$MODEL" in
  MiniGPT-4)            RUN="medical_minigpt4.py" ;;
  BLIP2)                RUN="medical_blip2.py" ;;
  InstructBLIP)         RUN="medical_instructblip.py" ;;
  LLaMA-Adapter-v2)     RUN="medical_llama_adapter2.py" ;;
  LLaVA)                RUN="medical_llava.py" ;;
  Otter)                RUN="medical_otter.py" ;;
  mPLUG-Owl)            RUN="medical_owl.py" ;;
  VPGTrans)             RUN="medical_vpgtrans.py" ;;
  llava-med)            RUN="$LAVAMED_RUNNER" ;;
  *)  echo "Unknown MODEL=$MODEL"; exit 1 ;;
esac

echo "Evaluating $MODEL on $TEST_PATH"

export PYTHONPATH="$ARENA_ROOT/Prefix_based_Score/LLaVA-Med:$PYTHONPATH"
export MODEL_PATH="$HOME/OmniMed/llava-v1.5-7b"
export MM_PROJECTOR_PATH="$HOME/OmniMed/llava-v1.5-7b/mm_projector.bin"
export VISION_TOWER="openai/clip-vit-large-patch14"

python "$RUN" --dataset_path "$TEST_PATH" --model-name "$MODEL_PATH" --mm-projector "$MODEL_PATH/mm_projector.bin" --vision-tower "$VISION_TOWER" --answers_base_path "$OUT_DIR"

if [[ -f "$OUT_DIR/${MODEL}.csv" ]]; then
    echo "result stored in $OUT_DIR/${MODEL}.csv"
else
    echo "runner finished but no CSV produced"
fi

echo Evaluating $MODEL 