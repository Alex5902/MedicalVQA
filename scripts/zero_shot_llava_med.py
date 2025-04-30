#!/usr/bin/env python3
"""
Run Multi-Modality-Arena evaluation (prefix-score + QA-score) on ONE specific
OmniMedVQA modality and collect the CSV files in ./results/.

Calls the underlying Python evaluation scripts directly.

Args
----
--omni_root     : directory that contains qa_items.json (+ Images/)
--arena_root    : .../Multi-Modality-Arena/MedicalEval
--model         : short model tag used for output filenames (default llava-med)
--results_dir   : where to drop per-modality CSVs (default ./results)
--modality      : The specific modality string to process
--model_path    : Path to the base LLM model (e.g., llava-v1.5-7b)
--vision_tower  : Path/ID of the vision tower (e.g., openai/clip-vit-large-patch14-336)
--mm_projector  : Path to the mm_projector.bin file
"""

import argparse
import os
import json
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

# ----------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def run_evaluation_script(python_script_path: Path, args_dict: dict, model_path_for_env: str):
    """
    Runs a specific Python evaluation script with provided arguments.
    Sets MODEL_PATH environment variable.
    """
    if not python_script_path.exists():
        print(f"ERROR: Evaluation script not found: {python_script_path}", file=sys.stderr)
        return False # Indicate failure

    cmd = [sys.executable, str(python_script_path)] # Use current python interpreter

    # Construct command line arguments from the dictionary
    for arg, value in args_dict.items():
        if value is not None: # Only add args that have a value
             # Handle boolean flags (like --answer-prompter)
             if isinstance(value, bool):
                 if value:
                     # argparse uses dashes, convert key underscores
                     cmd.append(f"--{arg.replace('_', '-')}")
                 # If False, we simply don't add the flag (store_true behavior)
             else:
                # argparse uses dashes, convert key underscores
                cmd.extend([f"--{arg.replace('_', '-')}", str(value)])

    print("\n$ " + " ".join(cmd), flush=True)
    try:
        # Set environment variable for model path, as eval scripts expect it
        env = os.environ.copy()
        env["MODEL_PATH"] = model_path_for_env # Use the passed model path
        print(f"DEBUG: Setting env MODEL_PATH={env['MODEL_PATH']}") # Add debug print
        subprocess.check_call(cmd, env=env)
        return True # Indicate success
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed: {' '.join(cmd)}\n{e}", file=sys.stderr)
        # Print the usage message from the failed script if possible (stderr)
        if e.stderr:
            print(f"--- Subprocess Stderr ---\n{e.stderr.decode(errors='ignore')}\n-------------------------", file=sys.stderr)
        return False # Indicate failure
    except Exception as e:
        print(f"ERROR: Unexpected error running script {python_script_path}: {e}", file=sys.stderr)
        return False # Indicate failure

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omni_root",     required=True)
    ap.add_argument("--arena_root",    required=True, help="…/Multi-Modality-Arena/MedicalEval")
    ap.add_argument("--model",         default="llava-med", help="Tag for output CSV filename (used by prefix script)")
    ap.add_argument("--results_dir",   default="./results")
    ap.add_argument("--modality",      required=True, help="Modality name to process")
    ap.add_argument("--model_path",    required=True, help="Path to base LLM model")
    ap.add_argument("--vision_tower",  required=True, help="Path/ID of vision tower")
    ap.add_argument("--mm_projector",  required=True, help="Path to mm_projector.bin")
    # Add other args needed by eval scripts if any (e.g., conv_mode)
    ap.add_argument("--conv_mode",     default="simple_legacy", help="Conversation mode for QA script")
    ap.add_argument("--answers_base_path", default="llava_med_output", help="Base dir for intermediate json/jsonl")

    args = ap.parse_args()

    target_modality = args.modality

    omni_root   = Path(args.omni_root).expanduser().resolve()
    arena_root  = Path(args.arena_root).expanduser().resolve()
    # Adjust arena_root if needed
    if not (arena_root / "Prefix_based_Score").exists() \
       and (arena_root / "MedicalEval" / "Prefix_based_Score").exists():
        arena_root = arena_root / "MedicalEval"
    results_dir = ensure_dir(args.results_dir)
    # Ensure the intermediate output base path exists
    answers_base_dir = ensure_dir(args.answers_base_path)


    qa_path = omni_root / "qa_items.json"
    if not qa_path.exists():
        sys.exit(f"ERROR: {qa_path} not found")

    print(f"Loading full dataset from {qa_path} to filter for modality: {target_modality}")
    qa_items = json.load(open(qa_path))
    modality_items = [itm for itm in qa_items
                      if itm.get("modality_type","").lower() == target_modality.lower()]

    if not modality_items:
        print(f"WARNING: No items found for modality '{target_modality}'. Skipping.")
        sys.exit(0)

    print(f"Found {len(modality_items)} items for modality {target_modality}")

    # Define evaluation scripts and their specific arguments expected by their argparse
    # Use underscores in keys here, run_evaluation_script converts to dashes for cmd line
    eval_scripts_config = [
        {
            "metric_subdir": "Prefix_based_Score",
            "script_name": "LLaVA-Med/llava/eval/model_med_eval_sp.py",
            "output_subdir": "Prefix_based_Score/results", # Relative path for finding CSV later (if needed)
            "args_def": {
                # Args defined in model_med_eval_sp.py's parser
                "model": args.model,
                "dataset_path": None, # Placeholder
                "mm_projector": args.mm_projector,
                "vision_tower": args.vision_tower,
                "image_root": str(omni_root / "OmniMedVQA"),
                "answers_base_path": str(answers_base_dir),
                "num_chunks": 1,
                "chunk_idx": 0,
                # model_name is NOT an argument for this script, it uses MODEL_PATH env var
                # conv_mode is NOT an argument
                # answer_prompter is NOT an argument
            }
        },
        {
            "metric_subdir": "Question-answering_Score",
            "script_name": "LLaVA-Med/llava/eval/model_med_eval.py",
            "output_subdir": "Question-answering_Score/results", # Relative path for finding CSV later (if needed)
            "args_def": {
                # Args defined in model_med_eval.py's parser
                "model_name": args.model_path,
                "question_file": None, # Placeholder (Correct arg name for QA script)
                "mm_projector": args.mm_projector,
                "vision_tower": args.vision_tower,
                "image_root": str(omni_root / "OmniMedVQA"),
                "answers_base_path": str(answers_base_dir),
                "conv_mode": args.conv_mode,
                "answer_prompter": False, # Explicitly disable second generate call
                "num_chunks": 1,
                "chunk_idx": 0,
                 # model is NOT an argument for this script
                 # dataset_path is NOT an argument (it uses question_file)
            }
        }
    ]

    # Create a temporary file specifically for this modality
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"omni_{target_modality.lower().replace(' ','_').replace('(','').replace(')','')}_"))
    mod_slug = target_modality.lower().replace(" ", "_").replace("(", "").replace(")", "") # Clean slug
    mod_json_path = tmp_dir / f"{mod_slug}.json"

    print(f"Writing {len(modality_items)} items to temporary file: {mod_json_path}")
    with open(mod_json_path, "w") as f:
        json.dump(modality_items, f)

    if mod_json_path.stat().st_size < 5:
        print(f"ERROR: Temporary JSON file {mod_json_path} is empty after writing. Skipping.")
        shutil.rmtree(tmp_dir)
        sys.exit(1)

    # --- Run metrics for THIS modality ---
    print(f"\n================ Processing Modality: {target_modality} ================", flush=True)
    for config in eval_scripts_config:
        metric_subdir = config["metric_subdir"]
        script_name = config["script_name"]
        script_path = arena_root / metric_subdir / script_name
        args_for_script = config["args_def"].copy() # Get the defined args

        # --- Set correct argument name for input file ---
        if config["metric_subdir"] == "Prefix_based_Score":
            args_for_script["dataset_path"] = str(mod_json_path)
            # Remove question_file if it accidentally exists
            args_for_script.pop("question_file", None)
        else: # Question-answering_Score
            args_for_script["question_file"] = str(mod_json_path)
            # Remove dataset_path if it accidentally exists
            args_for_script.pop("dataset_path", None)

        print(f"\n--- Running Metric: {metric_subdir} ---", flush=True)
        # Pass the actual model checkpoint path for setting the environment variable
        success = run_evaluation_script(script_path, args_for_script, args.model_path)


        if success:
            # Determine expected CSV name based on script convention
            if metric_subdir == "Prefix_based_Score":
                 # Prefix script uses args.model (--model) for filename
                expected_csv_name = f"{args.model}.csv"
            else: # Question-answering_Score
                 # QA script uses basename of --model-name (args.model_path) for filename
                 model_tag_for_csv = os.path.basename(args.model_path).replace("/", "_")
                 expected_csv_name = f"{model_tag_for_csv}.csv"

            # Define where the script *should* have saved its CSV output
            # This assumes scripts save to args.answers_base_path
            src_csv = answers_base_dir / expected_csv_name

            if src_csv.exists():
                dst_dir = ensure_dir(results_dir / mod_slug)
                dst_csv = dst_dir / f"{metric_subdir}.csv" # Rename consistently
                print(f"Copying result from {src_csv} to {dst_csv}")
                try:
                    # Use shutil.move if you want to move instead of copy
                    shutil.copy(src_csv, dst_csv)
                    print(f"✓ {metric_subdir} → {dst_csv}")
                    # Clean up the source CSV from answers_base_path after copying
                    os.remove(src_csv)
                    print(f"Removed source CSV: {src_csv}")
                except Exception as e:
                    print(f"ERROR: Failed to copy/remove {src_csv} to {dst_csv}: {e}", file=sys.stderr)
            else:
                print(f"WARNING: Expected result CSV not found at {src_csv} after running {script_name}")
        else:
             print(f"WARNING: Evaluation script {script_name} failed for modality {target_modality}.")


    # Clean up temporary directory
    print(f"Cleaning up temporary directory: {tmp_dir}")
    try:
        shutil.rmtree(tmp_dir)
    except Exception as e:
        print(f"WARNING: Failed to remove temporary directory {tmp_dir}: {e}", file=sys.stderr)

    print(f"\nFinished processing modality {target_modality}. Results are in {results_dir / mod_slug}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()