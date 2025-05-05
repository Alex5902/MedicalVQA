#!/usr/bin/env python3
"""
Run Multi-Modality-Arena evaluation (prefix-score) on ONE specific
OmniMedVQA modality using the BLIP-2 evaluation script and collect
the results in ./results/.

Calls the underlying medical_blip2.py script directly.

Args
----
--omni_root     : directory that contains qa_items.json or combined_dummy.json (+ Images/)
--arena_root    : .../Multi-Modality-Arena/MedicalEval (or parent dir if script is there)
--model_tag     : short model tag for output subdirs (e.g., blip2-flant5xl)
--results_dir   : where to drop per-modality result subdirectories (default ./results)
--modality      : The specific modality string to process
--input_json    : Name of the JSON file in omni_root (default: qa_items.json)
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

def run_evaluation_script(python_script_path: Path, args_dict: dict, python_exec: str):
    """
    Runs a specific Python evaluation script with provided arguments.
    """

    if not Path(python_exec).exists():
        print(f"ERROR: Specified Python executable not found: {python_exec}", file=sys.stderr)
        return False

    if not python_script_path.exists():
        print(f"ERROR: Evaluation script not found: {python_script_path}", file=sys.stderr)
        return False # Indicate failure

    cmd = [python_exec, str(python_script_path)]
    # cmd = [sys.executable, str(python_script_path)] # Use current python interpreter

    # Construct command line arguments from the dictionary
    for arg, value in args_dict.items():
        if value is not None: # Only add args that have a value
             # Handle boolean flags (like --some-flag) - unlikely for medical_blip2.py
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
        # No need to set MODEL_PATH env var for medical_blip2.py
        subprocess.check_call(cmd)
        return True # Indicate success
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed: {' '.join(cmd)}\n{e}", file=sys.stderr)
        if e.stderr:
            print(f"--- Subprocess Stderr ---\n{e.stderr.decode(errors='ignore')}\n-------------------------", file=sys.stderr)
        return False # Indicate failure
    except Exception as e:
        print(f"ERROR: Unexpected error running script {python_script_path}: {e}", file=sys.stderr)
        return False # Indicate failure

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omni_root",     required=True, help="Directory containing input JSON and optionally Images/")
    ap.add_argument("--image_base_path", required=True, help="Base directory containing the 'Images' folder")
    ap.add_argument("--arena_root",    required=True, help="Path to Multi-Modality-Arena/MedicalEval (or parent)")
    ap.add_argument("--model_tag",     default="blip2-flant5xl", help="Tag for output directory structure")
    ap.add_argument("--results_dir",   default="./results", help="Base directory for results")
    ap.add_argument("--modality",      required=True, help="Modality name to process")
    ap.add_argument("--input_json",    default="qa_items.json", help="Input JSON filename (e.g., qa_items.json or combined_dummy.json)")
    ap.add_argument("--python_executable", required=True, help="Full path to the python executable to use for subprocess")

    # --- Evaluation Type Specific Arguments ---
    ap.add_argument("--eval-type",     required=True, choices=['prefix', 'qa'], help="Type of evaluation to run ('prefix' or 'qa')")
    ap.add_argument("--batch-size",    type=int, default=4, help="Batch size (used primarily for QA evaluation)")

    args = ap.parse_args()

    target_modality = args.modality
    input_json_name = args.input_json # Capture which input file is used (real/dummy)
    eval_type = args.eval_type

    omni_root_for_json = Path(args.omni_root).expanduser().resolve()
    image_base_path_str = str(Path(args.image_base_path).expanduser().resolve())
    arena_root  = Path(args.arena_root).expanduser().resolve()
    python_exec_path = args.python_executable
    # Define results dir based on model tag and input json type (real/dummy)
    input_type_tag = "dummy" if "dummy" in input_json_name.lower() else "real"

    # --- Determine script path, arguments, and output details based on eval_type ---
    if eval_type == 'prefix':
        print("Configuring for Prefix Scoring...")
        script_subdir = "Prefix_based_Score"
        script_filename = "medical_blip2.py"
        target_script_path = arena_root / script_subdir / script_filename
        if not target_script_path.exists():
            # Fallback if structure is flat
             target_script_path = arena_root / script_filename
             if not target_script_path.exists():
                  sys.exit(f"ERROR: Cannot find {script_filename} in {arena_root/script_subdir} or {arena_root}")

        target_args_def = {
            "dataset_path": None, # Placeholder
            "answer_path": None,  # Placeholder
            "image_root": image_base_path_str, # medical_blip2.py expects --image-root
        }
        results_subdir_suffix = "prefix"
        run_title = "BLIP-2 Prefix Scoring"
        # Logic to determine expected output filename for prefix scoring
        def get_expected_prefix_filename(tmp_json_path_str):
            base_filename_part = tmp_json_path_str.replace('/', '_')
            if base_filename_part.endswith('.json'):
                base_filename_part = base_filename_part[:-5]
            return f"{base_filename_part}.json"
        expected_filename_logic = get_expected_prefix_filename
        clean_results_filename = f"{target_modality.lower().replace(' ','_').replace('(','').replace(')','')}_prefix_results.json"

    elif eval_type == 'qa':
        print("Configuring for Question Answering Scoring...")
        script_subdir = "Question-answering_Score" # Assumed subdir for QA script
        script_filename = "eval_medical.py"
        target_script_path = arena_root / script_subdir / script_filename
        if not target_script_path.exists():
             # Fallback if structure is flat (less likely based on repo structure)
             target_script_path = arena_root / script_filename
             if not target_script_path.exists():
                  sys.exit(f"ERROR: Cannot find {script_filename} in {arena_root/script_subdir} or {arena_root}")

        # Assumes eval_medical.py was modified to accept --image-base-path
        target_args_def = { # OK
            "model_name": "BLIP2",
            "dataset_path": None,
            "answer_path": None,
            "image_base_path": image_base_path_str, # eval_medical.py uses --image-base-path
            "batch_size": args.batch_size,
        }
        results_subdir_suffix = "qa"
        run_title = "BLIP-2 QA Scoring"
        # Assumes evaluate_medical_QA was modified to save this fixed name
        expected_filename_logic = lambda tmp_json_path_str: "qa_results.json"
        clean_results_filename = "qa_results.json" # Already clean

    else:
        # Should not happen due to choices in argparse, but good practice
        sys.exit(f"ERROR: Invalid --eval-type specified: {eval_type}")

    print(f"Found target script: {target_script_path}")

    results_base_dir = ensure_dir(Path(args.results_dir).expanduser().resolve() / f"{args.model_tag}_{input_type_tag}_{results_subdir_suffix}")

    # --- Load and Filter Data (Common Logic) ---
    qa_path = omni_root_for_json / input_json_name
    if not qa_path.exists():
        sys.exit(f"ERROR: Input JSON {qa_path} not found")

    print(f"Loading full dataset from {qa_path} to filter for modality: {target_modality}")
    try:
        qa_items = json.load(open(qa_path))
    except json.JSONDecodeError as e:
        sys.exit(f"ERROR: Failed to decode JSON from {qa_path}: {e}")
    except Exception as e:
        sys.exit(f"ERROR: Failed to load {qa_path}: {e}")

    modality_items = [itm for itm in qa_items
                      if itm.get("modality_type","").lower() == target_modality.lower()]

    if not modality_items:
        print(f"WARNING: No items found for modality '{target_modality}' in {input_json_name}. Skipping.")
        sys.exit(0)

    print(f"Found {len(modality_items)} items for modality {target_modality}")

    # --- Create Temporary File (Common Logic) ---
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"omni_{target_modality.lower().replace(' ','_').replace('(','').replace(')','')}_"))
    mod_slug = target_modality.lower().replace(" ", "_").replace("(", "").replace(")", "")
    mod_json_path = tmp_dir / f"{mod_slug}.json"

    print(f"Writing {len(modality_items)} items to temporary file: {mod_json_path}")
    with open(mod_json_path, "w") as f:
        json.dump(modality_items, f)

    if mod_json_path.stat().st_size < 5:
        print(f"ERROR: Temporary JSON file {mod_json_path} is empty after writing. Skipping.")
        shutil.rmtree(tmp_dir)
        sys.exit(1)

    # --- Run Evaluation (Common Logic) ---
    print(f"\n================ Processing Modality: {target_modality} ================", flush=True)

    args_for_script = target_args_def.copy() # Use the args defined in the if/else block
    args_for_script["dataset_path"] = str(mod_json_path)
    final_output_dir_for_modality = ensure_dir(results_base_dir / mod_slug)
    args_for_script["answer_path"] = str(final_output_dir_for_modality) # Pass final dir

    print(f"\n--- Running {run_title} ---", flush=True)
    success = run_evaluation_script(target_script_path, args_for_script, python_exec=python_exec_path)

    # --- Verify Output (Common Logic, uses variables set in if/else) ---
    if success:
        # Calculate expected filename using the logic determined earlier
        expected_output_filename = expected_filename_logic(str(mod_json_path))
        expected_output_path = final_output_dir_for_modality / expected_output_filename

        if expected_output_path.exists():
            print(f"✓ {run_title} results saved to: {expected_output_path}", flush=True)
            # Optional: Rename for clarity if needed
            try:
                target_path = final_output_dir_for_modality / clean_results_filename
                if expected_output_path != target_path:
                    expected_output_path.rename(target_path)
                    print(f"Renamed result to {clean_results_filename}")
                # else: # Optional: print if already clean
                #     print(f"Result file already has clean name: {clean_results_filename}")
            except Exception as e:
                print(f"Warning: Failed to rename result file {expected_output_path}: {e}")
        else:
             print(f"WARNING: {run_title} script succeeded but expected output file not found: {expected_output_path}")
    else:
        print(f"ERROR: {run_title} script {target_script_path} failed for modality {target_modality}.")
    # --- End Verify Output ---

    # --- Cleanup (Common Logic) ---
    print(f"Cleaning up temporary directory: {tmp_dir}")
    try:
        shutil.rmtree(tmp_dir)
    except Exception as e:
        print(f"WARNING: Failed to remove temporary directory {tmp_dir}: {e}", file=sys.stderr)

    print(f"\nFinished processing modality {target_modality} for {eval_type}. Results are in {final_output_dir_for_modality}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()