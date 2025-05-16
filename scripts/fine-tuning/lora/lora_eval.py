#!/usr/bin/env python3
"""
Run Multi-Modality-Arena evaluation on ONE specific
OmniMedVQA modality using a specified BLIP-2 evaluation script 
(potentially fine-tuned) and collect the results in ./results/.

Calls an underlying evaluation script (e.g., medical_blip2_finetuned.py).
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
        return False

    cmd = [python_exec, str(python_script_path)]
    for arg, value in args_dict.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{arg.replace('_', '-')}")
            else:
                cmd.extend([f"--{arg.replace('_', '-')}", str(value)])

    print("\n$ " + " ".join(cmd), flush=True)
    print(f"DEBUG Orchestrator: Full command list being executed: {cmd}", flush=True)
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed: {' '.join(cmd)}\n{e}", file=sys.stderr)
        if e.stderr:
            print(f"--- Subprocess Stderr ---\n{e.stderr.decode(errors='ignore')}\n-------------------------", file=sys.stderr)
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error running script {python_script_path}: {e}", file=sys.stderr)
        return False

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omni_root", required=True, help="Directory containing input JSON and optionally Images/")
    ap.add_argument("--image_base_path", required=True, help="Base directory containing the 'Images' folder")
    ap.add_argument("--arena_root", required=True, help="Path to Multi-Modality-Arena/MedicalEval (or parent)")
    ap.add_argument("--model_tag", default="blip2-flant5xl", help="Tag for output directory structure AND for naming within the results JSON if not overridden by checkpoint.")
    ap.add_argument("--results_dir", default="./results", help="Base directory for results")
    ap.add_argument("--modality", required=True, help="Modality name to process")
    ap.add_argument("--input_json", default="qa_items.json", help="Input JSON filename (e.g., qa_items.json or combined_dummy.json)")
    ap.add_argument("--python_executable", required=True, help="Full path to the python executable to use for subprocess")
    
    # --- Argument for fine-tuned checkpoint ---
    ap.add_argument("--ft_checkpoint_path", default=None, help="Path to the fine-tuned model checkpoint (.pth file), if evaluating a fine-tuned model.")

    # --- Evaluation Type Specific Arguments ---
    ap.add_argument("--eval-type", required=True, choices=['prefix', 'qa'], help="Type of evaluation to run ('prefix' or 'qa')")
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size (used for QA evaluation, and potentially by prefix if adapted)")
    ap.add_argument("--prompt_idx", type=int, default=4, help="Prompt index for prefix scoring (passed to medical_blip2_finetuned.py)")

    ap.add_argument("--lora_t5_adapter_path", default=None, help="Path to the T5 LoRA adapter directory for evaluation.")
    ap.add_argument("--lora_qformer_adapter_path", default=None, help="Path to the Q-Former LoRA adapter directory for evaluation.")


    args = ap.parse_args()

    target_modality = args.modality
    input_json_name = args.input_json
    eval_type = args.eval_type

    omni_root_for_json = Path(args.omni_root).expanduser().resolve()
    image_base_path_str = str(Path(args.image_base_path).expanduser().resolve())
    arena_root  = Path(args.arena_root).expanduser().resolve()
    python_exec_path = args.python_executable
    input_type_tag = "dummy" if "dummy" in input_json_name.lower() else "real"

    # Determine the model_name_tag for the underlying script.
    # If a checkpoint is provided, the underlying script might generate a more specific name.
    # Here, model_tag is mainly for the output directory structure.
    effective_model_name_tag_for_script = args.model_tag
    if args.ft_checkpoint_path and args.model_tag == "blip2-flant5xl": # If default tag and FT, let script derive
        # The medical_blip2_finetuned.py will create a more specific name based on checkpoint
        # but we can still use the base args.model_tag for the directory structure, or make it more specific.
        # For directory structure, let's make it specific if checkpoint is used.
        ckpt_basename = os.path.splitext(os.path.basename(args.ft_checkpoint_path))[0]
        dir_model_tag = f"{args.model_tag}-ft-{ckpt_basename}"
    else:
        dir_model_tag = args.model_tag


    if eval_type == 'prefix':
        print("Configuring for Prefix Scoring...")
        script_subdir = "Prefix_based_Score"
        # --- Point to your new/modified script ---
        script_filename = "medical_blip2_lora.py" # Or whatever you named your modified version
        
        target_script_path = arena_root / script_subdir / script_filename
        if not target_script_path.exists():
             target_script_path = arena_root / script_filename # Fallback
             if not target_script_path.exists():
                  sys.exit(f"ERROR: Cannot find {script_filename} in {arena_root/script_subdir} or {arena_root}")

        target_args_def = {
            "dataset_path": None,
            "answer_path": None,  # This will be the final_output_dir_for_modality
            "image_root": image_base_path_str,
            "model_name_tag": effective_model_name_tag_for_script, # Pass the tag
            "checkpoint_path": args.ft_checkpoint_path, # Pass the checkpoint path
            "prompt_idx": args.prompt_idx, # Pass prompt index
            "lora_t5_adapter_path": args.lora_t5_adapter_path,
            "lora_qformer_adapter_path": args.lora_qformer_adapter_path,
            "base_model_name": "blip2_t5", # Or make this configurable via lora_eval.py args
            "base_model_type": "pretrain_flant5xl" # Or make this configurable
        }
        results_subdir_suffix = "prefix"
        run_title = f"{args.model_tag} Prefix Scoring"
        
        # Expected output filename logic from medical_blip2_finetuned.py
        # It will be <mod_slug>_prefix_results.json
        def get_expected_prefix_filename(mod_slug_str):
            return f"{mod_slug_str}_prefix_results.json"
        expected_filename_logic = get_expected_prefix_filename
        # clean_results_filename will be set later using mod_slug

    elif eval_type == 'qa':
        print("Configuring for Question Answering Scoring...")
        # Note: If you use QA eval, you'll need to similarly modify 'eval_medical.py'
        # to accept and load the --ft_checkpoint_path.
        script_subdir = "Question-answering_Score"
        script_filename = "eval_medical.py" # This would also need modification for FT checkpoints
        target_script_path = arena_root / script_subdir / script_filename
        if not target_script_path.exists():
             target_script_path = arena_root / script_filename
             if not target_script_path.exists():
                  sys.exit(f"ERROR: Cannot find {script_filename} in {arena_root/script_subdir} or {arena_root}")

        target_args_def = {
            "model_name": "BLIP2", # This might need to be made dynamic or handled in eval_medical.py
            "dataset_path": None,
            "answer_path": None,
            "image_base_path": image_base_path_str,
            "batch_size": args.batch_size,
            "checkpoint_path": args.ft_checkpoint_path, # <<< ADD THIS if modifying eval_medical.py
            # Potentially add model_name_tag here too for eval_medical.py
        }
        results_subdir_suffix = "qa"
        run_title = f"{args.model_tag} QA Scoring"
        expected_filename_logic = lambda tmp_json_path_str: "qa_results.json" # Fixed by eval_medical.py
        # clean_results_filename will be set later

    else:
        sys.exit(f"ERROR: Invalid --eval-type specified: {eval_type}")

    print(f"Target evaluation script: {target_script_path}")

    # Use the potentially more specific dir_model_tag for results directory
    results_base_dir = ensure_dir(Path(args.results_dir).expanduser().resolve() / f"{dir_model_tag}_{input_type_tag}_{results_subdir_suffix}")

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

    print(f"\n================ Processing Modality: {target_modality} ================", flush=True)
    args_for_script = target_args_def.copy()
    args_for_script["dataset_path"] = str(mod_json_path)
    final_output_dir_for_modality = ensure_dir(results_base_dir / mod_slug)
    args_for_script["answer_path"] = str(final_output_dir_for_modality)

    print(f"\n--- Running {run_title} ---", flush=True)
    success = run_evaluation_script(target_script_path, args_for_script, python_exec=python_exec_path)

    if success:
        # For prefix scoring, the clean_results_filename depends on mod_slug
        if eval_type == 'prefix':
            clean_results_filename = f"{mod_slug}_prefix_results.json"
        elif eval_type == 'qa':
            clean_results_filename = "qa_results.json" # As assumed before

        expected_output_filename = expected_filename_logic(mod_slug) # Pass mod_slug for prefix
        expected_output_path = final_output_dir_for_modality / expected_output_filename
        
        if expected_output_path.exists():
            print(f"✓ {run_title} results found at: {expected_output_path}", flush=True) # Changed message slightly
            # The medical_blip2_finetuned.py now saves with the clean name directly.
            # So, renaming might not be necessary if expected_output_filename IS clean_results_filename.
            if expected_output_filename != clean_results_filename:
                 print(f"WARNING: Expected filename '{expected_output_filename}' differs from clean name '{clean_results_filename}'. This might indicate an issue or a need to adjust naming logic if files are not found.")
            # No rename needed if medical_blip2_finetuned.py saves with the correct final name.
        else:
             print(f"WARNING: {run_title} script succeeded but expected output file not found: {expected_output_path}")
    else:
        print(f"ERROR: {run_title} script {target_script_path} failed for modality {target_modality}.")

    print(f"Cleaning up temporary directory: {tmp_dir}")
    try:
        shutil.rmtree(tmp_dir)
    except Exception as e:
        print(f"WARNING: Failed to remove temporary directory {tmp_dir}: {e}", file=sys.stderr)

    print(f"\nFinished processing modality {target_modality} for {eval_type}. Results are in {final_output_dir_for_modality}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()