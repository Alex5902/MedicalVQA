#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import torch
from torch import nn
from tqdm import tqdm
from copy import deepcopy
import sys
import traceback
from collections import OrderedDict, defaultdict # Added defaultdict

# --- PEFT Imports ---
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        PeftModel,
        TaskType  # Assuming TaskType might be needed for LoraConfig
    )
    print("Successfully imported PEFT components.")
except ImportError:
    print("ERROR: Could not import PEFT. Please install with 'pip install peft'", file=sys.stderr)
    sys.exit(1)

# --- LAVIS Imports ---
try:
    from lavis.models import load_model_and_preprocess
    from lavis.models.blip2_models.blip2 import Blip2Base # For potential monkey-patch
    import contextlib

    @contextlib.contextmanager
    def no_op_maybe_autocast_merge(self, dtype=None): # Monkey-patch
        yield
    if hasattr(Blip2Base, 'maybe_autocast'): # Apply only if it exists
        Blip2Base.maybe_autocast = no_op_maybe_autocast_merge
        print("Applied no-op Blip2Base.maybe_autocast patch for merging.")
    else:
        print("Blip2Base.maybe_autocast not found, patch skipped (might be okay for merging).")
    print("Successfully imported LAVIS components.")
except ImportError:
    print("ERROR: Could not import LAVIS. Please ensure LAVIS is installed correctly.", file=sys.stderr)
    sys.exit(1)

# --- KnOTS Code Imports ---
import sys
from pathlib import Path

# --- KnOTS Code Imports ---
KNOTS_REPO_PATH = Path("/home/alex.ia/OmniMed/KnOTS") 
if KNOTS_REPO_PATH.exists() and KNOTS_REPO_PATH.is_dir():
    sys.path.insert(0, str(KNOTS_REPO_PATH))
    print(f"Added KnOTS repository to sys.path: {KNOTS_REPO_PATH}")
else:
    print(f"ERROR: KnOTS repository path not found at '{KNOTS_REPO_PATH}'. Please clone it and update the path.", file=sys.stderr)
    sys.exit(1)

# Now the imports should work if the files are at the root of KNOTS_REPO_PATH
try:
    from task_merger import SVDMerger 
    from ft_handlers import LoRAHandler
    from utils import get_mask_fn
    from masking_ops import masked_merge # Ensure this is importable or used internally by SVDMerger
    from collections import OrderedDict, defaultdict
    print("Successfully imported KnOTS components (SVDMerger, LoRAHandler, utils, masking_ops, collections).")
except ImportError as e:
    print(f"ERROR: Could not import KnOTS components even after adding to sys.path. Error: {e}", file=sys.stderr)
    print(f"Please check the file structure within '{KNOTS_REPO_PATH}' and ensure the modules are at the expected locations.", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

def get_qformer_attribute_name(model_instance):
    """Determines the attribute name of the Q-Former component."""
    if hasattr(model_instance, 'Qformer'):
        return 'Qformer'
    elif hasattr(model_instance, 'q_former'):
        return 'q_former'
    else:
        return None

def merge_qformer_knots_baseline(
    base_model_name: str,
    base_model_type: str,
    task_adapter_paths: list,
    qformer_lora_r: int,
    knots_param_handler_class, # e.g., LoRAHandler, but for SVDMerger it might expect GeneralHandler for DeltaW
    knots_svd_merge_config: dict, # Config for SVDMerger's merge method
    device: torch.device
):
    """
    Merges Q-Former LoRA adapters using KnOTS SVDMerger logic (SVD alignment + default merge).
    1. Calculates DeltaW for each Q-Former adapter using LoRAHandler.
    2. Initializes SVDMerger with these DeltaWs.
    3. Calls SVDMerger.transform() then SVDMerger.merge().
    4. Factorizes the merged DeltaW_full back into LoRA A and B matrices.
    """
    print(f"Loading base model for merging: {base_model_name} ({base_model_type})")
    base_model_cpu, vis_processors, _ = load_model_and_preprocess(
        name=base_model_name,
        model_type=base_model_type,
        is_eval=True,
        device="cpu"
    )
    base_model_cpu.eval()

    qformer_attr = get_qformer_attribute_name(base_model_cpu)
    if not qformer_attr:
        raise ValueError("Could not find Q-Former attribute ('Qformer' or 'q_former') on the base model.")
    
    original_qformer_component_cpu = getattr(base_model_cpu, qformer_attr)

    # --- 1. Get DeltaW for each task's Q-Former LoRA adapter ---
    # This list will contain state_dicts of DeltaW for the Q-Former layers from each task.
    task_qformer_delta_w_sds = []
    
    print("\n--- Extracting DeltaW from Q-Former LoRA Adapters ---")
    # Need a consistent set of layer names for DeltaW, get from first adapter
    # These names are like 'bert.encoder.layer.0.attention.self.query' (without .weight)
    qformer_lora_layer_base_names = None 

    for i, adapter_dir_path_str in enumerate(task_adapter_paths):
        adapter_path = Path(adapter_dir_path_str)
        print(f"  Processing adapter {i+1}/{len(task_adapter_paths)}: {adapter_path}")
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

        try:
            # Create a PeftModel of the Q-Former with the specific task's adapter
            qformer_for_task_peft = PeftModel.from_pretrained(
                deepcopy(original_qformer_component_cpu), # Use a fresh copy
                str(adapter_path),
                adapter_name=f"task_{i}"
            )
            qformer_for_task_peft.to(device) # LoRAHandler might do operations on device
            qformer_for_task_peft.eval()
            
            print(f"DEBUG: Keys in qformer_for_task_peft.state_dict() for adapter {adapter_path}:")
            for k_debug in list(qformer_for_task_peft.state_dict().keys()):
                if "lora" in k_debug: # Only print LoRA related keys
                    print(f"  {k_debug}")

            handler = LoRAHandler(qformer_for_task_peft) # LoRAHandler expects the PeftModel
            delta_w_sd = handler.get_ft_parameters() # Returns OrderedDict {layer_base_name: DeltaW_matrix}
            
            if qformer_lora_layer_base_names is None:
                qformer_lora_layer_base_names = list(delta_w_sd.keys())
            
            # Ensure all adapters have the same set of LoRA layers modified
            if set(delta_w_sd.keys()) != set(qformer_lora_layer_base_names):
                raise ValueError(f"Adapter {adapter_path} has different LoRA layers than the first adapter.")

            delta_w_sd_cpu = OrderedDict((k, v.cpu()) for k, v in delta_w_sd.items())
            task_qformer_delta_w_sds.append(delta_w_sd_cpu)
            print(f"    Extracted DeltaW for {len(delta_w_sd_cpu)} Q-Former LoRA layers.")
            
            del qformer_for_task_peft, handler, delta_w_sd # Clean up
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR: Failed to process adapter {adapter_path}: {e}", file=sys.stderr)
            traceback.print_exc()
            raise

    if not task_qformer_delta_w_sds:
        raise ValueError("No DeltaW state dicts were extracted. Check adapter paths and LoRAHandler.")

    # --- 2. Use KnOTS SVDMerger to merge these DeltaWs ---
    print("\n--- Merging Q-Former DeltaWs using KnOTS SVDMerger ---")
    
    # SVDMerger expects:
    # - finetuned_models: List of model instances (or can be state_dicts if param_handler works with them)
    # - pretrained_model: A single instance of the pretrained model (or its state_dict)
    # - param_handler: A class (like GeneralHandler) that can get parameters.
    # Since we already have DeltaWs, we can provide "dummy" finetuned_models as our DeltaW state_dicts,
    # and a "dummy" pretrained_model as a state_dict of zeros.
    # The `param_handler` for SVDMerger when given state_dicts should be GeneralHandler from KnOTS.
    
    try:
        from ft_handlers import GeneralHandler as KnotsGeneralHandler
    except ImportError:
        print("ERROR: Could not import GeneralHandler from KnOTS ft_handlers.py", file=sys.stderr)
        # Define a placeholder if not found, though this is risky
        class KnotsGeneralHandler:
            def __init__(self, model_sd): self.model_sd = model_sd
            def get_ft_parameters(self): return self.model_sd
        print("WARNING: Using a placeholder GeneralHandler. KnOTS merging might not work as expected.", file=sys.stderr)


    dummy_finetuned_models_for_svdmerger = [KnotsGeneralHandler(sd) for sd in task_qformer_delta_w_sds]
    
    # Create a "dummy" pretrained model state_dict of zeros, matching keys from the first DeltaW.
    # The keys in DeltaW are already the base LoRA layer names.
    ptm_zeros_sd = OrderedDict((k, torch.zeros_like(v)) for k, v in task_qformer_delta_w_sds[0].items())
    dummy_pretrained_model_for_svdmerger = KnotsGeneralHandler(ptm_zeros_sd) # Wrap in handler

    # Instantiate SVDMerger
    # It will call self.param_handler(self.pretrained_model).get_ft_parameters() for pt_params
    # and [self.param_handler(ft_model).get_ft_parameters() for ft_model in self.finetuned_models] for ftms_params
    # So, our handlers should just return the state_dicts we give them.
    
    svd_merger = SVDMerger(
        finetuned_models=task_qformer_delta_w_sds,  # Pass list of state_dicts directly
        pretrained_model=ptm_zeros_sd,              # Pass state_dict directly
        param_handler=KnotsGeneralHandler,          # The class that can init with state_dicts
        device=device,
        merge_config=knots_svd_merge_config
    )

    # The `transform` method calculates SVD and stores ingredients
    # It uses `self.get_task_directions` which is FT - PTM.
    # Since our "FT" is DeltaW and "PTM" is zeros, task_directions will be DeltaW.
    knots_svd_merge_config['svd_device'] = 'cpu'
    print("  Calling SVDMerger.transform() to compute SVD ingredients...")
    svd_merger.transform(merge_config=knots_svd_merge_config) # This populates svd_merger.ingredients
    print("  SVDMerger.transform() complete.")

    # The `merge` method uses the stored ingredients and the merge_config
    print(f"  Calling SVDMerger.merge() with config: {knots_svd_merge_config}")
    # SVDMerger.merge returns a *full model instance* with merged weights added to pretrained.
    # We need the merged_delta_w state_dict instead.
    # SVDMerger.merge() internally calculates `merged_sd` (which is DeltaW_merged_full)
    # before adding it to a base model. We need to capture that.
    
    # --- Replicating relevant parts of SVDMerger.merge to get merged_sd (DeltaW_merged_full) ---
    if svd_merger.ingredients is None:
        raise RuntimeError("SVDMerger.ingredients not populated. Call transform() first.")
    
    ingredients = deepcopy(svd_merger.ingredients)
    U_layerwise = ingredients['U']                     # Dict: {layer_name: U_tensor}
    task_Ss_layerwise = ingredients['task_Ss']         # List of Dicts: [{layer_name: S_tensor_task_i}, ...]
    task_sVs_layerwise = ingredients['task_sVs']       # List of Dicts: [{layer_name: sV_tensor_task_i}, ...]
                                                       # task_sVs are S_i @ V_i (components to merge)

    # Flatten task_sVs for masking and merging (as done in SVDMerger)
    # representation_helper is VectorOps
    flat_task_sVs_representations = svd_merger.representation_helper.directions_to_reps(task_sVs_layerwise)
    # flat_task_sVs_representations is now a list of flat vectors, one per task.

    # Get mask
    mask_fn_name = knots_svd_merge_config.get('mask_method', 'tv') # 'tv' is often a default in KnOTS configs
    mask_fn = get_mask_fn(mask_fn_name) # From knots.utils
    print(f"  Using mask_fn: {mask_fn_name}")
    
    # Pass relevant parts of knots_svd_merge_config to mask_fn
    masking_params = {k: v for k, v in knots_svd_merge_config.items() if k not in ['merging_type', 'mask_method']}
    masks_flat = mask_fn(flat_task_sVs_representations, **masking_params) # masks_flat is (num_tasks, D_flat)
    
    # Apply mask
    masked_sVs_flat = torch.vstack(flat_task_sVs_representations).clone().to(device) * masks_flat.to(device)
    
    # Perform merge
    # `masked_merge` is from masking_ops.py
    from masking_ops import masked_merge # Ensure this is correctly imported
    
    merged_sV_flat = masked_merge(
        vectors=masked_sVs_flat, # (num_tasks, D_flat)
        merge_func=knots_svd_merge_config.get('merging_type', 'mean'), # e.g., 'mean'
        weights=svd_merger.scaling_coeffs.to(device) if hasattr(svd_merger, 'scaling_coeffs') else None
    ) # merged_sV_flat is (D_flat)

    # Convert flat merged_sV back to state_dict structure
    # task_sVs_layerwise[0] is a template state_dict for one task {layer_name: sV_tensor}
    merged_sV_sd_layerwise = svd_merger.representation_helper.rep_to_state_dict(merged_sV_flat.cpu(), task_sVs_layerwise[0])

    # Reconstruct merged DeltaW for each layer
    merged_delta_w_full_sd_final = svd_merger.reconstruct_merged_sd(U_layerwise, merged_sV_sd_layerwise)
    
    # Convert from matrix format (used by SVDMerger) back to standard state_dict keys
    # ptm_reference_params in SVDMerger is used for this. We use our ptm_zeros_sd.
    merged_delta_w_full_sd_final = svd_merger.matrix_to_state_dict(merged_delta_w_full_sd_final, ptm_zeros_sd)
    print("  SVDMerger.merge() logic adapted, merged_delta_w_full_sd obtained.")
    # --- End of SVDMerger.merge() replication ---


    # --- 3. Factorize the merged DeltaW_full back into LoRA A and B ---
    print("\n--- Factorizing Merged Full DeltaW into LoRA A and B ---")
    merged_lora_A_sd = OrderedDict()
    merged_lora_B_sd = OrderedDict()

    # The keys in merged_delta_w_full_sd_final are the original LoRA layer base names
    for layer_base_name, delta_w_merged_full_layer_cpu in tqdm(merged_delta_w_full_sd_final.items(), desc="Factorizing layers"):
        delta_w_merged_full_layer = delta_w_merged_full_layer_cpu.to(device)
        try:
            U_factor, S_factor, V_factor_T = torch.linalg.svd(delta_w_merged_full_layer.to(torch.float64), full_matrices=False)
            U_factor = U_factor.to(torch.float32)
            S_factor = S_factor.to(torch.float32)
            V_factor_T = V_factor_T.to(torch.float32)

            # Keep only top 'qformer_lora_r' components
            U_r = U_factor[:, :qformer_lora_r]
            # S_r_diag = torch.diag(S_factor[:qformer_lora_r]) # Not needed for sqrt method
            V_r_T_top_r = V_factor_T[:qformer_lora_r, :] # Shape (r, in_features)
            
            sqrt_S_r = torch.sqrt(S_factor[:qformer_lora_r]) # Shape (r)

            # A_merged: (r, in_features) = diag(sqrt(S_r)) @ V_r_T_top_r
            A_merged_candidate = torch.diag(sqrt_S_r) @ V_r_T_top_r
            
            # B_merged: (out_features, r) = U_r @ diag(sqrt(S_r))
            B_merged_candidate = U_r @ torch.diag(sqrt_S_r)
            
            # PEFT stores weights as 'module_path.lora_A.default.weight'
            merged_lora_A_sd[f"{layer_base_name}.lora_A.default.weight"] = A_merged_candidate.cpu()
            merged_lora_B_sd[f"{layer_base_name}.lora_B.default.weight"] = B_merged_candidate.cpu()

        except Exception as e_factor:
            print(f"Error factorizing layer {layer_base_name}: {e_factor}", file=sys.stderr)
            traceback.print_exc()
            raise

    # --- 4. Create and Save New PEFT Adapter for the Q-Former component ---
    print("\n--- Creating and Saving Merged Q-Former PEFT Adapter ---")
    
    first_adapter_config_path = Path(task_adapter_paths[0]) / "adapter_config.json"
    if not first_adapter_config_path.exists():
        raise FileNotFoundError(f"Cannot find adapter_config.json in {task_adapter_paths[0]}")
    with open(first_adapter_config_path, 'r') as f:
        first_adapter_peft_config_dict = json.load(f)
    
    qformer_target_modules = first_adapter_peft_config_dict.get("target_modules")
    if not qformer_target_modules:
        raise ValueError("Could not determine target_modules for Q-Former from adapter config.")

    merged_lora_config = LoraConfig(
        r=qformer_lora_r,
        lora_alpha=first_adapter_peft_config_dict.get("lora_alpha", qformer_lora_r * 2),
        target_modules=qformer_target_modules,
        lora_dropout=first_adapter_peft_config_dict.get("lora_dropout", 0.05),
        bias=first_adapter_peft_config_dict.get("bias", "none"),
        task_type=None # Q-Former specific
    )

    # Get a fresh Q-Former component from the base model to apply the merged adapter to
    fresh_qformer_component_cpu = deepcopy(original_qformer_component_cpu)
    
    # Apply the LoRA config to make it a PeftModel
    # The model given to get_peft_model should be the actual nn.Module, not a PeftModel already
    merged_qformer_peft_model = get_peft_model(
        fresh_qformer_component_cpu,
        merged_lora_config,
        adapter_name="merged_knots_qformer_baseline" # Name for the new adapter
    )
    
    # Prepare the state_dict to load into this PeftModel
    # PEFT keys are typically `base_model.model.{original_module_path}.lora_A.default.weight`
    final_peft_state_dict_to_load = OrderedDict()
    adapter_name_for_peft_keys = "merged_knots_qformer_baseline" # This MUST match the adapter_name used in get_peft_model

    # Combine merged_lora_A_sd and merged_lora_B_sd for easier iteration
    temp_combined_factorized_sd = OrderedDict()
    temp_combined_factorized_sd.update(merged_lora_A_sd)
    temp_combined_factorized_sd.update(merged_lora_B_sd)

    print(f"\nDEBUG: Converting factorized LoRA keys to PEFT state_dict keys for adapter '{adapter_name_for_peft_keys}':")

    for key_from_factorization, tensor_val in temp_combined_factorized_sd.items():
        # key_from_factorization is like: "bert.encoder.layer.0.attention.self.query.lora_A.default.weight"
        
        parts = key_from_factorization.split('.')
        
        # Robustly find "lora_A" or "lora_B" and extract the base module path
        lora_type_str = None # Will be "A" or "B"
        lora_marker_idx = -1

        for i, part_name in enumerate(parts):
            if part_name == "lora_A":
                lora_type_str = "A"
                lora_marker_idx = i
                break
            elif part_name == "lora_B":
                lora_type_str = "B"
                lora_marker_idx = i
                break
        
        if lora_type_str is None or lora_marker_idx == -1:
            print(f"  WARNING: Could not parse lora_A/B from key: '{key_from_factorization}'. Skipping this key.")
            continue
            
        # The module path targeted by LoRA (e.g., "bert.encoder.layer.0.attention.self.query")
        target_module_path = ".".join(parts[:lora_marker_idx])
        
        # Ensure the rest of the key matches ".default.weight"
        expected_suffix = [f"lora_{lora_type_str}", "default", "weight"]
        actual_suffix = parts[lora_marker_idx:]
        
        if actual_suffix != expected_suffix:
            print(f"  WARNING: Key '{key_from_factorization}' does not have the expected '.lora_[A/B].default.weight' suffix. Actual suffix: {actual_suffix}. Skipping.")
            continue

        # Construct the key that PeftModel.load_state_dict expects for the new adapter.
        # Format for PeftModel created from a submodule (like QFormer):
        # "base_model.model.{TARGET_MODULE_PATH}.lora_{A_or_B}.{ADAPTER_NAME}.weight"
        peft_key = f"base_model.model.{target_module_path}.lora_{lora_type_str}.{adapter_name_for_peft_keys}.weight"
        
        final_peft_state_dict_to_load[peft_key] = tensor_val.cpu() # Ensure tensor is on CPU
        print(f"  Mapping: '{key_from_factorization}' -> '{peft_key}'")

    if not final_peft_state_dict_to_load:
        raise ValueError("Failed to construct any PEFT keys for loading. Check factorization and key parsing.")
    
    print(f"DEBUG: Example PEFT keys prepared for direct parameter setting: {list(final_peft_state_dict_to_load.keys())[:5]}")
    
    # Directly set the LoRA parameters in the merged_qformer_peft_model
    successful_sets = 0
    with torch.no_grad():
        for peft_key_name, tensor_value in final_peft_state_dict_to_load.items():
            # peft_key_name is like "base_model.model.module.path.lora_A.adapter.weight"
            
            current_module = merged_qformer_peft_model
            key_parts = peft_key_name.split('.')
            
            param_object_found = False
            try:
                # Navigate to the parent module of the parameter
                for part_idx, part_name in enumerate(key_parts[:-1]): # Iterate up to the second to last part
                    if hasattr(current_module, part_name):
                        current_module = getattr(current_module, part_name)
                    else:
                        print(f"  ERROR: Navigation failed. Module '{current_module.__class__.__name__}' has no attribute '{part_name}' for key '{peft_key_name}'.")
                        current_module = None # Mark as failed
                        break
                
                if current_module is not None:
                    param_name_leaf = key_parts[-1] # The actual parameter name (e.g., 'weight')
                    if hasattr(current_module, param_name_leaf):
                        param_to_set = getattr(current_module, param_name_leaf)
                        if isinstance(param_to_set, torch.nn.Parameter):
                            param_to_set.data.copy_(tensor_value.to(param_to_set.device, param_to_set.dtype))
                            # print(f"  Successfully set parameter: {peft_key_name}") # Can be verbose
                            successful_sets += 1
                            param_object_found = True
                        else:
                            print(f"  ERROR: Target '{peft_key_name}' is not a Parameter. Found type: {type(param_to_set)}")
                    else:
                        print(f"  ERROR: Parent module '{current_module.__class__.__name__}' has no parameter named '{param_name_leaf}' for key '{peft_key_name}'.")
                
                if not param_object_found:
                     print(f"  Failed to set parameter: {peft_key_name}")

            except Exception as e_set: # Catch any other exception during navigation/setting
                print(f"  ERROR: General exception while trying to set parameter '{peft_key_name}': {e_set}")
                traceback.print_exc(file=sys.stderr)


    print(f"Directly updated {successful_sets}/{len(final_peft_state_dict_to_load)} LoRA parameters in the new PeftModel.")
    if successful_sets != len(final_peft_state_dict_to_load):
        print("WARNING: Not all LoRA parameters were successfully set. Check DEBUG messages above.")


    # Mark non-LoRA parameters as not trainable for the saved adapter.
    # This part is still good to ensure correct state for saving.
    for n, p in merged_qformer_peft_model.named_parameters():
        # For the new adapter, only its LoRA weights should have requires_grad=True
        if adapter_name_for_peft_keys in n and "lora" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
            
    # Verify requires_grad status
    # print("DEBUG: Parameter requires_grad status after update:")
    # for n, p in merged_qformer_peft_model.named_parameters():
    #     if "lora" in n and adapter_name_for_peft_keys in n:
    #         print(f"  {n}: {p.requires_grad}")

    return merged_qformer_peft_model, qformer_attr


def main(args):
    if not torch.cuda.is_available() and args.device == "cuda":
        print("WARNING: CUDA specified but not available. Forcing CPU.", file=sys.stderr)
        args.device = "cpu"
    
    device = torch.device(args.device)
    print(f"Using device: {device}")

    if not args.task_adapter_paths:
        print("ERROR: No task adapter paths provided. Use --task_adapter_paths.", file=sys.stderr)
        sys.exit(1)
    if len(args.task_adapter_paths) < 2:
        print("ERROR: At least two adapters are required for merging.", file=sys.stderr)
        sys.exit(1)

    # This is the merge_config for SVDMerger.merge()
    # For "pure KnOTS" baseline (SVD alignment + default averaging of components):
    # - 'mask_method': 'tv' (from KnOTS utils, maps to tv_masking in masking_ops)
    # - 'topK': 100 (or 1.0, to ensure tv_masking effectively creates an all-ones mask)
    # - 'merging_type': 'mean' (for the final masked_merge step)
    knots_svd_merge_config = {
        "mask_method": args.knots_mask_method,  # e.g., "tv"
        "topK": args.knots_topK,                # e.g., 100 (for percent) or 1.0 (for fraction)
        "merging_type": args.knots_merging_type # e.g., "mean"
        # Add other params if SVDMerger.merge or its mask_fn/masked_merge require them
    }
    print(f"Using KnOTS SVDMerger configuration: {knots_svd_merge_config}")

    merged_qformer_peft_model, qformer_attr_name = merge_qformer_knots_baseline(
        base_model_name=args.base_model_name,
        base_model_type=args.base_model_type,
        task_adapter_paths=args.task_adapter_paths,
        qformer_lora_r=args.qformer_lora_r,
        knots_param_handler_class=None, # SVDMerger instantiates its own
        knots_svd_merge_config=knots_svd_merge_config,
        device=device
    )

    output_path = Path(args.output_merged_adapter_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the merged PEFT Q-Former model
    # This saves only the adapter weights and config for the Q-Former component
    merged_qformer_peft_model.save_pretrained(str(output_path))
    print(f"Successfully merged Q-Former LoRA adapter (KnOTS baseline) saved to: {output_path}")
    print(f"To use this merged adapter, load your base BLIP-2 model, get its '{qformer_attr_name}' component,")
    print(f"and then apply this saved adapter using: PeftModel.from_pretrained(qformer_component, '{output_path}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Q-Former LoRA adapters using KnOTS SVDMerger baseline logic.")
    parser.add_argument("--base_model_name", type=str, default="blip2_t5", help="Base LAVIS model name.")
    parser.add_argument("--base_model_type", type=str, default="pretrain_flant5xl", help="Specific model type for LAVIS.")
    parser.add_argument("--task_adapter_paths", type=str, nargs='+', required=True,
                        help="List of paths to the individual Q-Former LoRA adapter directories to be merged.")
    parser.add_argument("--qformer_lora_r", type=int, required=True,
                        help="The rank 'r' used when training the Q-Former LoRA adapters (e.g., 8).")
    parser.add_argument("--output_merged_adapter_path", type=str, required=True,
                        help="Path to save the merged Q-Former LoRA adapter directory.")
    
    # KnOTS SVDMerger specific parameters (for its .merge() method's merge_config)
    parser.add_argument("--knots_mask_method", type=str, default="tv",
                        help="Masking method for KnOTS SVDMerger (e.g., 'tv', 'ties'). 'tv' with topK=100 for baseline.")
    parser.add_argument("--knots_topK", type=float, default=100.0,
                        help="Top-K threshold for the mask_method (percentage if >1, fraction if <=1). E.g., 100 for no pruning with 'tv'.")
    parser.add_argument("--knots_merging_type", type=str, default="mean", choices=["mean", "sum"],
                        help="Final aggregation type for KnOTS SVDMerger (e.g., 'mean').")
    # Add other arguments if SVDMerger or its sub-functions take more from merge_config

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu).")

    args = parser.parse_args()
    main(args)