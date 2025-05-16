#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import time # For timing
import sys
import functools
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import importlib.util
import contextlib
import torch
import bitsandbytes.optim as bnb_optim
import torch.cuda.amp as amp # Import AMP components
import traceback # For printing full tracebacks on errors

# --- PEFT Imports ---
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        PeftModel,
        PeftConfig,
        TaskType,
        prepare_model_for_kbit_training # If using kbit training later, not used now
    )
    print("Successfully imported PEFT components.")
except ImportError:
    print("ERROR: Could not import PEFT. Please install with 'pip install peft'", file=sys.stderr)
    sys.exit(1)

# --- Start of Blip2Base.maybe_autocast monkey-patch ---
# This monkey patch disables the internal autocast within LAVIS's Blip2Base,
# ensuring that our global torch.amp.autocast context manages mixed precision consistently.
try:
    from lavis.models.blip2_models.blip2 import Blip2Base

    @contextlib.contextmanager
    def no_op_maybe_autocast(self, dtype=None): # dtype arg kept for signature match
        # This version completely disables internal autocast, relying on the global one.
        # print(f"DEBUG: NO-OP Blip2Base.maybe_autocast called by {type(self).__name__}, yielding without new autocast context.")
        yield # Yield without entering a new autocast context

    # Apply the monkey patch
    Blip2Base.maybe_autocast = no_op_maybe_autocast
    print("Successfully monkey-patched 'lavis.models.blip2_models.blip2.Blip2Base.maybe_autocast' to be a NO-OP.")

except ImportError:
    print("ERROR: Could not import Blip2Base for monkey-patching. Ensure LAVIS is installed correctly.", file=sys.stderr)
    sys.exit(1)
# --- End of Blip2Base.maybe_autocast monkey-patch ---

# --- Monkey-patching Block ---
# This needs to happen BEFORE VisionTransformer from lavis.models.eva_vit is imported
# by other LAVIS modules (like blip2.py)

# Path to your local modifications file
script_directory = Path(__file__).resolve().parent
local_mods_path = script_directory.parent / "local_lavis_mods" / "local_eva_vit_modifications.py"
local_eva_vit_modifications = None # Store the imported module globally

if local_mods_path.exists():
    print(f"Attempting to apply modifications from: '{local_mods_path}'")

    # Import the original modules that we will patch
    import lavis.models.eva_vit as original_eva_vit_module

    # Initialize local_mods_dir to None *before* the try block
    local_mods_dir = None # <--- Initialize here

    try:
        # Define local_mods_dir inside the try block (as it depends on local_mods_path)
        local_mods_dir = local_mods_path.parent
        # Add the directory of your local_mods_path to sys.path temporarily to allow import
        sys.path.insert(0, str(local_mods_dir))

        # Use importlib to handle potential ModuleNotFoundError more gracefully if the file exists but has errors
        # Note: Path to spec_from_file_location should be the full path to the file
        spec = importlib.util.spec_from_file_location("local_eva_vit_modifications", str(local_mods_path))
        local_eva_vit_modifications = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(local_eva_vit_modifications)

        # Perform the monkey-patching:
        # Replace the Block class in the original eva_vit module
        # with your ModifiedBlock class.
        # This also means ModifiedAttention will be used by ModifiedBlock.
        if hasattr(local_eva_vit_modifications, 'ModifiedBlock'):
            original_eva_vit_module.Block = local_eva_vit_modifications.ModifiedBlock
            print(f"Successfully monkey-patched 'lavis.models.eva_vit.Block' with 'ModifiedBlock'.")
        else:
            print("WARNING: 'ModifiedBlock' not found in local modifications. Patching skipped for Block.")

    except ImportError as e:
        print(f"ERROR: Could not import local modifications from '{local_mods_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
         print(f"ERROR: An unexpected error occurred during local modifications import/patching: {e}", file=sys.stderr)
         traceback.print_exc(file=sys.stderr)
         sys.exit(1)
    finally:
        # Clean up sys.path
        if local_mods_dir is not None and str(local_mods_dir) in sys.path:
             try:
                sys.path.pop(sys.path.index(str(local_mods_dir)))
             except ValueError:
                 pass 
else:
    print(f"WARNING: Local modifications file not found at '{local_mods_path}'. Using installed LAVIS version.")

# Global state to track the first detected NaN/Inf for debugging
nan_detected_module_info = {"name": None, "output_stats": None, "rank": None}

# Assuming LAVIS is in your environment
from lavis.models import load_model_and_preprocess
import torch.optim as optim

def setup_ddp():
    """Initializes DDP using environment variables and sets device."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ:
        rank_str = os.environ["RANK"]
        world_size_str = os.environ["WORLD_SIZE"]
        local_rank_str = os.environ["LOCAL_RANK"]
        if rank_str == '0':
            print(f"DDP: Found env vars: RANK={rank_str}, LOCAL_RANK={local_rank_str}, WORLD_SIZE={world_size_str}")
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(local_rank_str) 
        if rank == 0: 
            print(f"DDP: Process {rank}/{world_size} (local rank {local_rank}) successfully initialized process group.")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if rank == 0: 
            print(f"DDP: Process {rank} set to use device {device}.")
    else:
        print("Exiting: DDP environment variables (RANK, WORLD_SIZE, LOCAL_RANK) not found. Launch with torchrun.", file=sys.stderr)
        sys.exit(1) 
    return rank, world_size, device

def cleanup_ddp():
    """Cleans up DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()

# Helper function to get a module by its path string
def get_module_by_path(model_root, path_str):
    if isinstance(model_root, DDP):
        model_root = model_root.module
    try:
        module = model_root
        for part in path_str.split('.'):
            module = getattr(module, part)
        return module
    except AttributeError:
        return None

# Forward hook function
def get_nan_checking_hook(module_name, current_rank):
    def hook(module, input_tensors, output_tensors): 
        global nan_detected_module_info
        if nan_detected_module_info["name"] is not None and nan_detected_module_info["rank"] == current_rank:
             return 
        def check_tensor_for_nan(tensor, is_input):
             global nan_detected_module_info
             if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cuda':
                  return False
             if tensor.numel() > 0 and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
                 stats = {
                     "shape": list(tensor.shape), 
                     "min": tensor.min().item() if torch.isfinite(tensor).all() else float('nan'), 
                     "max": tensor.max().item() if torch.isfinite(tensor).all() else float('nan'),
                     "mean": tensor.mean().item() if torch.isfinite(tensor).all() else float('nan'),
                     "dtype": str(tensor.dtype),
                     "has_nan": torch.isnan(tensor).any().item(),
                     "has_inf": torch.isinf(tensor).any().item(),
                     "num_nans": torch.isnan(tensor).sum().item(),
                     "num_infs": torch.isinf(tensor).sum().item()
                 }
                 prefix = "INPUT to" if is_input else "OUTPUT of"
                 if current_rank == 0:
                     print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
                     print(f"!!! Rank {current_rank}: NaN/Inf DETECTED IN **{prefix.split(' ')[0]}** of module: {module_name} !!!", file=sys.stderr)
                     print(f"    Stats: {stats}", file=sys.stderr)
                     print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", file=sys.stderr)
                 nan_detected_module_info["name"] = f"{prefix} {module_name}"
                 nan_detected_module_info["output_stats"] = stats 
                 nan_detected_module_info["rank"] = current_rank
                 return True 
             return False 
        if isinstance(input_tensors, tuple):
            for i, t in enumerate(input_tensors):
                 if check_tensor_for_nan(t, is_input=True):
                       return 
        elif isinstance(input_tensors, torch.Tensor):
             if check_tensor_for_nan(input_tensors, is_input=True):
                  return 
        if nan_detected_module_info["name"] is not None and nan_detected_module_info["rank"] == current_rank:
             return 
        if isinstance(output_tensors, tuple):
            for i, t in enumerate(output_tensors):
                 if isinstance(t, torch.Tensor):
                      if check_tensor_for_nan(t, is_input=False):
                           pass 
        elif isinstance(output_tensors, torch.Tensor) and output_tensors.device.type == 'cuda':
             check_tensor_for_nan(output_tensors, is_input=False)
    return hook


# --- Custom Dataset for Training ---
class MedicalVqaFineTuningDataset(Dataset):
    def __init__(self, json_path, image_base_path, vis_processor, txt_processor,
                 max_input_len=128, max_target_len=64, prompt_type="default"):
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Loading data from: {json_path}")
        try:
            with open(json_path, 'r') as f:
                self.data = json.load(f)
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"  Loaded {len(self.data)} items.")
        except Exception as e:
            print(f"FATAL: Could not load or parse JSON from {json_path}: {e}", file=sys.stderr)
            raise
        self.image_base_path = Path(image_base_path)
        self.vis_processor = vis_processor
        self.txt_processor = txt_processor 
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.prompt_type = prompt_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        relative_img_path = item['image_path']
        full_img_path = self.image_base_path / relative_img_path
        try:
            raw_image = Image.open(full_img_path).convert("RGB")
            image = self.vis_processor(raw_image)
        except Exception as e:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Error loading image {full_img_path} for item {item.get('question_id', idx)}: {e}", file=sys.stderr)
            return None 
        question_text = item['question']
        gt_answer_text = item['gt_answer']
        if not question_text or not gt_answer_text:
             pass 
        return {
        "image": image,
        "text_input": question_text,    
        "text_output": gt_answer_text,  
        "question_id": item.get("question_id", f"idx_{idx}")
        }


# Custom collate_fn to handle None items (from image loading errors)
def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)


def main(args):
    rank, world_size, ddp_device = setup_ddp()
    is_main_process = (rank == 0)

    if torch.cuda.is_available():
        try:
            a = torch.randn(1000, 1000, device=ddp_device)
            b = torch.randn(1000, 1000, device=ddp_device)
            c = a @ b 
            if is_main_process:
                 print(f"Basic CUDA test successful on {ddp_device}. Result shape: {c.shape}")
            del a, b, c 
        except Exception as e:
            if is_main_process:
                print(f"Basic CUDA test failed on {ddp_device}: {e}", file=sys.stderr)
            cleanup_ddp()
            sys.exit(f"Basic CUDA test failed on {ddp_device}")
    elif is_main_process:
         print("CUDA not available, skipping basic CUDA test.")

    if is_main_process:
        print(f"Running in DDP mode with {world_size} GPU(s). Main process is RANK {rank}.")

    torch.manual_seed(args.seed + rank) 
    torch.cuda.manual_seed(args.seed + rank)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True      

    start_time = time.time()
    print(f"Using device: {ddp_device}")

    use_amp = not args.no_amp and ddp_device.type == 'cuda'
    amp_dtype = torch.float16 
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16 
        if is_main_process: print(f"Autocast dtype will be: {amp_dtype} (BF16 supported)")
    elif use_amp:
        amp_dtype = torch.float16 
        if is_main_process: print(f"Autocast dtype will be: {amp_dtype} (FP16, BF16 not supported or not primary)")
    else:
        amp_dtype = torch.float32 
        if is_main_process: print(f"Autocast dtype will be: {amp_dtype} (AMP disabled)")

    # --- Load Model and Processors ---
    if is_main_process:
        print(f"Loading base model: {args.base_model_name} ({args.base_model_type})")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=args.base_model_name, 
        model_type=args.base_model_type, 
        is_eval=False, 
        device=ddp_device # Load directly to target device to save memory on rank 0
    )

    if args.load_checkpoint:
        if not os.path.exists(args.load_checkpoint):
            if is_main_process:
                print(f"FATAL: Checkpoint file not found at {args.load_checkpoint}", file=sys.stderr)
            if dist.is_initialized(): 
                dist.barrier()
                cleanup_ddp()
            sys.exit(1)
        if is_main_process:
            print(f"Loading checkpoint weights from: {args.load_checkpoint}")
        try:
            checkpoint_state_dict = torch.load(args.load_checkpoint, map_location='cpu')
            model_state_to_load = {}
            for k, v in checkpoint_state_dict.items():
                name = k[7:] if k.startswith('module.') else k  
                model_state_to_load[name] = v
            load_msg = model.load_state_dict(model_state_to_load, strict=False)
            if is_main_process:
                print(f"Successfully loaded checkpoint weights. Load message: {load_msg}")
                if load_msg.missing_keys:
                    print(f"WARNING: Missing keys during checkpoint load: {load_msg.missing_keys}")
                if load_msg.unexpected_keys:
                    print(f"WARNING: Unexpected keys during checkpoint load: {load_msg.unexpected_keys}")
            model.to(ddp_device)
        except Exception as e:
            if is_main_process:
                print(f"FATAL: Failed to load checkpoint {args.load_checkpoint}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            if dist.is_initialized():
                dist.barrier()
                cleanup_ddp()
            sys.exit(1)
        if dist.is_initialized():
            dist.barrier()
        if is_main_process:
            print("Checkpoint loaded and model synchronized across DDP processes.")
    else:
        if is_main_process:
            print("No checkpoint provided via --load_checkpoint. Starting from base pre-trained model weights.")

    if is_main_process:
        print(f"Converting entire model to float32 before AMP context and PEFT.")
    model = model.to(dtype=torch.float32) 
    if next(model.parameters()).dtype != torch.float32:
        if is_main_process: print(f"Model initial dtype is {next(model.parameters()).dtype}, converting to float32.")
        model = model.to(dtype=torch.float32)
    elif is_main_process:
        print(f"Model initial data type after loading: {next(model.parameters()).dtype}") 


    # --- LoRA/PEFT Integration ---
    # This section should be BEFORE DDP wrapping
    # It modifies `model.llm_model` and `model.q_former` in place.

    # Utility to print module names for LoRA target_modules configuration
    if args.inspect_modules:
        if is_main_process:
            print("\n--- Inspecting T5 (LLM) Modules for LoRA ---")
            llm_component_to_inspect = None
            # Check for t5_model first as it's specific to Blip2T5
            if hasattr(model, 't5_model'): 
                llm_component_to_inspect = model.t5_model
                print("  Inspecting model.t5_model for T5 components...")
            elif hasattr(model, 'llm_model'): # Fallback for other models
                llm_component_to_inspect = model.llm_model
                print("  Inspecting model.llm_model for T5 components...")
            else:
                print("  Neither model.llm_model nor model.t5_model found for T5 inspection.")

            if llm_component_to_inspect:
                for name, _module in llm_component_to_inspect.named_modules():
                    if isinstance(_module, torch.nn.Linear):
                        print(f"  T5 Linear Layer: {name} (type: {type(_module).__name__})")
            
            print("\n--- Inspecting Q-Former Modules for LoRA ---")
            qformer_to_inspect = None
            if hasattr(model, 'Qformer'): # LAVIS BLIP-2 often uses model.Qformer
                qformer_to_inspect = model.Qformer
            elif hasattr(model, 'q_former'): 
                qformer_to_inspect = model.q_former
            else:
                print("  model.Qformer or model.q_former not found.")

            if qformer_to_inspect:
                for name, _module in qformer_to_inspect.named_modules():
                    if isinstance(_module, torch.nn.Linear):
                        print(f"  Q-Former Linear Layer: {name} (type: {type(_module).__name__})")
            
            print("-- Inspection Complete --\n")
        
        if dist.is_initialized(): dist.barrier()
        cleanup_ddp()
        sys.exit(0)

    is_any_lora_active_for_training = False

    # Determine the attribute name for the T5 language model
    t5_model_attr_name = None
    if hasattr(model, 't5_model'): # Prioritize t5_model for Blip2T5
        t5_model_attr_name = 't5_model'
    elif hasattr(model, 'llm_model'): # Fallback for other potential structures
        t5_model_attr_name = 'llm_model'

    if is_main_process and t5_model_attr_name:
        print(f"  Identified T5 model component attribute as: model.{t5_model_attr_name}")
    elif is_main_process:
        print(f"  CRITICAL WARNING: Could not identify T5 model component (t5_model or llm_model). T5 LoRA operations might fail.", file=sys.stderr)

    # Determine attribute name for Q-Former
    qformer_attr_name = None
    if hasattr(model, 'Qformer'):
        qformer_attr_name = 'Qformer'
    elif hasattr(model, 'q_former'):
        qformer_attr_name = 'q_former'

    if is_main_process and qformer_attr_name:
        print(f"  Identified Q-Former component attribute as: model.{qformer_attr_name}")
    elif is_main_process:
        print(f"  CRITICAL WARNING: Could not identify Q-Former component. Q-Former LoRA operations might fail.", file=sys.stderr)

    # 1. Load PREVIOUSLY trained LoRA adapters (and freeze them)
    if args.lora_load_previous_adapters_config:
        if is_main_process: print(f"Loading PREVIOUS LoRA adapter config from: {args.lora_load_previous_adapters_config}")
        try:
            with open(args.lora_load_previous_adapters_config, 'r') as f:
                previous_adapters_list = json.load(f)
        except Exception as e:
            if is_main_process: print(f"FATAL: Could not load or parse previous adapters JSON {args.lora_load_previous_adapters_config}: {e}", file=sys.stderr)
            if dist.is_initialized(): dist.barrier()
            cleanup_ddp()
            sys.exit(1)

        for adapter_info in previous_adapters_list:
            comp_path = adapter_info["path"]
            comp_name = adapter_info["name"]
            comp_component_key = adapter_info["component"].lower() # "t5" or "qformer"
            
            if is_main_process: print(f"  Loading previous adapter '{comp_name}' for '{comp_component_key}' from '{comp_path}' (frozen).")

            target_model_component_instance = None
            current_component_attr_name = None

            if comp_component_key == "t5":
                if t5_model_attr_name and hasattr(model, t5_model_attr_name):
                    target_model_component_instance = getattr(model, t5_model_attr_name)
                    current_component_attr_name = t5_model_attr_name
                else:
                    if is_main_process: print(f"ERROR: T5 model attribute ('{t5_model_attr_name}') not found on model for loading adapter '{comp_name}'. Skipping.", file=sys.stderr)
                    continue
            elif comp_component_key == "qformer":
                if qformer_attr_name and hasattr(model, qformer_attr_name):
                    target_model_component_instance = getattr(model, qformer_attr_name)
                    current_component_attr_name = qformer_attr_name
                else:
                    if is_main_process: print(f"ERROR: Q-Former attribute ('{qformer_attr_name}') not found on model for loading adapter '{comp_name}'. Skipping.", file=sys.stderr)
                    continue
            else:
                if is_main_process: print(f"ERROR: Unknown component '{comp_component_key}' for adapter '{comp_name}'. Skipping.", file=sys.stderr)
                continue
            
            try:
                peft_comp_instance = None
                if not isinstance(target_model_component_instance, PeftModel):
                    peft_comp_instance = PeftModel.from_pretrained(target_model_component_instance, comp_path, adapter_name=comp_name, is_trainable=False)
                else: 
                    target_model_component_instance.load_adapter(comp_path, adapter_name=comp_name, is_trainable=False)
                    peft_comp_instance = target_model_component_instance 
                
                if current_component_attr_name: # Ensure we have an attribute name to set
                    setattr(model, current_component_attr_name, peft_comp_instance)

                if is_main_process: print(f"    Adapter '{comp_name}' loaded for {comp_component_key} (model.{current_component_attr_name}).")

            except Exception as e:
                if is_main_process: print(f"ERROR loading previous adapter '{comp_name}' for {comp_component_key} (model.{current_component_attr_name}): {e}", file=sys.stderr)
    
    # 2. Configure and add NEW LoRA adapter for the CURRENT training stage
    if args.lora_train_current_adapter_name:
        is_any_lora_active_for_training = True
        if is_main_process: print(f"Configuring NEW LoRA adapter '{args.lora_train_current_adapter_name}' for the current training stage.")

        if is_main_process: print("  Freezing all base model parameters for LoRA training.")
        for param in model.parameters(): 
            param.requires_grad = False 

        # Configure T5 LoRA for current stage
        if args.lora_t5_target_modules: 
            if not (t5_model_attr_name and hasattr(model, t5_model_attr_name)):
                if is_main_process: print(f"ERROR: T5 model attribute ('{t5_model_attr_name}') not found for T5 LoRA configuration. Skipping T5 LoRA.", file=sys.stderr)
            else:
                current_t5_adapter_name = f"{args.lora_train_current_adapter_name}_t5"
                if is_main_process: print(f"  Configuring T5 LoRA (r={args.lora_r_t5}, alpha={args.lora_alpha_t5}) for adapter '{current_t5_adapter_name}' on model.{t5_model_attr_name}")
                t5_lora_config = LoraConfig(
                    r=args.lora_r_t5,
                    lora_alpha=args.lora_alpha_t5,
                    target_modules=args.lora_t5_target_modules,
                    lora_dropout=args.lora_dropout_t5,
                    bias=args.lora_bias_t5, 
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    modules_to_save=args.lora_t5_modules_to_save 
                )
                
                t5_component_instance = getattr(model, t5_model_attr_name)
                peft_t5_instance = None
                if not isinstance(t5_component_instance, PeftModel):
                    peft_t5_instance = get_peft_model(t5_component_instance, t5_lora_config, adapter_name=current_t5_adapter_name)
                else:
                    t5_component_instance.add_adapter(current_t5_adapter_name, t5_lora_config)
                    peft_t5_instance = t5_component_instance
                
                if peft_t5_instance:
                    peft_t5_instance.set_adapter(current_t5_adapter_name) 
                    setattr(model, t5_model_attr_name, peft_t5_instance)
                    if is_main_process: 
                        print(f"  Active T5 adapter on model.{t5_model_attr_name}: {peft_t5_instance.active_adapter}")
                        peft_t5_instance.print_trainable_parameters()
                elif is_main_process:
                    print(f"  ERROR: Failed to create/modify PeftModel for T5 component model.{t5_model_attr_name}", file=sys.stderr)


        # Configure Q-Former LoRA for current stage
        if args.lora_qformer_target_modules: 
            if not (qformer_attr_name and hasattr(model, qformer_attr_name)):
                if is_main_process: print(f"ERROR: Q-Former attribute ('{qformer_attr_name}') not found for Q-Former LoRA configuration. Skipping Q-Former LoRA.", file=sys.stderr)
            else:
                qformer_component_instance_to_adapt = getattr(model, qformer_attr_name)
                current_qformer_adapter_name = f"{args.lora_train_current_adapter_name}_qformer"
                if is_main_process: print(f"  Configuring Q-Former LoRA (r={args.lora_r_qformer}, alpha={args.lora_alpha_qformer}) for adapter '{current_qformer_adapter_name}' on model.{qformer_attr_name}")
                qformer_lora_config = LoraConfig(
                    r=args.lora_r_qformer,
                    lora_alpha=args.lora_alpha_qformer,
                    target_modules=args.lora_qformer_target_modules,
                    lora_dropout=args.lora_dropout_qformer,
                    bias=args.lora_bias_qformer,
                    task_type=None, 
                    modules_to_save=args.lora_qformer_modules_to_save
                )
                
                peft_qf_instance = None
                if not isinstance(qformer_component_instance_to_adapt, PeftModel):
                    peft_qf_instance = get_peft_model(qformer_component_instance_to_adapt, qformer_lora_config, adapter_name=current_qformer_adapter_name)
                else:
                    qformer_component_instance_to_adapt.add_adapter(current_qformer_adapter_name, qformer_lora_config)
                    peft_qf_instance = qformer_component_instance_to_adapt
                
                if peft_qf_instance:
                    peft_qf_instance.set_adapter(current_qformer_adapter_name)
                    setattr(model, qformer_attr_name, peft_qf_instance) 
                    if is_main_process: 
                        print(f"  Active Q-Former adapter on model.{qformer_attr_name}: {peft_qf_instance.active_adapter}")
                        peft_qf_instance.print_trainable_parameters()
                elif is_main_process:
                    print(f"  ERROR: Failed to create/modify PeftModel for Q-Former component model.{qformer_attr_name}", file=sys.stderr)
    
    if is_main_process and is_any_lora_active_for_training:
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        if total_params > 0 : # Avoid division by zero if model is empty for some reason
             print(f"PEFT Overall: Trainable params: {total_trainable_params} || All params: {total_params} || Trainable %: {100 * total_trainable_params / total_params:.4f}")
        else:
             print(f"PEFT Overall: Trainable params: {total_trainable_params} || All params: {total_params} (Model appears empty!)")


    # >>> ENABLE GRADIENT CHECKPOINTING <<<
    if is_main_process: print("Attempting to enable gradient checkpointing...")

    # T5 component  
    if t5_model_attr_name and hasattr(model, t5_model_attr_name):
        t5_component_for_gc = getattr(model, t5_model_attr_name)
        if isinstance(t5_component_for_gc, PeftModel):
            # For PeftModel, enable on the underlying base model.
            # The `base_model.model` attribute path is common in PEFT.
            if hasattr(t5_component_for_gc, 'base_model') and hasattr(t5_component_for_gc.base_model, 'model') and \
            hasattr(t5_component_for_gc.base_model.model, "gradient_checkpointing_enable"):
                try:
                    t5_component_for_gc.base_model.model.gradient_checkpointing_enable()
                    if is_main_process: print(f"  Successfully enabled gradient checkpointing for T5 base model (via PeftModel on model.{t5_model_attr_name}).")
                except Exception as e_gc_t5:
                    if is_main_process: print(f"  Failed to enable gradient checkpointing for T5 (via PeftModel on model.{t5_model_attr_name}): {e_gc_t5}")
            elif hasattr(t5_component_for_gc, "gradient_checkpointing_enable"): # If PeftModel itself supports it
                try:
                    t5_component_for_gc.gradient_checkpointing_enable()
                    if is_main_process: print(f"  Successfully enabled gradient checkpointing directly on T5 PeftModel (model.{t5_model_attr_name}).")
                except Exception as e_gc_t5_peft:
                    if is_main_process: print(f"  Failed to enable gradient checkpointing directly on T5 PeftModel (model.{t5_model_attr_name}): {e_gc_t5_peft}")
            else:
                if is_main_process: print(f"  T5 PeftModel (model.{t5_model_attr_name}) does not have a clear 'gradient_checkpointing_enable' method on base_model.model or itself.")
        elif hasattr(t5_component_for_gc, "gradient_checkpointing_enable"): # If it's not a PeftModel but has the method
            try:
                t5_component_for_gc.gradient_checkpointing_enable()
                if is_main_process: print(f"  Successfully enabled gradient checkpointing for model.{t5_model_attr_name}.")
            except Exception as e_gc_t5_direct:
                if is_main_process: print(f"  Failed to enable gradient checkpointing for model.{t5_model_attr_name}: {e_gc_t5_direct}")
        else:
            if is_main_process: print(f"  T5 model component (model.{t5_model_attr_name}) does not have a clear 'gradient_checkpointing_enable' method.")
    elif is_main_process:
        print("  T5 model component attribute not identified or model does not have it, skipping T5 gradient checkpointing.")


    # Visual encoder GC (should be unaffected by T5/QFormer LoRA)
    if hasattr(model, "visual_encoder") and hasattr(model.visual_encoder, "gradient_checkpointing_enable"):
        try:
            model.visual_encoder.gradient_checkpointing_enable()
            if is_main_process: print("  Successfully enabled gradient checkpointing for visual encoder.")
        except Exception as e_gc_vit:
            if is_main_process: print(f"  Failed to enable gradient checkpointing for visual encoder: {e_gc_vit}")
    elif is_main_process:
        print("  Visual encoder does not have 'gradient_checkpointing_enable' method.")


    if is_main_process:
        print(f"\n--- Runtime CUDA/cuDNN Info (inside script) ---")
        print(f"  torch.version.cuda: {torch.version.cuda}")
        print(f"  torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
        print(f"  torch.backends.cudnn.version(): {torch.backends.cudnn.version()}") # Corrected: Add parentheses
        if torch.cuda.is_available():
            print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
            print(f"  Current device: {ddp_device}")
            print(f"  torch.cuda.get_device_name({ddp_device.index}): {torch.cuda.get_device_name(ddp_device.index)}")
        else:
            print("  CUDA is not available according to torch.cuda.is_available()")
        print(f"--------------------------------------------------\n")

    blip_base_instance = None
    if hasattr(model, 'blip_model') and hasattr(model.blip_model, 'use_half_precision'): blip_base_instance = model.blip_model
    elif hasattr(model, 'inner_model') and hasattr(model.inner_model, 'use_half_precision'): blip_base_instance = model.inner_model
    elif hasattr(model, 'use_half_precision'): blip_base_instance = model
    if blip_base_instance is not None:
        if hasattr(blip_base_instance, 'use_half_precision'):
            if is_main_process: print(f"Found use_half_precision on {type(blip_base_instance).__name__}. Current value: {blip_base_instance.use_half_precision}")
            if blip_base_instance.use_half_precision: 
                if is_main_process: print("Setting use_half_precision to False to rely on global AMP.")
                blip_base_instance.use_half_precision = False
            # else: # This else branch is not strictly necessary, just for verbosity
            #     if is_main_process: print("'use_half_precision' is already False.")
        else:
            if is_main_process: print(f"WARNING: {type(blip_base_instance).__name__} does not have 'use_half_precision' attribute.")
    elif is_main_process: print("WARNING: Could not find Blip2Base instance to modify use_half_precision.")
    if is_main_process: print(f"Model data type after modifications: {next(model.parameters()).dtype}")

    if is_main_process:
        if hasattr(model, 'visual_encoder') and hasattr(model.visual_encoder, 'patch_embed') and hasattr(model.visual_encoder.patch_embed, 'proj'):
            print(f"Data type of visual_encoder.patch_embed.proj.weight: {model.visual_encoder.patch_embed.proj.weight.dtype}")
        if hasattr(model, 'visual_encoder') and hasattr(model.visual_encoder, 'patch_embed') and hasattr(model.visual_encoder.patch_embed, 'proj') and \
           hasattr(model.visual_encoder.patch_embed.proj, 'bias') and model.visual_encoder.patch_embed.proj.bias is not None:
             print(f"Data type of visual_encoder.patch_embed.proj.bias: {model.visual_encoder.patch_embed.proj.bias.dtype}")


    if is_main_process and args.print_model_structure: # Add CLI arg for this
        print("\n\nDEBUG: Full model structure:")
        print(model) 
        print("\n\nDEBUG: All named modules (use these paths for hooks):")
        model_for_named_modules = model.module if isinstance(model, DDP) else model
        for name, _module_obj in model_for_named_modules.named_modules():
            if name: 
                print(name)
        print("--- End of named modules ---\n\n")


    # --- Parameter Freezing and Counting (if not using LoRA for current training) ---
    total_params_count = sum(p.numel() for p in model.parameters()) 
    trainable_params_count = 0

    if not is_any_lora_active_for_training: 
        if args.unfreeze_all:
            if is_main_process: print("Unfreezing all model parameters for full fine-tuning (LoRA not active for training)...")
            for param in model.parameters():
                param.requires_grad = True
            trainable_params_count = total_params_count
        else: # Default partial fine-tuning if not unfreeze_all and no LoRA training
            if is_main_process: print("LoRA not active for training. Defaulting to tuning Q-Former and t5_proj.")
            if is_main_process: print("Freezing all model parameters initially...")
            for param in model.parameters():
                 param.requires_grad = False
            if is_main_process: print("Unfreezing Q-Former and t5_proj parameters for partial fine-tuning...")
            
            qformer_attr_name_to_check = 'Qformer' if hasattr(model, 'Qformer') else 'q_former' # Determine which attribute name is present
            # Iterate using model.named_parameters() to get correct names even if DDP wrapped (though DDP wrap is later)
            model_to_iterate_params = model.module if isinstance(model, DDP) else model # Should be before DDP
            for name, param in model_to_iterate_params.named_parameters():
                # Check if the parameter name corresponds to the Q-Former or the t5_proj layer
                # Need to be careful with naming, e.g. model.Qformer vs model.q_former
                is_qformer_param = False
                if hasattr(model_to_iterate_params, 'Qformer') and name.startswith('Qformer.'):
                    is_qformer_param = True
                elif hasattr(model_to_iterate_params, 'q_former') and name.startswith('q_former.'):
                    is_qformer_param = True
                
                if is_qformer_param or "t5_proj" in name:
                    param.requires_grad = True
                    trainable_params_count += param.numel()
        if is_main_process:
            print(f"Trainable parameters (non-LoRA): {trainable_params_count} / {total_params_count}")
            if trainable_params_count == 0 and not args.lora_train_current_adapter_name : # Only fatal if not planning LoRA
                print("FATAL: No trainable parameters (non-LoRA)! Check model structure and unfreeze logic.", file=sys.stderr)
                if dist.is_initialized(): dist.barrier(); cleanup_ddp(); sys.exit(1)
    else: 
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad) 
        if is_main_process:
            # This was already printed by PEFT summary, but good to have a consistent place
            print(f"Trainable parameters (LoRA active, from model.parameters()): {trainable_params_count} / {total_params_count}")
            if trainable_params_count == 0:
                print("FATAL: No trainable LoRA parameters! Check LoRA config and target_modules.", file=sys.stderr)
                if dist.is_initialized(): dist.barrier(); cleanup_ddp(); sys.exit(1)


    # --- DDP Model Wrapping ---
    if world_size > 1:
        model = DDP(model, device_ids=[ddp_device.index], find_unused_parameters=args.ddp_find_unused_parameters)
        if is_main_process: print(f"Model wrapped with DDP. find_unused_parameters={args.ddp_find_unused_parameters}")

    trainable_params_count_check = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process:
        print(f"Number of trainable parameters (after DDP wrap check): {trainable_params_count_check} / {total_params_count}")
        if trainable_params_count_check != trainable_params_count: # Compare with the count from before DDP
            print(f"WARNING: Trainable parameter count changed after DDP wrap! Before DDP (logic based): {trainable_params_count}, After DDP (summing requires_grad): {trainable_params_count_check}", file=sys.stderr)


    text_processor_for_training = None 
    tokenizer_found = False
    if "eval" in txt_processors:
         text_processor_for_training = txt_processors["eval"]
         if is_main_process: print("Using tokenizer from txt_processors['eval'].")
         tokenizer_found = True
    elif "train" in txt_processors: 
         text_processor_for_training = txt_processors["train"]
         if is_main_process: print("Using tokenizer from txt_processors['train'].")
         tokenizer_found = True
    if not tokenizer_found:
         if is_main_process:
              print(f"\nFATAL: Could not find a suitable tokenizer in txt_processors.", file=sys.stderr)
              print(f"Available keys in txt_processors: {list(txt_processors.keys())}", file=sys.stderr) # Print keys if dict
         if dist.is_initialized(): dist.barrier()
         cleanup_ddp(); sys.exit(1)

    model.train() 

    train_dataset = MedicalVqaFineTuningDataset( 
        args.train_json_path, args.image_base_path,
        vis_processors["eval"], text_processor_for_training,
        max_input_len=args.max_input_len, max_target_len=args.max_target_len
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, collate_fn=collate_fn_skip_none, sampler=train_sampler,
        pin_memory=True 
    )
    val_dataloader = None 
    if args.val_json_path:
        val_dataset = MedicalVqaFineTuningDataset(
            args.val_json_path, args.image_base_path,
            vis_processors["eval"], text_processor_for_training, 
            max_input_len=args.max_input_len, max_target_len=args.max_target_len
        )
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed)
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            num_workers=args.num_workers, collate_fn=collate_fn_skip_none, sampler=val_sampler,
            pin_memory=True
        )
    if is_main_process:
            print(f"Per-GPU Micro Batch Size: {args.batch_size}") 
            print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
            print(f"Effective Global Batch Size: {args.batch_size * world_size * args.gradient_accumulation_steps}")

    optimizer = bnb_optim.AdamW8bit( 
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    if is_main_process: print(f"Optimizer: {type(optimizer).__name__}, LR: {args.learning_rate}, WD: {args.weight_decay}")
    scaler = amp.GradScaler(enabled=use_amp) 
    if is_main_process: print(f"GradScaler enabled: {scaler.is_enabled()}")

    actual_model_for_vit_mod = model.module if isinstance(model, DDP) else model
    if local_eva_vit_modifications is not None and \
       hasattr(local_eva_vit_modifications, 'ModifiedAttention'):
        ModifiedAttentionClass = getattr(local_eva_vit_modifications, 'ModifiedAttention')
        if hasattr(actual_model_for_vit_mod, 'visual_encoder') and \
           hasattr(actual_model_for_vit_mod.visual_encoder, 'blocks'):
            if is_main_process:
                print("Attempting to set _block_idx on ModifiedAttention instances...")
            for i, blk in enumerate(actual_model_for_vit_mod.visual_encoder.blocks):
                if hasattr(blk, 'attn') and isinstance(blk.attn, ModifiedAttentionClass):
                    blk.attn._block_idx = i
                    if is_main_process and i < 5: 
                        print(f"  Set _block_idx = {i} for visual_encoder.blocks.{i}.attn (Type: {type(blk.attn).__name__})")
                elif is_main_process and i < 5: 
                    attn_type = type(blk.attn).__name__ if hasattr(blk, 'attn') else "NoAttnAttr"
                    print(f"  Skipping visual_encoder.blocks.{i}.attn: Not the patched ModifiedAttention (Actual type: {attn_type})")
        elif is_main_process:
            print("WARNING: Could not find visual_encoder.blocks to set _block_idx.")
    elif is_main_process:
        print(f"WARNING: local_eva_vit_modifications module not loaded or ModifiedAttention class not found. Skipping _block_idx setting.")

    if is_main_process and args.print_layer_norms_at_start: # Add CLI arg
        print("DEBUG: ln_vision parameters:")
        actual_model_for_ln_debug = model.module if isinstance(model, DDP) else model
        if hasattr(actual_model_for_ln_debug, 'ln_vision'):
            if torch.isfinite(actual_model_for_ln_debug.ln_vision.weight.data).all():
                 print(f"  ln_vision.weight: min={actual_model_for_ln_debug.ln_vision.weight.data.min()}, max={actual_model_for_ln_debug.ln_vision.weight.data.max()}, has_nan={torch.isnan(actual_model_for_ln_debug.ln_vision.weight.data).any()}")
            else:
                 print(f"  ln_vision.weight: Non-finite values detected. Dtype: {actual_model_for_ln_debug.ln_vision.weight.data.dtype}")
            if torch.isfinite(actual_model_for_ln_debug.ln_vision.bias.data).all():
                 print(f"  ln_vision.bias: min={actual_model_for_ln_debug.ln_vision.bias.data.min()}, max={actual_model_for_ln_debug.ln_vision.bias.data.max()}, has_nan={torch.isnan(actual_model_for_ln_debug.ln_vision.bias.data).any()}")
            else:
                 print(f"  ln_vision.bias: Non-finite values detected. Dtype: {actual_model_for_ln_debug.ln_vision.bias.data.dtype}")
        else:
            print("  Model does not have 'ln_vision' attribute.")


    hooks = []
    model_for_hooks = model.module if isinstance(model, DDP) else model
    current_rank = rank 
    
    # Determine number of T5 layers dynamically for hook registration
    num_t5_encoder_layers = 24 # Default for T5-XL, T5-XXL
    num_t5_decoder_layers = 24 # Default for T5-XL, T5-XXL
    if hasattr(model_for_hooks, 't5_model') and hasattr(model_for_hooks.t5_model, 'config'):
        if hasattr(model_for_hooks.t5_model.config, 'num_layers'): # T5, FlanT5
            num_t5_encoder_layers = model_for_hooks.t5_model.config.num_layers
            num_t5_decoder_layers = model_for_hooks.t5_model.config.num_decoder_layers if hasattr(model_for_hooks.t5_model.config, 'num_decoder_layers') else num_t5_encoder_layers
        elif hasattr(model_for_hooks.t5_model.config, 'num_hidden_layers'): # Some other T5 variants might use this
            num_t5_encoder_layers = model_for_hooks.t5_model.config.num_hidden_layers
            num_t5_decoder_layers = model_for_hooks.t5_model.config.num_hidden_layers


    modules_to_hook = {
        "vit_block_0_attn_qkv_out": "visual_encoder.blocks.0.attn.qkv",
        "vit_block_0_attn_proj_out": "visual_encoder.blocks.0.attn.proj", 
        "vit_block_19_attn_qkv_out": "visual_encoder.blocks.19.attn.qkv", # Assuming ViT-L/14 has ~24 blocks, 19 is near end
        "vit_block_19_attn_proj_out": "visual_encoder.blocks.19.attn.proj",
        "vit_block_38_attn_qkv_out": "visual_encoder.blocks.38.attn.qkv", # EVA-ViT-g has 40 blocks
        "vit_block_38_attn_proj_out": "visual_encoder.blocks.38.attn.proj",
        "vit_block_0_norm1_out": "visual_encoder.blocks.0.norm1",
        "vit_block_0_norm2_out": "visual_encoder.blocks.0.norm2",
        "vit_block_19_norm1_out": "visual_encoder.blocks.19.norm1",
        "vit_block_19_norm2_out": "visual_encoder.blocks.19.norm2",
        "vit_block_38_norm1_out": "visual_encoder.blocks.38.norm1",
        "vit_block_38_norm2_out": "visual_encoder.blocks.38.norm2",
        "ln_vision_out": "ln_vision", 
        "Qformer_bert_embeddings_norm_out": "Qformer.bert.embeddings.LayerNorm", # Adjusted for Qformer attribute
        "Qformer_output_before_t5_proj": "Qformer", # Adjusted for Qformer attribute
        "Qformer_crossattn_layer0_out_norm_out": "Qformer.bert.encoder.layer.0.crossattention.output.LayerNorm", # Adjusted
        "Qformer_crossattn_layer10_out_norm_out": "Qformer.bert.encoder.layer.10.crossattention.output.LayerNorm", # Adjusted
        "Qformer_selfattn_layer0_out_norm_out": "Qformer.bert.encoder.layer.0.attention.output.LayerNorm", # Adjusted
        "Qformer_selfattn_layer5_out_norm_out": "Qformer.bert.encoder.layer.5.attention.output.LayerNorm", # Adjusted
        "Qformer_selfattn_layer11_out_norm_out": "Qformer.bert.encoder.layer.11.attention.output.LayerNorm",# Adjusted
        "Qformer_intermediate_layer0_norm_out": "Qformer.bert.encoder.layer.0.output_query.LayerNorm", # Adjusted
        "Qformer_intermediate_layer5_norm_out": "Qformer.bert.encoder.layer.5.output_query.LayerNorm", # Adjusted
        "Qformer_intermediate_layer11_norm_out": "Qformer.bert.encoder.layer.11.output_query.LayerNorm", # Adjusted
        "t5_proj_out": "t5_proj",
        "t5_encoder_input_embeds": "t5_model.encoder.embed_tokens", 
        "t5_encoder_block_0_attn_out": "t5_model.encoder.block.0.layer.0.SelfAttention", 
        "t5_encoder_block_0_ffn_out": "t5_model.encoder.block.0.layer.1.DenseReluDense", 
        "t5_encoder_block_LAST_attn_out": f"t5_model.encoder.block.{num_t5_encoder_layers - 1}.layer.0.SelfAttention",
        "t5_encoder_block_LAST_ffn_out": f"t5_model.encoder.block.{num_t5_encoder_layers - 1}.layer.1.DenseReluDense",
        "t5_encoder_final_norm_out": "t5_model.encoder.final_layer_norm",
        "t5_decoder_crossattn_layer0_out_norm_out": "t5_model.decoder.block.0.layer.1.layer_norm", # T5 cross-attn is layer 1
        "t5_decoder_crossattn_layer11_out_norm_out": f"t5_model.decoder.block.{min(11, num_t5_decoder_layers -1)}.layer.1.layer_norm",
        "t5_decoder_crossattn_layerLAST_out_norm_out": f"t5_model.decoder.block.{num_t5_decoder_layers - 1}.layer.1.layer_norm",
        "t5_decoder_selfattn_layer0_out_norm_out": "t5_model.decoder.block.0.layer.0.layer_norm", # T5 self-attn is layer 0
        "t5_decoder_selfattn_layer11_out_norm_out": f"t5_model.decoder.block.{min(11, num_t5_decoder_layers -1)}.layer.0.layer_norm",
        "t5_decoder_selfattn_layerLAST_out_norm_out": f"t5_model.decoder.block.{num_t5_decoder_layers - 1}.layer.0.layer_norm",
        "t5_decoder_ffn_layer0_norm_out": "t5_model.decoder.block.0.layer.2.layer_norm", # T5 FFN is layer 2
        "t5_decoder_ffn_layer11_norm_out": f"t5_model.decoder.block.{min(11, num_t5_decoder_layers-1)}.layer.2.layer_norm",
        "t5_decoder_ffn_layerLAST_norm_out": f"t5_model.decoder.block.{num_t5_decoder_layers - 1}.layer.2.layer_norm",
        "t5_decoder_final_norm_out": "t5_model.decoder.final_layer_norm",
        "lm_head_out": "t5_model.lm_head", 
    }
    if args.enable_nan_debugging_hooks: # Add CLI arg
        if is_main_process: print("\nRegistering forward hooks for debugging...")
        for display_name, path_str in modules_to_hook.items():
            # Adjust path for Qformer if model attribute is 'q_former'
            if 'Qformer.' in path_str and hasattr(model_for_hooks, 'q_former') and not hasattr(model_for_hooks, 'Qformer'):
                path_str = path_str.replace('Qformer.', 'q_former.')

            module_to_hook = get_module_by_path(model_for_hooks, path_str)
            if module_to_hook:
                if is_main_process: print(f"  Hooking: {display_name} (path: {path_str})")
                hooks.append(module_to_hook.register_forward_hook(get_nan_checking_hook(display_name, current_rank)))
            else:
                if is_main_process: print(f"  WARNING: Could not find module for {display_name} at path {path_str}. Skipping hook.", file=sys.stderr)
        if is_main_process and not hooks:
            print("WARNING: No hooks were registered. Check module paths.")
        if is_main_process: print("Hooks registration complete.\n")
    elif is_main_process:
        print("NaN debugging hooks are disabled by CLI argument.")


    # --- Training Loop ---
    if is_main_process: print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    best_val_loss = float('inf')
    total_train_loss_epoch_sum = 0.0 # Renamed for clarity: sum of unscaled losses for the epoch
    total_samples_processed_epoch = 0 # Renamed for clarity: samples in epoch

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        if is_main_process: print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        model.train() # Ensure model is in training mode
        if world_size > 1: # DDP specific
            train_dataloader.sampler.set_epoch(epoch) 

        total_train_loss_epoch_sum = 0.0 # Reset for current epoch
        total_samples_processed_epoch = 0  # Reset for current epoch
        optimizer.zero_grad() 
        accumulation_step_count = 0 

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}", disable=not is_main_process)):
            if batch is None:
                if is_main_process: print(f"Warning: Skipping step {step+1} due to empty batch after filtering bad samples.", file=sys.stderr)
                continue

            images = batch["image"].to(ddp_device, non_blocking=True)
            text_input_batch = batch["text_input"] # List of strings
            text_output_batch = batch["text_output"] # List of strings

            if args.enable_nan_debugging_hooks:
                nan_info_list_from_ranks = [None for _ in range(world_size)] if world_size > 1 else [nan_detected_module_info]
                if world_size > 1:
                    dist.all_gather_object(nan_info_list_from_ranks, nan_detected_module_info) # Each rank sends its own dict
                
                # Check on main process
                if is_main_process:
                    first_nan_info_this_step = None
                    for rank_idx, info in enumerate(nan_info_list_from_ranks):
                        if info is not None and info["name"] is not None:
                            first_nan_info_this_step = info # Contains name, stats, and original rank
                            # We use rank_idx here as the reporting rank if info["rank"] wasn't set,
                            # but info["rank"] should be the source rank.
                            break 
                    if first_nan_info_this_step:
                        print(f"\nFATAL: NaN/Inf detected on Rank {first_nan_info_this_step.get('rank', 'unknown')} by hook in module {first_nan_info_this_step['name']}.", file=sys.stderr)
                        print(f"Problematic module output stats: {first_nan_info_this_step['output_stats']}", file=sys.stderr)
                        if world_size > 1: dist.barrier() 
                        for h in hooks: h.remove()
                        cleanup_ddp()
                        sys.exit(1) 
                # Reset local nan_detected_module_info on all ranks for the next forward pass by hooks
                nan_detected_module_info["name"] = None
                nan_detected_module_info["output_stats"] = None
                nan_detected_module_info["rank"] = None # Rank will be set by hook if issue found
                if world_size > 1: dist.barrier() # Ensure all ranks reset before next step


            with amp.autocast(enabled=use_amp, dtype=amp_dtype):
                samples = {
                    "image": images,
                    "text_input": text_input_batch, # Pass as list of strings
                    "text_output": text_output_batch # Pass as list of strings
                }
                try:
                    loss_output = model(samples)
                    if loss_output is None or "loss" not in loss_output:
                        if is_main_process: print(f"Warning: Invalid loss_output at step {step+1} on rank {rank}. Skipping batch. Output: {loss_output}", file=sys.stderr)
                        optimizer.zero_grad()
                        accumulation_step_count = 0 
                        continue
                    loss = loss_output["loss"]
                    if not torch.isfinite(loss):
                         if is_main_process:
                              print(f"\n!!! FATAL: Non-finite loss detected at Step {step+1} on rank {rank} BEFORE backward pass. Loss: {loss.item()}. !!!", file=sys.stderr)
                              try:
                                 torch.save(samples, f"bad_samples_rank_{rank}_step_{step+1}_loss.pt")
                                 print(f"    Saved problematic samples to bad_samples_rank_{rank}_step_{step+1}_loss.pt", file=sys.stderr)
                              except Exception as save_e: print(f"    Error saving samples: {save_e}", file=sys.stderr)
                         if world_size > 1: dist.barrier()
                         for h in hooks: h.remove(); cleanup_ddp(); sys.exit(1)
                    
                    original_loss_value = loss.item() 
                    total_train_loss_epoch_sum += original_loss_value * images.size(0) 
                    total_samples_processed_epoch += images.size(0)
                    scaled_loss = loss / args.gradient_accumulation_steps
                except Exception as e:
                    if is_main_process:
                         print(f"→ skipping step {step+1}, id={batch.get('question_id', ['N/A'])[0]} on rank {rank} due to forward pass exception: {e}", file=sys.stderr) # Get first ID if batch
                         traceback.print_exc(file=sys.stderr) 
                    optimizer.zero_grad()
                    accumulation_step_count = 0 
                    continue

            if use_amp:
                scaler.scale(scaled_loss).backward() 
            else:
                scaled_loss.backward()
            accumulation_step_count += 1

            is_final_accumulation_step_in_chunk = accumulation_step_count == args.gradient_accumulation_steps
            is_last_data_batch_in_epoch = (step + 1) == len(train_dataloader)

            if is_final_accumulation_step_in_chunk or (is_last_data_batch_in_epoch and accumulation_step_count > 0):
                if args.max_grad_norm > 0: 
                    if use_amp:
                         scaler.unscale_(optimizer) 
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad and p.grad is not None, model.parameters()), args.max_grad_norm)
                if use_amp:
                    scaler.step(optimizer) 
                    scaler.update() 
                else:
                    optimizer.step()
                optimizer.zero_grad()
                accumulation_step_count = 0 

            if (step + 1) % args.log_interval == 0 and is_main_process:
                 if 'original_loss_value' in locals() and torch.isfinite(torch.tensor(original_loss_value)).all(): 
                     print(f"  Epoch {epoch+1}, Step {step+1}/{len(train_dataloader)}, Micro Batch Loss: {original_loss_value:.4f}")
        
        # End of Training Epoch Loop
        avg_train_loss_epoch = total_train_loss_epoch_sum / total_samples_processed_epoch if total_samples_processed_epoch > 0 else 0.0

        avg_val_loss_epoch = float('inf') 
        if val_dataloader:
            if is_main_process: print(f"\n--- Starting Validation for Epoch {epoch+1} ---")
            model.eval() 
            total_val_loss_epoch_sum = 0.0
            val_samples_processed_epoch = 0
            with torch.no_grad(): 
                for step, batch in enumerate(tqdm(val_dataloader, desc=f"Validating Epoch {epoch+1}", disable=not is_main_process)):
                    if batch is None:
                        if is_main_process: print(f"Warning: Skipping validation step {step+1} due to empty batch.", file=sys.stderr)
                        continue
                    images = batch["image"].to(ddp_device, non_blocking=True)
                    text_input_batch = batch["text_input"]
                    text_output_batch = batch["text_output"]
                    with amp.autocast(enabled=use_amp, dtype=amp_dtype): 
                        samples = {"image": images, "text_input": text_input_batch, "text_output": text_output_batch}
                        try:
                            loss_output = model(samples)
                            if loss_output is None or "loss" not in loss_output:
                                if is_main_process: print(f"Warning: Invalid loss_output during validation at step {step+1}. Skipping. Output: {loss_output}", file=sys.stderr)
                                continue
                            loss = loss_output["loss"]
                            if not torch.isfinite(loss):
                                 if is_main_process: print(f"Warning: Non-finite validation loss at step {step+1} on rank {rank}. Loss: {loss.item()}.", file=sys.stderr)
                                 continue 
                        except Exception as e:
                            if is_main_process:
                                print(f"  → skipping validation sample {batch.get('question_id', ['N/A'])[0]} due to exception: {e}", file=sys.stderr)
                                traceback.print_exc(file=sys.stderr) 
                            continue 
                    total_val_loss_epoch_sum += loss.item() * images.size(0) 
                    val_samples_processed_epoch += images.size(0)
            avg_val_loss_epoch = total_val_loss_epoch_sum / val_samples_processed_epoch if val_samples_processed_epoch > 0 else float('inf')

        # Gather and Average Metrics across all GPUs
        current_epoch_avg_train_loss_global = avg_train_loss_epoch # Default for single GPU
        current_epoch_avg_val_loss_global = avg_val_loss_epoch   # Default for single GPU

        if world_size > 1:
            metrics_to_reduce = torch.tensor([total_train_loss_epoch_sum, total_samples_processed_epoch, 
                                              total_val_loss_epoch_sum if val_dataloader else 0.0, 
                                              val_samples_processed_epoch if val_dataloader else 0.0], 
                                             dtype=torch.float64, device=ddp_device)
            dist.all_reduce(metrics_to_reduce, op=dist.ReduceOp.SUM)
            
            global_sum_train_loss, global_sum_train_samples, global_sum_val_loss, global_sum_val_samples = metrics_to_reduce.tolist()
            
            current_epoch_avg_train_loss_global = global_sum_train_loss / global_sum_train_samples if global_sum_train_samples > 0 else 0.0
            if val_dataloader:
                current_epoch_avg_val_loss_global = global_sum_val_loss / global_sum_val_samples if global_sum_val_samples > 0 else float('inf')
            else: # No validation
                current_epoch_avg_val_loss_global = float('inf') # Or some other indicator like 0.0 if preferred when no val

        if is_main_process:
            print(f"Epoch {epoch+1} Global Average Training Loss: {current_epoch_avg_train_loss_global:.4f}")
            if val_dataloader: 
                print(f"Epoch {epoch+1} Global Average Validation Loss: {current_epoch_avg_val_loss_global:.4f}")
        
        # Checkpointing on main process
        if is_main_process:
            save_this_epoch = False
            if val_dataloader: # Save based on validation loss
                if current_epoch_avg_val_loss_global < best_val_loss:
                    best_val_loss = current_epoch_avg_val_loss_global
                    print(f"New best validation loss: {best_val_loss:.4f}. Saving model/adapters for epoch {epoch+1}...")
                    save_this_epoch = True
            elif epoch + 1 == args.epochs : # If no validation, save at the end of training
                 print(f"No validation. Saving model/adapters at final epoch {epoch+1}...")
                 save_this_epoch = True


            if save_this_epoch:
                actual_model_to_save_from = model.module if isinstance(model, DDP) else model
                if is_any_lora_active_for_training and args.lora_train_current_adapter_name:
                    lora_output_dir_epoch = Path(args.output_dir) / "lora_adapters"
                    lora_output_dir_epoch.mkdir(parents=True, exist_ok=True)
                    
                    # Save T5 adapter if configured
                    if args.lora_t5_target_modules and \
                    t5_model_attr_name and \
                    hasattr(actual_model_to_save_from, t5_model_attr_name) and \
                    isinstance(getattr(actual_model_to_save_from, t5_model_attr_name), PeftModel):
                        
                        t5_peft_model_to_save = getattr(actual_model_to_save_from, t5_model_attr_name)
                        # Check if the current adapter is indeed the one we are training
                        # This check is important if multiple adapters are loaded but only one is active for training
                        if hasattr(t5_peft_model_to_save, 'active_adapter') and \
                        t5_peft_model_to_save.active_adapter == f"{args.lora_train_current_adapter_name}_t5":
                            t5_adapter_save_path_epoch = lora_output_dir_epoch / f"{args.lora_train_current_adapter_name}_t5_epoch_{epoch+1}"
                            try:
                                t5_peft_model_to_save.save_pretrained(str(t5_adapter_save_path_epoch))
                                print(f"  T5 LoRA adapter for epoch {epoch+1} saved to {t5_adapter_save_path_epoch}")
                            except Exception as save_e: 
                                print(f"  ERROR saving T5 LoRA adapter for epoch {epoch+1} (model.{t5_model_attr_name}): {save_e}", file=sys.stderr)
                                traceback.print_exc(file=sys.stderr)
                        elif is_main_process:
                            active_adapter_name = getattr(t5_peft_model_to_save, 'active_adapter', 'UNKNOWN_ADAPTER')
                            print(f"  Skipping save for T5 adapter on model.{t5_model_attr_name}, active adapter is '{active_adapter_name}', expected '{args.lora_train_current_adapter_name}_t5'")
                    elif args.lora_t5_target_modules and is_main_process: # Log if T5 LoRA was intended but component is not PeftModel
                        if not (t5_model_attr_name and hasattr(actual_model_to_save_from, t5_model_attr_name)):
                            print(f"  WARNING: Cannot save T5 LoRA adapter. Attribute '{t5_model_attr_name}' not found on model.")
                        elif not isinstance(getattr(actual_model_to_save_from, t5_model_attr_name), PeftModel):
                            print(f"  WARNING: Cannot save T5 LoRA adapter. model.{t5_model_attr_name} is not a PeftModel instance.")
                        
                    # Save Q-Former adapter if configured
                    qformer_peft_model_instance_to_save = None
                    qformer_attr_name_save = None
                    if hasattr(actual_model_to_save_from, 'q_former') and isinstance(actual_model_to_save_from.q_former, PeftModel): 
                        qformer_peft_model_instance_to_save = actual_model_to_save_from.q_former
                        qformer_attr_name_save = 'q_former'
                    elif hasattr(actual_model_to_save_from, 'Qformer') and isinstance(actual_model_to_save_from.Qformer, PeftModel): 
                        qformer_peft_model_instance_to_save = actual_model_to_save_from.Qformer
                        qformer_attr_name_save = 'Qformer'

                    if args.lora_qformer_target_modules and qformer_peft_model_instance_to_save:
                        if qformer_peft_model_instance_to_save.active_adapter == f"{args.lora_train_current_adapter_name}_qformer":
                            qf_adapter_save_path_epoch = lora_output_dir_epoch / f"{args.lora_train_current_adapter_name}_qformer_epoch_{epoch+1}"
                            try:
                                qformer_peft_model_instance_to_save.save_pretrained(str(qf_adapter_save_path_epoch))
                                print(f"  Q-Former LoRA adapter for epoch {epoch+1} saved to {qf_adapter_save_path_epoch}")
                            except Exception as save_e: print(f"  ERROR saving Q-Former LoRA adapter for epoch {epoch+1}: {save_e}", file=sys.stderr)
                        # else: print(f"  Skipping save for QF adapter, active adapter is {qformer_peft_model_instance_to_save.active_adapter}, expected {args.lora_train_current_adapter_name}_qformer")
                else: # Save full model state_dict
                    full_model_save_path_epoch = Path(args.output_dir) / f"model_epoch_{epoch+1}.pth"
                    full_model_save_path_epoch.parent.mkdir(parents=True, exist_ok=True)
                    state_dict_to_save = actual_model_to_save_from.state_dict()
                    try:
                        torch.save(state_dict_to_save, full_model_save_path_epoch)
                        print(f"  Full model for epoch {epoch+1} saved to {full_model_save_path_epoch}")
                    except Exception as save_e: print(f"  ERROR saving full model for epoch {epoch+1}: {save_e}", file=sys.stderr)
            
            # Optionally save every epoch if args.save_every_epoch is set
            if args.save_every_epoch and not save_this_epoch: # Avoid double saving if already saved due to best_val
                # Similar saving logic as above, but for "every_epoch"
                # ... (can duplicate the saving block here with different path naming if needed)
                pass


        if is_main_process:
             print(f"Epoch {epoch+1} completed in {(time.time() - epoch_start_time):.2f} seconds.")
        if world_size > 1: dist.barrier()


    # --- Final Save ---
    if is_main_process:
        print("\nFine-tuning finished.")
        actual_model_final_save = model.module if isinstance(model, DDP) else model
        if is_any_lora_active_for_training and args.lora_train_current_adapter_name:
            lora_final_output_dir = Path(args.output_dir) / "lora_adapters"
            lora_final_output_dir.mkdir(parents=True, exist_ok=True)

            if args.lora_t5_target_modules and \
            t5_model_attr_name and \
            hasattr(actual_model_final_save, t5_model_attr_name) and \
            isinstance(getattr(actual_model_final_save, t5_model_attr_name), PeftModel):
                
                t5_peft_model_final_save = getattr(actual_model_final_save, t5_model_attr_name)
                if hasattr(t5_peft_model_final_save, 'active_adapter') and \
                t5_peft_model_final_save.active_adapter == f"{args.lora_train_current_adapter_name}_t5":
                    t5_final_adapter_path = lora_final_output_dir / f"{args.lora_train_current_adapter_name}_t5_final"
                    try: 
                        t5_peft_model_final_save.save_pretrained(str(t5_final_adapter_path))
                        print(f"Final T5 LoRA adapter saved to {t5_final_adapter_path}")
                    except Exception as e: 
                        print(f"Error saving final T5 LoRA (model.{t5_model_attr_name}): {e}", file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)
                elif is_main_process:
                    active_adapter_name = getattr(t5_peft_model_final_save, 'active_adapter', 'UNKNOWN_ADAPTER')
                    print(f"  Skipping final save for T5 adapter on model.{t5_model_attr_name}, active adapter is '{active_adapter_name}', expected '{args.lora_train_current_adapter_name}_t5'")
            elif args.lora_t5_target_modules and is_main_process:
                if not (t5_model_attr_name and hasattr(actual_model_final_save, t5_model_attr_name)):
                    print(f"  WARNING: Cannot save final T5 LoRA adapter. Attribute '{t5_model_attr_name}' not found on model.")
                elif not isinstance(getattr(actual_model_final_save, t5_model_attr_name), PeftModel):
                    print(f"  WARNING: Cannot save final T5 LoRA adapter. model.{t5_model_attr_name} is not a PeftModel instance.")

            qf_peft_final_save = None
            if hasattr(actual_model_final_save, 'q_former') and isinstance(actual_model_final_save.q_former, PeftModel): qf_peft_final_save = actual_model_final_save.q_former
            elif hasattr(actual_model_final_save, 'Qformer') and isinstance(actual_model_final_save.Qformer, PeftModel): qf_peft_final_save = actual_model_final_save.Qformer
            
            if args.lora_qformer_target_modules and qf_peft_final_save:
                if qf_peft_final_save.active_adapter == f"{args.lora_train_current_adapter_name}_qformer":
                    qf_final_adapter_path = lora_final_output_dir / f"{args.lora_train_current_adapter_name}_qformer_final"
                    try: qf_peft_final_save.save_pretrained(str(qf_final_adapter_path)); print(f"Final Q-Former LoRA adapter saved to {qf_final_adapter_path}")
                    except Exception as e: print(f"Error saving final QF LoRA: {e}", file=sys.stderr)
        elif args.save_full_model_at_end: # Only save full model if LoRA wasn't the primary mode or if explicitly requested
            final_full_model_path = Path(args.output_dir) / "final_model.pth"
            state_dict_to_save_final = actual_model_final_save.state_dict()
            try: torch.save(state_dict_to_save_final, final_full_model_path); print(f"Final full model saved to {final_full_model_path}")
            except Exception as save_e: print(f"ERROR: Failed to save final model: {save_e}", file=sys.stderr)
        
        print(f"Total script time: {(time.time() - start_time):.2f} seconds.")

    if args.enable_nan_debugging_hooks:
        if is_main_process: print("Removing forward hooks...") 
        for h in hooks: h.remove()
        if is_main_process: print("Hooks removed.")
    
    if world_size > 1: dist.barrier()
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BLIP-2 on a medical VQA modality, with LoRA support.")
    # Model args 
    model_group = parser.add_argument_group('Model Arguments')
    model_group.add_argument("--base_model_name", type=str, default="blip2_t5", help="Base LAVIS model name.")
    model_group.add_argument("--base_model_type", type=str, default="pretrain_flant5xl", help="Specific model type for LAVIS.")
    model_group.add_argument("--unfreeze_all", action="store_true", help="If set AND LoRA is NOT used for training, unfreeze all parameters. Overridden by LoRA.")
    model_group.add_argument("--load_checkpoint", type=str, default=None, help="Path to a full model .pth checkpoint to load (base weights before LoRA).")

    # Data args 
    data_group = parser.add_argument_group('Data Arguments')
    data_group.add_argument("--train_json_path", type=str, required=True, help="Path to the training JSON file.")
    data_group.add_argument("--val_json_path", type=str, default=None, help="Path to the validation JSON file.")
    data_group.add_argument("--image_base_path", type=str, required=True, help="Base directory for images.")
    data_group.add_argument("--max_input_len", type=int, default=128, help="Max input token length.")
    data_group.add_argument("--max_target_len", type=int, default=64, help="Max target token length.")

    # Training args 
    train_group = parser.add_argument_group('Training Arguments')
    train_group.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and LoRA adapters.")
    train_group.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    train_group.add_argument("--batch_size", type=int, default=1, help="Micro batch size per GPU.")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps.")
    train_group.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (for LoRA, typically higher than full FT).") 
    train_group.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay.")
    train_group.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    train_group.add_argument("--log_interval", type=int, default=50, help="Log interval (primarily epoch-based).")
    train_group.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    train_group.add_argument("--no_amp", action="store_true", help="Disable mixed-precision autocast.")
    train_group.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_group.add_argument("--ddp_find_unused_parameters", action="store_true", help="Set find_unused_parameters=True for DDP (can be useful for PEFT).")
    train_group.add_argument("--save_every_epoch", action="store_true", help="Save model/adapter checkpoint at the end of every epoch.")
    train_group.add_argument("--save_full_model_at_end", action="store_true", help="Also save the full model state_dict at the end, even if LoRA was trained.")


    # Debugging and Inspection Args
    debug_group = parser.add_argument_group('Debugging and Inspection')
    debug_group.add_argument("--inspect_modules", action="store_true", help="Print model's linear layer names for LoRA target_modules config and exit.")
    debug_group.add_argument("--print_model_structure", action="store_true", help="Print full model structure and named modules at startup.")
    debug_group.add_argument("--print_layer_norms_at_start", action="store_true", help="Print stats of key LayerNorm layers at startup.")
    debug_group.add_argument("--enable_nan_debugging_hooks", action="store_true", help="Enable forward hooks on key modules to detect NaNs/Infs.")


    # --- LoRA Specific Arguments ---
    lora_group = parser.add_argument_group('LoRA Arguments')
    lora_group.add_argument("--lora_load_previous_adapters_config", type=str, default=None,
                              help="Path to a JSON file defining previously trained LoRA adapters to load and freeze. "
                                   "Each entry: {'path': '/path/to/adapter_dir', 'name': 'adapter_unique_name', 'component': 't5'/'qformer'}")
    lora_group.add_argument("--lora_train_current_adapter_name", type=str, default=None, 
                              help="Base name for the NEW LoRA adapter to be trained in this stage (e.g., 'fundus', 'xray'). Suffixes '_t5'/'_qformer' will be added automatically.")

    # T5 LoRA config for current adapter
    lora_group.add_argument("--lora_t5_target_modules", type=str, nargs='+', default=None, help="List of T5 module names to apply LoRA to for the current adapter (e.g., q_proj v_proj).")
    lora_group.add_argument("--lora_r_t5", type=int, default=16, help="LoRA rank 'r' for T5 for the current adapter.")
    lora_group.add_argument("--lora_alpha_t5", type=int, default=32, help="LoRA alpha for T5 for the current adapter.")
    lora_group.add_argument("--lora_dropout_t5", type=float, default=0.05, help="LoRA dropout for T5 for the current adapter.")
    lora_group.add_argument("--lora_bias_t5", type=str, default="none", choices=['none', 'all', 'lora_only'], help="Bias type for T5 LoRA.")
    lora_group.add_argument("--lora_t5_modules_to_save", type=str, nargs='+', default=None, help="List of T5 modules to save and make trainable apart from LoRA layers (e.g. lm_head).")

    # Q-Former LoRA config for current adapter
    lora_group.add_argument("--lora_qformer_target_modules", type=str, nargs='+', default=None, help="List of Q-Former module names to apply LoRA to for the current adapter.")
    lora_group.add_argument("--lora_r_qformer", type=int, default=8, help="LoRA rank 'r' for Q-Former for the current adapter.")
    lora_group.add_argument("--lora_alpha_qformer", type=int, default=16, help="LoRA alpha for Q-Former for the current adapter.")
    lora_group.add_argument("--lora_dropout_qformer", type=float, default=0.05, help="LoRA dropout for Q-Former for the current adapter.")
    lora_group.add_argument("--lora_bias_qformer", type=str, default="none", choices=['none', 'all', 'lora_only'], help="Bias type for Q-Former LoRA.")
    lora_group.add_argument("--lora_qformer_modules_to_save", type=str, nargs='+', default=None, help="List of Q-Former modules to save and make trainable apart from LoRA layers.")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.lora_train_current_adapter_name:
        if not args.lora_t5_target_modules and not args.lora_qformer_target_modules:
            print("WARNING: --lora_train_current_adapter_name is provided, but neither --lora_t5_target_modules nor --lora_qformer_target_modules are set. No new LoRA adapter will be trained.", file=sys.stderr)
    if (args.lora_t5_target_modules or args.lora_qformer_target_modules) and not args.lora_train_current_adapter_name:
        print("ERROR: LoRA target modules specified but --lora_train_current_adapter_name is missing. This script will likely error or not train LoRA layers as intended.", file=sys.stderr)
        # Allowing to proceed for cases where this might be intentional for some debugging, but it's a strong warning.
        # Consider sys.exit(1) here if this should always be fatal.

    main(args)