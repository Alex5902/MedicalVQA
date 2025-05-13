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
local_mods_path = script_directory / "local_lavis_mods" / "local_eva_vit_modifications.py"
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

        # If your ModifiedAttention class has an attribute like _block_idx
        # and you want to set it during the hook registration later,
        # make sure the ModifiedBlock instantiates ModifiedAttention.

    except ImportError as e:
        print(f"ERROR: Could not import local modifications from '{local_mods_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
         print(f"ERROR: An unexpected error occurred during local modifications import/patching: {e}", file=sys.stderr)
         traceback.print_exc(file=sys.stderr)
         sys.exit(1)
    finally:
        # Clean up sys.path
        # Check if local_mods_dir was successfully defined (i.e., not None) before trying to remove it
        if local_mods_dir is not None and str(local_mods_dir) in sys.path:
             try:
                sys.path.pop(sys.path.index(str(local_mods_dir)))
             except ValueError:
                 pass # Should not happen if the check passes

else:
    print(f"WARNING: Local modifications file not found at '{local_mods_path}'. Using installed LAVIS version.")

# Global state to track the first detected NaN/Inf for debugging
# Added 'rank' to track which process detected it first
nan_detected_module_info = {"name": None, "output_stats": None, "rank": None}

# Assuming LAVIS is in your environment
from lavis.models import load_model_and_preprocess
import torch.optim as optim
# from transformers import get_linear_schedule_with_warmup # Optional scheduler

def setup_ddp():
    """Initializes DDP using environment variables and sets device."""
    # Get environment variables set by torchrun
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ:
        rank_str = os.environ["RANK"]
        world_size_str = os.environ["WORLD_SIZE"]
        local_rank_str = os.environ["LOCAL_RANK"]
        # Log initial env vars check
        # Only print this on main rank to avoid clutter
        if rank_str == '0':
            print(f"DDP: Found env vars: RANK={rank_str}, LOCAL_RANK={local_rank_str}, WORLD_SIZE={world_size_str}")

        # Initialize the distributed environment
        # init_method="env://" uses MASTER_ADDR and MASTER_PORT from os.environ
        # torchrun sets these automatically.
        dist.init_process_group(backend="nccl", init_method="env://")

        # Get rank and world_size from the initialized process group (should match env vars)
        # These are the official DDP rank/world_size after initialization
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # Get local_rank directly from the environment variable and convert to int
        local_rank = int(local_rank_str) # <<< Correct way to get local rank with torchrun

        if rank == 0: # Only print on main rank
            print(f"DDP: Process {rank}/{world_size} (local rank {local_rank}) successfully initialized process group.")

        # Set the current CUDA device for this process using the local_rank from env
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if rank == 0: # Only print on main rank
            print(f"DDP: Process {rank} set to use device {device}.")

    else:
        # This case should ideally not be hit when launched with torchrun correctly.
        print("Exiting: DDP environment variables (RANK, WORLD_SIZE, LOCAL_RANK) not found. Launch with torchrun.", file=sys.stderr)
        sys.exit(1) # Ensure it exits if env vars aren't set

    return rank, world_size, device

def cleanup_ddp():
    """Cleans up DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()

# Helper function to get a module by its path string
def get_module_by_path(model_root, path_str):
    # Handle DDP wrapper if present
    if isinstance(model_root, DDP):
        model_root = model_root.module

    try:
        module = model_root
        for part in path_str.split('.'):
            module = getattr(module, part)
        return module
    except AttributeError:
        # print(f"DEBUG: Could not find module at path '{path_str}'") # Too noisy during normal run
        return None

# Forward hook function
def get_nan_checking_hook(module_name, current_rank):
    """
    Creates a forward hook to check for NaNs/Infs in module inputs and outputs.
    Reports the first detected NaN/Inf on this rank.
    """
    def hook(module, input_tensors, output_tensors): # PyTorch hooks receive input and output (usually tuples)
        global nan_detected_module_info

        # --- Check Input(s) to the module ---
        # Only proceed if no NaN/Inf has been detected *on this rank* in this forward pass already
        if nan_detected_module_info["name"] is not None and nan_detected_module_info["rank"] == current_rank:
             return # Skip if we've already found a problem on this rank in this pass

        # Helper to check a single tensor for NaN/Inf and update global info
        def check_tensor_for_nan(tensor, is_input):
             global nan_detected_module_info
             # Only check CUDA tensors
             if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cuda':
                  return False

             if tensor.numel() > 0 and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
                 stats = {
                     "shape": list(tensor.shape), # Convert to list for printing
                     "min": tensor.min().item() if torch.isfinite(tensor).all() else float('nan'), # Only get stats if all finite
                     "max": tensor.max().item() if torch.isfinite(tensor).all() else float('nan'),
                     "mean": tensor.mean().item() if torch.isfinite(tensor).all() else float('nan'),
                     "dtype": str(tensor.dtype),
                     "has_nan": torch.isnan(tensor).any().item(),
                     "has_inf": torch.isinf(tensor).any().item(),
                     "num_nans": torch.isnan(tensor).sum().item(),
                     "num_infs": torch.isinf(tensor).sum().item()
                 }
                 prefix = "INPUT to" if is_input else "OUTPUT of"
                 # Report on main process if detected
                 if current_rank == 0:
                     print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
                     print(f"!!! Rank {current_rank}: NaN/Inf DETECTED IN **{prefix.split(' ')[0]}** of module: {module_name} !!!", file=sys.stderr)
                     print(f"    Stats: {stats}", file=sys.stderr)
                     print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", file=sys.stderr)

                 # Store info about this first failure on this rank
                 nan_detected_module_info["name"] = f"{prefix} {module_name}"
                 nan_detected_module_info["output_stats"] = stats # Use the same dict key
                 nan_detected_module_info["rank"] = current_rank
                 return True # Indicate that NaN/Inf was detected

             return False # Indicate no NaN/Inf detected


        # --- Check Input(s) ---
        # Input can be a single tensor or a tuple of tensors/other objects
        if isinstance(input_tensors, tuple):
            for i, t in enumerate(input_tensors):
                 if check_tensor_for_nan(t, is_input=True):
                       return # Stop checking inputs and outputs for this hook if a problem is found in input
        elif isinstance(input_tensors, torch.Tensor):
             if check_tensor_for_nan(input_tensors, is_input=True):
                  return # Stop checking inputs and outputs for this hook if a problem is found in input

        # --- Check Output(s) ---
        # This part only runs if no NaN/Inf was detected in the inputs by this hook
        if nan_detected_module_info["name"] is not None and nan_detected_module_info["rank"] == current_rank:
             return # Skip if input check on this hook found a problem

        # Output can be a single tensor or a tuple of tensors/other objects
        if isinstance(output_tensors, tuple):
            for i, t in enumerate(output_tensors):
                 if isinstance(t, torch.Tensor):
                      if check_tensor_for_nan(t, is_input=False):
                           # Don't return immediately after finding output NaN/Inf,
                           # allow other output tensors in the tuple to be checked if needed,
                           # but the global flag prevents re-reporting by this hook.
                           pass # Continue checking other output tensors in the tuple
        elif isinstance(output_tensors, torch.Tensor) and output_tensors.device.type == 'cuda':
             check_tensor_for_nan(output_tensors, is_input=False)

        # # Verbose logging of healthy outputs (can be very chatty)
        # else: # Only log if no NaN/Inf was detected by this hook
        #      if current_rank == 0 and isinstance(output_tensors, torch.Tensor) and output_tensors.numel() > 0:
        #          # Add size limit and finite check before printing min/max/mean for large tensors
        #          if output_tensors.numel() < 1000000 and torch.isfinite(output_tensors).all(): # Add size limit and finite check
        #              print(f"DEBUG Hook {module_name} (Rank {current_rank}): shape {list(output_tensors.shape)}, dtype {str(output_tensors.dtype)}, "
        #                    f"min {output_tensors.min().item():.4f}, max {output_tensors.max().item():.4f}, mean {output_tensors.mean().item():.4f}, "
        #                    f"has_nan {torch.isnan(output_tensors).any()}, has_inf {torch.isinf(output_tensors).any()}")
        #          else:
        #               print(f"DEBUG Hook {module_name} (Rank {current_rank}): shape {list(output_tensors.shape)}, dtype {str(output_tensors.dtype)}, (Stats skipped due to size or non-finite values)")


    return hook


# --- Custom Dataset for Training ---
class MedicalVqaFineTuningDataset(Dataset):
    def __init__(self, json_path, image_base_path, vis_processor, txt_processor,
                 max_input_len=128, max_target_len=64, prompt_type="default"):
        # Ensure this runs only on the main process or use a barrier if needed later
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
        self.txt_processor = txt_processor # This is the tokenizer from LAVIS
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
            # Apply visual transforms. Using eval processor might be appropriate
            # if it doesn't apply random augmentations meant only for pre-training.
            # If vis_processors has a 'train' key with augmentations, use that instead.
            # We'll stick to 'eval' as in your original code for now.
            image = self.vis_processor(raw_image)
        except Exception as e:
            # Log error but don't exit, collate_fn will skip None items
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Error loading image {full_img_path} for item {item.get('question_id', idx)}: {e}", file=sys.stderr)
            return None # Return None to be handled by collate_fn

        question_text = item['question']
        gt_answer_text = item['gt_answer']

        if not question_text or not gt_answer_text:
             # Non-fatal warning, item will likely cause issues later but might pass
             # print(f"WARNING: Empty text found for item {idx}. Q: '{question_text}', A: '{gt_answer_text}'", file=sys.stderr)
             pass # Suppress this common warning unless debugging specific text issues

        return {
        "image": image,
        "text_input": question_text,    # raw string
        "text_output": gt_answer_text,  # raw string
        # Add original ID for debugging, ensures it exists even if not in original json
        "question_id": item.get("question_id", f"idx_{idx}")
        }


# Custom collate_fn to handle None items (from image loading errors)
def collate_fn_skip_none(batch):
    # Filter out None values (from samples that failed to load)
    batch = [item for item in batch if item is not None]
    if not batch:
        # Return None if the entire micro-batch ended up being None
        # The training loop should check for this and skip the step.
        return None
    # Use default collate function for valid items
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)


def main(args):
    # Setup DDP and get rank, world_size, device
    rank, world_size, ddp_device = setup_ddp()
    is_main_process = (rank == 0)

    # --- Add this basic test ---
    if torch.cuda.is_available():
        try:
            a = torch.randn(1000, 1000, device=ddp_device)
            b = torch.randn(1000, 1000, device=ddp_device)
            c = a @ b # Simple matrix multiplication
            if is_main_process:
                 print(f"Basic CUDA test successful on {ddp_device}. Result shape: {c.shape}")
            del a, b, c # Clean up memory
        except Exception as e:
            if is_main_process:
                print(f"Basic CUDA test failed on {ddp_device}: {e}", file=sys.stderr)
            # Clean up DDP and exit if basic test fails on any rank
            cleanup_ddp()
            sys.exit(f"Basic CUDA test failed on {ddp_device}")
    elif is_main_process:
         print("CUDA not available, skipping basic CUDA test.")
    # --- End of basic test ---


    if is_main_process:
        print(f"Running in DDP mode with {world_size} GPU(s). Main process is RANK {rank}.")

    # Set seeds for reproducibility
    torch.manual_seed(args.seed + rank) # Add rank to seed for different data shuffles per rank
    torch.cuda.manual_seed(args.seed + rank)
    # Deterministic mode can slow down training, often set to False for performance unless specific debugging requires it
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True      # Usually True is better for performance

    # torch.autograd.set_detect_anomaly(True) # Keep this enabled during debugging NaN issues, disable for production

    start_time = time.time()
    print(f"Using device: {ddp_device}")

    # Determine AMP settings
    use_amp = not args.no_amp and ddp_device.type == 'cuda'
    amp_dtype = torch.float16 # Default to FP16
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16 # Prefer BF16 if supported for better stability
        if is_main_process: print(f"Autocast dtype will be: {amp_dtype} (BF16 supported)")
    elif use_amp:
        amp_dtype = torch.float16 # Fallback to FP16
        if is_main_process: print(f"Autocast dtype will be: {amp_dtype} (FP16, BF16 not supported or not primary)")
    else:
        amp_dtype = torch.float32 # AMP explicitly disabled or not on CUDA
        if is_main_process: print(f"Autocast dtype will be: {amp_dtype} (AMP disabled)")

    # --- Load Model and Processors ---
    if is_main_process:
        print(f"Loading base model: {args.base_model_name} ({args.base_model_type})")
    # is_eval=False is important for loading trainable components.
    # For T5 models, txt_processors might be a dict, e.g., txt_processors['text_input']
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=args.base_model_name, # e.g., "blip2_t5"
        model_type=args.base_model_type, # e.g., "pretrain_flant5xl"
        is_eval=False, # Load in training mode
        device=ddp_device
    )

    if is_main_process:
        print(f"Converting entire model to float32 before AMP context.")
    model = model.to(dtype=torch.float32)

    # Model should ideally be in FP32 initially when using AMP
    # Check the current dtype and set it to FP32 if it's not already
    if next(model.parameters()).dtype != torch.float32:
        if is_main_process: print(f"Model initial dtype is {next(model.parameters()).dtype}, converting to float32.")
        model = model.to(dtype=torch.float32)
    elif is_main_process:
        print(f"Model initial data type after loading: {next(model.parameters()).dtype}") # Log the actual dtype

    # >>> ENABLE GRADIENT CHECKPOINTING <<<
    if is_main_process:
        print("Attempting to enable gradient checkpointing...")

    # For the T5 model component
    if hasattr(model, "t5_model") and hasattr(model.t5_model, "gradient_checkpointing_enable"):
        try:
            model.t5_model.gradient_checkpointing_enable()
            if is_main_process: print("  Successfully enabled gradient checkpointing for T5 model.")
        except Exception as e_gc_t5:
            if is_main_process: print(f"  Failed to enable gradient checkpointing for T5: {e_gc_t5}")
    elif is_main_process:
        print("  T5 model does not have 'gradient_checkpointing_enable' method.")

    # For the visual encoder (EVA-ViT)
    # Hugging Face ViT models typically have this. LAVIS's EVA might too, or might need manual wrapping.
    # Check if the visual_encoder itself has the method
    if hasattr(model, "visual_encoder"):
        if hasattr(model.visual_encoder, "gradient_checkpointing_enable"):
            try:
                model.visual_encoder.gradient_checkpointing_enable()
                if is_main_process: print("  Successfully enabled gradient checkpointing for visual encoder.")
            except Exception as e_gc_vit:
                if is_main_process: print(f"  Failed to enable gradient checkpointing for visual encoder: {e_gc_vit}")
        elif hasattr(model.visual_encoder, 'blocks'): # If not, try to apply to its blocks manually
            if is_main_process: print("  Visual encoder does not have direct 'gradient_checkpointing_enable'. Attempting to wrap its blocks if applicable (not implemented here, requires torch.utils.checkpoint).")
            # Manual wrapping is more complex, let's hope the direct method exists.
            # If you need to do it manually, it would involve iterating model.visual_encoder.blocks
            # and changing their forward pass or wrapping them with torch.utils.checkpoint.checkpoint.
            # For now, we rely on a potential built-in method.
        elif is_main_process:
            print("  Visual encoder does not have 'gradient_checkpointing_enable' or 'blocks' attribute.")
    elif is_main_process:
        print("  Model does not have 'visual_encoder' attribute.")
    # >>> END GRADIENT CHECKPOINTING <<<



    if is_main_process:
        print(f"\n--- Runtime CUDA/cuDNN Info (inside script) ---")
        print(f"  torch.version.cuda: {torch.version.cuda}")
        print(f"  torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
        print(f"  torch.backends.cudnn.version: {torch.backends.cudnn.version()}")
        if torch.cuda.is_available():
            print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
            # Using ddp_device here which is specific to the current process's GPU
            print(f"  Current device: {ddp_device}")
            print(f"  torch.cuda.get_device_name({ddp_device.index}): {torch.cuda.get_device_name(ddp_device.index)}")
        else:
            print("  CUDA is not available according to torch.cuda.is_available()")
        print(f"--------------------------------------------------\n")


    # Try to find the Blip2Base instance. Path might vary slightly.
    blip_base_instance = None
    # Check common places for the use_half_precision flag
    if hasattr(model, 'blip_model') and hasattr(model.blip_model, 'use_half_precision'):
         blip_base_instance = model.blip_model
    elif hasattr(model, 'inner_model') and hasattr(model.inner_model, 'use_half_precision'): # Some wrappers might use inner_model
         blip_base_instance = model.inner_model
    elif hasattr(model, 'use_half_precision'): # If the main model object is Blip2Base
        blip_base_instance = model

    if blip_base_instance is not None:
        if hasattr(blip_base_instance, 'use_half_precision'):
            if is_main_process:
                print(f"Found use_half_precision on {type(blip_base_instance).__name__}. Current value: {blip_base_instance.use_half_precision}")
                # We set this to False to avoid double autocasting when using global AMP
                if blip_base_instance.use_half_precision: # Only set if it's True
                    if is_main_process: print("Setting use_half_precision to False to rely on global AMP.")
                    blip_base_instance.use_half_precision = False
                else:
                    if is_main_process: print("'use_half_precision' is already False.")
            else:
                 # Ensure non-main processes also set this flag if it exists and is True
                 if blip_base_instance.use_half_precision:
                      blip_base_instance.use_half_precision = False
        else:
            if is_main_process:
                print(f"WARNING: {type(blip_base_instance).__name__} does not have 'use_half_precision' attribute.")
    elif is_main_process:
        print("WARNING: Could not find Blip2Base instance to modify use_half_precision.")

    if is_main_process:
        # After potentially setting use_half_precision=False, check the *actual* model dtype again
        print(f"Model data type after modifications: {next(model.parameters()).dtype}")


    # Debug info about specific layer dtypes
    if is_main_process:
        if hasattr(model, 'visual_encoder') and hasattr(model.visual_encoder, 'patch_embed') and hasattr(model.visual_encoder.patch_embed, 'proj'):
            print(f"Data type of visual_encoder.patch_embed.proj.weight: {model.visual_encoder.patch_embed.proj.weight.dtype}")
        if hasattr(model, 'visual_encoder') and hasattr(model.visual_encoder.patch_embed.proj, 'bias') and model.visual_encoder.patch_embed.proj.bias is not None:
             print(f"Data type of visual_encoder.patch_embed.proj.bias: {model.visual_encoder.patch_embed.proj.bias.dtype}")


    if is_main_process:
        print("\n\nDEBUG: Full model structure:")
        print(model) # Prints a summary, good for top-level names

        print("\n\nDEBUG: All named modules (use these paths for hooks):")
        # Iterate over model directly before DDP wrap for hook paths
        model_for_named_modules = model # Check before DDP wrap
        for name, _module_obj in model_for_named_modules.named_modules():
            if name: # Filter out the root module which has an empty name string
                print(name)
        print("--- End of named modules ---\n\n")


    # --- Unfreeze Parameters ---
    # Calculate total parameters BEFORE any requires_grad changes and BEFORE DDP wrap
    total_params_count = sum(p.numel() for p in model.parameters())
    if is_main_process:
         print(f"Total model parameters: {total_params_count}")

    # Initialize trainable count before the conditional logic
    trainable_params_count = 0

    if args.unfreeze_all:
        if is_main_process: print("Unfreezing all model parameters for full fine-tuning...")
        for param in model.parameters(): # Iterate over model parameters directly
            param.requires_grad = True
        # After unfreezing all, the number of trainable parameters is the total number
        trainable_params_count = total_params_count # Assign in this branch
    else:
        if is_main_process: print("Warning: Not unfreezing all parameters (--unfreeze_all is not set). Defaulting to tuning Q-Former and t5_proj.")
        # Default is usually Q-Former + T5 projection layer for BLIP-2 fine-tuning
        if is_main_process: print("Freezing all model parameters initially...")
        for param in model.parameters(): # Iterate over model parameters directly
             param.requires_grad = False # Freeze everything

        # trainable_params_count is already initialized to 0 before the if/else
        if is_main_process: print("Unfreezing Q-Former and t5_proj parameters for partial fine-tuning...")
        for name, param in model.named_parameters(): # Iterate over named parameters to find Q-Former/t5_proj
            # Check if the parameter name corresponds to the Q-Former or the t5_proj layer
            if "Qformer" in name or "t5_proj" in name:
                param.requires_grad = True
                trainable_params_count += param.numel() # Accumulate in this branch
                # if is_main_process and param.numel() < 1000000: # Print small params for verification
                #      print(f"  Unfrozen: {name}")
            # You could optionally unfreeze other small parts like the lm_head if needed
            # elif "lm_head" in name:
            #     param.requires_grad = True
            #     trainable_params_count += param.numel()

    # After the if/else, trainable_params_count is guaranteed to be set.

    # Check if any parameters are trainable BEFORE DDP wrap
    if is_main_process:
        print(f"Number of trainable parameters: {trainable_params_count} / {total_params_count}")
        if trainable_params_count == 0:
            print("FATAL: No trainable parameters require gradients! Check model structure and unfreeze logic.", file=sys.stderr)
            cleanup_ddp()
            sys.exit(1)


    # --- DDP Model Wrapping ---
    # Wrap model with DDP *after* setting requires_grad
    if world_size > 1:
        # find_unused_parameters=False is important for performance if you are sure
        # all parameters requiring gradients are used. Set to True for debugging if needed.
        # Setting it to True might help debug issues where DDP complains about unused parameters.
        model = DDP(model, device_ids=[ddp_device.index], find_unused_parameters=True)
        if is_main_process: print(f"Model wrapped with DDP. find_unused_parameters=False")

    # Verify trainable parameters AGAIN after potential DDP wrap (state should be the same count)
    # Iterate over the DDP-wrapped model parameters for this check
    trainable_params_count_check = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process:
        # Reference the `total_params_count` that was calculated before DDP wrap
        # trainable_params_count_check should match trainable_params_count from before DDP wrap
        print(f"Number of trainable parameters (after DDP wrap check): {trainable_params_count_check} / {total_params_count}")
        if trainable_params_count_check != trainable_params_count:
            print(f"WARNING: Trainable parameter count changed after DDP wrap! Before: {trainable_params_count}, After: {trainable_params_count_check}", file=sys.stderr)


    # text_processor_for_training: Get the tokenizer instance
    text_processor_for_training = None
    tokenizer_found = False

    # Use the 'eval' key from txt_processors to get the tokenizer
    # This assumes the structure of txt_processors is similar to vis_processors
    if "eval" in txt_processors:
         text_processor_for_training = txt_processors["eval"]
         if is_main_process: print("Using tokenizer from txt_processors['eval'].")
         tokenizer_found = True
    # You could add a fallback to 'train' if 'eval' isn't the key for some reason
    elif "train" in txt_processors:
         text_processor_for_training = txt_processors["train"]
         if is_main_process: print("Using tokenizer from txt_processors['train'].")
         tokenizer_found = True
    # Add model-specific checks if needed, though 'eval'/'train' seems likely
    # elif hasattr(model, 't5_tokenizer'):
    #      text_processor_for_training = model.t5_tokenizer
    #      if is_main_process: print("Using T5 Tokenizer directly from model.")
    #      tokenizer_found = True
    # elif hasattr(model, 'opt_tokenizer'):
    #      text_processor_for_training = model.opt_tokenizer
    #      if is_main_process: print("Using OPT Tokenizer directly from model.")
    #      tokenizer_found = True


    # Check if a tokenizer was successfully assigned
    if not tokenizer_found: # Check the flag set in the conditional blocks
         if is_main_process:
              print(f"\nFATAL: Could not find a suitable tokenizer in txt_processors.", file=sys.stderr)
              print(f"Available keys in txt_processors: {txt_processors.keys()}", file=sys.stderr)
              print("Exiting. Please check the correct tokenizer key for your model type and LAVIS version.", file=sys.stderr)
         cleanup_ddp()
         sys.exit(1) # Exit with error

    # You should also verify that the loaded text_processor_for_training
    # has the necessary methods for your dataset's __getitem__ (like __call__ for tokenization)
    # print(f"DEBUG: Loaded text processor type: {type(text_processor_for_training)}")


    model.train() # Set model to training mode

    # --- Prepare Datasets and DataLoaders ---
    train_dataset = MedicalVqaFineTuningDataset(
        args.train_json_path, args.image_base_path,
        vis_processors["eval"], # Use 'eval' processor, usually suitable for fine-tuning data loading
        text_processor_for_training, # Use the found tokenizer
        max_input_len=args.max_input_len, max_target_len=args.max_target_len
    )
    # DDP: Use DistributedSampler
    # Add seed to sampler for reproducibility across runs with the same seed
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    # Batch size in DataLoader is PER GPU for DDP (micro-batch size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, collate_fn=collate_fn_skip_none, sampler=train_sampler,
        pin_memory=True # Enable pin_memory for faster data transfer to GPU
    )

    val_dataloader = None # Initialize to None
    if args.val_json_path:
        val_dataset = MedicalVqaFineTuningDataset(
            args.val_json_path, args.image_base_path,
            vis_processors["eval"], text_processor_for_training, # Use the found tokenizer
            max_input_len=args.max_input_len, max_target_len=args.max_target_len
        )
        # No shuffle for validation data
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


    # --- Optimizer and Scheduler ---
    # Use bitsandbytes' 8-bit AdamW optimizer
    # Filter parameters to only include those that require gradients
    optimizer = bnb_optim.AdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()), # Pass parameters from the DDP-wrapped model
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    if is_main_process: print(f"Optimizer: {type(optimizer).__name__}, LR: {args.learning_rate}, WD: {args.weight_decay}")

    # --- Initialize GradScaler ---
    # Initialize GradScaler only if AMP is enabled
    scaler = amp.GradScaler(enabled=use_amp) # <<< Correct use of the 'amp' alias

    if is_main_process: print(f"GradScaler enabled: {scaler.is_enabled()}")

    # --- Set _block_idx on ModifiedAttention instances (if applicable) ---
    actual_model_for_vit_mod = model.module if isinstance(model, DDP) else model

    # Check if the global 'local_eva_vit_modifications' module was loaded
    # and if it has the 'ModifiedAttention' class.
    # The 'local_eva_vit_modifications' variable is from the top-level import.
    if local_eva_vit_modifications is not None and \
       hasattr(local_eva_vit_modifications, 'ModifiedAttention'):
        
        # Get the ModifiedAttention class from the *globally* imported module
        ModifiedAttentionClass = getattr(local_eva_vit_modifications, 'ModifiedAttention')

        if hasattr(actual_model_for_vit_mod, 'visual_encoder') and \
           hasattr(actual_model_for_vit_mod.visual_encoder, 'blocks'):
            if is_main_process:
                print("Attempting to set _block_idx on ModifiedAttention instances...")
            for i, blk in enumerate(actual_model_for_vit_mod.visual_encoder.blocks):
                # Check if the block has an 'attn' attribute and if it's an instance
                # of the ModifiedAttentionClass from our single, global import.
                if hasattr(blk, 'attn') and isinstance(blk.attn, ModifiedAttentionClass):
                    blk.attn._block_idx = i
                    if is_main_process and i < 5: # Log for first few blocks on main process
                        # Confirm the type to be sure
                        print(f"  Set _block_idx = {i} for visual_encoder.blocks.{i}.attn (Type: {type(blk.attn).__name__})")
                elif is_main_process and i < 5: # If not ModifiedAttention or no attn
                    attn_type = type(blk.attn).__name__ if hasattr(blk, 'attn') else "NoAttnAttr"
                    # This print helps confirm if the patching worked or if the instance is of a different type
                    print(f"  Skipping visual_encoder.blocks.{i}.attn: Not the patched ModifiedAttention (Actual type: {attn_type})")
        elif is_main_process:
            print("WARNING: Could not find visual_encoder.blocks to set _block_idx.")
    elif is_main_process:
        # This 'else' branch executes if 'local_eva_vit_modifications' is None (file not found or import error at top)
        # or if 'ModifiedAttention' class is missing from the module.
        print(f"WARNING: local_eva_vit_modifications module not loaded or ModifiedAttention class not found. Skipping _block_idx setting.")
    # The original try-except for this block can be removed as we are not re-importing.
    # The error handling for the initial import is already at the top of the script.

    if is_main_process:
        print("DEBUG: ln_vision parameters:")
        # Check on actual model instance if DDP wrapped
        actual_model_for_ln_debug = model.module if isinstance(model, DDP) else model
        if hasattr(actual_model_for_ln_debug, 'ln_vision'):
            # Check for finite values before printing min/max/mean
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


    # --- Register Forward Hooks for Debugging ---
    hooks = []
    # Key modules to inspect. Adjust paths based on your `print(model)` or `model.named_modules()` output if needed.
    # Iterate over actual model instance if DDP wrapped to get correct paths for hooks
    model_for_hooks = model.module if isinstance(model, DDP) else model
    current_rank = rank # Pass rank to the hook factory


    modules_to_hook = {
        # Keep existing visual encoder hooks
        "vit_block_0_attn_qkv_out": "visual_encoder.blocks.0.attn.qkv",
        "vit_block_0_attn_proj_out": "visual_encoder.blocks.0.attn.proj", # After attention, before residual/norm

        "vit_block_19_attn_qkv_out": "visual_encoder.blocks.19.attn.qkv",
        "vit_block_19_attn_proj_out": "visual_encoder.blocks.19.attn.proj",

        "vit_block_38_attn_qkv_out": "visual_encoder.blocks.38.attn.qkv",
        "vit_block_38_attn_proj_out": "visual_encoder.blocks.38.attn.proj",

        # --- Add hooks for LayerNorms in the visual encoder blocks ---
        "vit_block_0_norm1_out": "visual_encoder.blocks.0.norm1",
        "vit_block_0_norm2_out": "visual_encoder.blocks.0.norm2",
        "vit_block_19_norm1_out": "visual_encoder.blocks.19.norm1",
        "vit_block_19_norm2_out": "visual_encoder.blocks.19.norm2",
        "vit_block_38_norm1_out": "visual_encoder.blocks.38.norm1",
        "vit_block_38_norm2_out": "visual_encoder.blocks.38.norm2",

        "ln_vision_out": "ln_vision", # Keep this final LN hook

        # --- Add hooks for key Qformer modules ---
        # Qformer Bert Embeddings LayerNorm
        "Qformer_bert_embeddings_norm_out": "Qformer.bert.embeddings.LayerNorm",
        # Qformer Cross Attention Output LayerNorm (where vision meets Qformer)
        # There are 6 layers of cross-attention (layers 0, 2, 4, 6, 8, 10 in BertEncoder.layer)
        "Qformer_crossattn_layer0_out_norm_out": "Qformer.bert.encoder.layer.0.crossattention.output.LayerNorm",
        "Qformer_crossattn_layer10_out_norm_out": "Qformer.bert.encoder.layer.10.crossattention.output.LayerNorm",
        # Qformer Self Attention Output LayerNorm
         "Qformer_selfattn_layer0_out_norm_out": "Qformer.bert.encoder.layer.0.attention.output.LayerNorm",
         "Qformer_selfattn_layer5_out_norm_out": "Qformer.bert.encoder.layer.5.attention.output.LayerNorm",
         "Qformer_selfattn_layer11_out_norm_out": "Qformer.bert.encoder.layer.11.attention.output.LayerNorm",
        # Qformer Intermediate Output LayerNorm (after FFN)
        "Qformer_intermediate_layer0_norm_out": "Qformer.bert.encoder.layer.0.output_query.LayerNorm", # This is the output of the intermediate+output dense layers
        "Qformer_intermediate_layer5_norm_out": "Qformer.bert.encoder.layer.5.output_query.LayerNorm",
        "Qformer_intermediate_layer11_norm_out": "Qformer.bert.encoder.layer.11.output_query.LayerNorm",


        # --- Add hooks for the t5_proj layer (mapping Qformer to T5 dimension) ---
        "t5_proj_out": "t5_proj",

        # --- Add hooks for key T5 modules ---
        # T5 Encoder final LayerNorm
        "t5_encoder_final_norm_out": "t5_model.encoder.final_layer_norm",
        # T5 Decoder Cross Attention LayerNorm (where Qformer/visual info meets T5 decoder)
        # T5-XL has 24 decoder blocks (0-23)
        "t5_decoder_crossattn_layer0_out_norm_out": "t5_model.decoder.block.0.layer.1.layer_norm",
        "t5_decoder_crossattn_layer11_out_norm_out": "t5_model.decoder.block.11.layer.1.layer_norm",
        "t5_decoder_crossattn_layer23_out_norm_out": "t5_model.decoder.block.23.layer.1.layer_norm",
         # T5 Decoder Self Attention LayerNorm
        "t5_decoder_selfattn_layer0_out_norm_out": "t5_model.decoder.block.0.layer.0.layer_norm",
        "t5_decoder_selfattn_layer11_out_norm_out": "t5_model.decoder.block.11.layer.0.layer_norm",
        "t5_decoder_selfattn_layer23_out_norm_out": "t5_model.decoder.block.23.layer.0.layer_norm",
        # T5 Decoder FFN LayerNorm
        "t5_decoder_ffn_layer0_norm_out": "t5_model.decoder.block.0.layer.2.layer_norm",
        "t5_decoder_ffn_layer11_norm_out": "t5_model.decoder.block.11.layer.2.layer_norm",
        "t5_decoder_ffn_layer23_norm_out": "t5_model.decoder.block.23.layer.2.layer_norm",
        # T5 Decoder final LayerNorm
        "t5_decoder_final_norm_out": "t5_model.decoder.final_layer_norm",

        # --- Add a hook for the final language model head ---
        "lm_head_out": "t5_model.lm_head", # Output before final loss calculation
    }

    if is_main_process: print("\nRegistering forward hooks for debugging...")
    for display_name, path_str in modules_to_hook.items():
        # Use the actual model instance for hooks (unwrapped if DDP is used)
        module_to_hook = get_module_by_path(model_for_hooks, path_str)
        if module_to_hook:
            if is_main_process: print(f"  Hooking: {display_name} (path: {path_str})")
            hooks.append(module_to_hook.register_forward_hook(get_nan_checking_hook(display_name, current_rank)))
        else:
            if is_main_process: print(f"  WARNING: Could not find module for {display_name} at path {path_str}. Skipping hook.", file=sys.stderr)
    if is_main_process and not hooks:
        print("WARNING: No hooks were registered. Check module paths.")
    if is_main_process: print("Hooks registration complete.\n")


    # --- Training Loop ---
    if is_main_process: print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    best_val_loss = float('inf')

    # Variables to track total loss for epoch average (unscaled)
    total_train_loss = 0.0
    total_samples_processed = 0 # This should track actual samples, not batches

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        if is_main_process: print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        model.train()
        train_dataloader.sampler.set_epoch(epoch) # Important for DDP to shuffle differently each epoch

        # Reset epoch loss trackers at the start of each epoch
        total_train_loss = 0.0
        total_samples_processed = 0

        # Reset gradient accumulation state
        optimizer.zero_grad() # Zero gradients at the start of the epoch
        accumulation_step_count = 0 # Reset accumulation counter

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}", disable=not is_main_process)):
            if batch is None:
                if is_main_process: print(f"Warning: Skipping step {step+1} due to empty batch after filtering bad samples.", file=sys.stderr)
                # If a batch is skipped, it doesn't contribute to accumulation, so we don't increment accumulation_step_count
                continue

            # Transfer data to the correct device
            images = batch["image"].to(ddp_device, non_blocking=True)
            text_input_batch = batch["text_input"]
            text_output_batch = batch["text_output"]

            # --- Check for NaNs/Infs detected by hooks *before* the forward pass ---
            # This check is done via a DDP all_reduce to see if any rank detected NaN/Inf
            # We need to collect nan_detected_module_info from all ranks first to see if ANY rank had a problem
            nan_info_list = [None for _ in range(world_size)] if is_main_process else None
            dist.gather_object(nan_detected_module_info, nan_info_list, dst=0)

            if is_main_process:
                 first_nan_info = None
                 for info in nan_info_list:
                      if info is not None and info["name"] is not None:
                           first_nan_info = info
                           break # Found the first rank with a problem

                 if first_nan_info:
                      print(f"\nFATAL: NaN/Inf detected on Rank {first_nan_info['rank']} by hook in module {first_nan_info['name']}.", file=sys.stderr)
                      print(f"Problematic module output stats: {first_nan_info['output_stats']}", file=sys.stderr)
                      # Coordinate exit across all ranks - A barrier is needed before exiting
                      dist.barrier() # Ensure all ranks reach here
                      # Clean up hooks and DDP before exiting
                      for h in hooks: h.remove()
                      cleanup_ddp()
                      sys.exit(1) # Exit immediately on all ranks
                 # Reset global info on main process after checking all ranks for this step
                 nan_detected_module_info["name"] = None
                 nan_detected_module_info["output_stats"] = None
                 nan_detected_module_info["rank"] = None
            else:
                 # Non-main ranks send their info and then wait at the barrier
                 if nan_detected_module_info["name"] is not None:
                      # Ensure rank is recorded if detected locally
                      nan_detected_module_info["rank"] = rank
                 # Send the locally detected info to the main process
                 dist.gather_object(nan_detected_module_info, nan_info_list, dst=0)
                 # Wait for main process to check and decide whether to exit
                 dist.barrier()
                 # If main process decided to exit, the sys.exit(1) will be called above
                 # If not exiting, reset local info and continue
                 nan_detected_module_info["name"] = None
                 nan_detected_module_info["output_stats"] = None
                 nan_detected_module_info["rank"] = None


            # --- Forward Pass and Loss Calculation (with AMP) ---
            # Use autocast context manager for mixed precision
            with amp.autocast(enabled=use_amp, dtype=amp_dtype): # <<< Corrected: Removed device_type
                samples = {
                    "image": images,
                    "text_input": text_input_batch,
                    "text_output": text_output_batch
                }

                try:
                    # Pass the whole batch (size args.batch_size = 1) as the micro-batch
                    loss_output = model(samples)

                    if loss_output is None or "loss" not in loss_output:
                        if is_main_process: print(f"Warning: Invalid loss_output at step {step+1} on rank {rank}. Skipping batch. Output: {loss_output}", file=sys.stderr)
                        # If model output is bad, skip this micro-batch. Zero gradients for this accumulation chunk.
                        optimizer.zero_grad()
                        accumulation_step_count = 0 # Reset accumulation count for the current chunk
                        continue

                    loss = loss_output["loss"]

                    # Check for non-finite loss immediately after calculation
                    if not torch.isfinite(loss):
                         if is_main_process:
                              print(f"\n!!! FATAL: Non-finite loss detected at Step {step+1} on rank {rank} BEFORE backward pass. Loss: {loss.item()}. !!!", file=sys.stderr)
                              print(f"    Model output keys: {loss_output.keys()}", file=sys.stderr)
                              try:
                                 # Save problematic inputs
                                 torch.save(samples, f"bad_samples_rank_{rank}_step_{step+1}_loss.pt")
                                 print(f"    Saved problematic samples to bad_samples_rank_{rank}_step_{step+1}_loss.pt", file=sys.stderr)
                              except Exception as save_e:
                                 print(f"    Error saving samples: {save_e}", file=sys.stderr)
                         # Clean up hooks and DDP before exiting on all ranks
                         for h in hooks: h.remove()
                         cleanup_ddp()
                         sys.exit(1) # Exit immediately

                    # If loss is finite, accumulate it (unscaled)
                    original_loss_value = loss.item() # Store the unscaled loss value
                    total_train_loss += original_loss_value * images.size(0) # Sum unscaled loss * actual batch size (1)
                    total_samples_processed += images.size(0)

                    # Scale loss for gradient accumulation
                    # This scaled loss is used for the backward pass
                    scaled_loss = loss / args.gradient_accumulation_steps


                except Exception as e:
                    # Catch exceptions during forward pass, log, and skip this micro-batch
                    if is_main_process:
                         print(f"→ skipping step {step+1}, id={batch.get('question_id', 'N/A')} on rank {rank} due to exception: {e}", file=sys.stderr)
                         traceback.print_exc(file=sys.stderr) # Print traceback for better debugging
                    # Zero gradients for this potentially partial accumulation chunk
                    optimizer.zero_grad()
                    accumulation_step_count = 0 # Reset accumulation count
                    continue


            # --- Backward Pass ---
            # The backward pass is performed on the scaled loss
            if use_amp:
                # Scales the loss and calls backward() to create scaled gradients
                scaler.scale(scaled_loss).backward() # <<< Correct use of scaler.scale()
            else:
                # No scaler needed, backward on the manually scaled loss
                scaled_loss.backward()

            # Increment accumulation step count for this successful micro-batch
            accumulation_step_count += 1

            # --- Optimizer Step and Gradient Zeroing ---
            # Check if it's time to perform an optimizer step
            # This happens when accumulation_step_count reaches the target (args.gradient_accumulation_steps)
            # OR if this is the very last micro-batch of the epoch AND accumulation_step_count > 0
            is_final_accumulation_step_in_chunk = accumulation_step_count == args.gradient_accumulation_steps
            is_last_data_batch_in_epoch = (step + 1) == len(train_dataloader)

            if is_final_accumulation_step_in_chunk or (is_last_data_batch_in_epoch and accumulation_step_count > 0):

                # Perform the optimizer step and gradient zeroing

                # --- Gradient Clipping ---
                if args.max_grad_norm > 0: # Check if clipping is enabled (max_grad_norm > 0)
                    # Unscale gradients before clipping if using AMP
                    # This is crucial because clip_grad_norm_ operates on the gradient tensor directly
                    # and is not aware of the scaler's scaling factor.
                    if use_amp:
                         scaler.unscale_(optimizer) # Gradients are now unscaled

                    # Apply gradient clipping to parameters that require gradients
                    # Filter parameters to avoid issues with frozen layers that have no gradients (which would have grad=None)
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad and p.grad is not None, model.parameters()), args.max_grad_norm)


                # --- Optimizer Step and Scaler Update ---
                if use_amp:
                    # scaler.step(optimizer) performs the optimizer step.
                    # It internally checks if the gradients are valid (not NaN/Inf).
                    # If gradients are valid, it applies the unscaled gradients (it may unscale internally again,
                    # but calling scaler.unscale_ beforehand is the standard practice for clipping).
                    # If gradients are NaN/Inf, it skips the optimizer step.
                    scaler.step(optimizer) # <<< Correct use of scaler.step()

                    # scaler.update() adjusts the scaling factor for the next step.
                    # It should be called after scaler.step().
                    scaler.update() # <<< Correct use of scaler.update()
                else:
                    # If not using AMP, simply perform the optimizer step
                    optimizer.step()

                # --- Zero Gradients ---
                # Zero gradients *after* the optimizer step
                optimizer.zero_grad()

                # Reset accumulation step count after a successful step
                accumulation_step_count = 0 # Reset accumulation step counter for the next chunk


            # Log per-batch/per-step stats if log_interval is used
            # This should happen after each micro-batch if desired, regardless of accumulation step.
            # if (step + 1) % args.log_interval == 0 and is_main_process:
            #     # Note: 'loss' here is the scaled loss for the micro-batch
            #     # Printing the unscaled loss might be more intuitive: original_loss_value
            #      if 'original_loss_value' in locals() and torch.isfinite(torch.tensor(original_loss_value)).all(): # Check if loss was successfully calculated and is finite
            #          print(f"  Epoch {epoch+1}, Step {step+1}/{len(train_dataloader)}, Micro Batch Loss: {original_loss_value:.4f}")


        # --- End of Training Epoch Loop ---

        # The average loss calculation for the epoch now correctly uses the sum of unscaled losses
        avg_train_loss = total_train_loss / total_samples_processed if total_samples_processed > 0 else 0

        # --- Validation Loop ---
        # Only run validation if a validation path is provided
        avg_val_loss = float('inf') # Initialize before validation on this rank
        if val_dataloader:
            if is_main_process: print(f"\n--- Starting Validation for Epoch {epoch+1} ---")
            model.eval() # Set model to evaluation mode
            total_val_loss = 0
            val_items_processed = 0
            with torch.no_grad(): # No gradient calculation during validation
                for step, batch in enumerate(tqdm(val_dataloader, desc=f"Validating Epoch {epoch+1}", disable=not is_main_process)):
                    if batch is None:
                        if is_main_process: print(f"Warning: Skipping validation step {step+1} due to empty batch.", file=sys.stderr)
                        continue

                    images = batch["image"].to(ddp_device, non_blocking=True)
                    text_input_batch = batch["text_input"]
                    text_output_batch = batch["text_output"]

                    # Validation also uses autocast if AMP is enabled, for consistent inference dtype
                    with amp.autocast(enabled=use_amp, dtype=amp_dtype): # <<< Corrected: Removed device_type
                        samples = {
                            "image": images,
                            "text_input": text_input_batch,
                            "text_output": text_output_batch
                        }
                        try:
                            loss_output = model(samples)
                            if loss_output is None or "loss" not in loss_output:
                                if is_main_process: print(f"Warning: Invalid loss_output during validation at step {step+1}. Skipping batch. Output: {loss_output}", file=sys.stderr)
                                continue
                            loss = loss_output["loss"]
                            # Check for non-finite loss during validation as well
                            if not torch.isfinite(loss):
                                 if is_main_process: print(f"Warning: Non-finite validation loss detected at step {step+1} on rank {rank}. Loss: {loss.item()}.", file=sys.stderr)
                                 continue # Skip accumulation for this batch

                        except Exception as e:
                            if is_main_process:
                                print(f"  → skipping validation sample {batch.get('question_id', 'N/A')} because of exception: {e}", file=sys.stderr)
                                traceback.print_exc(file=sys.stderr) # Print traceback
                            continue # Skip batch due to exception

                    # Only accumulate valid, finite losses
                    total_val_loss += loss.item() * images.size(0) # Sum unscaled validation loss * batch size (1)
                    val_items_processed += images.size(0)

            # Calculate average validation loss on this rank
            avg_val_loss = total_val_loss / val_items_processed if val_items_processed > 0 else float('inf') # Use float('inf') if no val samples were processed


        # --- Gather and Average Metrics across all GPUs ---
        if world_size > 1:
            # Need to sum total_train_loss, total_samples_processed, total_val_loss, val_items_processed
            # across all ranks. Use torch.distributed.all_reduce for efficiency.

            # Create tensors for metrics
            # Using float64 for sums to reduce risk of overflow with large datasets
            train_metrics = torch.tensor([total_train_loss, total_samples_processed], dtype=torch.float64, device=ddp_device)
            # Ensure val_metrics is created even if val_dataloader is None, with zero values for reduction
            val_metrics_tensor = torch.tensor([total_val_loss, val_items_processed], dtype=torch.float64, device=ddp_device)


            # All-reduce to sum metrics across all ranks
            dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_metrics_tensor, op=dist.ReduceOp.SUM) # Use the tensor here

            # Extract summed metrics on all ranks
            total_train_loss_all, total_train_samples_all = train_metrics.tolist()
            total_val_loss_all, total_val_samples_all = val_metrics_tensor.tolist() # Extract from the tensor


            # Calculate global average losses on all ranks
            # Ensure division by zero is handled
            avg_train_loss_global = total_train_loss_all / total_train_samples_all if total_train_samples_all > 0 else 0
            # If validation was run (checked by val_dataloader presence earlier), calculate avg_val_loss
            # Otherwise, it remains float('inf') as initialized or 0 if val_dataloader was None
            avg_val_loss_global = total_val_loss_all / total_val_samples_all if total_val_samples_all > 0 else (float('inf') if val_dataloader else 0.0)


            # Print global averages only on the main process
            if is_main_process:
                print(f"Epoch {epoch+1} Global Average Training Loss: {avg_train_loss_global:.4f}")
                if val_dataloader: # Only print validation if it was run
                    print(f"Epoch {epoch+1} Global Average Validation Loss: {avg_val_loss_global:.4f}")

            # Use the global average validation loss for saving logic
            current_avg_val_loss = avg_val_loss_global

        else: # Not DDP
            # If not DDP, the calculated avg_train_loss and avg_val_loss are already global
            if is_main_process: # This will always be True if not DDP
                 print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")
                 if val_dataloader: # Only print validation if it was run
                    print(f"Epoch {epoch+1} Average Validation Loss: {avg_val_loss:.4f}")
            current_avg_val_loss = avg_val_loss # Use the local average as the global one


        # --- Checkpointing (only on main process) ---
        if is_main_process:
            # Save best model based on validation loss IF validation was performed
            if val_dataloader and current_avg_val_loss < best_val_loss:
                best_val_loss = current_avg_val_loss
                print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                save_path = Path(args.output_dir) / f"best_model_epoch_{epoch+1}.pth"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                # Save state_dict without DDP wrapper by accessing model.module
                state_dict_to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                # Example of filtering trainable params (uncomment if needed):
                # actual_model_state = model.module if isinstance(model, DDP) else model
                # state_dict_to_save = {k: v for k, v in actual_model_state.state_dict().items() if actual_model_state.get_parameter(k).requires_grad}

                try:
                    torch.save(state_dict_to_save, save_path)
                    print(f"Model saved to {save_path}")
                except Exception as save_e:
                    print(f"ERROR: Failed to save model checkpoint to {save_path}: {save_e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)


            # Optionally save epoch checkpoints regardless of validation loss
            # epoch_save_path = Path(args.output_dir) / f"model_epoch_{epoch+1}.pth"
            # epoch_save_path.parent.mkdir(parents=True, exist_ok=True)
            # state_dict_to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            # try:
            #     torch.save(state_dict_to_save, epoch_save_path)
            #     print(f"Epoch {epoch+1} model saved to {epoch_save_path}")
            # except Exception as save_e:
            #     print(f"ERROR: Failed to save epoch checkpoint to {epoch_save_path}: {save_e}", file=sys.stderr)
            #     traceback.print_exc(file=sys.stderr)


        if is_main_process:
             print(f"Epoch {epoch+1} completed in {(time.time() - epoch_start_time):.2f} seconds.")

        # Ensure all processes wait here before starting next epoch
        # This is essential for DDP synchronization
        dist.barrier()


    # --- Final Save and Cleanup ---
    # All ranks should clean up DDP
    if is_main_process:
        print("\nFine-tuning finished.")
        final_save_path = Path(args.output_dir) / "final_model.pth"
        # Save state_dict without DDP wrapper
        state_dict_to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        try:
            torch.save(state_dict_to_save, final_save_path)
            print(f"Final model saved to {final_save_path}")
        except Exception as save_e:
             print(f"ERROR: Failed to save final model to {final_save_path}: {save_e}", file=sys.stderr)
             traceback.print_exc(file=sys.stderr)

        print(f"Total script time: {(time.time() - start_time):.2f} seconds.")

    # Clean up hooks on all ranks
    if is_main_process: print("Removing forward hooks...")
    for h in hooks:
        h.remove()
    if is_main_process: print("Hooks removed.")

    # Ensure all processes finish logging/saving before cleaning up DDP
    dist.barrier()
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Fine-tune BLIP-2 on a medical VQA modality.")
    # Model args
    parser.add_argument("--base_model_name", type=str, default="blip2_t5", help="Base LAVIS model name (e.g., blip2_t5, blip2_opt).")
    parser.add_argument("--base_model_type", type=str, default="pretrain_flant5xl", help="Specific model type for LAVIS (e.g., pretrain_flant5xl, pretrain_opt2.7b).")
    parser.add_argument("--unfreeze_all", action="store_true", help="If set, unfreeze all parameters for full fine-tuning. Otherwise, tunes Q-Former and t5_proj by default.")

    # Data args
    parser.add_argument("--train_json_path", type=str, required=True, help="Path to the training JSON file.")
    parser.add_argument("--val_json_path", type=str, default=None, help="Path to the validation JSON file (optional).")
    parser.add_argument("--image_base_path", type=str, required=True, help="Base directory where images (e.g., 'Images/' folder) are located.")
    parser.add_argument("--max_input_len", type=int, default=128, help="Max length for tokenized input question/prompt.")
    parser.add_argument("--max_target_len", type=int, default=64, help="Max length for tokenized target answer.")

    # Training args
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save fine-tuned model checkpoints.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training micro batch size per GPU.") # Renamed for clarity
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps.") # Changed default to 8 as you were using it
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for AdamW.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping. Set to 0 or negative to disable.")
    # parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler.") # Scheduler is commented out
    # Logging interval is complex with accumulation, rely on epoch average
    parser.add_argument("--log_interval", type=int, default=50, help="Log training loss every N steps (Note: Logging is primarily epoch-based with accumulation).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    # parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use if CUDA is available (-1 for CPU).") # Removed, torchrun handles this via LOCAL_RANK
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed-precision autocast (fp32 only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")


    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)