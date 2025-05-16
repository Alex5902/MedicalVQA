#!/usr/bin/env python3
import os
import json
import numpy as np
from tqdm import tqdm # Kept for potential future use if data_all is large
from PIL import Image
from scipy.special import softmax
from lavis.models import load_model_and_preprocess
import torch
from types import MethodType
import argparse
import sys 
import traceback

# --- PEFT Import ---
from peft import PeftModel

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# --- Monkey-patches for Blip2Base and ViT (if needed during evaluation) ---
# It's generally good practice to apply the same patches used during training
# if they affect model architecture or forward pass behavior that's also relevant at eval time.
# For now, assuming these are mainly for training stability and might not be strictly needed
# for evaluation if not using AMP or if the issues don't manifest in eval.
# If you find NaNs or inconsistencies, consider adding them here too.
# For simplicity in this eval script, I'll omit them, but be aware.
# --- Start of Blip2Base.maybe_autocast monkey-patch (OPTIONAL FOR EVAL) ---
try:
    from lavis.models.blip2_models.blip2 import Blip2Base
    import contextlib

    @contextlib.contextmanager
    def no_op_maybe_autocast_eval(self, dtype=None):
        yield 
    # To avoid conflicts if this script is imported, check if it's already patched
    if not hasattr(Blip2Base, '_maybe_autocast_original_eval'):
        Blip2Base._maybe_autocast_original_eval = Blip2Base.maybe_autocast
        Blip2Base.maybe_autocast = no_op_maybe_autocast_eval
        print("Successfully monkey-patched 'Blip2Base.maybe_autocast' for evaluation (if not already by training script).")
except ImportError:
    print("Warning: Could not import Blip2Base for eval monkey-patching.")
# --- End of Blip2Base.maybe_autocast monkey-patch ---


@torch.no_grad()
def forward_lm(self, samples):
    """
    Custom forward_lm for prefix scoring.
    Calculates the loss for generating `samples["text_output"]` (candidate answer)
    given `samples["image"]` and `samples["text_input"]` (prefix/question).
    """
    image = samples["image"]
    
    # Ensure model components are on the correct device
    # This should ideally be handled once after model loading.
    # self.visual_encoder.to(image.device)
    # self.ln_vision.to(image.device)
    # self.Qformer.to(image.device)
    # self.t5_proj.to(image.device)
    # self.t5_model.to(image.device)

    # Use the model's configured autocast (which is now a no-op if patched globally)
    # or manage autocast explicitly here.
    # Forcing bfloat16 for consistency with training if available.
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and image.device.type == 'cuda' else torch.float16
    use_amp_for_vit = image.device.type == 'cuda' # Enable AMP for ViT parts if on CUDA

    with torch.cuda.amp.autocast(enabled=use_amp_for_vit, dtype=amp_dtype):
        image_embeds = self.ln_vision(self.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
    
    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    
    with torch.cuda.amp.autocast(enabled=use_amp_for_vit, dtype=amp_dtype): # QFormer might also benefit
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        
    atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

    # Explicit AMP for T5 model part
    use_amp_for_t5 = image.device.type == 'cuda'
    with torch.cuda.amp.autocast(enabled=use_amp_for_t5, dtype=amp_dtype):
        # Tokenize prefix (text_input)
        prefix_tokenized = self.t5_tokenizer(
            samples["text_input"],
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len, # Or a separate max_prefix_len
            return_tensors="pt"
        ).to(image.device)

        # Tokenize candidate (text_output)
        candidate_tokenized = self.t5_tokenizer(
            samples["text_output"],
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len, # Or a separate max_candidate_len
            return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, prefix_tokenized.attention_mask], dim=1)

        # Prepare targets for loss calculation (mask out padding)
        targets = candidate_tokenized.input_ids.clone()
        targets[targets == self.t5_tokenizer.pad_token_id] = -100 # Standard ignore_index for CrossEntropyLoss

        # Prepare inputs for T5 encoder
        # The t5_model.encoder.embed_tokens is part of t5_model, so it's under its autocast
        inputs_embeds = self.t5_model.encoder.embed_tokens(prefix_tokenized.input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
        
        # Prepare decoder inputs (shifted right)
        decoder_input_ids = self.t5_model._shift_right(candidate_tokenized.input_ids)

        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=candidate_tokenized.attention_mask,
            return_dict=True,
            labels=targets, # T5 calculates loss internally if labels are provided
        )
        loss = outputs.loss
        
    # Handle potential NaN/Inf in loss, common in mixed precision if not perfectly stable
    if loss is not None and not torch.isfinite(loss):
        print(f"WARNING: Non-finite loss ({loss.item()}) detected in forward_lm. Replacing with a large value.", file=sys.stderr)
        # You might want to log the inputs that caused this for debugging
        # For now, return a very high loss value.
        loss = torch.tensor(float('inf'), device=loss.device, dtype=loss.dtype)


    return {"loss": loss}


def load_candidates_medical(data):
    a,b,c,d = data.get('option_A'), data.get('option_B'), data.get('option_C'), data.get('option_D')
    answer_list = [a, b]
    if c is not None:
        answer_list.append(c)
    if d is not None:
        answer_list.append(d)
    return [str(cand) if cand is not None else "" for cand in answer_list]


def load_prompt(question, idx=4):
    prompts = ["{}".format(question),
               "{} Answer:".format(question),
               "{} The answer is".format(question),
               "Question: {} Answer:".format(question),
               "Question: {} The answer is".format(question)
               ]
    return prompts[idx]
 
@torch.no_grad()
def test(model, vis_processors, dataset_path_str=None, model_name_for_output='blip2_eval', prompt_idx=4, save_path_dir_str='', image_root_str=''):
    data_all = json.load(open(dataset_path_str))
    cnt = 0
    correct = 0
    res = []
    
    # Determine if running in main process for tqdm (if DDP were used for eval)
    is_main_eval_process = not dist.is_initialized() or dist.get_rank() == 0

    for data_idx, data in enumerate(tqdm(data_all, desc=f"Evaluating {os.path.basename(dataset_path_str)}", disable=not is_main_eval_process)):
        cnt += 1
        question = data['question']
        candidates = load_candidates_medical(data)
        answer = str(data['gt_answer']) 
        img_path_relative = data['image_path']
        img_path_full = os.path.join(image_root_str, img_path_relative)
        
        item_id_for_log = data.get('question_id', data_idx)

        if not os.path.exists(img_path_full):
            if is_main_eval_process: print(f"WARNING: Image file not found: {img_path_full} for item {item_id_for_log}. Skipping item.")
            data['model_pred'] = "IMAGE_NOT_FOUND"
            data['confidence_scores_neg_log_likelihood'] = "[]"
            data['is_correct'] = 'no'
            res.append(data)
            continue
        
        prefix_prompt = load_prompt(question, prompt_idx)
        candidate_losses = []
        try:
            raw_image = Image.open(img_path_full).convert("RGB")
            image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        except Exception as e:
            if is_main_eval_process: print(f"ERROR: Failed to load or process image {img_path_full} for item {item_id_for_log}: {e}. Skipping item.")
            data['model_pred'] = "IMAGE_LOAD_ERROR"
            data['confidence_scores_neg_log_likelihood'] = "[]"
            data['is_correct'] = 'no'
            res.append(data)
            continue
            
        for candidate_text in candidates:
            samples = {
                "image": image_tensor,
                "text_input": [prefix_prompt], 
                "text_output": [str(candidate_text) if candidate_text is not None else ""], 
            }
            try:
                outputs = model.forward_lm(samples) # Call the bound method
                loss = outputs["loss"]
                # Ensure loss is a scalar float, handle potential issues
                if loss is None: # Should not happen if forward_lm is robust
                    candidate_losses.append(float('inf'))
                    if is_main_eval_process: print(f"WARNING: Loss is None for item {item_id_for_log}, candidate '{candidate_text}'.", file=sys.stderr)
                elif not torch.isfinite(loss).item(): # Check for NaN/Inf from loss tensor
                    candidate_losses.append(float('inf')) # Assign a very high loss
                    if is_main_eval_process: print(f"WARNING: Non-finite loss ({loss.item()}) for item {item_id_for_log}, candidate '{candidate_text}'.", file=sys.stderr)
                else:
                    candidate_losses.append(loss.item())
            except Exception as e:
                if is_main_eval_process: print(f"ERROR during model forward for item {item_id_for_log}, candidate '{candidate_text}': {e}")
                traceback.print_exc(file=sys.stderr) 
                candidate_losses.append(float('inf')) 

        data['confidence_scores_neg_log_likelihood'] = str(candidate_losses)
        pred = "ERROR" # Default prediction in case of all errors
        
        if not candidate_losses or all(l == float('inf') for l in candidate_losses) :
            pred = "ALL_CANDIDATES_ERROR"
        else:
            pred_idx = np.argmin(candidate_losses) # Lower loss is better
            pred = candidates[pred_idx]
            
            try:
                # Softmax requires finite values. Replace inf with a large number.
                finite_losses = np.array([l if l != float('inf') else 1e9 for l in candidate_losses])
                if not np.all(np.isinf(finite_losses)): # Avoid softmax if all were effectively errors
                    probabilities = softmax(np.reciprocal(finite_losses + 1e-9)) # Add epsilon to avoid div by zero if loss is 0
                    data['probabilities_from_loss'] = str(probabilities.tolist())
            except Exception as e_prob:
                if is_main_eval_process: print(f"Warning: Could not calculate probabilities for item {item_id_for_log}: {e_prob}")
                data['probabilities_from_loss'] = "[]"


        data['model_pred'] = str(pred) # Ensure prediction is string
        data['is_correct'] = 'yes' if str(pred) == answer else 'no'
        if str(pred) == answer:
            correct += 1
        res.append(data)
        
    acc = correct / cnt if cnt > 0 else 0
    print(f"Accuracy for {os.path.basename(dataset_path_str)}: {acc:.4f} ({correct}/{cnt})")
        
    final_res = {'model_name': model_name_for_output, 
                 'dataset_name': os.path.basename(dataset_path_str), 
                 'accuracy': acc,
                 'correct_count': correct,
                 'total_count': cnt,
                 'predictions': res}
    
    base_dataset_filename = os.path.basename(dataset_path_str)
    if base_dataset_filename.endswith('.json'):
        base_dataset_filename = base_dataset_filename[:-5]
    output_json_filename = f"{base_dataset_filename}_prefix_results.json"
    full_save_path = os.path.join(save_path_dir_str, output_json_filename)
    
    with open(full_save_path, 'w') as f:
        json.dump(final_res, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {full_save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="BLIP-2 LoRA/Full Model Prefix Scoring Evaluation")
    
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSON dataset file for a specific modality")
    parser.add_argument("--answer_path", type=str, required=True, help="Directory to save output JSON results")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory containing the 'Images' folder")
    
    parser.add_argument("--base_model_name", type=str, default="blip2_t5", help="Base LAVIS model name (e.g., blip2_t5).")
    parser.add_argument("--base_model_type", type=str, default="pretrain_flant5xl", help="Specific model type for LAVIS (e.g., pretrain_flant5xl).")
    parser.add_argument("--model_name_tag", type=str, default="blip2_eval", help="Tag for naming output files, will be combined with checkpoint/LoRA info.")
    
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a FULL fine-tuned model checkpoint (.pth). IGNORED if LoRA paths are given.")
    parser.add_argument("--prompt_idx", type=int, default=4, help="Index of the prompt to use")

    # --- NEW LoRA ARGUMENTS ---
    parser.add_argument("--lora_t5_adapter_path", type=str, default=None, help="Path to the trained T5 LoRA adapter directory.")
    parser.add_argument("--lora_qformer_adapter_path", type=str, default=None, help="Path to the trained Q-Former LoRA adapter directory.")
    
    # For single GPU evaluation, is_main_process is effectively true.
    # If you ever adapt this for DDP evaluation, this would need proper handling.
    parser.add_argument("--is_main_process", action="store_true", default=True, help="Flag indicating if this is the main process (for logging). Default true for single GPU eval.")
    
    args = parser.parse_args()
    return args

def run(args):
    print(f"Loading base model architecture: {args.base_model_name}, type: {args.base_model_type}")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=args.base_model_name, 
        model_type=args.base_model_type, 
        is_eval=True, 
        device=device # Global device variable
    )
    model.max_txt_len = 512 # Set a default, can be overridden if needed

    output_model_name_tag = args.model_name_tag # Start with the base tag from orchestrator

    # --- LoRA Adapter Loading or Full Checkpoint Loading ---
    if args.lora_t5_adapter_path or args.lora_qformer_adapter_path:
        if args.is_main_process: print("LoRA adapter paths provided, applying LoRA adapters...")
        if args.checkpoint_path and args.is_main_process:
            print(f"WARNING: --checkpoint_path '{args.checkpoint_path}' is provided but will be IGNORED because LoRA adapter paths are also provided.")

        # Determine T5 and Q-Former attribute names
        t5_model_attr_name = 't5_model' if hasattr(model, 't5_model') else ('llm_model' if hasattr(model, 'llm_model') else None)
        qformer_attr_name = 'Qformer' if hasattr(model, 'Qformer') else ('q_former' if hasattr(model, 'q_former') else None)

        lora_tags = []
        # Apply T5 LoRA adapter
        if args.lora_t5_adapter_path:
            if t5_model_attr_name and hasattr(model, t5_model_attr_name):
                if os.path.exists(args.lora_t5_adapter_path):
                    if args.is_main_process: print(f"  Loading T5 LoRA adapter from: {args.lora_t5_adapter_path} onto model.{t5_model_attr_name}")
                    t5_component = getattr(model, t5_model_attr_name)
                    if isinstance(t5_component, PeftModel): t5_component = t5_component.get_base_model()
                    
                    t5_peft_model = PeftModel.from_pretrained(t5_component, args.lora_t5_adapter_path, adapter_name="eval_t5_adapter")
                    setattr(model, t5_model_attr_name, t5_peft_model)
                    getattr(model, t5_model_attr_name).set_adapter("eval_t5_adapter")
                    if args.is_main_process: print(f"    Successfully applied T5 LoRA adapter.")
                    lora_tags.append(Path(args.lora_t5_adapter_path).parent.name) # e.g. fundus_t5_final
                elif args.is_main_process: print(f"  WARNING: T5 LoRA adapter path not found: {args.lora_t5_adapter_path}")
            elif args.is_main_process: print(f"  WARNING: T5 component '{t5_model_attr_name}' not found on model, cannot apply T5 LoRA.")

        # Apply Q-Former LoRA adapter
        if args.lora_qformer_adapter_path:
            if qformer_attr_name and hasattr(model, qformer_attr_name):
                if os.path.exists(args.lora_qformer_adapter_path):
                    if args.is_main_process: print(f"  Loading Q-Former LoRA adapter from: {args.lora_qformer_adapter_path} onto model.{qformer_attr_name}")
                    qformer_component = getattr(model, qformer_attr_name)
                    if isinstance(qformer_component, PeftModel): qformer_component = qformer_component.get_base_model()
                    
                    qformer_peft_model = PeftModel.from_pretrained(qformer_component, args.lora_qformer_adapter_path, adapter_name="eval_qf_adapter")
                    setattr(model, qformer_attr_name, qformer_peft_model)
                    getattr(model, qformer_attr_name).set_adapter("eval_qf_adapter")
                    if args.is_main_process: print(f"    Successfully applied Q-Former LoRA adapter.")
                    lora_tags.append(Path(args.lora_qformer_adapter_path).parent.name)
                elif args.is_main_process: print(f"  WARNING: Q-Former LoRA adapter path not found: {args.lora_qformer_adapter_path}")
            elif args.is_main_process: print(f"  WARNING: Q-Former component '{qformer_attr_name}' not found on model, cannot apply Q-Former LoRA.")
        
        if lora_tags: # Update output tag if LoRA was applied
            output_model_name_tag = f"{args.model_name_tag}_lora_{'_'.join(lora_tags)}"

    elif args.checkpoint_path: 
        if args.is_main_process: print(f"Loading full fine-tuned checkpoint from: {args.checkpoint_path}")
        # ... (your existing full checkpoint loading logic) ...
        if not os.path.exists(args.checkpoint_path):
            print(f"ERROR: Checkpoint file not found at {args.checkpoint_path}", file=sys.stderr); sys.exit(1)
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint) # Handles if 'model' key exists or if it's raw state_dict
            
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k 
                new_state_dict[name] = v
            
            load_msg = model.load_state_dict(new_state_dict, strict=False)
            if args.is_main_process: 
                print(f"  Full checkpoint loaded. Message: {load_msg}")
                if load_msg.missing_keys: print(f"  WARNING: Missing keys: {load_msg.missing_keys}")
                if load_msg.unexpected_keys: print(f"  WARNING: Unexpected keys: {load_msg.unexpected_keys}")
            
            # Update output tag for full fine-tune
            ckpt_name_part = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
            output_model_name_tag = f"{args.model_name_tag}_ft_{ckpt_name_part}"

        except Exception as e:
            if args.is_main_process: print(f"ERROR: Failed to load checkpoint {args.checkpoint_path}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr); sys.exit(1)
    else:
        if args.is_main_process: print("Evaluating base pre-trained model (no checkpoint or LoRA adapters specified).")
        # output_model_name_tag remains args.model_name_tag

    model.eval()
    model.to(device) # Ensure final model is on device

    # Bind the custom forward_lm method
    model.forward_lm = MethodType(forward_lm, model)
    # Ensure t5_tokenizer is available on the model instance for forward_lm
    # LAVIS Blip2T5 models usually have self.t5_tokenizer already.
    if not hasattr(model, 't5_tokenizer'):
        print("Manually setting t5_tokenizer on model instance from txt_processors.")
        # txt_processors is usually a dict, e.g., {'train': ..., 'eval': ...}
        # The 'eval' processor's tokenizer is typically what's needed.
        if "eval" in txt_processors and hasattr(txt_processors["eval"], 'tokenizer'):
            model.t5_tokenizer = txt_processors["eval"].tokenizer
        elif "train" in txt_processors and hasattr(txt_processors["train"], 'tokenizer'): # Fallback
             model.t5_tokenizer = txt_processors["train"].tokenizer
        else:
            print("ERROR: Could not find t5_tokenizer in txt_processors. forward_lm might fail.", file=sys.stderr)
            # Attempt to get it from the T5 model itself if possible (Hugging Face style)
            if hasattr(model, t5_model_attr_name or "") and hasattr(getattr(model, t5_model_attr_name or "", None), 'tokenizer'):
                model.t5_tokenizer = getattr(model, t5_model_attr_name).tokenizer
                print("Found tokenizer on t5_model component.")
            else:
                 print("FATAL: T5 Tokenizer not found!", file=sys.stderr)
                 sys.exit(1)


    print(f"Running evaluation for dataset: {args.dataset_path}")
    print(f"Results will be saved in directory: {args.answer_path} with model tag: {output_model_name_tag}")
    print(f"Using image root: {args.image_root}")
    
    test(model, 
         vis_processors, 
         dataset_path_str=args.dataset_path, 
         model_name_for_output=output_model_name_tag, 
         prompt_idx=args.prompt_idx, 
         save_path_dir_str=args.answer_path,
         image_root_str=args.image_root)

if __name__ == "__main__":
    args = parse_args()
    run(args)