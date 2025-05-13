#!/usr/bin/env python3
import os
import json
# import pandas # Not used, can be removed
import random # Not used, can be removed
# import pickle # Not used, can be removed
import numpy as np
from tqdm import tqdm # Not used in test(), but could be if data_all is large
from PIL import Image
from scipy.special import softmax
# import requests # Not used, can be removed
from lavis.models import load_model_and_preprocess
import torch
# import pdb # Not used, can be removed
from types import MethodType
# from PIL import Image # Duplicate import
from io import BytesIO # Not used if images are loaded from path
import argparse
import sys # For sys.exit

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def forward_lm(self, samples):
    image = samples["image"]
    # Ensure model is on the correct device, though it should be already
    # self.to(image.device) # Usually done once after loading

    with self.maybe_autocast(): # LAVIS's internal autocast
        image_embeds = self.ln_vision(self.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = self.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )

    inputs_t5 = self.t5_proj(query_output.last_hidden_state)
    atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

    # Use global autocast for T5 part if desired, or rely on LAVIS's maybe_autocast
    # Using bfloat16 as in your training script for consistency if supported
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and image.device.type == 'cuda' else torch.float16
    
    # For prefix scoring, we construct the full prompt and then get the loss for the "text_output" part
    # The original forward_lm was designed for training-like loss calculation.
    # For prefix scoring, we want the perplexity/loss of the candidate completion.
    
    # We need to combine prefix (text_input) and candidate (text_output) for the T5 model's input
    # and then calculate the loss only over the candidate part.
    
    full_input_text = [inp + outp for inp, outp in zip(samples["text_input"], samples["text_output"])]
    prefix_lengths = [len(self.t5_tokenizer.encode(inp)) for inp in samples["text_input"]]

    with torch.cuda.amp.autocast(enabled=(image.device.type == 'cuda'), dtype=amp_dtype): # Global AMP for T5
        input_tokens = self.t5_tokenizer(
            full_input_text, # Tokenize the combined prefix + candidate
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len, # Ensure this is appropriate
            return_tensors="pt",
        ).to(image.device)

        # Create labels: mask out the prefix part, only calculate loss on the candidate part
        targets = input_tokens.input_ids.clone()
        for i in range(targets.size(0)): # Iterate over batch
            # Mask prefix tokens (excluding the special start token for the target if any, but here it's direct continuation)
            # The prefix_lengths here are for the *tokenized* prefix.
            # We need to be careful if the tokenizer adds special tokens.
            # A simpler way for prefix scoring is to get logits and manually calculate cross-entropy for target tokens.
            # However, to reuse the model's loss calculation, we mask.
            # Let's assume prefix_lengths is the number of tokens in the *prefix only*.
            # The 'start_loc' passed in samples is for the original tokenization of prefix.
            # This part needs to be robust.
            
            # The original `forward_lm` in BLIP2T5 expects separate input and output for labels.
            # Let's adapt to that structure for consistency with how BLIP-2 calculates loss.
            
            # Tokenize prefix (text_input)
            prefix_tokenized = self.t5_tokenizer(
                samples["text_input"],
                padding="longest", # Should already be handled if batching, but good for single
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

        targets = candidate_tokenized.input_ids.masked_fill(
            candidate_tokenized.input_ids == self.t5_tokenizer.pad_token_id, -100
        )

        inputs_embeds = self.t5_model.encoder.embed_tokens(prefix_tokenized.input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
        
        # Decoder inputs should start with pad_token_id for T5 generation loss style
        decoder_input_ids = self.t5_model._shift_right(candidate_tokenized.input_ids)

        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=candidate_tokenized.attention_mask,
            return_dict=True,
            labels=targets, # Calculate loss against the candidate tokens
        )
        loss = outputs.loss
        # The loss is already an average over the sequence.
        # For perplexity-like scoring, this average negative log likelihood is fine.
    return {"loss": loss}


def load_candidates_medical(data):
    a,b,c,d = data.get('option_A'), data.get('option_B'), data.get('option_C'), data.get('option_D')
    answer_list = [a, b]
    if c is not None:
        answer_list.append(c)
    if d is not None:
        answer_list.append(d)
    # Ensure candidates are strings
    return [str(cand) if cand is not None else "" for cand in answer_list]


def load_prompt(question, idx=4): # Defaulting to prompt_idx 4 from original script
    prompts = ["{}".format(question),
               "{} Answer:".format(question),
               "{} The answer is".format(question),
               "Question: {} Answer:".format(question),
               "Question: {} The answer is".format(question)
               ]
    return prompts[idx]

# bytes2PIL is not used if loading images from path
# def bytes2PIL(bytes_img):
#     '''Transform bytes image to PIL.
#     Args:
#         bytes_img: Bytes image.
#     '''
#     pil_img = Image.open(BytesIO(bytes_img)).convert("RGB")
#     return pil_img
 
@torch.no_grad()
def test(model, vis_processors, dataset_path_str=None, model_name_for_output='blip2', prompt_idx=4, save_path_dir_str='', image_root_str=''):
    # dataset_path_str is the full path to the JSON file for the current modality
    data_all = json.load(open(dataset_path_str))
    cnt = 0
    correct = 0
    
    res = []
    
    # Wrap data_all with tqdm if you want a progress bar for items
    # for data in tqdm(data_all, desc=f"Evaluating {os.path.basename(dataset_path_str)}"):
    for data_idx, data in enumerate(data_all):
        cnt += 1
        question = data['question']
        candidates = load_candidates_medical(data)
        answer = str(data['gt_answer']) # Ensure ground truth is also string
        img_path_relative = data['image_path']
        img_path_full = os.path.join(image_root_str, img_path_relative)
        
        if not os.path.exists(img_path_full):
            print(f"WARNING: Image file not found: {img_path_full} for item {data.get('question_id', data_idx)}. Skipping item.")
            # Add a placeholder to res or skip, ensure consistent output structure
            data['model_pred'] = "IMAGE_NOT_FOUND"
            data['confidence'] = "[]"
            data['is_correct'] = 'no'
            res.append(data)
            continue
        
        prefix_prompt = load_prompt(question, prompt_idx)
        # The 'start_loc' logic from original script might not be directly needed if forward_lm is adapted well.
        # prefix_tokens = model.t5_tokenizer(prefix_prompt, return_tensors="pt", truncation=True, max_length=512)
        # start_loc = prefix_tokens.input_ids.size(1) 
        
        candidate_losses = []
        try:
            raw_image = Image.open(img_path_full).convert("RGB")
            image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        except Exception as e:
            print(f"ERROR: Failed to load or process image {img_path_full} for item {data.get('question_id', data_idx)}: {e}. Skipping item.")
            data['model_pred'] = "IMAGE_LOAD_ERROR"
            data['confidence'] = "[]"
            data['is_correct'] = 'no'
            res.append(data)
            continue
            
        for candidate_text in candidates:
            # The forward_lm function expects "text_input" (prefix) and "text_output" (candidate)
            # The loss returned will be the cross-entropy loss for generating candidate_text given the image and prefix_prompt.
            # Lower loss means higher probability.
            samples = {
                "image": image_tensor,
                "text_input": [prefix_prompt], # Batch of 1
                "text_output": [str(candidate_text) if candidate_text is not None else ""], # Batch of 1, ensure string
                # "start_loc": start_loc # Not directly used by the revised forward_lm
            }
            try:
                outputs = model.forward_lm(samples)
                loss = outputs["loss"]
                candidate_losses.append(loss.item())
            except Exception as e:
                print(f"ERROR during model forward for item {data.get('question_id', data_idx)}, candidate '{candidate_text}': {e}")
                traceback.print_exc(file=sys.stderr) # Print full traceback for debugging
                candidate_losses.append(float('inf')) # Assign a very high loss for failed candidates


        data['confidence_scores_neg_log_likelihood'] = str(candidate_losses) # Store raw losses
        
        if any(l == float('inf') for l in candidate_losses) or not candidate_losses:
            pred = "MODEL_FORWARD_ERROR" # Or handle as per requirement
        else:
            # Lower loss is better. To use argmax for probability, take reciprocal then softmax.
            # Or directly use argmin on the losses.
            pred_idx = np.argmin(candidate_losses)
            pred = candidates[pred_idx]
            
            # If you want probability-like scores from losses:
            # Handle potential inf values before reciprocal if any error occurred
            valid_losses = [l if l != float('inf') else 1e9 for l in candidate_losses] # Replace inf with large number
            if not all(l == 1e9 for l in valid_losses): # Check if all were errors
                 probabilities = softmax(np.reciprocal(valid_losses)) # Higher prob for lower loss
                 data['probabilities_from_loss'] = str(probabilities.tolist())


        # print(f"Item: {data.get('question_id', data_idx)}, Q: {question[:30]}..., Candidates: {candidates}, Losses: {candidate_losses}, Pred: {pred}, GT: {answer}")

        data['model_pred'] = pred
        data['is_correct'] = 'yes' if pred == answer else 'no'
        if pred == answer:
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
    
    # Construct output filename based on the input dataset filename part
    base_dataset_filename = os.path.basename(dataset_path_str)
    if base_dataset_filename.endswith('.json'):
        base_dataset_filename = base_dataset_filename[:-5]
    
    output_json_filename = f"{base_dataset_filename}_prefix_results.json" # Consistent with orchestrator expectation
    
    full_save_path = os.path.join(save_path_dir_str, output_json_filename)
    
    with open(full_save_path, 'w') as f:
        json.dump(final_res, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {full_save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="BLIP-2 Prefix Scoring Evaluation")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSON dataset file for a specific modality")
    parser.add_argument("--answer_path", type=str, required=True, help="Directory to save output JSON results")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory containing the 'Images' folder (e.g., path to OmniMedVQA/OmniMedVQA/)")
    parser.add_argument("--model_name_tag", type=str, default="blip2_t5_pretrain_flant5xl", help="Base model name for output (e.g., blip2_t5_flant5xl or blip2_t5_flant5xl_finetuned_fundus)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the fine-tuned model checkpoint (.pth file), if evaluating a fine-tuned model.")
    # Add arguments for base model name and type if you want to make them configurable
    # parser.add_argument("--base_model_name", type=str, default="blip2_t5")
    # parser.add_argument("--base_model_type", type=str, default="pretrain_flant5xl")
    parser.add_argument("--prompt_idx", type=int, default=4, help="Index of the prompt to use from load_prompt function.")

    args = parser.parse_args()
    return args

def run(args):
    # These are the base architecture details
    base_model_name = "blip2_t5" 
    base_model_type = "pretrain_flant5xl"
    
    print(f"Loading base model architecture: {base_model_name}, type: {base_model_type}")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=base_model_name, 
        model_type=base_model_type, 
        is_eval=True, 
        device=device
    )

    # Store the original max_txt_len from the loaded txt_processors, if available
    # Or use a sensible default. The T5 tokenizer itself doesn't store it this way.
    # The model.max_txt_len is usually for the QFormer part.
    # For T5, it's often controlled by tokenizer's max_length_single_sentence.
    # Let's assume a reasonable max length for inputs to T5.
    # model.max_txt_len = getattr(txt_processors.get("eval", None), 'max_seq_length', 512) # Example
    model.max_txt_len = 512 # A common max length for T5 inputs during generation/scoring

    # If a fine-tuned checkpoint is provided, load its weights
    output_model_name_tag = args.model_name_tag # Default to argument
    if args.checkpoint_path:
        print(f"Loading fine-tuned weights from: {args.checkpoint_path}")
        if not os.path.exists(args.checkpoint_path):
            print(f"ERROR: Checkpoint file not found at {args.checkpoint_path}", file=sys.stderr)
            sys.exit(1)
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
            # Determine if the checkpoint is the state_dict itself or a dict containing it
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded fine-tuned checkpoint. Load message: {msg}")
            if msg.missing_keys:
                print(f"WARNING: Missing keys: {msg.missing_keys}")
            if msg.unexpected_keys:
                print(f"WARNING: Unexpected keys: {msg.unexpected_keys}")
            # Update the model name tag for output if it's the default and a checkpoint is loaded
            if args.model_name_tag == "blip2_t5_pretrain_flant5xl" and args.checkpoint_path:
                ckpt_name_part = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
                output_model_name_tag = f"blip2_t5_ft_{ckpt_name_part}"

        except Exception as e:
            print(f"ERROR: Failed to load checkpoint {args.checkpoint_path}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
    else:
        print("No fine-tuned checkpoint provided. Evaluating the base pre-trained model.")

    model.eval() # Ensure model is in evaluation mode
    # The model should already be on `device` from load_model_and_preprocess

    # The original script bound forward_lm to the model instance.
    # Ensure your model instance has this method.
    # If you're using the standard Blip2T5 from LAVIS, it should have a `forward_lm`
    # or a similar method for calculating loss for generation tasks.
    # The provided `forward_lm` function seems to be a custom one or one from an older LAVIS version.
    # Let's bind your custom `forward_lm` to the loaded model instance.
    model.forward_lm = MethodType(forward_lm, model)
    # Also ensure the model has access to its t5_tokenizer, which it should by default
    # model.t5_tokenizer = txt_processors["eval"].tokenizer # Or however it's accessed

    # The save path for results is now the directory passed by --answer-path
    # The test() function will construct the specific output filename inside this directory.
    # No need to create model_type subdir here if orchestrator handles it.
    # answer_path_dir = args.answer_path
    # os.makedirs(answer_path_dir, exist_ok=True) # Ensure orchestrator creates the final dir

    print(f"Running evaluation for dataset: {args.dataset_path}")
    print(f"Results will be saved in directory: {args.answer_path}")
    print(f"Using image root: {args.image_root}")
    
    test(model, 
         vis_processors, 
         dataset_path_str=args.dataset_path, 
         model_name_for_output=output_model_name_tag, 
         prompt_idx=args.prompt_idx, 
         save_path_dir_str=args.answer_path, # Pass the directory
         image_root_str=args.image_root)

if __name__ == "__main__":
    args = parse_args()
    # The orchestrator script (run_blip2_modality.py) should ensure
    # that args.answer_path is the final directory for this specific modality.
    # Example: ./results/blip2-flant5xl-ft-fundus_real_prefix/fundus_photography/
    # So, medical_blip2_finetuned.py just writes its output JSON into that directory.
    run(args)