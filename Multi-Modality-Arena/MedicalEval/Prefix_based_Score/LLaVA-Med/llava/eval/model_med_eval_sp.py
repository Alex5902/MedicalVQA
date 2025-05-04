import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import torch.nn as nn
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, IGNORE_INDEX # Add constants
import sys
import os
import json
from tqdm import tqdm
from copy import deepcopy
from llava import LlavaLlamaForCausalLM
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from PIL import Image
import math
from scipy.special import softmax
import numpy as np
from torch.nn import CrossEntropyLoss


def load_candidates_medical(data):
    a,b,c,d = data.get('option_A'), data.get('option_B'), data.get('option_C'), data.get('option_D')
    answer_list = [a, b]
    if c is not None:
        answer_list.append(c)
    if d is not None:
        answer_list.append(d)
    return answer_list
    

def load_prompt(question, idx=4):
    prompts = ["{}".format(question),
               "{} Answer:".format(question),
               "{} The answer is".format(question),
               "Question: {} Answer:".format(question),
               "Question: {} The answer is".format(question)
               ]
    return prompts[idx]


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"




detail_describe_instructions = [
    "Describe the following image in detail.",
    "Provide a detailed description of the given image.",
    "Give an elaborate explanation of the image you see.",
    "Share a comprehensive rundown of the presented image.",
    "Offer a thorough analysis of the image.",
    "Explain the various aspects of the image before you.",
    "Clarify the contents of the displayed image with great detail.",
    "Characterize the image using a well-detailed description.",
    "Break down the elements of the image in a detailed manner.",
    "Walk through the important details of the image.",
    "Portray the image with a rich, descriptive narrative.",
    "Narrate the contents of the image with precision.",
    "Analyze the image in a comprehensive and detailed manner.",
    "Illustrate the image through a descriptive explanation.",
    "Examine the image closely and share its details.",
    "Write an exhaustive depiction of the given image.",
]

concise_describe_instructions = [
    "Describe the following image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the following image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo below.",
    "Write a terse but informative summary of the following picture.",
    "Create a compact narrative representing the image presented.",
]

prompt_pool = detail_describe_instructions + concise_describe_instructions

prompt_pool = [ "Describe the following image in detail."]


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def eval_model(args, question_file, answers_base_path): 
     # Model Setup
    disable_torch_init()
    model_path = os.path.expanduser(args.model_name)
    model_name = get_model_name_from_path(model_path)
    device = 'cuda' # Or handle device selection more robustly

    # Use the official LLaVA loader
    try:
        print(f"Loading pretrained model: {model_path}")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None, # Loading a full model, not LoRA
            model_name=model_name,
            device='cpu' # Directly specify device if load_pretrained_model supports it
            # load_4bit= getattr(args, 'load_4bit', False), # Pass quantization flags if you add them
            # load_8bit= getattr(args, 'load_8bit', False)
        )
        #         # --- >>> ADD SPECIAL TOKEN HANDLING <<< ---
        # # Ensure the image token is known and maps to IMAGE_TOKEN_INDEX (-200)
        # # Check if the token exists and what its ID is
        # current_image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        # print(f"Tokenizer initially maps '{DEFAULT_IMAGE_TOKEN}' to ID: {current_image_token_id}")

        # if current_image_token_id == tokenizer.unk_token_id or current_image_token_id != IMAGE_TOKEN_INDEX:
        #     print(f"WARNING: Tokenizer did not correctly map '{DEFAULT_IMAGE_TOKEN}'. Adding/forcing mapping.")
        #     # Add the token if it doesn't exist
        #     num_added = tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
        #     print(f"Added {num_added} new token(s). New tokenizer size: {len(tokenizer)}")
        #     # Explicitly set the ID for the image token in the tokenizer's internal mapping if possible
        #     # (This part is less standard and might depend on tokenizer implementation,
        #     # but adding it as a special token *should* often be sufficient if LLaVA's
        #     # internal logic looks for IMAGE_TOKEN_INDEX directly)

        #     # Resize embeddings if new tokens were added
        #     if num_added > 0:
        #          model.resize_token_embeddings(len(tokenizer))
        #          # Optionally initialize new embeddings
        #          # input_embeddings = model.get_input_embeddings().weight.data
        #          # output_embeddings = model.get_output_embeddings().weight.data
        #          # input_embeddings_avg = input_embeddings[:-num_added].mean(dim=0, keepdim=True)
        #          # output_embeddings_avg = output_embeddings[:-num_added].mean(dim=0, keepdim=True)
        #          # input_embeddings[-num_added:] = input_embeddings_avg
        #          # if model.get_output_embeddings() is not None: # Check if output embeddings exist
        #          #     output_embeddings[-num_added:] = output_embeddings_avg


        #     # Verify the mapping again
        #     final_image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        #     print(f"Tokenizer finally maps '{DEFAULT_IMAGE_TOKEN}' to ID: {final_image_token_id}")
        #     # Note: Even after adding, convert_tokens_to_ids might not return -200 directly.
        #     # The key is that the model's internal logic in prepare_inputs... uses IMAGE_TOKEN_INDEX == -200.
        #     # Adding the token ensures it's not UNK, and LLaVA should handle the rest.

        # # --- End Special Token Handling ---

        # If load_pretrained_model doesn't handle device, uncomment below:
        # model.to(device)
        model.to('cuda')
        print("Model moved to GPU.")
        print("Model, Tokenizer, Image Processor loaded successfully.")
        print("--- Model Configuration ---")
        print(model.config)
        print(f"Has vision tower: {hasattr(model, 'get_vision_tower')}")
        if hasattr(model, 'get_vision_tower'):
            try:
                print(f"Vision Tower object: {model.get_vision_tower()}")
            except: pass # Handle potential errors
        print(f"Has mm_projector: {hasattr(model.model, 'mm_projector')}")
        if hasattr(model.model, 'mm_projector'):
            print(f"Projector type: {type(model.model.mm_projector)}")
        print("-------------------------")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        # Exit or re-raise if model loading fails catastrophically
        sys.exit(f"Failed to load model {model_path}. Exiting.")


    # We might still need image_token_len for prompt construction later
    try:
        # Need to get vision_config after loading
        vision_config = model.get_vision_tower().config
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
        print(f"Image token length calculated: {image_token_len}")
    except Exception as e:
        print(f"Warning: Could not determine image_token_len from loaded model. Error: {e}")
        # Set a default or handle cases where it might not be needed depending on prompt logic
        image_token_len = 256 # Default fallback if needed, adjust as necessary

    questions = json.load(open(os.path.expanduser(args.dataset_path), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    answers_file = os.path.join(answers_base_path, 'tmp', os.path.dirname(args.dataset_path).replace('/', '_') + 'pred.jsonl')
    # answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # os.makedirs(os.path.join(os.path.dirname(answers_file), "images"), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    print('start inference...', flush=True)
    res = []
    cnt = 0
    tot = 0
    skipped_items = 0
    for i, line in enumerate(tqdm(questions)):
        try:
            # tot += 1
            candidates = load_candidates_medical(line)
            question_text, gt_ans, raw_image = preprocess_input(line, args)
            if raw_image is None: # Skip if image failed to load
                print(f"WARNING: Skipping item {i} (ID: {line.get('id', 'N/A')}) due to image loading failure.", file=sys.stderr)
                skipped_items += 1
                continue
            tot += 1

            base_prefix_text = load_prompt(question_text)
            prefix_with_image_token = DEFAULT_IMAGE_TOKEN + "\n" + base_prefix_text
            # prefix_tokens = tokenizer(prefix)
            prefix_combined_tokens = tokenizer(prefix_with_image_token)
            # start_loc = len(prefix_tokens.input_ids)
            start_loc_for_loss = len(prefix_combined_tokens.input_ids)
            candidate_scores = []  # pred scores of candidates

            images = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values']
            images_check = images.cpu().float() # Move to CPU, ensure float32 for stats
            print(f"DEBUG Image Tensor Stats: Shape={images_check.shape}, Mean={images_check.mean().item():.4f}, Std={images_check.std().item():.4f}, Min={images_check.min().item():.4f}, Max={images_check.max().item():.4f}")
            # Put back on GPU if needed for model call
            images = images.half().cuda()

            # for candidate in candidates:
            for candidate_idx, candidate in enumerate(candidates):
                prompt = prefix_with_image_token + " {}.".format(candidate)
                # input_ids = tokenizer(prompt).input_ids
                # input_ids = torch.as_tensor([input_ids]).cuda()
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

                # --- >>> Manually Replace Token ID with IMAGE_TOKEN_INDEX (-200) <<< ---
                # Expected position of the token corresponding to <image> is index 1 (after BOS)
                expected_image_token_index_pos = 1
                if input_ids.shape[1] > expected_image_token_index_pos:
                    original_id_at_pos1 = input_ids[0, expected_image_token_index_pos].item()
                    input_ids[0, expected_image_token_index_pos] = IMAGE_TOKEN_INDEX # Force -200
                    print(f"DEBUG Item {i}: Replaced token ID {original_id_at_pos1} with {IMAGE_TOKEN_INDEX} at index {expected_image_token_index_pos}", flush=True)
                else:
                    print(f"WARNING Item {i}: input_ids too short for image token replacement!", flush=True)
                # --- End Manual Replacement ---
                
                targets = input_ids.clone()
                # Mask the prefix part up to where the candidate answer starts
                # Use IGNORE_INDEX (-100)
                targets[0, :start_loc_for_loss] = IGNORE_INDEX
                # prompt = prefix + " {}.".format(candidate)
                # prompt_tokens =tokenizer(prompt, return_tensors="pt")
                # lang_t = prompt_tokens["input_ids"]
                
                # prefix_tokens = tokenizer(prefix, return_tensors="pt")  
                # lang_t1 = prefix_tokens["input_ids"]
                # lang_diff = lang_t.shape[1] - lang_t1.shape[1]

                if candidate_idx == 0: # Print only for the first candidate per item
                    print(f"\nDEBUG Item {i} Candidate 0:", flush=True)
                    print(f"  IMAGE_TOKEN_INDEX = {IMAGE_TOKEN_INDEX}", flush=True)
                    print(f"  input_ids shape: {input_ids.shape}", flush=True)
                    print(f"  input_ids values: {input_ids[0].tolist()}", flush=True) # Show the actual token IDs
                    print(f"  Does input_ids contain IMAGE_TOKEN_INDEX? {(input_ids == IMAGE_TOKEN_INDEX).any().item()}", flush=True)
                
                with torch.inference_mode():
                    # --- >>> FIX 2: Pass 'targets' as the 'labels' argument <<< ---
                    outputs = model(
                        input_ids=input_ids,    # Contains -200
                        labels=targets,         # Pass targets (prefix masked) here
                        use_cache=False,
                        images=images,
                        return_dict=True
                    )

                # --- >>> FIX 3: Get loss DIRECTLY from model output <<< ---
                loss = outputs.loss # This loss is calculated internally by the model

                # --- >>> FIX 4: REMOVE the entire external loss calculation block <<< ---
                # REMOVE {
                # logits = outputs.logits
                # targets = input_ids.clone()
                # targets[0, :start_loc_for_loss] = IGNORE_INDEX
                # shift_logits = logits[..., :-1, :].contiguous()
                # shift_labels = targets[..., 1:].contiguous()
                # shift_logits = shift_logits.view(-1, model.config.vocab_size)
                # shift_labels = shift_labels.view(-1)
                # active_loss = shift_labels != IGNORE_INDEX
                # active_logits = shift_logits[active_loss]
                # active_labels = shift_labels[active_loss]
                # if active_labels.numel() > 0:
                #      loss_fct = CrossEntropyLoss()
                #      loss = loss_fct(active_logits, active_labels)
                # else:
                #      loss = torch.tensor(float('inf'), device=device)
                # } REMOVE

                # --- Check loss and append ---
                if loss is None:
                    print(f"WARNING: Model returned None loss item {i} cand '{candidate}'.", flush=True)
                    candidate_scores.append(float('inf'))
                elif torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: Loss NaN/Inf item {i} cand '{candidate}'.", flush=True)
                    candidate_scores.append(float('inf'))
                else:
                    candidate_scores.append(loss.item())

            ori_candidate = candidate_scores # Assign original scores list

            if not ori_candidate or all(s == float('inf') for s in ori_candidate):
                 print(f"WARNING: No valid scores for item {i}. Assigning ERROR prediction.", flush=True)
                 pred = "ERROR: No valid scores"
            else:
                # Filter valid scores and indices for softmax/argmax
                valid_scores = np.array([s for s in ori_candidate if s != float('inf')])
                valid_indices = [idx for idx, s in enumerate(ori_candidate) if s != float('inf')]

                if len(valid_scores) > 0:
                    # Lower loss is better, so use reciprocal before softmax
                    probabilities = softmax(np.reciprocal(valid_scores))
                    best_valid_idx = np.argmax(probabilities)
                    # Map back to original candidate list index
                    pred_idx = valid_indices[best_valid_idx]
                    pred = candidates[pred_idx]
                else: # Should not be reachable if first check passes
                    pred = "ERROR: Filtering failed"

            print(f"Candidates: {candidates}, Scores: {ori_candidate}, Pred: {pred}", flush=True)

            # --- Save results ---
            save_dict = deepcopy(line)
            save_dict['model_pred'] = pred
            save_dict['prompt_question'] = base_prefix_text # Save base text prefix
            save_dict['confidence'] = str(ori_candidate) # Save original losses
            save_dict['is_correct'] = 'yes' if pred == gt_ans else 'no'
            if pred == gt_ans:
                cnt += 1

            ans_file.write(json.dumps(save_dict) + "\n")
            res.append(save_dict)
            ans_file.flush()
        except Exception as e: 
            print(f"\nERROR processing item {i} (ID: {line.get('id', 'N/A')}) in Prefix Score")
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
            print("Skipping this item due to error...")
            continue

    ans_file.close()
    
    final_save_dict = {
        "model_name": model_name,
        "dataset_name": args.dataset_path,
        "correct_precentage" :cnt / tot,
        "pred_dict" : res
    }
    with open(os.path.join(answers_base_path, os.path.dirname(args.dataset_path).replace('/', '_') + '.json'), 'w') as f:
        json.dump(final_save_dict, f, indent=4, ensure_ascii=False)


def preprocess_input(entity, args) -> tuple:
    a,b,c,d = entity.get('option_A'), entity.get('option_B'), entity.get('option_C'), entity.get('option_D')
    answer_list = [a, b]
    if c is not None:
        answer_list.append(c)
    if d is not None:
        answer_list.append(d)
    q_str = entity['question'] + f'Here are {len(answer_list)} candidate answers:' + str(answer_list)+' Only return what you think is the correct answer from the candidate answers, do not return any other irrelevant text!'
    ans_str = entity['gt_answer']
    
    image_file = entity.get('image_path')
    if args.image_root and image_file: # Check if image_root is provided and path exists
        image_full_path = os.path.join(args.image_root, image_file)
    else:
        # Assume image_path is absolute or relative to CWD if image_root not given
        image_full_path = image_file

    if not image_full_path or not os.path.exists(image_full_path):
         print(f"ERROR: Image path not found or invalid: {image_full_path} (Original: {image_file})", file=sys.stderr)
         # Decide how to handle: skip, raise error? For now, let read_img_from_url handle it potentially.
         # Or explicitly return None/raise error here if needed.
         # Let's try to open it anyway, maybe it's a URL or special path handled later.
         pass # Let the existing error handling catch it later if truly invalid

    image = read_img_from_url(image_full_path) # Pass the constructed path
    return q_str, ans_str, image
    

from PIL import Image
from io import BytesIO

def read_img_from_url(url):
    img = Image.open(url)
    return img
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="VLP_web_data/llava_med/llava_med_in_text_60k")
    parser.add_argument("--dataset-path", type=str, default="/path/to/dataset")
    parser.add_argument("--image-root", type=str, default=None, help="Root directory for image paths")
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple_legacy")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answer-prompter", default=True, action="store_true")
    parser.add_argument('--answers-base-path', type=str, default="llava_med_output")
    args = parser.parse_args()

    print('start', flush=True)
    
    os.makedirs(args.answers_base_path, exist_ok=True)
    eval_model(args, args.dataset_path, args.answers_base_path)
                
    print('finish', flush=True)
    