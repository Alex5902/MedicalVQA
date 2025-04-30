import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm
from copy import deepcopy

import sys
from pathlib import Path
# Calculate the path to the directory containing the 'llava' package
# This assumes the script is in .../LLaVA-Med/llava/eval/
script_dir = Path(__file__).parent # .../LLaVA-Med/llava/eval
llava_dir = script_dir.parent    # .../LLaVA-Med/llava
llava_med_dir = llava_dir.parent # .../LLaVA-Med
# Add the directory *containing* the 'llava' package to sys.path
if str(llava_med_dir) not in sys.path:
    sys.path.insert(0, str(llava_med_dir))
    print(f"DEBUG: Added {llava_med_dir} to sys.path")

from llava import LlavaLlamaForCausalLM
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from PIL import Image
import math
from scipy.special import softmax
import numpy as np
from torch.nn import CrossEntropyLoss

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 os.pardir,  # …/eval
                 os.pardir,  # …/llava
                 os.pardir,  # …/LLaVA-Med
                 os.pardir,  # …/Prefix_based_Score
                 os.pardir   # …/MedicalEval
                 )
)

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
        "mm_hidden_size": 4096
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)

def resize_and_pad(img: Image.Image, target_size: int, pad_color=(0, 0, 0)) -> Image.Image:
    """Resizes image to fit within target_size while preserving aspect ratio, then pads."""
    original_width, original_height = img.size

    # Calculate new size preserving aspect ratio
    ratio = min(target_size / original_width, target_size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Resize using BILINEAR or BICUBIC. BICUBIC is often preferred for quality.
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

    # Create padded background
    new_img = Image.new("RGB", (target_size, target_size), pad_color)

    # Calculate paste position (center)
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2

    # Paste the resized image onto the center of the square background
    new_img.paste(img, (paste_x, paste_y))

    return new_img

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
    # Model
    # disable_torch_init()
    model_name = os.environ.get("MODEL_PATH", os.path.expanduser(args.model_name))
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, use_fast=False)
    if args.mm_projector is None:
        patch_config(model_name)
        
        print(model_name)
        
        # load the entire multi-modal model onto GPU in fp16
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_cache=True,
            local_files_only=True,
        )
        print(f"[DEBUG] model_name = {model_name}")

        expected_vision_tower_for_processor = "openai/clip-vit-large-patch14" # Verify this path is correct for 224x224
        print(f"INFO: Forcing image processor loading from: {expected_vision_tower_for_processor} (Model expects 224x224)")

        # HF knows which CLIP tower you need
        image_processor = CLIPImageProcessor.from_pretrained(expected_vision_tower_for_processor)

        # now pull the actual vision module back out for config
        vision_tower = model.model.vision_tower[0]

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    else:
        # in case of using a pretrained model with only a MLP projector weights
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, use_cache=True, local_files_only=True)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        print(f"INFO: Loading Vision Tower specified by args: {args.vision_tower}")
        if not args.vision_tower:
            raise ValueError("Vision tower path must be provided via --vision-tower")
        vision_tower = CLIPVisionModel.from_pretrained(args.vision_tower, torch_dtype=torch.float16).cuda()
        model.model.vision_tower = [vision_tower] # Assign loaded tower to model

        # Load image processor corresponding to the loaded vision tower
        print(f"INFO: Loading image processor corresponding to vision tower: {args.vision_tower}")
        image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower, torch_dtype=torch.float16)

        # Get config from the loaded vision tower (will have image_size=336)
        vision_config = vision_tower.config
        print(f"INFO: Vision Tower Config - Image Size: {vision_config.image_size}, Patch Size: {vision_config.patch_size}")

        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        # load the raw dict
        ckpt = torch.load(args.mm_projector, map_location="cpu")
        print("CHECKPOINT KEYS:", ckpt.keys())
        for k,v in ckpt.items():
            print(k, v.shape)

        import torch.nn as nn
        # Load the state dict from the provided projector path
        print(f"INFO: Loading MM Projector weights from: {args.mm_projector}")
        if not args.mm_projector or not os.path.exists(args.mm_projector):
            raise ValueError(f"MM Projector weights file not found or not specified: {args.mm_projector}")
        ckpt = torch.load(args.mm_projector, map_location="cpu")

        # Determine projector dimensions
        # Input dimension comes from the vision tower's hidden size
        # Output dimension matches the language model's hidden size
        vision_hidden_dim = vision_config.hidden_size
        llm_hidden_dim = model.config.hidden_size

        # Check if the checkpoint has the expected 2-layer keys
        if "model.mm_projector.0.weight" in ckpt and "model.mm_projector.2.weight" in ckpt:
            print("INFO: Detected 2-layer MLP projector weights in checkpoint.")
            # Define the 2-layer MLP structure (matches LLaVA v1.5 standard)
            # Linear -> GELU -> Linear
            mm_projector = nn.Sequential(
                nn.Linear(vision_hidden_dim, llm_hidden_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_hidden_dim, llm_hidden_dim, bias=True)
            )
            # Load the weights, renaming keys if necessary (from ckpt format to Sequential format)
            state_dict_to_load = {}
            for key, value in ckpt.items():
                if key.startswith("model.mm_projector."):
                    # Rename 'model.mm_projector.0.weight' -> '0.weight', etc.
                    new_key = key.replace("model.mm_projector.", "")
                    state_dict_to_load[new_key] = value
            mm_projector.load_state_dict(state_dict_to_load)
            print("INFO: Successfully loaded 2-layer MLP projector weights.")
        elif "weight" in ckpt: # Fallback for single layer? (Less likely for LLaVA v1.5)
            print("WARN: Detected single-layer projector weights. Loading as Linear.")
            weight = ckpt["weight"]
            bias = ckpt.get("bias", None)
            out_dim, in_dim = weight.shape
            if in_dim != vision_hidden_dim or out_dim != llm_hidden_dim:
                print(f"WARN: Projector dimensions mismatch! Ckpt: {in_dim} -> {out_dim}, Expected: {vision_hidden_dim} -> {llm_hidden_dim}")
            mm_projector = torch.nn.Linear(in_dim, out_dim)
            state = {"weight": weight}
            if bias is not None: state["bias"] = bias
            mm_projector.load_state_dict(state)
        else:
            raise ValueError(f"Could not find expected projector weights ('model.mm_projector.0.weight' or 'weight') in {args.mm_projector}")

        # Assign the loaded projector to the model
        model.model.mm_projector = mm_projector.cuda().half()

        model.model.vision_tower = [vision_tower]
###
    questions = json.load(open(os.path.expanduser(question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    answers_file = os.path.join(answers_base_path, 'tmp', os.path.dirname(question_file).replace('/', '_') + 'pred.jsonl')
    # answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # os.makedirs(os.path.join(os.path.dirname(answers_file), "images"), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    print('start inference...', flush=True)
    res = []
    cnt = 0
    tot = 0
    for i, line in enumerate(tqdm(questions)):
        try:
            tot += 1
            candidates = load_candidates_medical(line)
            question, gt_ans, raw_image = preprocess_input(line)
            prefix = load_prompt(question)
            prefix_tokens = tokenizer(prefix)
            start_loc = len(prefix_tokens.input_ids)
            candidate_scores = []  # pred scores of candidates

            # 1. Get target size from model config
            vision_cfg = model.model.vision_tower[0].config
            model_image_size = vision_cfg.image_size # e.g., 224

            # 2. Resize and pad the original PIL image
            padded_image = resize_and_pad(raw_image, model_image_size, pad_color=(0,0,0)) # Use black padding
            # print(f"DEBUG: Original PIL size: {raw_image.size}, Padded PIL size: {padded_image.size}")

            # 3. Pass the *padded square* PIL image to the processor for normalization/tensor conversion
            pixel_values = image_processor(
                padded_image,
                return_tensors="pt"
            ).pixel_values

            # pixel_values = image_processor(
            # raw_image, # Pass the original, non-square image
            # return_tensors='pt'
            # )['pixel_values']

            # 4. Verification (Shape should now consistently match model_image_size)
            # print(f"DEBUG: pixel_values shape after processor: {pixel_values.shape}")
            # if pixel_values.shape[-2:] != (model_image_size, model_image_size):
            #     raise ValueError(f"Processor output size {pixel_values.shape[-2:]} doesn't match model expected size {(model_image_size, model_image_size)}. Check padding or processor config.")

            # 5. Move to device and cast dtype
            images = pixel_values.to(next(model.parameters()).device, dtype=torch.float16)
            # print(f"DEBUG: Final images tensor shape: {images.shape}, dtype: {images.dtype}, device: {images.device}")

            candidate_losses = {}
            # Get original question text and map keys to full option text
            question_text = line['question'] # Raw question
            option_texts = { "A": line.get('option_A'), "B": line.get('option_B') }
            if line.get('option_C') is not None: option_texts["C"] = line['option_C']
            if line.get('option_D') is not None: option_texts["D"] = line['option_D']

            # Use the original script's prefix style
            prefix = f"Question: {question_text} The answer is" # Use f-string for clarity

            # Tokenize the prefix once to find its length accurately, including special tokens
            prefix_inputs = tokenizer(prefix, return_tensors="pt").to(images.device)
            # If tokenizer adds BOS, start_loc should account for it.
            # Assuming BOS is added, length is shape[1]. If no BOS, might need -1? Test this.
            start_loc = prefix_inputs.input_ids.shape[1]

            for candidate_key, candidate_full_text in option_texts.items():
                if candidate_full_text is None: continue

                # Construct prompt: Prefix + " " + OptionText + "."
                option_suffix = f" {candidate_full_text}." # Add space and period like original
                prompt = prefix + option_suffix

                # Tokenize the full prompt
                inputs = tokenizer(prompt, return_tensors="pt").to(images.device)
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask

                # Calculate length of the added suffix tokens
                # Tokenize the suffix *without* special tokens
                option_suffix_tokens = tokenizer(option_suffix, add_special_tokens=False).input_ids
                option_len = len(option_suffix_tokens)

                if option_len == 0:
                    print(f"WARN: Option suffix '{option_suffix}' tokenized to zero tokens. Skipping.")
                    candidate_losses[candidate_key] = float('inf')
                    continue

                # Prepare labels: mask everything except the option suffix tokens
                labels = input_ids.clone()
                labels[0, :start_loc] = -100 # Mask prefix part

                # Mask tokens *after* the option suffix
                end_mask_idx = min(start_loc + option_len, labels.shape[1])
                labels[0, end_mask_idx:] = -100

                # Sanity check
                if start_loc >= end_mask_idx or (labels[0, start_loc:end_mask_idx] == -100).all():
                     print(f"ERROR: Labeling issue for option {candidate_key}. All target labels masked.")
                     print(f"Input IDs shape: {input_ids.shape}, Start Loc: {start_loc}, Option Len: {option_len}, End Mask Idx: {end_mask_idx}")
                     candidate_losses[candidate_key] = float('inf')
                     continue

                # Perform forward pass (same as before)
                with torch.inference_mode():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=images,
                        labels=labels,
                        return_dict=True,
                        use_cache=False
                    )
                    loss = outputs.loss

                # Store loss (same as before)
                if loss is not None:
                    loss_item = loss.item()
                    if math.isnan(loss_item) or math.isinf(loss_item):
                         print(f"WARN: Encountered {loss_item} loss for option {candidate_key}. Assigning inf.")
                         candidate_losses[candidate_key] = float('inf')
                    else:
                        candidate_losses[candidate_key] = loss_item
                        print(f"DEBUG: Candidate: {candidate_key}, Loss: {loss_item:.4f}")
                else:
                    print(f"ERROR: Loss is None for candidate {candidate_key}. Check model output or label masking.")
                    candidate_losses[candidate_key] = float('inf')

            # --- End ADD --- - Continue below with prediction logic
            if not candidate_losses or all(l == float('inf') for l in candidate_losses.values()):
                predicted_answer_key = "ERROR"
                min_loss = float('inf')
                predicted_answer_text = "Error in loss calculation" # Keep this
                is_correct = False
                ori_candidate_losses_str = str(candidate_losses)
                print(f"WARN: Could not determine prediction for question {i} due to errors or infinite losses.")
            else:
                predicted_answer_key = min(candidate_losses, key=candidate_losses.get)
                min_loss = candidate_losses[predicted_answer_key]

                # --- CHANGE THIS LINE ---
                # Map the predicted key back to the actual answer text using the correct dictionary
                predicted_answer_text = option_texts[predicted_answer_key] # Use option_texts
                # --- END CHANGE ---

                ori_candidate_losses_str = str({k: f"{v:.4f}" if v != float('inf') else 'inf' for k, v in candidate_losses.items()})

                print(f"INFO: Losses: {ori_candidate_losses_str}")
                print(f"INFO: Predicted Key: {predicted_answer_key} (Min Loss: {min_loss:.4f}) -> Answer: {predicted_answer_text}")
                print(f"INFO: Ground Truth Answer: {gt_ans}")

                is_correct = (predicted_answer_text == gt_ans)
                if is_correct:
                    cnt += 1

            # Store results (Keep this block the same, but ensure prefix is just question_text if needed)
            save_dict = deepcopy(line)
            save_dict['model_pred_key'] = predicted_answer_key
            save_dict['model_pred'] = predicted_answer_text
            save_dict['prompt_question'] = question_text # Save the raw question as the 'prefix'
            save_dict['candidate_losses'] = ori_candidate_losses_str
            save_dict['is_correct'] = 'yes' if is_correct else 'no'

            ans_file.write(json.dumps(save_dict) + "\n")
            res.append(save_dict)
            ans_file.flush()
        except Exception as e:
            print(f"[{i}] skipped – {type(e).__name__}: {e}", flush=True)
            continue
    ans_file.close()
    
    final_save_dict = {
        "model_name": model_name,
        "dataset_name": question_file,
        "correct_percentage" : cnt / tot if tot > 0 else 0.0, # Use correct spelling and handle division by zero
        "pred_dict" : res
    }
    with open(os.path.join(answers_base_path, os.path.dirname(question_file).replace('/', '_') + '.json'), 'w') as f:
        json.dump(final_save_dict, f, indent=4, ensure_ascii=False)

    import pandas as pd

    df = pd.DataFrame(res)

    out_dir = os.path.abspath(answers_base_path)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{args.model}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV results to {csv_path}", flush=True)

def preprocess_input(entity) -> tuple:
    a,b,c,d = entity.get('option_A'), entity.get('option_B'), entity.get('option_C'), entity.get('option_D')
    answer_list = [a, b]
    if c is not None:
        answer_list.append(c)
    if d is not None:
        answer_list.append(d)
    q_str = entity['question'] + f'Here are {len(answer_list)} candidate answers:' + str(answer_list)+' Only return what you think is the correct answer from the candidate answers, do not return any other irrelevant text!'
    ans_str = entity['gt_answer']
    
    image_url = entity.get('image_path')
    # image = read_img_from_url(image_url)
    image = read_img_from_url(image_url, args.image_root)

    return q_str, ans_str, image
    

from PIL import Image
from io import BytesIO

# def read_img_from_url(url):
#     img = Image.open(url)
#     return img

def read_img_from_url(url, root):
    full_path = os.path.join(root, url)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")
    return Image.open(full_path).convert("RGB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=str, default="/home/alex.ia/OmniMed/OmniMedVQA/OmniMedVQA", help="Top-level folder that contains the Images/ subdir",)
    parser.add_argument("--model", type=str, default="llava-med")
    parser.add_argument("--model-name", type=str, default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--dataset-path", type=str, default="/path/to/dataset")
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
    