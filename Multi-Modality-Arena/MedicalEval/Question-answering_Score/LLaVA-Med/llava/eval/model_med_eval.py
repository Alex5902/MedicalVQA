import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import random
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
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
import difflib
from PIL import Image
import math

def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

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

def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    
    # Return the index of the most similar string
    return most_similar_index


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
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.mm_projector is None:
        patch_config(model_name)
        
        print(model_name)
        if "BiomedCLIP" in model_name or "biomed_clip" in model_name:
            model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True, device_map=None)
            model = model.to_empty(device='cuda').to(dtype=torch.float16)
            # model = model.to(torch.float16)
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
            
            openai_vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
            vision_config = openai_vision_tower.config
            vision_tower = model.model.vision_tower[0]
            vision_tower = vision_tower.to_empty(device='cuda').to(dtype=torch.float16)
            setattr(vision_tower, 'config', vision_config)
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True, device_map=None)
            model = model.to_empty(device='cuda').to(dtype=torch.float16)
            # Explicitly load the processor for the expected 224x224 size
            expected_vision_tower_for_processor = "openai/clip-vit-large-patch14" # Verify this path is correct for 224x224
            print(f"INFO: Forcing image processor loading from: {expected_vision_tower_for_processor} (Model expects 224x224)")
            image_processor = CLIPImageProcessor.from_pretrained(expected_vision_tower_for_processor, torch_dtype=torch.float16) # <-- REPLACED LINE
            vision_tower = model.model.vision_tower[0]
            vision_tower = vision_tower.to_empty(device='cuda').to(dtype=torch.float16)
            

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        model.resize_token_embeddings(len(tokenizer))

        # import pdb; pdb.set_trace()
        vision_config = vision_tower.config
        image_size  = vision_config.image_size       # e.g. 224
        patch_size  = vision_config.patch_size       # e.g. 14
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end

        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        # image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
        
    else:
        # in case of using a pretrained model with only a MLP projector weights
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True, device_map=None)
        model = model.to_empty(device='cuda').to(dtype=torch.float16)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        model.resize_token_embeddings(len(tokenizer))
        # Load vision tower specified by args (should be the ...-336 version)
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
        image_size  = vision_config.image_size       # e.g. 224
        patch_size  = vision_config.patch_size       # e.g. 14
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        import torch.nn as nn

        mm_hidden = model.config.hidden_size  # e.g. 4096
        mm_projector = nn.Sequential(
            nn.Linear(vision_config.hidden_size,   mm_hidden),
            nn.GELU(),
            nn.Linear(        mm_hidden,           mm_hidden),
        )

        ckpt = torch.load(args.mm_projector, map_location='cpu')
        new_state = { k.replace("model.mm_projector.", ""): v
            for k,v in ckpt.items() }
        # now load into your Sequential
        mm_projector.load_state_dict(new_state)

        model.model.mm_projector = mm_projector.cuda().half()
        model.model.vision_tower = [vision_tower]

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2


    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    # import pdb; pdb.set_trace()
    questions = json.load(open(os.path.expanduser(question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    answers_file = os.path.join(answers_base_path, 'tmp', os.path.dirname(question_file).replace('/', '_') + 'pred.jsonl')
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    print('start inference...', flush=True)
    res = []
    print(f"Loaded {len(questions)} questions from {question_file}")
    for i, line in enumerate(tqdm(questions)):
        try:
            print(f"Processing sample {i}: {line}")
            # Use new preprocess_input, pass args.image_root explicitly
            prompt_text_for_llava, gt_answer_text, raw_image = preprocess_input(line, args.image_root)

            # Store the original question for saving later
            original_question = line['question']

            # --- Image Processing (Keep the padding logic) ---
            vision_cfg = model.model.vision_tower[0].config
            model_image_size = vision_cfg.image_size
            patch_size = vision_cfg.patch_size

            padded_image = resize_and_pad(raw_image, model_image_size, pad_color=(0,0,0))
            print(f"DEBUG: Original PIL size: {raw_image.size}, Padded PIL size: {padded_image.size}")

            pixel_values = image_processor(
                padded_image,
                return_tensors="pt"
            ).pixel_values

            print(f"DEBUG: pixel_values shape after processor: {pixel_values.shape}")
            if pixel_values.shape[-2:] != (model_image_size, model_image_size):
                raise ValueError(f"Processor output size {pixel_values.shape[-2:]} doesn't match model expected size {(model_image_size, model_image_size)}. Check padding or processor config.")

            images = pixel_values.to(next(model.parameters()).device, dtype=torch.float16)
            print(f"DEBUG: Final images tensor shape: {images.shape}, dtype: {images.dtype}, device: {images.device}")
            # --- End Image Processing ---

            # --- Prepare Prompt for SINGLE Generation ---
            # Compute patch token length
            _, _, H, W = pixel_values.shape
            image_token_len = (H // patch_size) * (W // patch_size)

            # Splice patch tokens into the prompt text
            # Add the image token placeholder first, then replace it
            prompt_with_placeholder = prompt_text_for_llava + "\n" + DEFAULT_IMAGE_TOKEN # Add placeholder

            if getattr(model.config, "mm_use_im_start_end", False):
                qs_with_patch_tokens = prompt_with_placeholder.replace(
                    DEFAULT_IMAGE_TOKEN,
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
                )
            else:
                 qs_with_patch_tokens = prompt_with_placeholder.replace(
                    DEFAULT_IMAGE_TOKEN,
                    DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                 )

            # Prepare conversation template
            # Use the selected conv_mode (e.g., 'simple_legacy', 'v1', etc.)
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs_with_patch_tokens) # Add user message (prompt + patches)
            conv.append_message(conv.roles[1], None) # Add assistant role marker to prompt generation
            prompt_for_model = conv.get_prompt()

            # Tokenize for generation
            input_ids = tokenizer([prompt_for_model], return_tensors="pt").input_ids.cuda()

            # --- SINGLE Generate Call ---
            # Use stop words appropriate for the conversation template
            stop_str = conv.sep if conv.sep_style != conv_templates['simple_legacy'].sep_style else None
            keywords = [stop_str] if stop_str is not None else []
            # Add "###" as a keyword if using the template separator, but be cautious
            # keywords.append("###") # Uncomment cautiously if needed and separator is ###

            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids) if keywords else None

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True, # Or False for greedy decoding
                    temperature=0.7, # Adjust as needed
                    # max_new_tokens=128, # Reduced token limit
                    max_new_tokens=64, # Even shorter if answers are typically brief
                    use_cache=True, # Caching helps generation speed
                    stopping_criteria=[stopping_criteria] if stopping_criteria else None) # Pass list

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')

            # Decode the generated tokens
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

            # --- Clean up the generated output ---
            # Remove potential stop words/separators from the end
            if stop_str and outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)].strip()
            # Add specific keywords if needed
            # for keyword in keywords:
            #      if outputs.endswith(keyword):
            #           outputs = outputs[:-len(keyword)].strip()

            # Remove prefixes added by some conversation templates if necessary
            # This depends heavily on the specific conv_mode template
            # Example for simple_legacy might involve removing "Response:" if present
            if args.conv_mode == 'simple_legacy':
                 if outputs.lower().startswith('response:'):
                      outputs = outputs[len('response:'):].strip()
                 elif outputs.lower().startswith('assistant:'): # Check common patterns
                      outputs = outputs[len('assistant:'):].strip()

            outputs = outputs.strip() # Final strip

            # --- SECOND GENERATE CALL IS REMOVED ---

            # --- Save results ---
            save_dict = deepcopy(line)
            save_dict['model_pred'] = outputs # Use the cleaned output from the single generate call
            # Save the original question text, not the full formatted prompt
            save_dict['prompt_question'] = original_question

            ans_file.write(json.dumps(save_dict) + "\n")
            res.append(save_dict)
            ans_file.flush()

        except Exception as e:
            print(f"[{i}] skipped – {type(e).__name__}: {e}", flush=True)
            # Optional: Add traceback for debugging
            import traceback
            traceback.print_exc()
            continue
        
    ans_file.close()
    
    pred_dict, correct_precentage = MedicalEval(res)
    final_save_dict = {
        "model_name": model_name,
        "dataset_name": question_file,
        "correct_precentage" :correct_precentage,
        "pred_dict" : pred_dict
    }
    
    with open(os.path.join(answers_base_path, os.path.dirname(question_file).replace('/', '_') + '.json'), 'w') as f:
        json.dump(final_save_dict, f, indent=4, ensure_ascii=False)

    import pandas as pd
    # turn your list of predictions into a DataFrame
    df = pd.DataFrame(res)

    out_dir = os.path.abspath(answers_base_path)
    os.makedirs(out_dir, exist_ok=True)

    # use a tag that always exists
    model_tag = os.path.basename(args.model_name).replace("/", "_")
    csv_path = os.path.join(out_dir, f"{model_tag}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved QA CSV results to {csv_path}", flush=True)


def preprocess_input(entity, image_root_path) -> tuple:
    """
    Prepares inputs based on Figure 3's QA prompt format.
    Returns:
        tuple: (prompt_text_for_llava, gt_answer_text, raw_pil_image)
    """
    question_text = entity['question']
    gt_ans = entity['gt_answer']
    option_a = entity.get('option_A')
    option_b = entity.get('option_B')
    option_c = entity.get('option_C')
    option_d = entity.get('option_D')

    # Construct the options string exactly as in Figure 3 (e.g., "A:CT, B:MRI,...")
    options_list = []
    if option_a is not None: options_list.append(f"A:{option_a}")
    if option_b is not None: options_list.append(f"B:{option_b}")
    if option_c is not None: options_list.append(f"C:{option_c}")
    if option_d is not None: options_list.append(f"D:{option_d}")
    options_str = ", ".join(options_list) # Join with comma and space

    # Construct the prompt using the Figure 3 template
    # Using "***" as separator instead of "###" to avoid potential conflict
    # with stopping criteria if '###' is used as a stop word. Adjust if needed.
    prompt_template = (
        "This is a medical Question with several Options, and there is only one "
        "correct answer among these options. Please select the correct answer for the "
        "question. Remember, you can only select one option. The Question is:{question}"
        "*** The candidate Options are:{options}" # Using *** separator
    )
    # Format the prompt text (without image tokens for now)
    prompt_text_for_llava = prompt_template.format(question=question_text, options=options_str)

    # Load image
    image_url = entity.get('image_path')
    # Pass image_root_path explicitly to avoid reliance on global args
    image = read_img_from_url(image_url, image_root_path)

    return prompt_text_for_llava, gt_ans, image
    

from PIL import Image
import sys
from io import BytesIO

# def read_img_from_url(url):
#     base_path = "/home/alex.ia/OmniMed/OmniMedVQA/OmniMedVQA"
#     full_path = os.path.join(base_path, url)
    
#     if not os.path.exists(full_path):
#         raise FileNotFoundError(f"Image not found: {full_path}")
    
#     img = Image.open(full_path).convert("RGB")
    
#     if img.size == (0, 0):
#         raise ValueError(f"Image is empty: {full_path}")
    
#     return img
def read_img_from_url(url, root):
    full_path = os.path.join(root, url)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")
    return Image.open(full_path).convert("RGB")

def MedicalEval(pred_dict: list) -> tuple:
    tot = len(pred_dict)
    if tot == 0:
        return pred_dict, 0.0
    succ = 0
    for data in pred_dict:
        try:
            a,b,c,d = data.get('option_A'), data.get('option_B'), data.get('option_C'), data.get('option_D')
            answer_list = [a, b]
            if c is not None:
                answer_list.append(c)
            if d is not None:
                answer_list.append(d)
            
            if answer_list[find_most_similar_index(answer_list, data['model_pred'])] == data['gt_answer']:
                succ += 1
                data['is_correct'] = 'yes'
            else:
                data['is_correct'] = 'no'
        except:
            continue
        
    return pred_dict, succ/tot
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=str, default="/home/alex.ia/OmniMed/OmniMedVQA/OmniMedVQA", help="Top-level folder that contains the Images/ subdir",)
    parser.add_argument("--model-name", type=str, default="VLP_web_data/llava_med/llava_med_in_text_60k")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
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
    eval_model(args, args.question_file, args.answers_base_path)
    print('finish', flush=True)