import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import torch.nn as nn
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN # Add constants
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
        # If load_pretrained_model doesn't handle device, uncomment below:
        # model.to(device)
        model.to('cuda')
        print("Model moved to GPU.")
        print("Model, Tokenizer, Image Processor loaded successfully.")
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
    for i, line in enumerate(tqdm(questions)):
        try:
            tot += 1
            candidates = load_candidates_medical(line)
            question_text, gt_ans, raw_image = preprocess_input(line)
            prefix = load_prompt(question_text)
            prefix_tokens = tokenizer(prefix)
            start_loc = len(prefix_tokens.input_ids)
            candidate_scores = []  # pred scores of candidates

            images = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values']
            images = images.half().cuda()

            for candidate in candidates:
                prompt = prefix + " {}.".format(candidate)
                input_ids = tokenizer(prompt).input_ids
                input_ids = torch.as_tensor([input_ids]).cuda()
                
                
                prompt = prefix + " {}.".format(candidate)
                prompt_tokens =tokenizer(prompt, return_tensors="pt")
                lang_t = prompt_tokens["input_ids"]
                
                prefix_tokens = tokenizer(prefix, return_tensors="pt")  
                lang_t1 = prefix_tokens["input_ids"]
                lang_diff = lang_t.shape[1] - lang_t1.shape[1]
                
                with torch.inference_mode():
                    out = model(input_ids,use_cache=True,images=images)
                logits = out.logits
                targets =  input_ids
                targets[0,:start_loc]=-100
                targets[0,start_loc+lang_diff:]=-100
                shift_logits = logits[...,:-1,:].contiguous()
                shift_labels = targets[...,1:].contiguous()
                loss_fct = CrossEntropyLoss(reduction="mean")
                vocab_size = model.config.vocab_size # Get vocab size from model config
                loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1)) # Use vocab_size
            
                candidate_scores.append(loss.item())
            ori_candidate = candidate_scores
            candidate_scores = softmax(np.reciprocal(candidate_scores))
            pred = candidates[np.argmax(candidate_scores)]   
            print(candidates, candidate_scores, flush=True)
            save_dict = deepcopy(line)
            save_dict['model_pred'] = pred
            save_dict['prompt_question'] = prefix
            save_dict['confidence'] = str(ori_candidate)
            save_dict['is_correct'] = 'yes' if pred == gt_ans else 'no'
            if pred == gt_ans:
                cnt  += 1

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


def preprocess_input(entity) -> tuple:
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
    