import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from copy import deepcopy
import difflib
from PIL import Image
from io import BytesIO
from pathlib import Path

def bytes2PIL(bytes_img):
    '''Transform bytes image to PIL.
    Args:
        bytes_img: Bytes image.
    '''
    pil_img = Image.open(BytesIO(bytes_img)).convert("RGB")
    return pil_img

def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()
 
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


def MedicalEval(pred_dict: list) -> tuple:
    tot = len(pred_dict)
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


def evaluate_medical_QA(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    batch_size=1,
    answer_path='./answers',
    image_base_path=None
):
    
    if image_base_path is None:
        raise ValueError("image_base_path must be provided to evaluate_medical_QA")
    
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    
    for batch in tqdm(dataloader, desc="Running inference"):
        try:
            image_paths_relative = batch['image_path'] # Get relative paths
            questions = batch['question']
            entities = batch['entity']

            imgs = []
            valid_indices = [] # Keep track of items successfully loaded in this batch

            for i, img_path_rel in enumerate(image_paths_relative):
                full_img_path = os.path.join(image_base_path, img_path_rel)
                try:
                    # --- CHANGE: Open the full_img_path, not the relative one ---
                    img = Image.open(full_img_path).convert("RGB")
                    imgs.append(img)
                    valid_indices.append(i) # Mark this index as valid
                except FileNotFoundError:
                    print(f"WARNING: Image file not found at {full_img_path}. Skipping item {i} in batch.")
                    # Don't append to imgs, index i won't be in valid_indices
                except Exception as e:
                    print(f"ERROR: Failed to load image {full_img_path} due to {e}. Skipping item {i} in batch.")
                    # Don't append to imgs, index i won't be in valid_indices

            # --- ADD: Check if any images were loaded in the batch ---
            if not valid_indices:
                print("WARNING: No valid images loaded in this batch. Skipping.")
                continue # Skip to the next batch

            # --- ADD: Select only the valid questions and entities for this batch ---
            valid_questions = [questions[i] for i in valid_indices]
            valid_entities = [entities[i] for i in valid_indices]

            # Generate answers only for valid images/questions
            # Note: `imgs` now only contains successfully loaded images
            outputs = model.batch_generate(imgs, valid_questions)
            print(outputs)

            for entity, output in zip(valid_entities, outputs):
                answer_dict = deepcopy(entity)
                answer_dict['model_pred'] = output
                predictions.append(answer_dict)

        except Exception as e:
            # --- CHANGE: More informative error for batch failure ---
            print(f"ERROR during batch processing: {e}. Skipping batch.")
            # You might want to log which entities were in the failed batch if possible
            continue

    # --- CHANGE: Fix result saving ---
    # The 'answer_path' argument now directly represents the final output directory
    output_dir = Path(answer_path)
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure the directory exists

    # Use a fixed, predictable filename within that directory
    output_filename = "qa_results.json" # <<< Use a fixed name
    final_output_path = output_dir / output_filename
    # --- End result saving fix ---

    # Run evaluation on the collected predictions
    final_dict, correct_precentage = MedicalEval(predictions)

    # Construct a more reliable dataset name if possible
    dataset_name_modality = "unknown_modality"
    if dataset.data:
         dataset_name_modality = dataset.data[0].get("modality_type", "unknown_modality")

    save_dict = {
        "model_name": model_name,
        "dataset_name": dataset_name_modality, # Use the extracted modality type
        "correct_precentage" :correct_precentage,
        "pred_dict" : final_dict # Use the evaluated dict
    }
    
    print(f"Saving QA results to: {final_output_path}") # <<< Print the correct final path
    with open(final_output_path, "w") as f:
        json.dump(save_dict, f, indent=4, ensure_ascii=False) # Added ensure_ascii=False

    print(f'QA Accuracy:{correct_precentage}')
    return correct_precentage

def pred_medical_QA(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    batch_size=1,
    answer_path='./answers'
):
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    for batch in tqdm(dataloader, desc="Running inference"):
        try:
            print(batch['question'])
            image_paths = batch['image_path']
            imgs = []
            for img_path in image_paths:
                imgs.append(Image.open(img_path).convert("RGB"))
            outputs = model.batch_generate(imgs, batch['question'])
            print(outputs)
            for entity, output in zip(batch['entity'], outputs):
                answer_dict = deepcopy(entity)
                answer_dict['model_pred'] = output
                predictions.append(answer_dict)
        except:
            print(f'error: {batch}')
            continue

    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    
    save_dict = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "pred_dict" : predictions
    }

    answer_path = os.path.join(answer_dir, f"{model_name}_{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(save_dict, indent=4))
        
    return save_dict
