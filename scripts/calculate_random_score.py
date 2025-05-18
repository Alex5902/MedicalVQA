import json
import os

def calculate_expected_random_score(file_path):
    """
    Calculates the expected random score for a VQA dataset in JSON format.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        float: The expected random score as a percentage, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None

    if not isinstance(data, list) or not data:
        print(f"Warning: No data or unexpected data format in {file_path}")
        return 0.0 # Or None, depending on how you want to handle empty/bad files

    total_random_probability_sum = 0.0
    num_questions = 0

    option_keys = ["option_A", "option_B", "option_C", "option_D"] 
    # If you expect more options (E, F, etc.), extend this list
    # For this problem, D seems to be the max from the example.

    for item in data:
        if not isinstance(item, dict):
            print(f"Warning: Skipping non-dictionary item in {file_path}: {item}")
            continue

        num_options = 0
        for key in option_keys:
            if key in item and item[key] is not None and str(item[key]).strip() != "":
                # We check if the key exists and has a non-empty value.
                # The problem implies options are present if the key exists.
                # You might simplify to `if key in item:` if all options are guaranteed non-null/empty.
                num_options += 1
        
        if num_options > 0:
            random_chance_for_question = 1.0 / num_options
            total_random_probability_sum += random_chance_for_question
            num_questions += 1
        else:
            # This case should ideally not happen if questions always have options.
            print(f"Warning: Question {item.get('question_id', 'N/A')} has 0 valid options. Skipping.")


    if num_questions == 0:
        print(f"No valid questions found in {file_path} to calculate score.")
        return 0.0 # Or None

    average_random_score = total_random_probability_sum / num_questions
    return average_random_score * 100

# --- Main script execution ---
if __name__ == "__main__":
    base_path = "/home/alex.ia/OmniMed/finetuning_datasets/OmniMedVQA_Splits_ImgLevel"
    
    test_json_files = [
        os.path.join(base_path, "Fundus_Photography/test.json"),
        os.path.join(base_path, "Microscopy_Images/test.json"),
        os.path.join(base_path, "X-Ray/test.json")
    ]

    print("Calculating expected random scores for test sets:\n")
    for test_file in test_json_files:
        print(f"Processing: {test_file}")
        score = calculate_expected_random_score(test_file)
        if score is not None:
            print(f"Expected random score: {score:.2f}%\n")
        else:
            print(f"Could not calculate score for {test_file}\n")