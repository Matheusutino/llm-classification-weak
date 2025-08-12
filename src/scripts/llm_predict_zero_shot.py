import json
import time 
import random
import argparse
import pandas as pd
from tqdm import tqdm 
from src.core.predictors.predictor_manager import PredictionManager
from src.core.utils import get_prompts, check_file_exists, create_directory, get_last_element_from_path, save_json, extract_json_block, extract_think_block

random.seed(42)

def llm_predict_zero_shot(dataset_path: str, service: str, prompt_name: str, model_name, max_tokens, temperature, seed, max_attempts):
    # Read the dataset
    dataset = pd.read_csv(dataset_path)

    dataset_name = get_last_element_from_path(dataset_path)
    results_path = f"datasets/llm_predict/zero_shot/{dataset_name[:-4]}"
    model_id = model_name.split("/")[-1]

    check_file_exists(f'{results_path}/{model_id}.csv')

    # Initialize the managers
    prediction_manager = PredictionManager(service=service, model_name=model_name)

    user_prompt, system_prompt = get_prompts(prompt_name)

    classes = dataset['class'].unique()
    categories = '\n'.join([f"- {c}" for c in classes]).strip()

    results = []

    start_time = time.time()
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Processando textos para o dataset {dataset_name[:-4]}"):
        text = row['text']
        # Format the user prompt with the specific text and classes
        formatted_user_prompt = user_prompt.format(text=text, categories=categories)
        
        use_fixed_temperature = True
        valid_result = False
        attempt = -1

        while not valid_result and attempt < max_attempts: 
            attempt += 1
            try:
                if use_fixed_temperature:
                    current_temperature = temperature
                    use_fixed_temperature = False
                else:
                    current_temperature = random.uniform(0.0, 2.0)
                
                result = prediction_manager.predict(
                    system_prompt=system_prompt,
                    user_prompt=formatted_user_prompt,
                    max_tokens=max_tokens,
                    temperature=current_temperature,
                    seed=seed
                )

                print(result)

                if result is None:
                    print("Resultado None, tentando novamente...")
                    continue

                think_content, result_clean = extract_think_block(result)
                result_only_json_str = extract_json_block(result_clean)

                json_obj = json.loads(result_only_json_str)
                required_keys = ['category', 'explanation', 'confidence']

                if all(key in json_obj for key in required_keys):
                    if json_obj['category'] in classes.tolist():
                        dataset.at[idx, 'response'] = result
                        dataset.at[idx, 'predicted_class'] = json_obj['category']
                        dataset.at[idx, 'explanation'] = json_obj['explanation']
                        dataset.at[idx, 'confidence'] = json_obj['confidence']
                        dataset.at[idx, 'temperature'] = current_temperature
                        dataset.at[idx, 'attempt'] = attempt
                        dataset.at[idx, 'reasoning'] = think_content
                        valid_result = True
                    else:
                        print(f"Resultado '{json_obj['category']}' não está nas classes, tentando novamente...")
                else:
                    missing = [k for k in required_keys if k not in json_obj]
                    print(f"JSON inválido, faltando chave(s): {missing}")
            
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"Erro ao processar tentativa {attempt}: {e}")


        # If the result is still not valid after 10 attempts, fill with 'other'
        if not valid_result:
            dataset.at[idx, 'response'] = result
            dataset.at[idx, 'predicted_class'] = None
            dataset.at[idx, 'explanation'] = None
            dataset.at[idx, 'confidence'] = None
            dataset.at[idx, 'temperature'] = None
            dataset.at[idx, 'attempt'] = attempt
            dataset.at[idx, 'reasoning'] = None
        
    end_time = time.time()
    processing_time = end_time - start_time 

    config_data = {
        "dataset": dataset_name,
        "service": service,
        "model": model_name,
        "prompt_name": prompt_name,
        "max_tokens": max_tokens,
        "initial_temperature": temperature,
        "seed": seed,
        "max_attempts": max_attempts,
        "processing_time_seconds": processing_time
    }

    results_df = pd.DataFrame(results)
    dataset = pd.concat([dataset, results_df], axis=1)

    create_directory(results_path)
    print(results_path)
    save_json(config_data, f"{results_path}/{model_id}.json")
    dataset.to_csv(f'{results_path}/{model_id}.csv', index=False)

if __name__ == "__main__":
    # Initialize the ArgumentParser
    parser = argparse.ArgumentParser(description="Run predictions with LLM.")
    
    # Add required arguments 
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--service', type=str, help='The service for prediction')
    parser.add_argument('--prompt_name', type=str, help='Name of the prompt to use')
    parser.add_argument('--model_name', type=str, help='Model name for prediction (optional)')
    parser.add_argument('--max_tokens', type=float, default=1024, help='Max Tokens for llm inference (optional)')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for llm inference (optional)')
    parser.add_argument('--seed', type=int, default=0, help='Seed for llm inference (optional)')
    parser.add_argument('--max_attempts', type=int, default=10, help='Max attempts')

    # Parse the arguments
    args = parser.parse_args()

    # Call llm_predict with required arguments and optional kwargs
    llm_predict_zero_shot(
        dataset_path = args.dataset_path, 
        service = args.service, 
        prompt_name = args.prompt_name, 
        model_name = args.model_name,
        max_tokens= args.max_tokens,
        temperature = args.temperature,
        seed = args.seed,
        max_attempts = args.max_attempts
    )

