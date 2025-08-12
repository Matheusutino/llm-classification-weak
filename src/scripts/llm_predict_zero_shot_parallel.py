import concurrent.futures
import json
import time 
import random
import argparse
import pandas as pd
from tqdm import tqdm 
from src.core.predictors.predictor_manager import PredictionManager
from src.core.utils import get_prompts, check_file_exists, create_directory, get_last_element_from_path, save_json, extract_json_block, extract_think_block

random.seed(42)

def process_row(idx, row, classes, user_prompt, system_prompt, prediction_manager, max_tokens, temperature, seed, max_attempts):
    text = row['text']
    categories = '\n'.join([f"- {c}" for c in classes]).strip()
    formatted_user_prompt = user_prompt.format(text=text, categories=categories)

    use_fixed_temperature = True
    valid_result = False
    attempt = -1
    result = None
    think_content = None

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

            if result is None:
                print(f"Resultado None para idx {idx}, tentando novamente...")
                continue

            think_content, result_clean = extract_think_block(result)
            result_only_json_str = extract_json_block(result_clean)
            json_obj = json.loads(result_only_json_str)
            required_keys = ['category', 'explanation', 'confidence']

            if all(key in json_obj for key in required_keys):
                if json_obj['category'] in classes.tolist():
                    return {
                        'idx': idx,
                        'response': result,
                        'predicted_class': json_obj['category'],
                        'explanation': json_obj['explanation'],
                        'confidence': json_obj['confidence'],
                        'temperature': current_temperature,
                        'attempt': attempt,
                        'reasoning': think_content
                    }
                else:
                    print(f"Resultado '{json_obj['category']}' não está nas classes para idx {idx}, tentando novamente...")
            else:
                missing = [k for k in required_keys if k not in json_obj]
                print(f"JSON inválido, faltando chave(s): {missing} para idx {idx}")

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"Erro ao processar tentativa {attempt} para idx {idx}: {e}")

    # Se não obteve resultado válido, retorna valores padrão
    return {
        'idx': idx,
        'response': result,
        'predicted_class': None,
        'explanation': None,
        'confidence': None,
        'temperature': None,
        'attempt': attempt,
        'reasoning': None
    }

def llm_predict_zero_shot_parallel(dataset_path: str, service: str, prompt_name: str, model_name, max_tokens, temperature, seed, max_attempts, max_workers=5):
    dataset = pd.read_csv(dataset_path)

    dataset_name = get_last_element_from_path(dataset_path)
    results_path = f"datasets/llm_predict/zero_shot/{dataset_name[:-4]}"
    model_id = model_name.split("/")[-1]

    check_file_exists(f'{results_path}/{model_id}.csv')

    prediction_manager = PredictionManager(service=service, model_name=model_name)

    user_prompt, system_prompt = get_prompts(prompt_name)
    classes = dataset['class'].unique()

    create_directory(results_path)

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, row in dataset.iterrows():
            futures.append(executor.submit(
                process_row,
                idx, row, classes, user_prompt, system_prompt,
                prediction_manager, max_tokens, temperature, seed, max_attempts
            ))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processando textos paralelamente"):
            res = future.result()
            results.append(res)

    # Ordena resultados pelo índice para manter a ordem original
    results = sorted(results, key=lambda x: x['idx'])

    # Atualiza o dataset com as previsões
    for r in results:
        idx = r['idx']
        dataset.at[idx, 'response'] = r['response']
        dataset.at[idx, 'predicted_class'] = r['predicted_class']
        dataset.at[idx, 'explanation'] = r['explanation']
        dataset.at[idx, 'confidence'] = r['confidence']
        dataset.at[idx, 'temperature'] = r['temperature']
        dataset.at[idx, 'attempt'] = r['attempt']
        dataset.at[idx, 'reasoning'] = r['reasoning']

    # Salva o resultado e configurações
    config_data = {
        "dataset": dataset_name,
        "service": service,
        "model": model_name,
        "prompt_name": prompt_name,
        "max_tokens": max_tokens,
        "initial_temperature": temperature,
        "seed": seed,
        "max_attempts": max_attempts,
    }

    save_json(config_data, f"{results_path}/{model_id}.json")
    dataset.to_csv(f'{results_path}/{model_id}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions with LLM in parallel.")
    
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--service', type=str, help='The service for prediction')
    parser.add_argument('--prompt_name', type=str, help='Name of the prompt to use')
    parser.add_argument('--model_name', type=str, help='Model name for prediction (optional)')
    parser.add_argument('--max_tokens', type=float, default=1024, help='Max Tokens for llm inference (optional)')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for llm inference (optional)')
    parser.add_argument('--seed', type=int, default=0, help='Seed for llm inference (optional)')
    parser.add_argument('--max_attempts', type=int, default=10, help='Max attempts')
    parser.add_argument('--max_workers', type=int, default=80, help='Número máximo de threads para execução paralela')

    args = parser.parse_args()

    llm_predict_zero_shot_parallel(
        dataset_path=args.dataset_path,
        service=args.service,
        prompt_name=args.prompt_name,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        max_attempts=args.max_attempts,
        max_workers=args.max_workers
    )

