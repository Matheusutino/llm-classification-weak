import random
import argparse
import pandas as pd
from tqdm import tqdm 
from src.core.messages.message_manager import MessageManager
from src.core.predictors.predictor_manager import PredictionManager
from src.core.utils import get_prompts, check_directory_exists, create_directory, get_last_element_from_path

def llm_predict(dataset_path: str, service: str, message_type: str, prompt_name: str, **kwargs):
    # Read the dataset
    dataset = pd.read_csv(dataset_path)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)[0:1000]

    dataset_name = get_last_element_from_path(dataset_path)
    model_identifier = kwargs['filename'] if kwargs['filename'] else kwargs['model_name'] 
    results_path = f"datasets/llm_predict/{model_identifier}"
    check_directory_exists(results_path)

    # Initialize the managers
    message_manager = MessageManager(message_type=message_type)
    prediction_manager = PredictionManager(service=service, **kwargs)

    user_prompt, system_prompt = get_prompts(prompt_name)

    classes = '\n'.join(dataset['class'].unique().tolist())
    # examples = dataset.groupby('class').first().reset_index()
    # few_shot_examples = "\n".join([f"Text: {row['text']}\nCategory: {row['class']}" for _, row in examples.iterrows()])
    # classes = dataset['class'].unique().tolist()

    results = []

    for text in tqdm(dataset['text'].tolist(), desc="Processando textos"):
        # Format the user prompt with the specific text and classes
        # formatted_user_prompt = user_prompt.format(text=text, few_shot_examples=few_shot_examples, classes=classes)
        formatted_user_prompt = user_prompt.format(text=text, classes=classes)
        message = message_manager.generate_message(formatted_user_prompt, system_prompt)
        
        use_fixed_temperature = True
        # Loop até encontrar um resultado válido
        valid_result = False
        while not valid_result:
            if use_fixed_temperature:
                temperature = kwargs["temperature"]
                use_fixed_temperature = False  # Só use a fixa uma vez
            else:
                temperature = random.uniform(0.0, 1.0)
                
            result = prediction_manager.predict(message, temperature = temperature).strip()

            # Verifique se o result está nas classes
            if result in classes:
                valid_result = True
                results.append(result)
            else:
                print(f"Resultado '{result}' não está nas classes, tentando novamente...")
    
    dataset['predict_llm'] = results

    create_directory(results_path)
    dataset.to_csv(f'{results_path}/{dataset_name}', index=False)

if __name__ == "__main__":
    # Initialize the ArgumentParser
    parser = argparse.ArgumentParser(description="Run predictions with LLM.")
    
    # Add required arguments (sem `--`)
    parser.add_argument('dataset_path', type=str, help='Path to the dataset CSV file')
    parser.add_argument('service', type=str, help='The service for prediction')
    parser.add_argument('message_type', type=str, help='Message type for generating messages')
    parser.add_argument('prompt_name', type=str, help='Name of the prompt to use')

    # Add optional arguments (com `--`)
    parser.add_argument('--repo_id', type=str, help='Repository ID for the model (optional)')
    parser.add_argument('--filename', type=str, help='Filename for input data (optional)')
    parser.add_argument('--model_name', type=str, help='Model name for prediction (optional)')
    parser.add_argument('--temperature', type=float, default=0.3, help='Temperature for llm inference (optional)')

    # Parse the arguments
    args = parser.parse_args()

    # Call llm_predict with required arguments and optional kwargs
    llm_predict(
        dataset_path = args.dataset_path, 
        service = args.service, 
        message_type = args.message_type, 
        prompt_name = args.prompt_name, 
        repo_id = args.repo_id,
        filename = args.filename,
        model_name = args.model_name,
        temperature = args.temperature
    )

