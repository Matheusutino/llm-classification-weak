import argparse
import pandas as pd
from tqdm import tqdm 
from src.core.messages.message_manager import MessageManager
from src.core.predictors.predictor_manager import PredictionManager
from src.core.utils import get_prompts, create_directory

def llm_predict(dataset_path: str, service: str, message_type: str, prompt_name: str, **kwargs):
    # Read the dataset
    dataset = pd.read_csv(dataset_path)[0:3]

    results_path = "results"
    create_directory(results_path)

    # Initialize the managers
    message_manager = MessageManager(message_type=message_type)
    prediction_manager = PredictionManager(service=service, **kwargs)

    user_prompt, system_prompt = get_prompts(prompt_name)

    classes = ', '.join(dataset['class'].unique().tolist())

    results = []

    for text in tqdm(dataset['text'], desc="Processando textos"):
        user_prompt = user_prompt.format(text = text, classes = classes)
        message = message_manager.generate_message(user_prompt, system_prompt)
        result = prediction_manager.predict(message)

        results.append(result)
    
    dataset['predict_llm'] = results

    dataset.to_csv(f'{results_path}/results.csv', index=False)

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
        model_name = args.model_name
    )

