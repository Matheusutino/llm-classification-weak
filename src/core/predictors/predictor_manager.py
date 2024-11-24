from typing import List, Dict, Union
from src.core.predictors.openai_predictor import OpenAIPredictor
from src.core.predictors.maritaca_ai import MaritacaAIPredictor
from src.core.predictors.llama_cpp_predictor import LlamaCppPredictor
from src.core.predictors.ollama_predictor import OllamaPredictor

class PredictionManager:
    """Manager class to handle predictions for a single model type."""

    def __init__(self, service: str, **kwargs):
        """
        Initializes the PredictionManager with the appropriate prediction class based on the model_name.

        Args:
            model_name (str): The name of the model ('openai' or the Hugging Face model name).
            **kwargs: Additional keyword arguments such as 'openai_api_key' and 'device'.
        """
        if service.lower() == 'openai':
            self.predictor = OpenAIPredictor(model_name=kwargs.get('model_name'))
        elif service.lower() == 'maritaca_ai':
            self.predictor = MaritacaAIPredictor(model_name=kwargs.get('model_name'))
        elif service.lower() == 'llama_cpp':
            repo_id = kwargs.get('repo_id')
            filename = kwargs.get('filename')
            device = kwargs.get('device', 'gpu')
            self.predictor = LlamaCppPredictor(repo_id = repo_id, filename = filename, device=device)
        elif service.lower() == 'ollama':
            self.predictor = OllamaPredictor(model_name=kwargs.get('model_name'))
        else:
            raise ValueError(f"Unsupported service: {service}. Please choose from 'openai', 'maritaca_ai', 'llama_cpp', or 'huggingface'.")


    def predict(self, messages: Union[str, List[Dict[str, str]]], temperature: float = 0.3, **kwargs) -> str:
        """Generates a prediction using the initialized predictor.
        
        Args:
            input_data (str): The input data for which the prediction should be generated.
            **kwargs: Additional arguments for prediction (e.g., max_tokens, temperature).
        
        Returns:
            str: The prediction result.
        """
        return self.predictor.predict(messages, temperature = temperature, **kwargs)