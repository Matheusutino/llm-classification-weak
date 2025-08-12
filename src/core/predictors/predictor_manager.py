import os
from src.core.predictors.openai_predictor import OpenAIPredictor
from src.core.predictors.ollama_predictor import OllamaPredictor
from src.core.predictors.nvidia_predictor import NVIDIAPredictor
from src.core.predictors.together_predictor import TogetherPredictor
from src.core.predictors.openrouter_predictor import OpenRouterPredictor
from src.core.predictors.vllm_predictor import VLLMPredictor

class PredictionManager:
    """Manager class to handle predictions for a single model type."""

    def __init__(self, service: str, model_name: str):
        """
        Initializes the PredictionManager with the appropriate prediction class based on the service.

        Args:
            service (str): The name of the service ('openai', 'nvidia', 'openrouter', or 'ollama').
            **kwargs: Additional keyword arguments such as 'model_name' and 'device'.
        """
        if service.lower() == 'openai':
            self.predictor = OpenAIPredictor(model_name=model_name)
        elif service.lower() == 'nvidia':
            self.predictor = NVIDIAPredictor(model_name=model_name)
        elif service.lower() == 'openrouter':
            self.predictor = OpenRouterPredictor(model_name=model_name)
        elif service.lower() == 'together':
            self.predictor = TogetherPredictor(model_name=model_name)
        elif service.lower() == 'ollama':
            self.predictor = OllamaPredictor(model_name=model_name)
        elif service.lower() == 'vllm':
            self.predictor = VLLMPredictor(model_path=model_name)
        else:
            raise ValueError(f"Unsupported service: {service}. Please choose from 'openai', 'nvidia', 'openrouter', 'ollama'.")

    def predict(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, seed: int) -> str:
        """
        Generates a prediction using the initialized predictor.

        Args:
            system_prompt (str): The system-level instruction or context.
            user_prompt (str): The user input or question.
            max_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.
            seed (int): Random seed (used only in Ollama).

        Returns:
            str: The generated response.
        """
        return self.predictor.predict(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed
        )
    
    def predict_batch(self, batch_prompts, max_tokens: int, temperature: float, seed: int) -> str:
        """
        Generates a prediction using the initialized predictor.

        Args:
            system_prompt (str): The system-level instruction or context.
            user_prompt (str): The user input or question.
            max_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.
            seed (int): Random seed (used only in Ollama).

        Returns:
            str: The generated response.
        """
        return self.predictor.predict_batch(
            batch_prompts=batch_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed
        )

