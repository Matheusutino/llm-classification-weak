import ollama
from src.core.predictors.base import PredictionModel
from src.core.utils import get_value_by_key_json

class OllamaPredictor(PredictionModel):
    """Prediction model implementation for Ollama."""

    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        Initializes the OllamaPredictor with a model name and device.

        Args:
            model_name (str): The name of the Ollama model.
            device (str): Device to use for inference, 'cpu' or 'gpu'.
        """
        self.device = device
        self.model_name = model_name
        self.num_ctx = get_value_by_key_json(file_path="configs/context_lenght.json", key = model_name)
        ollama.pull(self.model_name)

    def predict(self, messages, max_tokens: int = 64, temperature: float = 0.3):
        """
        Generates text based on the input prompt using the Ollama model.

        Args:
            messages (str): The formatted prompt for Ollama, including system and user sections.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature. Lower values make the output more deterministic.

        Returns:
            str: The generated text.
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options= {
                    'num_predict': max_tokens,
                    'temperature': temperature,
                    'num_ctx': self.num_ctx
                }
            )
            return response['message']['content']
        except Exception as e:
            raise RuntimeError(f"Error generating text: {e}")
