import llama_cpp
from huggingface_hub import hf_hub_download
from src.core.predictors.base import PredictionModel

class LlamaCppPredictor(PredictionModel):
    """Prediction model implementation for llama.cpp."""

    def __init__(self, repo_id: str, filename: str, device: str = 'gpu', verbose: bool = False):
        """
        Initializes the LlamaCppPredictor with a model path and device.

        Args:
            model_name (str): The path to the Llama model file (.bin or .gguf).
            device (str): Device to use for inference, 'cpu' or 'cuda'.
        """
        self.device = device

        if self.device == 'cpu':
            n_gpu_layers = 0 
        elif self.device == 'gpu':
            n_gpu_layers = -1  
        else:
            raise ValueError(f"Invalid device: {device}. Choose 'cpu' or 'gpu'")

        self.model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.model = llama_cpp.Llama(
            model_path=self.model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048,
            verbose=verbose
        )

    def predict(self, messages, max_tokens: int = 64, temperature: float = 0.3):
        """
        Generates text based on the input prompt using the Llama model.

        Args:
            messages (str): The formatted prompt for Llama, including system and user sections.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature. Lower values make the output more deterministic.
            do_sample (bool): Whether to use sampling or greedy decoding.

        Returns:
            str: The generated text.
        """
        try:
            output = self.model(
                messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return output['choices'][0]["text"]
        except Exception as e:
            raise RuntimeError(f"Error generating text: {e}")