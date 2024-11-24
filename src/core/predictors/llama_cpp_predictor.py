import llama_cpp
from huggingface_hub import hf_hub_download
from src.core.predictors.base import PredictionModel
from src.core.utils import get_value_by_key_json

class LlamaCppPredictor(PredictionModel):
    """Prediction model implementation for llama.cpp."""

    def __init__(self, repo_id: str, filename: str, device: str = 'gpu', verbose: bool = True):
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
        
        self.n_ctx = get_value_by_key_json(file_path="configs/context_lenght.json", key = repo_id)

        self.model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.model = llama_cpp.Llama(
            model_path=self.model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=verbose
        )

    def truncate_prompt(self, prompt: str) -> str:
        """
        Tokeniza um prompt, trunca para o tamanho máximo permitido pelo contexto e detokeniza de volta para string.

        Args:
            prompt: Texto de entrada para truncamento.

        Returns:
            Texto truncado.
        """
        if not isinstance(prompt, str):
            raise ValueError(f"Invalid prompt type: {type(prompt)}. Expected string.")
        
        # Converte o texto para bytes conforme esperado pela função tokenize
        prompt_bytes = prompt.encode("utf-8")
        
        try:
            # Tokeniza
            tokens = self.model.tokenize(prompt_bytes, add_bos=True)
            
            # Trunca os tokens
            truncated_tokens = tokens[:self.n_ctx]
            
            # Detokeniza de volta para bytes e converte para string
            truncated_prompt = self.model.detokenize(truncated_tokens).decode("utf-8")
            
            return truncated_prompt
        except Exception as e:
            raise RuntimeError(f"Error during truncation: {e}") from e



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
            messages = self.truncate_prompt(messages)
            
            output = self.model(
                messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return output['choices'][0]["text"]
        except Exception as e:
            raise RuntimeError(f"Error generating text: {e}")