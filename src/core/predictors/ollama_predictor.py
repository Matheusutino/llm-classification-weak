import ollama
from src.core.predictors.base import PredictionModel
from src.core.utils import get_value_by_key_json

class OllamaPredictor(PredictionModel):
    def __init__(self, model_name: str, num_ctx:int = 4096, device: str = 'cpu'):
        self.device = device
        self.model_name = model_name
        self.num_ctx = num_ctx
        #self.num_ctx = get_value_by_key_json(file_path="configs/context_lenght.json", key=model_name)
        ollama.pull(self.model_name)

    def predict(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, seed: int, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature,
                    'num_ctx': self.num_ctx,
                    'seed': seed
                }
            )
            return response['message']['content'].strip()
        except Exception as e:
            raise RuntimeError(f"Error generating text: {e}")


