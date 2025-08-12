import os
from dotenv import load_dotenv, find_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)
from together import Together
from src.core.predictors.base import PredictionModel

# Load environment variables
load_dotenv(find_dotenv())

class TogetherPredictor(PredictionModel):
    """Predictor using Together.ai's API."""

    def __init__(self, model_name: str):
        os.getenv("TOGETHER_API_KEY")
        self.client = Together()
        self.model = model_name  # Model name passed as a parameter

    retry(wait=wait_random_exponential(min=2, max=5), stop=stop_after_attempt(5))
    def predict(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, seed: int, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
