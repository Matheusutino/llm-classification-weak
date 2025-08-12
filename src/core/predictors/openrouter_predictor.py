import os
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv
from src.core.predictors.base import PredictionModel

load_dotenv(find_dotenv())

class OpenRouterPredictor(PredictionModel):
    def __init__(self, model_name: str, api_key: str = None, base_url: str = "https://openrouter.ai/api/v1"):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @retry(wait=wait_random_exponential(min=2, max=5), stop=stop_after_attempt(5))
    def predict(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, seed: int, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenRouter API error: {e}")

