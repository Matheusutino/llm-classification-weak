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

# Carrega variáveis de ambiente
load_dotenv(find_dotenv())

class NVIDIAPredictor(PredictionModel):
    def __init__(self, model_name: str, api_key: str = None, base_url: str = "https://integrate.api.nvidia.com/v1"):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @retry(wait=wait_random_exponential(min=2, max=5), stop=stop_after_attempt(5))
    def predict(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"NVIDIA API error: {e}")

