from abc import ABC, abstractmethod

class PredictionModel(ABC):
    """Abstract base class for prediction models."""

    @abstractmethod
    def predict(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """
        Abstract method to perform prediction.

        Args:
            system_prompt (str): The system message.
            user_prompt (str): The user message.
            max_tokens (int): Maximum tokens for generation.
            temperature (float): Sampling temperature.

        Returns:
            str: The prediction result.
        """
        pass
