from src.core.messages.openai_message import OpenAIMessageGenerator
from src.core.messages.str_message import StrMessageGenerator
from src.core.messages.llama_message import LlamaMessageGenerator
from src.core.messages.phi35_message import Phi35MessageGenerator
from src.core.messages.aya_expanse_message import AyaExpanseMessageGenerator
from src.core.messages.granite_message import GraniteMessageGenerator
from src.core.messages.mistral_message import MistralMessageGenerator
from src.core.messages.qwen25_message import Qwen25MessageGenerator

class MessageManager:
    """Manager class to handle message generation for different model types."""

    def __init__(self, message_type: str):
        """Initializes the MessageManager with supported message generator classes."""
        self.message_generators = {
            'openai': OpenAIMessageGenerator(),
            'str': StrMessageGenerator(),
            'sabia-7b': OpenAIMessageGenerator(),
            'llama-3.2': LlamaMessageGenerator(),
            'phi-3.5': Phi35MessageGenerator(),
            'aya_expanse': AyaExpanseMessageGenerator(),
            'granite': GraniteMessageGenerator(),
            'mistral': MistralMessageGenerator(),
            'qwen-2.5': Qwen25MessageGenerator()
        }

        self.message_generator = self.message_generators.get(message_type, None)

    def generate_message(self, prompt: str, specialist: str = None) -> list:
        """Generates a message according to the specified model type.
        
        Args:
            prompt (str): The input text for which the message should be generated.
            specialist (str, optional): The specialist content to include in the message.
        
        Returns:
            list: The formatted message as a list of dictionaries.
        """
        if not self.message_generator:
            raise ValueError("Unsupported model type. Choose between 'openai' or 'huggingface'.")

        return self.message_generator.generate_message(prompt, specialist)