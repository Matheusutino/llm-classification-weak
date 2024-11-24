from src.core.messages.base import MessageGenerator

class Qwen25MessageGenerator(MessageGenerator):

    def generate_message(self, prompt: str, specialist: str) -> str:
        message = f"<|im_start|>system\n{specialist}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"

        return message
