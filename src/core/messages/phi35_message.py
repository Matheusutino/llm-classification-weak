from src.core.messages.base import MessageGenerator

class Phi35MessageGenerator(MessageGenerator):

    def generate_message(self, prompt: str, specialist = None) -> str:
        message = ""

        if specialist:
            message += f"<|system|>{specialist}<|end|>"

        message += f"<|user|>{prompt}<|end|><|assistant|>"
        return message
