from src.core.messages.base import MessageGenerator

class GraniteMessageGenerator(MessageGenerator):

    def generate_message(self, prompt: str, specialist=None) -> str:
        """
        Generates a message using the Granite message format.

        Args:
            prompt (str): The user's input message.
            system_prompt (str, optional): The system's initial context or instructions. Defaults to an empty string.

        Returns:
            str: A formatted string adhering to the Granite message template.
        """
         
        specialist = specialist if specialist else ""
        
        message = f"<|start_of_role|>system<|end_of_role|>{specialist}<|end_of_text|><|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|><|start_of_role|>assistant<|end_of_role|>"
        
        return message
