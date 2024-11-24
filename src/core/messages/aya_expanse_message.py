from src.core.messages.base import MessageGenerator

class AyaExpanseMessageGenerator(MessageGenerator):

    def generate_message(self, prompt: str, specialist=None) -> str:
        specialist = specialist if specialist else ""
        prompt = prompt

        message = f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{specialist}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        
        return message
