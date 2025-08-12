import os
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from src.core.predictors.base import PredictionModel
from src.core.utils import get_value_by_key_json

class VLLMPredictor(PredictionModel):
    def __init__(self, model_path: str, device: str = 'cuda'):
        # Carregar .env
        load_dotenv()
        os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")

        self.device = device
        self.model_path = model_path
        self.num_ctx = get_value_by_key_json(
            file_path="configs/context_lenght.json", key=model_path
        )

        self.llm = LLM(
            model=model_path,
            max_model_len=self.num_ctx,
            gpu_memory_utilization=0.6,
            max_num_seqs=1,
            quantization="awq"
        )

    def predict(
        self,
        prompts: dict,
        max_tokens: int,
        temperature: float,
        seed: int,
        **kwargs
    ) -> str:
        """
        Recebe um único prompt no formato:
            {"system_prompt": "Texto do sistema", "user_prompt": "Texto do usuário"}
        Retorna a resposta gerada como string.
        """
        responses = self.predict_batch(
            batch_prompts=[prompts],
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            **kwargs
        )
        return responses[0] if responses else ""

    def predict_batch(
        self,
        batch_prompts: list[dict],
        max_tokens: int,
        temperature: float,
        seed: int,
        **kwargs
    ) -> list[str]:
        """
        Recebe lista de prompts, cada um no formato:
            {"system_prompt": "...", "user_prompt": "..."}
        Retorna lista de respostas (strings).
        """

        batch_messages = []
        for prompts in batch_prompts:
            messages = [
                {"role": "system", "content": prompts["system_prompt"]},
                {"role": "user", "content": prompts["user_prompt"]}
            ]
            batch_messages.append(messages)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed
        )

        try:
            outputs = self.llm.chat(batch_messages, sampling_params=sampling_params, use_tqdm=False)
            responses = [output.outputs[0].text.strip() for output in outputs]
            return responses
        except Exception as e:
            raise RuntimeError(f"Error generating text with vLLM: {e}")
