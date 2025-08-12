#!/bin/bash

# Lista de datasets (por enquanto, só um)
DATASETS=("datasets/original/Dmoz-Health.csv")

# Service fixo
SERVICE="openrouter"

# Prompt fixo
PROMPT="zero_shot"

# Lista de modelos
MODELS=(
    "meta-llama/llama-3.2-3b-instruct"
    "liquid/lfm-7b"
    "deepseek/deepseek-r1-0528-qwen3-8b"
    "google/gemma-2-9b-it"  # ruim
    "mistralai/mistral-nemo"
    "cognitivecomputations/dolphin3.0-r1-mistral-24b"
    "qwen/qwen3-32b"
    "openai/gpt-oss-20b"
    "meta-llama/llama-3.3-70b-instruct"
    "meta-llama/llama-4-scout"
)

# Parâmetros adicionais 
MAX_TOKENS=1024 # Que o modelo pode gerar
TEMPERATURE=0.0
MAX_ATTEMPTS=10
MAX_WORKERS=200

# Loop para rodar todos os datasets e modelos
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "Rodando com dataset: $dataset e modelo: $model"
        python3 -m src.scripts.llm_predict_zero_shot_parallel \
            --dataset_path "$dataset" \
            --service "$SERVICE" \
            --prompt_name "$PROMPT" \
            --model_name "$model" \
            --max_tokens "$MAX_TOKENS" \
            --temperature "$TEMPERATURE" \
            --max_attempts "$MAX_ATTEMPTS" \
            --max_workers "$MAX_WORKERS"
    done
done
