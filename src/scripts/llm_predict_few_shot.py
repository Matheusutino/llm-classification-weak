import re
import json
import time
import random
import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from jinja2 import Template
from sentence_transformers import SentenceTransformer
from src.core.messages.message_manager import MessageManager
from src.core.predictors.predictor_manager import PredictionManager
from src.core.utils import get_prompts, check_file_exists, create_directory, get_last_element_from_path, save_json

random.seed(42)

def generate_embeddings(df, model_name='all-MiniLM-L12-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
    df['embedding'] = list(embeddings)
    return df

def apply_kmeans_clustering(df):
    num_clusters = df['class'].nunique()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df['embedding'].tolist())
    return df, kmeans.cluster_centers_

def select_representative_instances(df, cluster_centers, num_clusters, m):
    n = int(m * (len(df) / num_clusters))
    df['is_train'] = False  # Inicializa como False para todos
    
    for cluster_id in range(num_clusters):
        cluster_samples = df[df['cluster'] == cluster_id]
        similarities = cosine_similarity(cluster_samples['embedding'].tolist(), [cluster_centers[cluster_id]])[:, 0]
        sorted_indices = np.argsort(similarities)[::-1] 
        selected = cluster_samples.iloc[sorted_indices[:n]]
        df.loc[selected.index, 'is_train'] = True  # Marca os selecionados como True
    
    return df

def get_similar_examples(text_embedding, selected_instances, top_n=5):
    similarities = cosine_similarity([text_embedding], selected_instances['embedding'].tolist())[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return selected_instances.iloc[top_indices]

def llm_predict_few_shot(dataset_path: str, service: str, message_type: str, prompt_name: str, **kwargs):
    dataset = pd.read_csv(dataset_path)
    dataset_name = get_last_element_from_path(dataset_path)
    model_identifier = kwargs.get('model_name', kwargs.get('filename', 'error'))
    results_path = f"datasets/llm_predict/few_shot/{dataset_name[:-4]}/{model_identifier.split("/")[-1]}"
    check_file_exists(f'{results_path}/{model_identifier.split("/")[-1]}.csv')
    
    message_manager = MessageManager(message_type=message_type)
    prediction_manager = PredictionManager(service=service, **kwargs)
    user_prompt, system_prompt = get_prompts(prompt_name)
    user_prompt = Template(user_prompt)
    
    dataset = generate_embeddings(dataset)
    dataset, cluster_centers = apply_kmeans_clustering(dataset)
    dataset = select_representative_instances(dataset, cluster_centers, dataset['class'].nunique(), m = kwargs['m'])
    
    categories = '\n'.join([f"- {c}" for c in dataset['class'].unique()]).strip()
    results = []
    start_time = time.time()
    
    selected_instances = dataset[dataset['is_train']]
    dataset_unlabeled = dataset[~dataset['is_train']]
    
    for _, row in tqdm(dataset_unlabeled.iterrows(), desc="Processando textos"):
        try:
            similar_examples = get_similar_examples(row['embedding'], selected_instances, top_n=5)
            examples = '\n\n'.join([f"Text: {ex['text']}\n\nCategory: {ex['class']}" for _, ex in similar_examples.iterrows()])
            formatted_user_prompt = user_prompt.render(text=row['text'], categories=categories, examples=examples)
            message = message_manager.generate_message(formatted_user_prompt, system_prompt)
            
            valid_result = False
            attempt = 0
            while not valid_result and attempt < 10:
                attempt += 1
                temperature = kwargs['temperature'] if attempt == 1 else random.uniform(0.0, 1.0)
                result = prediction_manager.predict(message, temperature=temperature, seed=kwargs['seed'])

                if '<think>' in result and '</think>' in result:
                    # Caso normal: pega entre <think>...</think>
                    think_match = re.search(r'<think>(.*?)</think>', result, re.DOTALL)
                    think_content = think_match.group(1).strip() if think_match else None
                    # Remove o bloco <think>...</think> do texto
                    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()

                elif '</think>' in result and '<think>' not in result:
                    # Só tem </think>: assume que tudo antes dela é o think_content
                    parts = result.split('</think>')
                    think_content = parts[0].strip()
                    # O restante do texto após </think> será usado como JSON
                    result = parts[1].strip() if len(parts) > 1 else ''

                else:
                    # Nenhuma tag encontrada
                    think_content = None
                    result = result.strip()
                result = result.replace('```json', '').replace('```', '')
                
                try:
                    json_obj = json.loads(result)
                    if json_obj['category'] in dataset['class'].tolist():
                        row['predicted_class'] = json_obj['category']
                        row['explanation'] = json_obj['explanation']
                        row['confidence'] = json_obj['confidence']
                        row['reasoning'] = think_content
                        valid_result = True
                except json.JSONDecodeError:
                    pass
            
            if not valid_result:
                row['predicted_class'] = 'other'
                row['explanation'] = 'other'
                row['confidence'] = 1
                row['reasoning'] = 'other'
                
            results.append(row)
        except Exception as e:
            print(f"Erro inesperado: {e}")
    
    end_time = time.time()
    time_data = {"dataset": dataset_name, "model": model_identifier.split("/")[-1], "processing_time_seconds": end_time - start_time}
    dataset_results = pd.DataFrame(results).drop(['embedding'], axis=1)
    create_directory(results_path)
    save_json(time_data, f"{results_path}/{model_identifier.split("/")[-1]}.json")
    selected_instances.to_csv(f'{results_path}/{model_identifier.split("/")[-1]}_train.csv', index=False)
    dataset_results.to_csv(f'{results_path}/{model_identifier.split("/")[-1]}_test.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions with LLM (few-shot).")
    parser.add_argument('dataset_path', type=str, help='Caminho para o arquivo CSV do dataset')
    parser.add_argument('service', type=str, help='Serviço de predição')
    parser.add_argument('message_type', type=str, help='Tipo de mensagem para geração de mensagens')
    parser.add_argument('prompt_name', type=str, help='Nome do prompt a ser utilizado')
    parser.add_argument('--repo_id', type=str, help='ID do repositório para o modelo (opcional)')
    parser.add_argument('--filename', type=str, help='Nome do arquivo para os dados de entrada (opcional)')
    parser.add_argument('--model_name', type=str, help='Nome do modelo para a predição (opcional)')
    parser.add_argument('--m', type=float, default=0.1, help='Porcentagem dos dados que será utilizado para treinamento.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperatura para inferência LLM (opcional)')
    parser.add_argument('--seed', type=int, default=0, help='Semente para inferência LLM (opcional)')
    args = parser.parse_args()
    llm_predict_few_shot(**vars(args))