import os
import re
import json
import yaml
from glob import glob

def read_model_config(config_path: str, service:str):
    """
    Lê o arquivo de configuração dos modelos (JSON) e retorna a lista de modelos para 'ollama'.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config.get(service, [])

def get_datasets_from_directory(dataset_dir: str):
    """
    Retorna uma lista de caminhos para os arquivos CSV de dataset presentes no diretório especificado.
    """
    return glob(os.path.join(dataset_dir, "*.csv"))

def check_directory_exists(directory_path: str) -> None:
    """
    Checks if a directory exists and raises an error if it does.

    Args:
        directory_path (str): The path to the directory to check.

    Raises:
        FileExistsError: If the specified directory already exists.
    """
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        raise FileExistsError(f"The directory '{directory_path}' already exists.")
    
def check_file_exists(file_path: str) -> None:
    """
    Checks if a file exists and raises an error if it does.

    Args:
        file_path (str): The path to the file to check.

    Raises:
        FileExistsError: If the specified file already exists.
    """
    if os.path.exists(file_path) and os.path.isfile(file_path):
        raise FileExistsError(f"The file '{file_path}' already exists.")

def create_directory(directory_path: str) -> bool:
    """Create a directory if it does not exist. 

    Args:
        directory_path (str): The path of the directory to create.

    Returns:
        bool: True if the directory was successfully created or already exists, False if there was an error.
    
    Raises:
        OSError: If an error occurs while creating the directory.
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except OSError as e:
        print(f"Error creating directory: {e}")
        return False
    
def get_last_element_from_path(file_path: str) -> str:
    """Extract the last element from a given file path.

    Args:
        file_path (str): The file path from which to extract the last element.

    Returns:
        str: The last element of the path, or an empty string if the path is empty.
    """
    return os.path.basename(file_path)

import json

def save_json(data, file_path):
    """
    Saves the given dictionary to a JSON file.

    Args:
        data (dict): The dictionary containing data to be saved.
        file_path (str): The path where the JSON file should be saved.

    Returns:
        bool: True if the file was saved successfully, False otherwise.
    
    Example:
        data = {"name": "John", "age": 30}
        file_path = "output.json"
        success = save_json(data, file_path)
    """
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        return True
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False


def read_json(file_path: str) -> dict:
    """Reads a JSON file and returns its content as a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The content of the JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_prompts(prompt_name: str, path_prompts: str = "configs/prompts.yaml"):
    # Lê o arquivo YAML
    with open(path_prompts, 'r') as file:
        prompts = yaml.safe_load(file)

    # Verifica se o prompt_name está presente no dicionário
    if prompt_name in prompts:  # Verifica se a chave do prompt_name existe no YAML
        user_prompt = prompts[prompt_name].get("user_prompt", "User prompt not found.")
        system_prompt = prompts[prompt_name].get("system_prompt", "System prompt not found.")
        return user_prompt, system_prompt
    else:
        return f"Prompt '{prompt_name}' not found."

def get_value_by_key_json(file_path: str, key: str) -> str:
    """Reads a JSON file and retrieves the value associated with a given key.

    Args:
        file_path (str): The path to the JSON file.
        key (str): The key whose value needs to be fetched.

    Returns:
        any: The value associated with the key, or a message if the key is not found.
    """
    # Read the JSON file
    data = read_json(file_path)
    
    # Return the value for the given key
    return data.get(key, "Key not found")

def extract_json_block(text: str) -> str:
    """
    Extracts the content of a JSON block from a string.
    First tries to find a markdown-style ```json block```.
    If not found, falls back to extracting the first {...} block.

    Args:
        text (str): The input string that may contain a JSON block.

    Returns:
        str or None: The extracted JSON content as a string, or None if not found.
    """
    # Tenta extrair bloco entre ```json ... ```
    pattern_md = r"```json(.*?)```"
    match_md = re.search(pattern_md, text, re.DOTALL)
    if match_md:
        return match_md.group(1).strip()

    # Caso não encontre, tenta extrair o primeiro bloco {...}
    pattern_braces = r"\{.*?\}"
    match_braces = re.search(pattern_braces, text, re.DOTALL)
    if match_braces:
        return match_braces.group(0).strip()

    return None
    

def extract_think_block(text: str) -> tuple[str | None, str]:
    """
    Extracts a <think>...</think> block from the input text and returns the remaining content.

    Handles three cases:
    1. Both <think> and </think> are present → extract content between them.
    2. Only </think> is present → assume everything before it is the think block.
    3. No <think> tags found → return None for the think block.

    Args:
        text (str): The input string containing <think> block and other content.

    Returns:
        tuple:
            - think_content (str or None): The extracted think block content, or None if not found.
            - cleaned_text (str): The remaining content after removing the <think> block.
    """
    if '<think>' in text and '</think>' in text:
        # Case 1: Normal case — content is between <think> and </think>
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        think_content = match.group(1).strip() if match else None
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    elif '</think>' in text and '<think>' not in text:
        # Case 2: Only </think> is present — assume everything before it is <think>
        parts = text.split('</think>')
        think_content = parts[0].strip()
        cleaned_text = parts[1].strip() if len(parts) > 1 else ''

    else:
        # Case 3: No <think> block found
        think_content = None
        cleaned_text = text.strip()

    return think_content, cleaned_text

