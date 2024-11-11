import os
import json
import numpy as np
import pandas as pd
from typing import Any, Union

def create_directory(directory_path: str) -> bool:
    """Create a directory if it does not exist. Raise an error if it already exists.

    Args:
        directory_path (str): The path of the directory to create.

    Returns:
        bool: True if the directory was successfully created.
    
    Raises:
        FileExistsError: If the directory already exists.
        OSError: If an error occurs while creating the directory.
    """
    # if os.path.exists(directory_path):
    #     raise FileExistsError(f"The directory '{directory_path}' already exists.")
    
    try:
        os.makedirs(directory_path)
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

def get_prompts(prompt_name: str, path_prompts: str = "configs/prompts.json"):
    prompts = read_json(path_prompts)
    
    # Verifica se o prompt_name está presente no dicionário
    if prompt_name in prompts:
        user_prompt = prompts[prompt_name].get("user_prompt", "User prompt not found.")
        system_prompt = prompts[prompt_name].get("system_prompt", "System prompt not found.")
        return user_prompt, system_prompt
    else:
        return f"Prompt '{prompt_name}' not found."