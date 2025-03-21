import json
import yaml
import os

def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def json_to_dict(json_str):
    """
    Convert a JSON string to a Python dictionary.
    
    Args:
        json_str (str): A string containing JSON data
        
    Returns:
        dict: The parsed JSON data as a Python dictionary
    """
    # Remove any markdown code block markers if present
    json_str = json_str.strip('`').strip('json').strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None