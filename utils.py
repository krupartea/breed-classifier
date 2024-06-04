import json
from pathlib import Path


def load_json(path):
    path = Path(path)
    with path.open("r") as f:
        result = json.load(f)
    
    return result