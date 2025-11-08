import json
from typing import List, Dict

def load_truthfulqa(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        return json.load(f)

def to_bool_label(z: int) -> str:
    return 'True' if int(z) == 1 else 'False'
