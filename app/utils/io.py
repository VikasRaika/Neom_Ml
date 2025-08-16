import json
from pathlib import Path
from typing import Dict, List

def load_users(path: str = "sample_data/synthetic_users.json") -> Dict[str, List[float]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Users file not found at {p.resolve()}")
    raw = p.read_text().strip()
    if not raw:
        raise ValueError(f"Users file is empty: {p.resolve()}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {p.resolve()}: {e}") from e
    return {u["id"]: u["psychometrics"] for u in data}
