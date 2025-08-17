import json
from pathlib import Path
from typing import Dict, Tuple, List, Any
import os, json
from pathlib import Path


TRAIT_ALIASES = ("traits", "psychometrics", "big5", "vector")

def _coerce_traits(rec: dict) -> List[float] | None:
    """Find a 5-D numeric vector under one of the known keys and coerce to floats."""
    for key in TRAIT_ALIASES:
        vec = rec.get(key)
        if isinstance(vec, list) and len(vec) == 5:
            try:
                return [float(x) for x in vec]
            except (TypeError, ValueError):
                return None
    return None

def load_users(path: str | None = None) -> Dict[str, dict]:
    """
    Load users JSON and return a dict keyed by user id with normalized schema:
      { "<id>": {"traits": [f,f,f,f,f], ...}, ... }

    Accepts any of:
      - {"users": [ { "id": "...", "traits": [...] }, ... ]}
      - [ { "id": "...", "traits": [...] }, ... ]
      - already keyed: { "user_1": {...}, "user_2": {...} }
    """
    p = Path(path or os.getenv("USERS_PATH", "sample_data/synthetic_users.json"))
    data = json.loads(p.read_text())

    # 1) If already keyed, normalize trait key.
    if isinstance(data, dict) and not isinstance(data.get("users"), list):
        users_by_id = {}
        for uid, rec in data.items():
            if not isinstance(rec, dict): 
                continue
            traits = _coerce_traits(rec)
            if traits is None:
                continue
            r = dict(rec)
            r["traits"] = traits
            users_by_id[uid] = r
        if users_by_id:
            return users_by_id
        # fall through if nothing valid

    # 2) If wrapped list or bare list, collect records.
    if isinstance(data, dict) and isinstance(data.get("users"), list):
        records = data["users"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Unsupported users JSON structure")

    users_by_id: Dict[str, dict] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        uid = rec.get("id") or rec.get("user_id")
        if not uid:
            continue
        traits = _coerce_traits(rec)
        if traits is None:
            continue
        r = dict(rec)
        r["traits"] = traits
        users_by_id[uid] = r

    if not users_by_id:
        raise ValueError("No valid users with 5-D numeric traits found in users JSON")

    return users_by_id

def get_default_user_pair(users: Dict[str, dict]) -> Tuple[str, str]:
    """
    Return a deterministic pair of user IDs that both have valid 5-D traits.
    """
    valid_ids = sorted(
        uid for uid, rec in users.items()
        if isinstance(rec.get("traits"), list) and len(rec["traits"]) == 5
    )
    if len(valid_ids) < 2:
        raise ValueError("Need at least two users with 5-D traits")
    return valid_ids[0], valid_ids[1]

def list_user_ids(users: Dict[str, dict]) -> List[str]:
    return sorted(users.keys())
