# app/services/match.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from app.services.vectorize import make_topic_vector, fuse_vectors, cosine


def load_users(path: str | None = None) -> Dict[str, dict]:
    """
    Load users from JSON and return a dict keyed by user id.
    Accepts either:
      - {"users": [ { "id": "...", "traits": [...] }, ... ]}
      - [ { "id": "...", "traits": [...] }, ... ]
      - or keyed already: { "user_1": {...}, "user_2": {...} }
    """
    p = Path(path or os.getenv("USERS_PATH", "sample_data/synthetic_users.json"))
    data = json.loads(p.read_text())

    # Already keyed?
    if isinstance(data, dict) and any(isinstance(v, dict) and "traits" in v for v in data.values()):
        return data

    # Wrapped list?
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
        users_by_id[uid] = rec

    if not users_by_id:
        raise ValueError("No users with 'id'/'user_id' found in users JSON")

    return users_by_id


def get_default_user_pair(users: Dict[str, dict]) -> Tuple[str, str]:
    """
    Pick the first two users that have 5-D 'traits'.
    """
    ids = [uid for uid, rec in users.items() if isinstance(rec.get("traits"), list) and len(rec["traits"]) == 5]
    if len(ids) < 2:
        raise RuntimeError("Need at least two users with 5-D traits in users JSON")
    return ids[0], ids[1]


def compute_match(user1_traits: np.ndarray,
                  user2_traits: np.ndarray,
                  topic_vec: np.ndarray,
                  *,
                  scale1: float,
                  scale2: float) -> dict:
    """
    Build per-user fused vectors with user-specific topic scales and compute cosine similarity.
    """
    fused1 = fuse_vectors(user1_traits, topic_vec, topic_scale=scale1)
    fused2 = fuse_vectors(user2_traits, topic_vec, topic_scale=scale2)

    sim = cosine(fused1, fused2)

    if   sim >= 0.80: label = "high"
    elif sim >= 0.55: label = "medium"
    else:             label = "low"

    return {
        "score": round(sim, 4),
        "label": label,
        "detail": f"combined_dim={fused1.shape[0]}; scales=({scale1},{scale2})",
    }
