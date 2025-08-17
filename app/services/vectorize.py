# app/services/vectorize.py
from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# one shared encoder; MiniLM returns 384-d vectors
_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def make_topic_vector(topics: List[str]) -> np.ndarray:
    """
    Encode each topic phrase; return a single 384-d mean-pooled, L2-normalized embedding.
    """
    topics = [t.strip() for t in topics if isinstance(t, str) and t.strip()]
    if not topics:
        # zero vector (no topics) — caller can decide how to treat it
        return np.zeros(384, dtype=np.float32)

    embs = _encoder.encode(topics, normalize_embeddings=True)
    vec = np.mean(np.asarray(embs, dtype=np.float32), axis=0)
    # re-normalize mean-pooled vector to unit length (unless zero)
    n = np.linalg.norm(vec)
    return (vec / n) if n > 0 else vec


def fuse_vectors(user_traits: np.ndarray, topic_vec: np.ndarray, *, topic_scale: float = 1.0) -> np.ndarray:
    """
    Build a fused feature for similarity:
    - user_traits: (5,)
    - topic_vec:   (384,)  (already L2-normalized)
    - topic_scale: user-specific interest multiplier
    - We DO NOT add across spaces; we concatenate [traits || scaled_topic].
    """
    user_traits = np.asarray(user_traits, dtype=np.float32).reshape(-1)
    topic_vec = np.asarray(topic_vec, dtype=np.float32).reshape(-1)

    if topic_vec.shape[0] != 384:
        raise ValueError(f"topic_vec must be 384-d, got {topic_vec.shape}")

    scaled_topic = topic_vec * float(topic_scale)
    fused = np.concatenate([user_traits, scaled_topic], axis=0)
    return fused


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
