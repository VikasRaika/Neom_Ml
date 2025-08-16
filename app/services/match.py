import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Standard cosine similarity in [0,1] for non-negative vectors.
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    sim = float(np.dot(a, b) / denom)
    # keep it in [0,1] for interpretability (optional clip)
    return max(0.0, min(1.0, sim))

def interpret(score: float) -> str:
    """
    Simple buckets for a human-readable label.
    """
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"
