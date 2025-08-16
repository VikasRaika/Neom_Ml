import numpy as np
from typing import List

def make_topic_vector(topics: List[str], weights: List[float]) -> np.ndarray:
    """
    Represent the conversation topics by their normalized weights.
    Assumes weights correspond 1:1 with topics.
    """
    arr = np.array(weights, dtype=float)
    s = arr.sum()
    if s > 0:
        arr = arr / s
    return arr

def fuse_vectors(psychometrics: List[float],
                 topic_vec: np.ndarray,
                 alpha: float = 0.5,
                 beta: float = 0.5) -> np.ndarray:
    """
    Concatenate weighted psychometrics and topic vector:
      combined = [alpha * psychometrics, beta * topic_vec]
    """
    p = alpha * np.array(psychometrics, dtype=float)
    t = beta * topic_vec.astype(float)
    return np.concatenate([p, t], axis=0)
