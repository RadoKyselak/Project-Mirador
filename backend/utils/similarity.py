from typing import List
from math import sqrt

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = 0.0
    mag_vec1_sq = 0.0
    mag_vec2_sq = 0.0

    for v1, v2 in zip(vec1, vec2):
        dot_product += v1 * v2
        mag_vec1_sq += v1**2
        mag_vec2_sq += v2**2

    mag_vec1 = sqrt(mag_vec1_sq)
    mag_vec2 = sqrt(mag_vec2_sq)

    if mag_vec1 == 0 or mag_vec2 == 0:
        return 0.0

    return dot_product / (mag_vec1 * mag_vec2)
