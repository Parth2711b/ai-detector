"""
features.py

This file contains ONLY analysis logic.
It does NOT know about FastAPI, requests, or files.
It only processes raw audio bytes.
"""

from collections import Counter
import math
import numpy as np

# -------------------------------------------------
# Feature state object
# Keeps running statistics across chunks
# -------------------------------------------------
class FeatureState:
    def __init__(self):
        self.total_bytes = 0
        self.non_zero_bytes = 0
        self.byte_frequency = Counter()


# -------------------------------------------------
# Analyze a single chunk
# -------------------------------------------------
def analyze_chunk(chunk: bytes, state: FeatureState) -> None:
    """
    Processes a chunk and updates state incrementally.
    Does NOT store the chunk.
    """

    state.total_bytes += len(chunk)

    for b in chunk:
        if b != 0:
            state.non_zero_bytes += 1
        state.byte_frequency[b] += 1


# -------------------------------------------------
# Helper: Shannon entropy
# -------------------------------------------------
def calculate_entropy(freq: Counter, total: int) -> float:
    """
    Measures randomness of byte distribution.
    """

    if total == 0:
        return 0.0

    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


# -------------------------------------------------
# Finalize features after all chunks are processed
# -------------------------------------------------
def finalize_features(state: FeatureState) -> dict:
    """
    Converts accumulated state into final features.
    """

    entropy = calculate_entropy(state.byte_frequency, state.total_bytes)

    return {
        "total_bytes": state.total_bytes,
        "non_zero_bytes": state.non_zero_bytes,
        "zero_bytes": state.total_bytes - state.non_zero_bytes,
        "entropy": entropy,
    }


# -------------------------------------------------
# Public API: Extract features from raw bytes
# -------------------------------------------------
def extract_features(audio_bytes: bytes) -> np.ndarray:
    """
    Extracts features from raw audio bytes.
    Returns a fixed-length numeric feature vector.
    """

    state = FeatureState()
    chunk_size = 8192

    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        analyze_chunk(chunk, state)

    features_dict = finalize_features(state)

    # Fixed-order feature vector (important for ML)
    feature_vector = np.array([
        features_dict["total_bytes"],
        features_dict["non_zero_bytes"],
        features_dict["zero_bytes"],
        features_dict["entropy"],
    ], dtype=np.float32)

    return feature_vector
