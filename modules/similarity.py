"""
similarity.py — Multi-Image Similarity & Forensic Comparison Engine
Day 4 of build plan

Uses deep learning feature embeddings + cosine similarity to:
- Compare multiple uploaded images
- Detect duplicates and near-duplicates
- Find similarity scores between image pairs
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
from itertools import combinations

# -------------------------------------------------------------------
# We use TensorFlow MobileNetV2 to extract image embeddings
# Embeddings = high-dimensional feature vectors that represent image content
# Similar images → similar vectors → high cosine similarity
# -------------------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras import Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# -------------------------------------------------------------------
# Embedding model loader (strip final classification layer)
# -------------------------------------------------------------------
_embedding_model = None

def get_embedding_model():
    """
    Load MobileNetV2 without the top classification layer.
    Output is a 1280-dim feature vector per image.
    """
    global _embedding_model
    if _embedding_model is None and TF_AVAILABLE:
        base = MobileNetV2(weights="imagenet", include_top=False,
                           pooling="avg", input_shape=(224, 224, 3))
        _embedding_model = base
    return _embedding_model


# -------------------------------------------------------------------
# EXTRACT EMBEDDING FROM SINGLE IMAGE
# -------------------------------------------------------------------
def extract_embedding(image: Image.Image) -> np.ndarray:
    """
    Convert PIL image → 1280-dim feature vector using MobileNetV2.
    
    Returns:
        numpy array of shape (1280,) — the image embedding
    """
    if not TF_AVAILABLE:
        # Fallback: use simple pixel histogram as pseudo-embedding
        return _histogram_embedding(image)

    model = get_embedding_model()

    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    embedding = model.predict(arr, verbose=0)[0]  # shape: (1280,)
    return embedding


def _histogram_embedding(image: Image.Image) -> np.ndarray:
    """
    Fallback: compute color histogram as image embedding.
    Less accurate but works without TensorFlow.
    """
    img = image.convert("RGB").resize((64, 64))
    arr = np.array(img)
    # Compute histogram for each channel, normalize
    hist = []
    for channel in range(3):
        h, _ = np.histogram(arr[:, :, channel], bins=32, range=(0, 256))
        hist.extend(h / h.sum())
    return np.array(hist, dtype=np.float32)


# -------------------------------------------------------------------
# COMPUTE SIMILARITY BETWEEN TWO EMBEDDINGS
# -------------------------------------------------------------------
def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two image embeddings.
    
    Returns:
        float between 0 and 1 (1 = identical, 0 = completely different)
    """
    if SKLEARN_AVAILABLE:
        sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    else:
        # Manual cosine similarity
        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        sim = dot / (norm + 1e-8)
    return float(np.clip(sim, 0, 1))


# -------------------------------------------------------------------
# MAIN: COMPARE MULTIPLE IMAGES
# -------------------------------------------------------------------
def compare_images(images: List[Image.Image], names: List[str] = None) -> dict:
    """
    Compare a list of images and compute pairwise similarity.

    Args:
        images: list of PIL Images
        names: optional list of names for each image

    Returns:
        dict with:
            - embeddings: list of embedding vectors
            - similarity_matrix: NxN similarity scores
            - pairs: list of (img1_name, img2_name, similarity, relationship)
            - duplicates: list of duplicate pairs
            - summary: text summary
    """
    n = len(images)
    if n < 2:
        return {
            "embeddings": [],
            "similarity_matrix": [],
            "pairs": [],
            "duplicates": [],
            "summary": "Upload at least 2 images to compare."
        }

    if names is None:
        names = [f"Image {i+1}" for i in range(n)]

    # Extract embeddings for all images
    embeddings = []
    for img in images:
        emb = extract_embedding(img)
        embeddings.append(emb)

    # Build NxN similarity matrix
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i][j] = 1.0
            else:
                sim_matrix[i][j] = compute_similarity(embeddings[i], embeddings[j])

    # Build pairwise comparison list
    pairs = []
    duplicates = []

    for i, j in combinations(range(n), 2):
        sim = sim_matrix[i][j]
        relationship = _classify_relationship(sim)

        pair = {
            "img1": names[i],
            "img2": names[j],
            "img1_idx": i,
            "img2_idx": j,
            "similarity": round(sim * 100, 1),
            "relationship": relationship,
        }
        pairs.append(pair)

        if sim >= 0.92:
            duplicates.append(pair)

    # Sort by similarity descending
    pairs.sort(key=lambda x: x["similarity"], reverse=True)

    summary = _build_summary(pairs, duplicates, n)

    return {
        "embeddings": embeddings,
        "similarity_matrix": sim_matrix.tolist(),
        "names": names,
        "pairs": pairs,
        "duplicates": duplicates,
        "total_images": n,
        "summary": summary,
    }


# -------------------------------------------------------------------
# CLASSIFY RELATIONSHIP BASED ON SIMILARITY SCORE
# -------------------------------------------------------------------
def _classify_relationship(sim: float) -> str:
    """
    Map similarity score to human-readable relationship type.
    These thresholds are calibrated for MobileNetV2 embeddings.
    """
    if sim >= 0.98:
        return "🔴 Exact Duplicate"
    elif sim >= 0.92:
        return "🟠 Near Duplicate"
    elif sim >= 0.80:
        return "🟡 Very Similar"
    elif sim >= 0.65:
        return "🟢 Similar Content"
    elif sim >= 0.45:
        return "🔵 Partially Related"
    else:
        return "⚪ Different"


# -------------------------------------------------------------------
# BUILD SUMMARY TEXT
# -------------------------------------------------------------------
def _build_summary(pairs, duplicates, n_images):
    lines = [f"Analyzed {n_images} images ({len(pairs)} comparisons made)"]

    if duplicates:
        lines.append(f"⚠️ Found {len(duplicates)} duplicate/near-duplicate pair(s)!")
        for d in duplicates:
            lines.append(f"  • {d['img1']} ↔ {d['img2']} ({d['similarity']}% similar)")
    else:
        lines.append("✅ No duplicate images detected.")

    if pairs:
        most_similar = pairs[0]
        least_similar = pairs[-1]
        lines.append(f"Most similar pair: {most_similar['img1']} & {most_similar['img2']} "
                     f"({most_similar['similarity']}%)")
        lines.append(f"Most different pair: {least_similar['img1']} & {least_similar['img2']} "
                     f"({least_similar['similarity']}%)")

    return "\n".join(lines)


# -------------------------------------------------------------------
# GENERATE SIMILARITY HEATMAP DATA (for Plotly visualization)
# -------------------------------------------------------------------
def get_heatmap_data(similarity_result: dict) -> dict:
    """
    Prepare data for a Plotly heatmap visualization.
    """
    return {
        "z": similarity_result["similarity_matrix"],
        "x": similarity_result["names"],
        "y": similarity_result["names"],
        "colorscale": "RdYlGn",
        "zmin": 0,
        "zmax": 1,
    }