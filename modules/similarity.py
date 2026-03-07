"""
similarity.py — Image Similarity and Comparison Engine
Uses deep learning embeddings to detect duplicate and similar images
No changes needed - this module was already correct
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional

try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Global model cache
_embedding_model = None


# ═══════════════════════════════════════════════════════════════════
# MODEL INITIALIZATION
# ═══════════════════════════════════════════════════════════════════

def get_embedding_model():
    """
    Load and cache MobileNetV2 model for feature extraction.
    Uses the model without top classification layer to get embeddings.
    
    Returns:
        Keras model or None if TensorFlow unavailable
    """
    global _embedding_model
    
    if _embedding_model is None and TF_AVAILABLE:
        # Load MobileNetV2 without classification layer
        # Output is 1280-dimensional feature vector
        _embedding_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg"  # Global average pooling
        )
    
    return _embedding_model


# ═══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_features(image: Image.Image) -> Optional[np.ndarray]:
    """
    Extract deep learning feature vector from image.
    
    Uses MobileNetV2 pre-trained on ImageNet to generate a 
    1280-dimensional embedding vector that captures visual features.
    
    Args:
        image: PIL Image object
        
    Returns:
        Numpy array of shape (1280,) or None if TensorFlow unavailable
    """
    if not TF_AVAILABLE:
        return None
    
    model = get_embedding_model()
    if model is None:
        return None
    
    try:
        # Resize image to 224x224 (MobileNetV2 input size)
        img_resized = image.convert("RGB").resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for MobileNetV2
        img_preprocessed = preprocess_input(img_array)
        
        # Extract features
        features = model.predict(img_preprocessed, verbose=0)
        
        # Flatten to 1D array and normalize
        features = features.flatten()
        features = features / np.linalg.norm(features)  # L2 normalization
        
        return features
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# SIMILARITY CALCULATION
# ═══════════════════════════════════════════════════════════════════

def cosine_similarity(
    features1: np.ndarray, 
    features2: np.ndarray
) -> float:
    """
    Calculate cosine similarity between two feature vectors.
    
    Cosine similarity measures the cosine of the angle between vectors.
    Returns value between -1 and 1, where:
    - 1.0 = identical vectors (same direction)
    - 0.0 = orthogonal vectors (perpendicular)
    - -1.0 = opposite vectors
    
    For normalized vectors, this is equivalent to dot product.
    
    Args:
        features1: First feature vector (normalized)
        features2: Second feature vector (normalized)
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Dot product of normalized vectors
    similarity = np.dot(features1, features2)
    
    # Ensure result is in [0, 1] range
    similarity = max(0.0, min(1.0, similarity))
    
    return float(similarity)


def euclidean_distance(
    features1: np.ndarray,
    features2: np.ndarray
) -> float:
    """
    Calculate Euclidean (L2) distance between feature vectors.
    
    Alternative to cosine similarity. Measures straight-line distance
    between points in feature space.
    
    Args:
        features1: First feature vector
        features2: Second feature vector
        
    Returns:
        Euclidean distance (lower = more similar)
    """
    return float(np.linalg.norm(features1 - features2))


# ═══════════════════════════════════════════════════════════════════
# SIMILARITY INTERPRETATION
# ═══════════════════════════════════════════════════════════════════

def interpret_similarity(similarity: float) -> str:
    """
    Convert similarity score to human-readable relationship description.
    
    Thresholds:
    - 0.95+: Exact duplicate or near-identical
    - 0.85-0.94: Very similar (same scene/subject)
    - 0.70-0.84: Similar content
    - 0.50-0.69: Some similarity
    - Below 0.50: Different images
    
    Args:
        similarity: Cosine similarity score (0.0-1.0)
        
    Returns:
        Relationship description string
    """
    if similarity >= 0.95:
        return "Exact duplicate or near-identical"
    elif similarity >= 0.85:
        return "Very similar (likely same scene)"
    elif similarity >= 0.70:
        return "Similar content"
    elif similarity >= 0.50:
        return "Some similarity"
    else:
        return "Different images"


# ═══════════════════════════════════════════════════════════════════
# DUPLICATE DETECTION
# ═══════════════════════════════════════════════════════════════════

def is_duplicate(
    similarity: float,
    threshold: float = 0.92
) -> bool:
    """
    Determine if similarity score indicates a duplicate image.
    
    Default threshold of 0.92 (92%) catches near-duplicates while
    avoiding false positives from similar but distinct images.
    
    Args:
        similarity: Cosine similarity score (0.0-1.0)
        threshold: Minimum similarity to consider duplicate (default 0.92)
        
    Returns:
        True if images are duplicates
    """
    return similarity >= threshold


# ═══════════════════════════════════════════════════════════════════
# MULTI-IMAGE COMPARISON
# ═══════════════════════════════════════════════════════════════════

def compare_images(
    images: List[Image.Image],
    names: Optional[List[str]] = None
) -> Dict:
    """
    Compare multiple images and compute pairwise similarity matrix.
    
    Extracts features from all images and computes similarity between
    every pair. Also identifies duplicate images.
    
    Args:
        images: List of PIL Image objects to compare
        names: Optional list of image names (defaults to "Image 1", "Image 2", etc.)
        
    Returns:
        Dictionary containing:
        - similarity_matrix: 2D list of similarity scores
        - pairs: List of all pairwise comparisons sorted by similarity
        - duplicates: List of duplicate pairs (similarity >= 0.92)
        - names: List of image names
        - total_images: Number of images compared
        - feature_vectors: List of extracted feature vectors
    """
    if not TF_AVAILABLE:
        return _no_tf_result(len(images))
    
    # Generate default names if not provided
    if names is None:
        names = [f"Image {i+1}" for i in range(len(images))]
    
    num_images = len(images)
    
    # ───────────────────────────────────────────────────────────────
    # STEP 1: Extract features from all images
    # ───────────────────────────────────────────────────────────────
    feature_vectors = []
    
    for i, image in enumerate(images):
        features = extract_features(image)
        
        if features is None:
            # Feature extraction failed - use random vector
            print(f"Warning: Feature extraction failed for {names[i]}")
            features = np.random.randn(1280)
            features = features / np.linalg.norm(features)
        
        feature_vectors.append(features)
    
    # ───────────────────────────────────────────────────────────────
    # STEP 2: Compute similarity matrix
    # ───────────────────────────────────────────────────────────────
    similarity_matrix = []
    
    for i in range(num_images):
        row = []
        for j in range(num_images):
            if i == j:
                # Same image - similarity is 1.0
                similarity = 1.0
            else:
                # Different images - compute cosine similarity
                similarity = cosine_similarity(
                    feature_vectors[i],
                    feature_vectors[j]
                )
            row.append(similarity)
        similarity_matrix.append(row)
    
    # ───────────────────────────────────────────────────────────────
    # STEP 3: Generate pairwise comparisons
    # ───────────────────────────────────────────────────────────────
    pairs = []
    duplicates = []
    
    for i in range(num_images):
        for j in range(i + 1, num_images):  # Only upper triangle (avoid duplicates)
            similarity = similarity_matrix[i][j]
            similarity_percent = round(similarity * 100, 1)
            
            pair_info = {
                "img1": names[i],
                "img2": names[j],
                "similarity": similarity_percent,
                "similarity_raw": similarity,
                "relationship": interpret_similarity(similarity)
            }
            
            pairs.append(pair_info)
            
            # Check if this is a duplicate
            if is_duplicate(similarity):
                duplicates.append(pair_info)
    
    # Sort pairs by similarity (highest first)
    pairs.sort(key=lambda x: x["similarity_raw"], reverse=True)
    
    # ───────────────────────────────────────────────────────────────
    # STEP 4: Return complete comparison results
    # ───────────────────────────────────────────────────────────────
    return {
        "similarity_matrix": similarity_matrix,
        "pairs": pairs,
        "duplicates": duplicates,
        "names": names,
        "total_images": num_images,
        "feature_vectors": feature_vectors,
    }


# ═══════════════════════════════════════════════════════════════════
# SINGLE PAIR COMPARISON
# ═══════════════════════════════════════════════════════════════════

def compare_two_images(
    image1: Image.Image,
    image2: Image.Image,
    name1: str = "Image 1",
    name2: str = "Image 2"
) -> Dict:
    """
    Compare two images and return similarity score.
    
    Convenience function for comparing just two images without
    creating a full similarity matrix.
    
    Args:
        image1: First PIL Image
        image2: Second PIL Image
        name1: Name for first image
        name2: Name for second image
        
    Returns:
        Dictionary with similarity info:
        - similarity: Percentage similarity (0-100)
        - similarity_raw: Raw similarity score (0.0-1.0)
        - relationship: Interpretation of similarity
        - is_duplicate: Boolean duplicate flag
    """
    if not TF_AVAILABLE:
        return {
            "similarity": 0,
            "similarity_raw": 0.0,
            "relationship": "TensorFlow not available",
            "is_duplicate": False,
            "error": "TensorFlow not installed"
        }
    
    # Extract features
    features1 = extract_features(image1)
    features2 = extract_features(image2)
    
    if features1 is None or features2 is None:
        return {
            "similarity": 0,
            "similarity_raw": 0.0,
            "relationship": "Feature extraction failed",
            "is_duplicate": False,
            "error": "Could not extract features"
        }
    
    # Calculate similarity
    similarity = cosine_similarity(features1, features2)
    similarity_percent = round(similarity * 100, 1)
    
    return {
        "similarity": similarity_percent,
        "similarity_raw": similarity,
        "relationship": interpret_similarity(similarity),
        "is_duplicate": is_duplicate(similarity),
        "img1_name": name1,
        "img2_name": name2,
    }


# ═══════════════════════════════════════════════════════════════════
# FIND SIMILAR IMAGES
# ═══════════════════════════════════════════════════════════════════

def find_similar_images(
    query_image: Image.Image,
    image_database: List[Image.Image],
    database_names: Optional[List[str]] = None,
    top_k: int = 5,
    min_similarity: float = 0.5
) -> List[Dict]:
    """
    Find images similar to query image from a database.
    
    Use case: Given one image, find the most similar images from a collection.
    Example: "Find images similar to this surveillance photo"
    
    Args:
        query_image: Image to search for
        image_database: List of images to search through
        database_names: Optional names for database images
        top_k: Maximum number of results to return
        min_similarity: Minimum similarity threshold (0.0-1.0)
        
    Returns:
        List of dictionaries with similar images and scores
    """
    if not TF_AVAILABLE:
        return []
    
    # Generate default names
    if database_names is None:
        database_names = [f"Image {i+1}" for i in range(len(image_database))]
    
    # Extract query features
    query_features = extract_features(query_image)
    if query_features is None:
        return []
    
    # Compare query to each database image
    results = []
    
    for i, db_image in enumerate(image_database):
        db_features = extract_features(db_image)
        
        if db_features is None:
            continue
        
        similarity = cosine_similarity(query_features, db_features)
        
        # Only include if meets minimum threshold
        if similarity >= min_similarity:
            results.append({
                "name": database_names[i],
                "index": i,
                "similarity": round(similarity * 100, 1),
                "similarity_raw": similarity,
                "relationship": interpret_similarity(similarity)
            })
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x["similarity_raw"], reverse=True)
    
    # Return top K results
    return results[:top_k]


# ═══════════════════════════════════════════════════════════════════
# CLUSTERING SIMILAR IMAGES
# ═══════════════════════════════════════════════════════════════════

def cluster_similar_images(
    images: List[Image.Image],
    names: Optional[List[str]] = None,
    similarity_threshold: float = 0.75
) -> List[List[int]]:
    """
    Group similar images into clusters.
    
    Uses similarity threshold to group images that are similar to each other.
    Useful for organizing large collections of images.
    
    Args:
        images: List of PIL Images
        names: Optional image names
        similarity_threshold: Minimum similarity to cluster together (default 0.75)
        
    Returns:
        List of clusters, where each cluster is a list of image indices
    """
    if not TF_AVAILABLE:
        return [[i] for i in range(len(images))]  # Each image in own cluster
    
    num_images = len(images)
    
    # Get similarity matrix
    comparison = compare_images(images, names)
    sim_matrix = comparison["similarity_matrix"]
    
    # Track which images have been clustered
    clustered = [False] * num_images
    clusters = []
    
    for i in range(num_images):
        if clustered[i]:
            continue
        
        # Start new cluster with this image
        cluster = [i]
        clustered[i] = True
        
        # Find all images similar to this one
        for j in range(num_images):
            if i == j or clustered[j]:
                continue
            
            if sim_matrix[i][j] >= similarity_threshold:
                cluster.append(j)
                clustered[j] = True
        
        clusters.append(cluster)
    
    return clusters


# ═══════════════════════════════════════════════════════════════════
# ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════

def _no_tf_result(num_images: int) -> Dict:
    """
    Return empty result when TensorFlow is not available.
    
    Args:
        num_images: Number of images that were attempted
        
    Returns:
        Empty comparison result dictionary
    """
    # Create identity matrix (all 1s on diagonal, 0s elsewhere)
    similarity_matrix = [
        [1.0 if i == j else 0.0 for j in range(num_images)]
        for i in range(num_images)
    ]
    
    return {
        "similarity_matrix": similarity_matrix,
        "pairs": [],
        "duplicates": [],
        "names": [f"Image {i+1}" for i in range(num_images)],
        "total_images": num_images,
        "feature_vectors": [],
        "error": "TensorFlow not installed - similarity analysis unavailable"
    }


# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_duplicate_count(comparison_result: Dict) -> int:
    """
    Get number of duplicate pairs found.
    
    Args:
        comparison_result: Output from compare_images()
        
    Returns:
        Number of duplicate pairs
    """
    return len(comparison_result.get("duplicates", []))


def get_most_similar_pair(comparison_result: Dict) -> Optional[Dict]:
    """
    Get the most similar pair of images.
    
    Args:
        comparison_result: Output from compare_images()
        
    Returns:
        Dictionary with pair info, or None if no pairs
    """
    pairs = comparison_result.get("pairs", [])
    
    if not pairs:
        return None
    
    return pairs[0]  # Already sorted by similarity


def get_least_similar_pair(comparison_result: Dict) -> Optional[Dict]:
    """
    Get the least similar pair of images.
    
    Args:
        comparison_result: Output from compare_images()
        
    Returns:
        Dictionary with pair info, or None if no pairs
    """
    pairs = comparison_result.get("pairs", [])
    
    if not pairs:
        return None
    
    return pairs[-1]  # Last in sorted list


def format_similarity_report(comparison_result: Dict) -> str:
    """
    Generate a formatted text report of similarity analysis.
    
    Args:
        comparison_result: Output from compare_images()
        
    Returns:
        Multi-line formatted report string
    """
    lines = []
    
    lines.append("=" * 60)
    lines.append("IMAGE SIMILARITY ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    total = comparison_result.get("total_images", 0)
    lines.append(f"Total Images Analyzed: {total}")
    lines.append(f"Total Pairs Compared: {len(comparison_result.get('pairs', []))}")
    lines.append(f"Duplicates Found: {get_duplicate_count(comparison_result)}")
    lines.append("")
    
    # Most similar pair
    most_similar = get_most_similar_pair(comparison_result)
    if most_similar:
        lines.append("Most Similar Pair:")
        lines.append(f"  {most_similar['img1']} ↔ {most_similar['img2']}")
        lines.append(f"  Similarity: {most_similar['similarity']}%")
        lines.append(f"  Relationship: {most_similar['relationship']}")
        lines.append("")
    
    # List duplicates
    duplicates = comparison_result.get("duplicates", [])
    if duplicates:
        lines.append("Duplicate Images:")
        for dup in duplicates:
            lines.append(f"  • {dup['img1']} ↔ {dup['img2']} ({dup['similarity']}%)")
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)