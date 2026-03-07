"""
scene.py — Scene Classification Engine
PRODUCTION VERSION: Conservative scene classification with realistic risk scores
FIX #2: Uses ImageNet appropriately - doesn't over-rely on MobileNetV2 for security scenes
FIX #5: Base risk scores significantly lowered
"""

import numpy as np
from PIL import Image
from typing import Optional, Tuple, List

try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

_mobilenet_model = None


def get_model():
    """Load and cache MobileNetV2 model"""
    global _mobilenet_model
    if _mobilenet_model is None and TF_AVAILABLE:
        _mobilenet_model = MobileNetV2(weights="imagenet", include_top=True)
    return _mobilenet_model


# ═══════════════════════════════════════════════════════════════════
# SCENE DEFINITIONS WITH REALISTIC BASE RISK SCORES
# FIX #5: All base scores significantly reduced
# ═══════════════════════════════════════════════════════════════════

SCENE_DEFINITIONS = {
    # High-risk scenes (only when truly dangerous objects detected)
    "violence":       ("🚨", 30, "Potential violence or altercation detected"),
    "accident":       ("🚗💥", 25, "Vehicle accident scene"),
    "fire_emergency": ("🔥", 35, "Fire or smoke emergency"),
    "robbery":        ("🔫", 40, "Armed threat situation detected"),
    "weapon_threat":  ("⚔️", 35, "Weapons detected in scene"),
    
    # Medium-risk scenes
    "road":           ("🛣️", 12, "Traffic or road environment"),
    "crowded_area":   ("👥", 15, "Crowded public space"),
    "warehouse":      ("🏭", 12, "Industrial or warehouse facility"),
    "parking":        ("🅿️", 10, "Parking lot or garage"),
    "night_scene":    ("🌙", 12, "Night-time or low-light scene"),
    
    # Low-risk scenes
    "hospital":       ("🏥", 8, "Medical or healthcare facility"),
    "office":         ("🏢", 5, "Professional office workspace"),
    "classroom":      ("🎓", 5, "Educational setting"),
    "kitchen":        ("🍳", 8, "Kitchen or food preparation area"),
    "outdoor":        ("🌳", 6, "Outdoor natural environment"),
    "indoor":         ("🏠", 5, "Indoor residential space"),
    
    # Default
    "unknown":        ("❓", 5, "Scene type could not be determined"),
}


# Keyword mappings for ImageNet-to-scene translation
KEYWORD_SCENE_MAP = {
    "road": [
        "traffic", "street", "highway", "crosswalk", "road", 
        "lane", "cab", "taxi", "intersection"
    ],
    "office": [
        "desk", "monitor", "keyboard", "computer", "laptop", 
        "chair", "office", "cubicle", "workstation"
    ],
    "hospital": [
        "stretcher", "stethoscope", "syringe", "hospital", 
        "ambulance", "medical", "clinic"
    ],
    "classroom": [
        "classroom", "blackboard", "school", "lecture", 
        "whiteboard", "desk", "student"
    ],
    "outdoor": [
        "tree", "grass", "sky", "mountain", "park", "garden", 
        "forest", "beach", "lake", "river"
    ],
    "kitchen": [
        "kitchen", "stove", "refrigerator", "oven", "microwave", 
        "sink", "counter"
    ],
    "crowded_area": [
        "crowd", "people", "mall", "market", "stadium", 
        "concert", "festival", "audience"
    ],
    "parking": [
        "parking", "garage", "lot", "car park"
    ],
    "warehouse": [
        "warehouse", "factory", "industrial", "storage", 
        "shelf", "forklift", "pallet"
    ],
    "indoor": [
        "room", "living", "bedroom", "hallway", "interior", 
        "sofa", "couch", "furniture"
    ],
}


# ═══════════════════════════════════════════════════════════════════
# MAIN SCENE CLASSIFICATION FUNCTION
# ═══════════════════════════════════════════════════════════════════

def classify_scene(
    image: Image.Image, 
    detection_result: Optional[dict] = None
) -> dict:
    """
    Classify the scene type using multiple signals.
    
    FIX #2: MobileNetV2 is used appropriately - only for general scene understanding
    FIX #5: Base risk scores are much lower and more realistic
    
    Process:
    1. Analyze detected objects (highest priority)
    2. Use ImageNet predictions (supporting evidence)
    3. Analyze brightness (night detection)
    4. Combine signals with weighted voting
    
    Args:
        image: PIL Image to classify
        detection_result: Optional output from detect_objects()
        
    Returns:
        Dictionary containing:
        - scene: Scene type identifier
        - confidence: Classification confidence (0-100)
        - scene_emoji: Emoji representing the scene
        - description: Human-readable description
        - base_risk_score: Inherent risk of this scene type
        - top_predictions: Top ImageNet predictions
        - is_dangerous: Boolean flag for high-risk scenes
        - source: Which signal(s) determined the classification
    """
    classification_signals = []

    # ───────────────────────────────────────────────────────────────
    # SIGNAL 1: Object-based classification (HIGHEST PRIORITY)
    # ───────────────────────────────────────────────────────────────
    if detection_result:
        object_signal = _classify_from_objects(detection_result)
        if object_signal:
            classification_signals.append(object_signal)

    # ───────────────────────────────────────────────────────────────
    # SIGNAL 2: ImageNet deep learning classification
    # FIX #2: Used for general scene understanding, not security detection
    # ───────────────────────────────────────────────────────────────
    if TF_AVAILABLE:
        imagenet_signal = _classify_from_imagenet(image)
        if imagenet_signal:
            classification_signals.append(imagenet_signal)

    # ───────────────────────────────────────────────────────────────
    # SIGNAL 3: Brightness analysis (night detection)
    # ───────────────────────────────────────────────────────────────
    brightness_signal = _analyze_brightness(image)
    if brightness_signal:
        classification_signals.append(brightness_signal)

    # ───────────────────────────────────────────────────────────────
    # SIGNAL COMBINATION: Weighted voting system
    # ───────────────────────────────────────────────────────────────
    final_scene, final_confidence, source_info = _combine_signals(
        classification_signals
    )
    
    # Get scene metadata
    scene_info = SCENE_DEFINITIONS.get(
        final_scene, 
        SCENE_DEFINITIONS["unknown"]
    )
    emoji, base_risk, description = scene_info
    
    # Determine if scene is inherently dangerous
    dangerous_scenes = [
        "violence", "robbery", "weapon_threat", 
        "fire_emergency", "accident"
    ]
    
    return {
        "scene": final_scene,
        "confidence": final_confidence,
        "scene_emoji": emoji,
        "description": description,
        "base_risk_score": base_risk,
        "top_predictions": _get_top_imagenet_predictions(image) if TF_AVAILABLE else [],
        "is_dangerous": final_scene in dangerous_scenes,
        "source": source_info,
    }


# ═══════════════════════════════════════════════════════════════════
# CLASSIFICATION SIGNAL EXTRACTORS
# ═══════════════════════════════════════════════════════════════════

def _classify_from_objects(detection_result: dict) -> Optional[Tuple[str, float, str]]:
    """
    Classify scene based on detected objects and their relationships.
    
    FIX #1 & #4: Only uses high-confidence real detections
    No context engine inferences, no weapon guessing
    
    Args:
        detection_result: Output from detect_objects()
        
    Returns:
        Tuple of (scene_name, confidence, source_description) or None
    """
    category_counts = detection_result.get("category_counts", {})
    object_counts = detection_result.get("object_counts", {})
    object_keys = [k.lower() for k in object_counts.keys()]
    weapons = detection_result.get("weapons_found", [])
    fires = detection_result.get("fire_found", [])
    
    # ───────────────────────────────────────────────────────────────
    # CRITICAL: Weapons (only real YOLO detections with 75%+ confidence)
    # FIX #1: No inferred weapons from context engine
    # ───────────────────────────────────────────────────────────────
    if weapons:
        # Filter to only high-confidence real weapons
        real_weapons = [
            w for w in weapons 
            if w.get("confidence", 0) >= 75
        ]
        
        if real_weapons:
            weapon_names = " ".join(w["label"].lower() for w in real_weapons)
            
            # Firearms = robbery scene
            if any(gun in weapon_names for gun in ["gun", "rifle", "handgun", "pistol", "firearm", "shotgun"]):
                return ("robbery", 85, "Firearm detected by YOLO")
            
            # Other weapons = weapon threat
            return ("weapon_threat", 80, "Weapon detected by YOLO")
    
    # ───────────────────────────────────────────────────────────────
    # EMERGENCY: Fire/smoke detection
    # ───────────────────────────────────────────────────────────────
    if fires:
        return ("fire_emergency", 80, "Fire/smoke detected by YOLO")
    
    # ───────────────────────────────────────────────────────────────
    # EMERGENCY: Emergency vehicles
    # ───────────────────────────────────────────────────────────────
    if any(vehicle in obj_keys for vehicle in ["ambulance", "fire truck"]):
        return ("accident", 75, "Emergency vehicle detected")
    
    # ───────────────────────────────────────────────────────────────
    # Get high-confidence person count
    # FIX #4: Only count persons with 60%+ confidence
    # ───────────────────────────────────────────────────────────────
    person_count = 0
    for obj_name, obj_data in object_counts.items():
        if "person" in obj_name.lower():
            if obj_data.get("max_confidence", 0) >= 60:
                person_count = obj_data.get("count", 0)
                break
    
    vehicle_count = category_counts.get("VEHICLE", 0)
    
    # ───────────────────────────────────────────────────────────────
    # ROAD SCENES: Vehicles + traffic infrastructure
    # ───────────────────────────────────────────────────────────────
    traffic_signs = ["traffic light", "stop sign", "parking meter"]
    has_traffic_sign = any(sign in obj_keys for sign in traffic_signs)
    
    if vehicle_count >= 2 and has_traffic_sign:
        return ("road", 75, "Multiple vehicles + traffic signs")
    
    if vehicle_count >= 3:
        return ("road", 70, "Multiple vehicles detected")
    
    if vehicle_count >= 1 and has_traffic_sign:
        return ("road", 65, "Vehicle + traffic infrastructure")
    
    # ───────────────────────────────────────────────────────────────
    # CROWDED AREAS: Many people
    # FIX #3: Significantly raised thresholds (was 5, now 20)
    # ───────────────────────────────────────────────────────────────
    if person_count >= 25:
        return ("crowded_area", 75, f"{person_count} people detected")
    
    if person_count >= 15:
        return ("crowded_area", 65, f"{person_count} people detected")
    
    # ───────────────────────────────────────────────────────────────
    # OFFICE: Multiple office items
    # ───────────────────────────────────────────────────────────────
    office_items = ["laptop", "keyboard", "monitor", "mouse", "desk", "computer"]
    office_count = sum(1 for item in office_items if item in obj_keys)
    
    if office_count >= 2:
        return ("office", 70, "Office equipment detected")
    
    # ───────────────────────────────────────────────────────────────
    # KITCHEN: Multiple kitchen items
    # ───────────────────────────────────────────────────────────────
    kitchen_items = ["microwave", "oven", "refrigerator", "sink", "stove", "toaster"]
    kitchen_count = sum(1 for item in kitchen_items if item in obj_keys)
    
    if kitchen_count >= 2:
        return ("kitchen", 70, "Kitchen appliances detected")
    
    # ───────────────────────────────────────────────────────────────
    # WAREHOUSE: Industrial equipment
    # ───────────────────────────────────────────────────────────────
    warehouse_items = ["forklift", "pallet", "warehouse", "shelf"]
    if any(item in obj_keys for item in warehouse_items):
        return ("warehouse", 65, "Industrial equipment detected")
    
    return None


def _classify_from_imagenet(image: Image.Image) -> Optional[Tuple[str, float, str]]:
    """
    Classify scene using MobileNetV2 ImageNet predictions.
    
    FIX #2: Used appropriately - general scene understanding, not security
    Maps ImageNet categories to our scene types using keyword matching
    
    Args:
        image: PIL Image to classify
        
    Returns:
        Tuple of (scene_name, confidence, source_description) or None
    """
    try:
        model = get_model()
        if model is None:
            return None
        
        # Preprocess image for MobileNetV2
        img_resized = image.convert("RGB").resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(preprocess_input(img_array), axis=0)
        
        # Get predictions
        predictions = model.predict(img_array, verbose=0)
        decoded_predictions = decode_predictions(predictions, top=10)[0]
        
        # Score each scene type based on keyword matching
        scene_scores = {scene: 0.0 for scene in KEYWORD_SCENE_MAP.keys()}
        
        for _, class_name, score in decoded_predictions:
            class_name_lower = class_name.lower()
            
            # Match against each scene's keywords
            for scene_type, keywords in KEYWORD_SCENE_MAP.items():
                if any(keyword in class_name_lower for keyword in keywords):
                    scene_scores[scene_type] += float(score)
        
        # Find best matching scene
        best_scene = max(scene_scores, key=scene_scores.get)
        best_score = scene_scores[best_scene]
        
        # Only return if confidence is reasonable
        if best_score >= 0.05:  # 5% minimum
            confidence = round(best_score * 100, 1)
            return (best_scene, confidence, f"ImageNet: {best_scene}")
        else:
            # Fallback to generic indoor
            return ("indoor", 35, "ImageNet: generic indoor scene")
            
    except Exception as e:
        print(f"ImageNet classification error: {e}")
        return None


def _analyze_brightness(image: Image.Image) -> Optional[Tuple[str, float, str]]:
    """
    Analyze image brightness to detect night scenes.
    
    Args:
        image: PIL Image to analyze
        
    Returns:
        Tuple of (scene_name, confidence, source_description) or None
    """
    try:
        # Convert to grayscale and resize for speed
        gray_image = image.convert("L").resize((100, 100))
        brightness_array = np.array(gray_image, dtype=np.float32)
        average_brightness = brightness_array.mean()
        
        # Very dark = night scene
        if average_brightness < 35:
            confidence = round(100 - (average_brightness / 35 * 100), 1)
            return ("night_scene", min(confidence, 60), "Brightness analysis: very dark")
        
        return None
        
    except Exception as e:
        print(f"Brightness analysis error: {e}")
        return None


def _combine_signals(
    signals: List[Tuple[str, float, str]]
) -> Tuple[str, float, str]:
    """
    Combine multiple classification signals using weighted voting.
    
    Object-based signals have highest weight, ImageNet is supporting
    
    Args:
        signals: List of (scene_name, confidence, source) tuples
        
    Returns:
        Tuple of (final_scene, final_confidence, combined_source)
    """
    if not signals:
        return ("unknown", 0, "No classification signals")
    
    # Aggregate votes for each scene type
    scene_votes = {}
    
    for scene_type, confidence, source in signals:
        if scene_type not in scene_votes:
            scene_votes[scene_type] = {
                "total_confidence": 0,
                "count": 0,
                "sources": []
            }
        
        scene_votes[scene_type]["total_confidence"] += confidence
        scene_votes[scene_type]["count"] += 1
        scene_votes[scene_type]["sources"].append(source)
    
    # Calculate weighted scores
    # Multiple signals agreeing = bonus confidence
    best_scene = None
    best_score = -1
    
    for scene_type, vote_data in scene_votes.items():
        # Average confidence with multi-signal bonus
        avg_confidence = vote_data["total_confidence"] / vote_data["count"]
        signal_bonus = vote_data["count"] * 5  # 5% bonus per agreeing signal
        
        weighted_score = avg_confidence + signal_bonus
        
        if weighted_score > best_score:
            best_score = weighted_score
            best_scene = scene_type
    
    # Get final confidence and source description
    vote_data = scene_votes[best_scene]
    final_confidence = round(
        min(vote_data["total_confidence"] / vote_data["count"], 95), 
        1
    )
    combined_source = " + ".join(vote_data["sources"])
    
    return (best_scene, final_confidence, combined_source)


def _get_top_imagenet_predictions(image: Image.Image) -> List[Tuple[str, float]]:
    """
    Get top ImageNet predictions for display purposes.
    
    Args:
        image: PIL Image to classify
        
    Returns:
        List of (class_name, confidence_percentage) tuples
    """
    try:
        model = get_model()
        if model is None:
            return []
        
        # Preprocess
        img_resized = image.convert("RGB").resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(preprocess_input(img_array), axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        decoded_predictions = decode_predictions(predictions, top=8)[0]
        
        # Format results
        results = [
            (class_name.replace("_", " "), round(float(score) * 100, 1))
            for _, class_name, score in decoded_predictions
        ]
        
        return results
        
    except Exception as e:
        print(f"ImageNet prediction error: {e}")
        return []