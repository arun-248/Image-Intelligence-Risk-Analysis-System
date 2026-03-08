

"""
scene.py — Enhanced Scene Classification with Threat Detection
Uses object detection + ImageNet + visual patterns to understand scenes
Specially tuned to detect threats, violence, military vs civilian contexts
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple


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
# SCENE DEFINITIONS (UPDATED WITH BETTER RISK SCORES)
# ═══════════════════════════════════════════════════════════════════

SCENE_DEFINITIONS = {
    # High-threat scenes
    "violence":       ("🚨", 45, "Active violence or physical altercation"),
    "robbery":        ("🔫", 55, "Armed robbery or threat situation"),
    "weapon_threat":  ("⚔️", 50, "Weapons detected in threatening context"),
    "accident":       ("🚗💥", 40, "Vehicle accident or collision"),
    "fire_emergency": ("🔥", 50, "Fire, smoke, or explosion emergency"),
    
    # Medium-threat scenes
    "road":           ("🛣️", 18, "Traffic or road environment"),
    "crowded_area":   ("👥", 25, "High crowd density area"),
    "warehouse":      ("🏭", 20, "Industrial or warehouse facility"),
    "parking":        ("🅿️", 15, "Parking lot or garage"),
    "night_scene":    ("🌙", 20, "Night-time or low-light scene"),
    
    # Low-threat scenes
    "hospital":       ("🏥", 12, "Medical or healthcare facility"),
    "office":         ("🏢", 8, "Professional office workspace"),
    "classroom":      ("🎓", 8, "Educational setting"),
    "kitchen":        ("🍳", 12, "Kitchen or food preparation area"),
    "outdoor":        ("🌳", 10, "Outdoor natural environment"),
    "indoor":         ("🏠", 8, "Indoor residential space"),
    "military":       ("🪖", 35, "Military or law enforcement context"),
    
    # Default
    "unknown":        ("❓", 8, "Scene type could not be determined"),
}


KEYWORD_SCENE_MAP = {
    "road": ["traffic", "street", "highway", "crosswalk", "road", "lane", "cab", "taxi"],
    "office": ["desk", "monitor", "keyboard", "computer", "laptop", "chair", "office"],
    "hospital": ["stretcher", "stethoscope", "syringe", "hospital", "ambulance", "medical"],
    "classroom": ["classroom", "blackboard", "school", "lecture", "whiteboard"],
    "outdoor": ["tree", "grass", "sky", "mountain", "park", "garden", "forest", "beach"],
    "kitchen": ["kitchen", "stove", "refrigerator", "oven", "microwave", "sink"],
    "crowded_area": ["crowd", "people", "mall", "market", "stadium", "concert"],
    "parking": ["parking", "garage", "lot"],
    "warehouse": ["warehouse", "factory", "industrial", "storage", "shelf"],
    "indoor": ["room", "living", "bedroom", "hallway", "interior", "sofa", "couch"],
    "military": ["tank", "helicopter", "soldier", "uniform", "military", "army"],
}


def classify_scene(image: Image.Image, detection_result: Optional[dict] = None) -> dict:
    """
    Enhanced scene classification with threat detection.
    
    Process:
    1. Check for weapons/threats first (highest priority)
    2. Analyze detected objects and context
    3. Use ImageNet for general scene understanding
    4. Determine if military vs civilian context
    
    Args:
        image: PIL Image to classify
        detection_result: Output from detect_objects()
        
    Returns:
        Dictionary with scene type, confidence, risk score, and threat indicators
    """
    classification_signals = []

    # ═══════════════════════════════════════════════════════════════
    # PRIORITY 1: Check for weapons and threats
    # ═══════════════════════════════════════════════════════════════
    if detection_result:
        threat_signal = _classify_from_threats(detection_result)
        if threat_signal:
            classification_signals.append(threat_signal)
        
        # Object-based classification
        object_signal = _classify_from_objects(detection_result)
        if object_signal:
            classification_signals.append(object_signal)

    # ═══════════════════════════════════════════════════════════════
    # SIGNAL 2: ImageNet classification
    # ═══════════════════════════════════════════════════════════════
    if TF_AVAILABLE:
        imagenet_signal = _classify_from_imagenet(image)
        if imagenet_signal:
            classification_signals.append(imagenet_signal)

    # ═══════════════════════════════════════════════════════════════
    # SIGNAL 3: Brightness analysis
    # ═══════════════════════════════════════════════════════════════
    brightness_signal = _analyze_brightness(image)
    if brightness_signal:
        classification_signals.append(brightness_signal)

    # ═══════════════════════════════════════════════════════════════
    # COMBINE SIGNALS
    # ═══════════════════════════════════════════════════════════════
    final_scene, final_confidence, source_info = _combine_signals(classification_signals)
    
    # Get scene metadata
    scene_info = SCENE_DEFINITIONS.get(final_scene, SCENE_DEFINITIONS["unknown"])
    emoji, base_risk, description = scene_info
    
    # Determine if scene is dangerous
    dangerous_scenes = ["violence", "robbery", "weapon_threat", "fire_emergency", "accident"]
    
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
# THREAT-BASED CLASSIFICATION (HIGHEST PRIORITY)
# ═══════════════════════════════════════════════════════════════════

def _classify_from_threats(detection_result: dict) -> Optional[Tuple[str, float, str]]:
    """
    Classify based on detected weapons and violence indicators.
    This has highest priority - overrides all other signals.
    """
    weapons = detection_result.get("weapons_found", [])
    violence = detection_result.get("violence_indicators", [])
    objs = detection_result.get("object_counts", {})
    
    # ───────────────────────────────────────────────────────────────
    # WEAPONS DETECTED
    # ───────────────────────────────────────────────────────────────
    if weapons:
        # Filter to high-confidence weapons
        real_weapons = [w for w in weapons if w.get("confidence", 0) >= 50]
        
        if real_weapons:
            weapon_names = " ".join(w["label"].lower() for w in real_weapons)
            max_conf = max(w["confidence"] for w in real_weapons)
            
            # Check if military context (uniforms, vehicles, outdoor organized)
            is_military = _detect_military_context(objs, detection_result)
            
            if is_military:
                return ("military", min(max_conf, 85), "Armed military/law enforcement personnel detected")
            
            # Firearms in civilian context = robbery
            if any(gun in weapon_names for gun in ["gun", "rifle", "handgun", "pistol", "firearm"]):
                return ("robbery", min(max_conf + 10, 95), f"Firearm detected: {real_weapons[0]['label']}")
            
            # Other weapons = weapon threat
            return ("weapon_threat", min(max_conf, 90), f"Weapon detected: {real_weapons[0]['label']}")
    
    # ───────────────────────────────────────────────────────────────
    # VIOLENCE INDICATORS
    # ───────────────────────────────────────────────────────────────
    if violence:
        violence_types = [v.get("type", "") for v in violence]
        
        if "potential_victim" in violence_types:
            return ("violence", 75, "Potential victim detected (person lying down)")
        
        if "aggressive_crowd" in violence_types:
            return ("violence", 65, "Aggressive crowd formation detected")
        
        if "violent_scene_coloring" in violence_types:
            return ("violence", 60, "Violence indicators in scene coloring")
    
    return None


def _detect_military_context(objs: Dict, detection_result: dict) -> bool:
    """
    Detect if this is military/law enforcement context vs civilian threat.
    
    Indicators of military context:
    - Military vehicles (helicopter, tank, truck)
    - Uniforms (multiple people in organized formation)
    - Outdoor organized setting
    - Multiple armed individuals in formation
    """
    obj_keys = [k.lower() for k in objs.keys()]
    
    # Military vehicles
    military_vehicles = ["helicopter", "airplane", "truck", "tank"]
    has_military_vehicle = any(v in obj_keys for v in military_vehicles)
    
    # Large organized group (10+ people suggests training/operation)
    person_count = 0
    for obj_name, obj_data in objs.items():
        if "person" in obj_name.lower():
            person_count = obj_data.get("count", 0)
            break
    
    # Military context indicators
    is_outdoor = detection_result.get("category_counts", {}).get("OUTDOOR", 0) > 0
    large_group = person_count >= 10
    
    return (has_military_vehicle or (large_group and is_outdoor))


# ═══════════════════════════════════════════════════════════════════
# OBJECT-BASED CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

def _classify_from_objects(detection_result: dict) -> Optional[Tuple[str, float, str]]:
    """Classify scene based on detected objects"""
    category_counts = detection_result.get("category_counts", {})
    object_counts = detection_result.get("object_counts", {})
    obj_keys = [k.lower() for k in object_counts.keys()]
    fires = detection_result.get("fire_found", [])
    
    # Fire/smoke
    if fires:
        return ("fire_emergency", 85, "Fire/smoke detected")
    
    # Emergency vehicles
    if any(vehicle in obj_keys for vehicle in ["ambulance", "fire truck"]):
        return ("accident", 80, "Emergency vehicle detected")
    
    # Get high-confidence person count
    person_count = 0
    for obj_name, obj_data in object_counts.items():
        if "person" in obj_name.lower():
            if obj_data.get("max_confidence", 0) >= 50:
                person_count = obj_data.get("count", 0)
                break
    
    vehicle_count = category_counts.get("VEHICLE", 0)
    
    # Road scenes
    traffic_signs = ["traffic light", "stop sign", "parking meter"]
    has_traffic_sign = any(sign in obj_keys for sign in traffic_signs)
    
    if vehicle_count >= 2 and has_traffic_sign:
        return ("road", 80, "Multiple vehicles + traffic infrastructure")
    
    if vehicle_count >= 3:
        return ("road", 75, "Multiple vehicles detected")
    
    # Crowded areas (lowered threshold from 25 to 12)
    if person_count >= 12:
        return ("crowded_area", 75, f"{person_count} people detected")
    
    # Office
    office_items = ["laptop", "keyboard", "monitor", "mouse", "desk"]
    office_count = sum(1 for item in office_items if item in obj_keys)
    if office_count >= 2:
        return ("office", 75, "Office equipment detected")
    
    # Kitchen
    kitchen_items = ["microwave", "oven", "refrigerator", "sink", "stove"]
    kitchen_count = sum(1 for item in kitchen_items if item in obj_keys)
    if kitchen_count >= 2:
        return ("kitchen", 75, "Kitchen appliances detected")
    
    # Warehouse
    warehouse_items = ["forklift", "pallet", "shelf"]
    if any(item in obj_keys for item in warehouse_items):
        return ("warehouse", 70, "Industrial equipment detected")
    
    return None


def _classify_from_imagenet(image: Image.Image) -> Optional[Tuple[str, float, str]]:
    """Classify scene using ImageNet predictions"""
    try:
        model = get_model()
        if model is None:
            return None
        
        img_resized = image.convert("RGB").resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(preprocess_input(img_array), axis=0)
        
        predictions = model.predict(img_array, verbose=0)
        decoded_predictions = decode_predictions(predictions, top=10)[0]
        
        scene_scores = {scene: 0.0 for scene in KEYWORD_SCENE_MAP.keys()}
        
        for _, class_name, score in decoded_predictions:
            class_name_lower = class_name.lower()
            for scene_type, keywords in KEYWORD_SCENE_MAP.items():
                if any(keyword in class_name_lower for keyword in keywords):
                    scene_scores[scene_type] += float(score)
        
        best_scene = max(scene_scores, key=scene_scores.get)
        best_score = scene_scores[best_scene]
        
        if best_score >= 0.05:
            confidence = round(best_score * 100, 1)
            return (best_scene, confidence, f"ImageNet: {best_scene}")
        else:
            return ("indoor", 40, "ImageNet: generic indoor scene")
            
    except Exception as e:
        print(f"ImageNet classification error: {e}")
        return None


def _analyze_brightness(image: Image.Image) -> Optional[Tuple[str, float, str]]:
    """Detect night scenes based on brightness"""
    try:
        gray_image = image.convert("L").resize((100, 100))
        brightness_array = np.array(gray_image, dtype=np.float32)
        average_brightness = brightness_array.mean()
        
        if average_brightness < 40:
            confidence = round(100 - (average_brightness / 40 * 100), 1)
            return ("night_scene", min(confidence, 70), "Very dark scene detected")
        
        return None
        
    except Exception as e:
        print(f"Brightness analysis error: {e}")
        return None


def _combine_signals(signals: List[Tuple[str, float, str]]) -> Tuple[str, float, str]:
    """Combine multiple classification signals"""
    if not signals:
        return ("unknown", 0, "No classification signals")
    
    votes = {}
    for scene_type, confidence, source in signals:
        if scene_type not in votes:
            votes[scene_type] = {"total_confidence": 0, "count": 0, "sources": []}
        
        votes[scene_type]["total_confidence"] += confidence
        votes[scene_type]["count"] += 1
        votes[scene_type]["sources"].append(source)
    
    best_scene = None
    best_score = -1
    
    for scene_type, vote_data in votes.items():
        avg_confidence = vote_data["total_confidence"] / vote_data["count"]
        signal_bonus = vote_data["count"] * 8  # Increased from 5
        weighted_score = avg_confidence + signal_bonus
        
        if weighted_score > best_score:
            best_score = weighted_score
            best_scene = scene_type
    
    vote_data = votes[best_scene]
    final_confidence = round(min(vote_data["total_confidence"] / vote_data["count"], 95), 1)
    combined_source = " + ".join(vote_data["sources"])
    
    return (best_scene, final_confidence, combined_source)


def _get_top_imagenet_predictions(image: Image.Image) -> List[Tuple[str, float]]:
    """Get top ImageNet predictions for display"""
    try:
        model = get_model()
        if model is None:
            return []
        
        img_resized = image.convert("RGB").resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(preprocess_input(img_array), axis=0)
        
        predictions = model.predict(img_array, verbose=0)
        decoded_predictions = decode_predictions(predictions, top=8)[0]
        
        results = [
            (class_name.replace("_", " "), round(float(score) * 100, 1))
            for _, class_name, score in decoded_predictions
        ]
        
        return results
        
    except Exception as e:
        print(f"ImageNet prediction error: {e}")
        return []