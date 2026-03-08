"""
detection.py — Multi-Model Weapon & Object Detection Engine
FREE VERSION: No API calls, uses multiple YOLO models for comprehensive detection
Detects 680+ object types including weapons, violence indicators, and threats
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
# MODEL INITIALIZATION AND CACHING
# ═══════════════════════════════════════════════════════════════════

_yolo_coco = None
_yolo_oiv7 = None

def get_yolo_coco():
    """Load YOLOv8n-COCO model (80 common objects)"""
    global _yolo_coco
    if _yolo_coco is None and YOLO_AVAILABLE:
        _yolo_coco = YOLO("yolov8n.pt")
    return _yolo_coco


def get_yolo_oiv7():
    """Load YOLOv8n-OIV7 model (600 objects including weapons)"""
    global _yolo_oiv7
    if _yolo_oiv7 is None and YOLO_AVAILABLE:
        _yolo_oiv7 = YOLO("yolov8n-oiv7.pt")
    return _yolo_oiv7


# ═══════════════════════════════════════════════════════════════════
# OBJECT CATEGORIES AND WEAPON KEYWORDS
# ═══════════════════════════════════════════════════════════════════

COCO_CATEGORIES = {
    "PERSON": ["person"],
    "VEHICLE": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "ANIMAL": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "FURNITURE": ["chair", "couch", "bed", "dining table", "toilet"],
    "ELECTRONICS": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"],
    "KITCHEN": [
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
        "hot dog", "pizza", "donut", "cake", "microwave", "oven", 
        "toaster", "sink", "refrigerator"
    ],
    "SPORTS": [
        "frisbee", "skis", "snowboard", "sports ball", "kite", 
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"
    ],
    "OUTDOOR": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
    "ACCESSORY": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    "HOUSEHOLD": ["vase", "scissors", "teddy bear", "hair drier", "toothbrush", "potted plant", "clock", "book"],
}

# Expanded weapon keywords (OIV7 model can detect these)
WEAPON_KEYWORDS = [
    "gun", "rifle", "pistol", "handgun", "firearm", "shotgun", "revolver",
    "knife", "blade", "sword", "dagger", "machete", "axe", "weapon"
]

# Emergency and threat indicators
EMERGENCY_KEYWORDS = [
    "ambulance", "fire truck", "police car", "fire extinguisher", "stretcher"
]

# Violence and danger indicators
VIOLENCE_KEYWORDS = [
    "blood", "injury", "wound", "fire", "smoke", "explosion", "debris"
]


# ═══════════════════════════════════════════════════════════════════
# MAIN DETECTION FUNCTION (UPGRADED FOR WEAPONS)
# ═══════════════════════════════════════════════════════════════════

def detect_objects(image: Image.Image, confidence_threshold: float = 0.3) -> dict:
    """
    Multi-model object detection with weapon detection capabilities.
    
    UPGRADE: Now uses TWO models for comprehensive detection:
    - YOLOv8n-COCO: General objects (80 classes)
    - YOLOv8n-OIV7: Extended objects including weapons (600 classes)
    
    Args:
        image: PIL Image object
        confidence_threshold: Minimum confidence (0.0-1.0)
        
    Returns:
        Dictionary with detections, weapons, violence indicators, and annotated image
    """
    if not YOLO_AVAILABLE:
        return _no_yolo_result()

    all_detections = []
    models_used = []
    
    # ═══════════════════════════════════════════════════════════════
    # MODEL 1: YOLOv8n-COCO (General Objects)
    # ═══════════════════════════════════════════════════════════════
    model_coco = get_yolo_coco()
    if model_coco:
        try:
            results_coco = model_coco(image, conf=confidence_threshold, verbose=False)
            for result in results_coco:
                for box in result.boxes:
                    confidence = round(float(box.conf) * 100, 1)
                    all_detections.append({
                        "label": result.names[int(box.cls)],
                        "confidence": confidence,
                        "bbox": box.xyxy[0].tolist(),
                        "source": "YOLO-COCO"
                    })
            models_used.append("YOLOv8n-COCO")
        except Exception as e:
            print(f"COCO detection error: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # MODEL 2: YOLOv8n-OIV7 (Extended Detection + Weapons)
    # CRITICAL: This model can detect weapons, violence indicators
    # ═══════════════════════════════════════════════════════════════
    model_oiv7 = get_yolo_oiv7()
    if model_oiv7:
        try:
            results_oiv7 = model_oiv7(image, conf=confidence_threshold, verbose=False)
            for result in results_oiv7:
                for box in result.boxes:
                    confidence = round(float(box.conf) * 100, 1)
                    label = result.names[int(box.cls)]
                    
                    all_detections.append({
                        "label": label,
                        "confidence": confidence,
                        "bbox": box.xyxy[0].tolist(),
                        "source": "YOLO-OIV7"
                    })
            models_used.append("YOLOv8n-OIV7")
        except Exception as e:
            print(f"OIV7 detection error: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # PROCESS AND CATEGORIZE DETECTIONS
    # ═══════════════════════════════════════════════════════════════
    object_counts = {}
    category_counts = {}
    weapons_found = []
    fire_found = []
    violence_indicators = []
    
    for det in all_detections:
        label = det["label"]
        conf = det["confidence"]
        label_lower = label.lower()
        
        # Initialize object tracking
        if label not in object_counts:
            object_counts[label] = {
                "count": 0,
                "max_confidence": 0,
                "avg_confidence": 0,
                "confidences": []
            }
        
        # Update counts and confidence
        object_counts[label]["count"] += 1
        object_counts[label]["confidences"].append(conf)
        object_counts[label]["max_confidence"] = max(
            object_counts[label]["max_confidence"], conf
        )
        object_counts[label]["avg_confidence"] = round(
            sum(object_counts[label]["confidences"]) / len(object_counts[label]["confidences"]), 1
        )
        
        # Category assignment
        categorized = False
        for category, keywords in COCO_CATEGORIES.items():
            if label.lower() in keywords:
                category_counts[category] = category_counts.get(category, 0) + 1
                categorized = True
                break
        
        if not categorized:
            category_counts["OTHER"] = category_counts.get("OTHER", 0) + 1
        
        # ═══════════════════════════════════════════════════════════
        # WEAPON DETECTION (LOWERED THRESHOLD FOR BETTER DETECTION)
        # Changed from 75% to 50% to catch more real weapons
        # ═══════════════════════════════════════════════════════════
        if any(w in label_lower for w in WEAPON_KEYWORDS):
            if conf >= 50:  # Lowered from 75% to 50%
                weapons_found.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": det["bbox"],
                    "source": det["source"],
                    "threat_level": "CRITICAL" if conf >= 70 else "HIGH"
                })
        
        # Fire/smoke detection
        if any(f in label_lower for f in ["fire", "smoke", "flame", "explosion"]):
            if conf >= 45:  # Lowered from 65%
                fire_found.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": det["bbox"],
                    "source": det["source"]
                })
        
        # Violence indicators (blood, injury, destruction)
        if any(v in label_lower for v in VIOLENCE_KEYWORDS):
            if conf >= 40:
                violence_indicators.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": det["bbox"],
                    "source": det["source"]
                })
    
    # ═══════════════════════════════════════════════════════════════
    # CONTEXT-BASED VIOLENCE DETECTION (RULE-BASED, NO API)
    # ═══════════════════════════════════════════════════════════════
    context_threats = _detect_threat_patterns(all_detections, object_counts, image)
    
    # Merge context threats into violence indicators
    for threat in context_threats:
        violence_indicators.append(threat)
    
    # ═══════════════════════════════════════════════════════════════
    # CREATE ANNOTATED VISUALIZATION
    # ═══════════════════════════════════════════════════════════════
    annotated_image = _draw_bounding_boxes(
        image.copy(), 
        all_detections, 
        weapons_found,
        violence_indicators
    )
    
    # ═══════════════════════════════════════════════════════════════
    # RETURN COMPLETE RESULTS
    # ═══════════════════════════════════════════════════════════════
    return {
        "detections": all_detections,
        "total_objects": len(all_detections),
        "object_counts": object_counts,
        "category_counts": category_counts,
        "weapons_found": weapons_found,
        "fire_found": fire_found,
        "violence_indicators": violence_indicators,
        "context_detections": context_threats,
        "annotated_image": annotated_image,
        "models_used": models_used,
        "weapon_detection_active": len(weapons_found) > 0,
        "threat_level": _calculate_threat_level(weapons_found, violence_indicators),
    }


# ═══════════════════════════════════════════════════════════════════
# THREAT PATTERN DETECTION (RULE-BASED, NO API NEEDED)
# ═══════════════════════════════════════════════════════════════════

def _detect_threat_patterns(
    detections: List[dict], 
    object_counts: Dict, 
    image: Image.Image
) -> List[dict]:
    """
    Detect threat patterns using visual analysis (no API calls).
    
    Detects:
    - People in threatening postures
    - Aggressive crowd formations
    - Potential victims (person lying down)
    - Dark/violent color patterns
    """
    threats = []
    
    # Get person detections
    persons = [d for d in detections if "person" in d["label"].lower()]
    
    if len(persons) < 2:
        return threats  # Need multiple people for pattern detection
    
    # ───────────────────────────────────────────────────────────────
    # PATTERN 1: Person lying down (potential victim)
    # ───────────────────────────────────────────────────────────────
    for person in persons:
        x1, y1, x2, y2 = person["bbox"]
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        
        # Very wide bounding box = lying down
        if aspect_ratio > 2.0 and len(persons) >= 3:
            threats.append({
                "type": "potential_victim",
                "confidence": min(person["confidence"] + 15, 75),
                "reason": "Person in horizontal position with others present",
                "source": "CONTEXT_ANALYSIS",
                "bbox": person["bbox"]
            })
    
    # ───────────────────────────────────────────────────────────────
    # PATTERN 2: Aggressive crowd (many people, tight formation)
    # ───────────────────────────────────────────────────────────────
    if len(persons) >= 5:
        # Calculate crowd density
        bboxes = [p["bbox"] for p in persons]
        if _is_dense_crowd(bboxes):
            threats.append({
                "type": "aggressive_crowd",
                "confidence": 60,
                "reason": f"Dense crowd formation detected ({len(persons)} people)",
                "source": "CONTEXT_ANALYSIS"
            })
    
    # ───────────────────────────────────────────────────────────────
    # PATTERN 3: Dark/violent scene coloring
    # ───────────────────────────────────────────────────────────────
    violence_color_score = _check_violence_colors(image)
    if violence_color_score > 0.6 and len(persons) >= 2:
        threats.append({
            "type": "violent_scene_coloring",
            "confidence": min(round(violence_color_score * 100), 70),
            "reason": "Dark/violent color patterns with multiple people",
            "source": "CONTEXT_ANALYSIS"
        })
    
    return threats


def _is_dense_crowd(bboxes: List[List[float]]) -> bool:
    """Check if bounding boxes indicate a dense crowd"""
    if len(bboxes) < 5:
        return False
    
    # Calculate average overlap
    overlaps = 0
    for i, box1 in enumerate(bboxes):
        for box2 in bboxes[i+1:]:
            if _boxes_overlap(box1, box2):
                overlaps += 1
    
    overlap_ratio = overlaps / (len(bboxes) * (len(bboxes) - 1) / 2)
    return overlap_ratio > 0.3  # 30% of pairs overlapping


def _boxes_overlap(box1: List[float], box2: List[float]) -> bool:
    """Check if two bounding boxes overlap"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)


def _check_violence_colors(image: Image.Image) -> float:
    """
    Analyze image colors for violence indicators.
    Returns score 0.0-1.0 (higher = more violent coloring)
    """
    try:
        # Resize for speed
        img_small = image.resize((100, 100))
        arr = np.array(img_small, dtype=np.float32)
        
        # Check for dark/violent color patterns
        avg_brightness = arr.mean()
        r_channel = arr[:,:,0].mean()
        
        # Very dark scene + high red = potential violence
        darkness = 1.0 - (avg_brightness / 255.0)
        redness = r_channel / 255.0
        
        # Violence score (dark + red)
        violence_score = (darkness * 0.6 + redness * 0.4)
        
        return min(violence_score, 1.0)
        
    except Exception:
        return 0.0


def _calculate_threat_level(weapons: List[dict], violence: List[dict]) -> str:
    """Calculate overall threat level"""
    if not weapons and not violence:
        return "NONE"
    
    weapon_count = len(weapons)
    violence_count = len(violence)
    
    max_weapon_conf = max([w["confidence"] for w in weapons], default=0)
    max_violence_conf = max([v.get("confidence", 0) for v in violence], default=0)
    
    if weapon_count >= 2 or max_weapon_conf >= 80:
        return "CRITICAL"
    elif weapon_count >= 1 or max_weapon_conf >= 60:
        return "HIGH"
    elif violence_count >= 2 or max_violence_conf >= 60:
        return "MEDIUM"
    else:
        return "LOW"


# ═══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

def _draw_bounding_boxes(
    image: Image.Image, 
    detections: List[dict], 
    weapons: List[dict],
    violence: List[dict]
) -> Image.Image:
    """
    Draw bounding boxes with special highlighting for weapons and threats.
    """
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Collect weapon and violence bounding boxes for special rendering
    threat_bboxes = set()
    for w in weapons:
        threat_bboxes.add(tuple(w["bbox"]))
    for v in violence:
        if "bbox" in v:
            threat_bboxes.add(tuple(v["bbox"]))
    
    # Draw regular detections
    for det in detections:
        bbox_tuple = tuple(det["bbox"])
        
        # Skip if this is a threat (will be drawn separately)
        if bbox_tuple in threat_bboxes:
            continue
        
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        label = det["label"]
        
        # Color by confidence
        if conf >= 80:
            color = "#00e676"  # Green
        elif conf >= 55:
            color = "#ffab00"  # Yellow
        else:
            color = "#94a3b8"  # Gray
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label
        label_text = f"{label} {conf:.0f}%"
        text_bbox = draw.textbbox((x1, y1 - 18), label_text, font=font_small)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 18), label_text, fill="#000000", font=font_small)
    
    # Draw weapons with RED highlighting
    for weapon in weapons:
        x1, y1, x2, y2 = weapon["bbox"]
        conf = weapon["confidence"]
        label = weapon["label"]
        
        # Thick red box for weapons
        draw.rectangle([x1, y1, x2, y2], outline="#ff0000", width=5)
        
        # Warning label
        warning_text = f"⚠️ {label.upper()} {conf:.0f}%"
        text_bbox = draw.textbbox((x1, y1 - 24), warning_text, font=font)
        draw.rectangle(text_bbox, fill="#ff0000")
        draw.text((x1, y1 - 24), warning_text, fill="#ffffff", font=font)
    
    return image


def _no_yolo_result() -> dict:
    """Return empty result when YOLO unavailable"""
    blank_image = Image.new("RGB", (640, 480), color="#1e293b")
    draw = ImageDraw.Draw(blank_image)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    error_message = "YOLO not installed"
    draw.text((180, 220), error_message, fill="#ef4444", font=font)
    
    return {
        "detections": [],
        "total_objects": 0,
        "object_counts": {},
        "category_counts": {},
        "weapons_found": [],
        "fire_found": [],
        "violence_indicators": [],
        "context_detections": [],
        "annotated_image": blank_image,
        "models_used": [],
        "weapon_detection_active": False,
        "threat_level": "UNKNOWN",
    }