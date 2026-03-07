"""
detection.py — YOLOv8 Object Detection Engine
PRODUCTION VERSION: Context engine completely removed to eliminate false positives
Only reports what YOLO actually sees with high confidence
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
    """Load and cache YOLO COCO model (80 common object classes)"""
    global _yolo_coco
    if _yolo_coco is None and YOLO_AVAILABLE:
        _yolo_coco = YOLO("yolov8n.pt")
    return _yolo_coco


def get_yolo_oiv7():
    """Load and cache YOLO Open Images V7 model (600 diverse classes)"""
    global _yolo_oiv7
    if _yolo_oiv7 is None and YOLO_AVAILABLE:
        _yolo_oiv7 = YOLO("yolov8n-oiv7.pt")
    return _yolo_oiv7


# ═══════════════════════════════════════════════════════════════════
# OBJECT CATEGORIES AND KEYWORD MAPPINGS
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
    "HOUSEHOLD": [
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush", 
        "potted plant", "clock", "book"
    ],
}

# Weapon keywords for detection (only from YOLO, not inferred)
WEAPON_KEYWORDS = [
    "gun", "rifle", "pistol", "handgun", "firearm", "shotgun", "revolver",
    "knife", "blade", "sword", "dagger", "machete", "axe"
]

# Emergency vehicle keywords
EMERGENCY_KEYWORDS = [
    "ambulance", "fire truck", "police car", "fire extinguisher", "stretcher"
]


# ═══════════════════════════════════════════════════════════════════
# MAIN DETECTION FUNCTION
# ═══════════════════════════════════════════════════════════════════

def detect_objects(image: Image.Image, confidence_threshold: float = 0.3) -> dict:
    """
    Run multi-model object detection on an image.
    
    FIX #1: Context engine COMPLETELY REMOVED - no more false weapon detections
    FIX #4: Only objects above confidence_threshold are counted
    
    Args:
        image: PIL Image object
        confidence_threshold: Minimum confidence (0.0-1.0) to count detection
        
    Returns:
        Dictionary containing:
        - detections: List of all detected objects with confidence scores
        - total_objects: Count of detections
        - object_counts: Dict with count, max_confidence, avg_confidence per object
        - category_counts: Counts by category (PERSON, VEHICLE, etc)
        - weapons_found: List of weapon detections (YOLO only, NOT inferred)
        - fire_found: List of fire/smoke detections
        - annotated_image: Image with bounding boxes drawn
        - models_used: List of model names used
    """
    if not YOLO_AVAILABLE:
        return _no_yolo_result()

    all_detections = []
    models_used = []
    
    # ───────────────────────────────────────────────────────────────
    # STEP 1: Run YOLO COCO Detection (80 common classes)
    # ───────────────────────────────────────────────────────────────
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
    
    # ───────────────────────────────────────────────────────────────
    # STEP 2: Run YOLO Open Images V7 Detection (600 classes)
    # ───────────────────────────────────────────────────────────────
    model_oiv7 = get_yolo_oiv7()
    if model_oiv7:
        try:
            results_oiv7 = model_oiv7(image, conf=confidence_threshold, verbose=False)
            for result in results_oiv7:
                for box in result.boxes:
                    confidence = round(float(box.conf) * 100, 1)
                    all_detections.append({
                        "label": result.names[int(box.cls)],
                        "confidence": confidence,
                        "bbox": box.xyxy[0].tolist(),
                        "source": "YOLO-OIV7"
                    })
            models_used.append("YOLOv8n-OIV7")
        except Exception as e:
            print(f"OIV7 detection error: {e}")
    
    # ───────────────────────────────────────────────────────────────
    # STEP 3: Process and Categorize Detections
    # ───────────────────────────────────────────────────────────────
    object_counts = {}
    category_counts = {}
    weapons_found = []
    fire_found = []
    
    for det in all_detections:
        label = det["label"]
        conf = det["confidence"]
        
        # Initialize object tracking with confidence data
        if label not in object_counts:
            object_counts[label] = {
                "count": 0,
                "max_confidence": 0,
                "avg_confidence": 0,
                "confidences": []
            }
        
        # Update counts and confidence tracking
        object_counts[label]["count"] += 1
        object_counts[label]["confidences"].append(conf)
        object_counts[label]["max_confidence"] = max(
            object_counts[label]["max_confidence"], 
            conf
        )
        object_counts[label]["avg_confidence"] = round(
            sum(object_counts[label]["confidences"]) / len(object_counts[label]["confidences"]), 
            1
        )
        
        # Category assignment
        categorized = False
        for category, keywords in COCO_CATEGORIES.items():
            if label.lower() in keywords:
                category_counts[category] = category_counts.get(category, 0) + 1
                categorized = True
                break
        
        # If not categorized, add to OTHER
        if not categorized:
            category_counts["OTHER"] = category_counts.get("OTHER", 0) + 1
        
        # ───────────────────────────────────────────────────────────
        # FIX #1 & #4: Weapon detection - ONLY from YOLO, HIGH CONFIDENCE ONLY
        # No context engine, no guessing from shadows/umbrellas
        # ───────────────────────────────────────────────────────────
        if any(w in label.lower() for w in WEAPON_KEYWORDS):
            if conf >= 75:  # Only report weapons with 75%+ confidence
                weapons_found.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": det["bbox"],
                    "source": det["source"]
                })
        
        # Fire/smoke detection (high confidence only)
        if any(f in label.lower() for f in ["fire", "smoke", "flame"]):
            if conf >= 65:  # Only report fire with 65%+ confidence
                fire_found.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": det["bbox"],
                    "source": det["source"]
                })
    
    # ───────────────────────────────────────────────────────────────
    # STEP 4: Create Annotated Visualization
    # ───────────────────────────────────────────────────────────────
    annotated_image = _draw_bounding_boxes(
        image.copy(), 
        all_detections, 
        weapons_found
    )
    
    # ───────────────────────────────────────────────────────────────
    # STEP 5: Return Complete Detection Results
    # ───────────────────────────────────────────────────────────────
    return {
        "detections": all_detections,
        "total_objects": len(all_detections),
        "object_counts": object_counts,
        "category_counts": category_counts,
        "weapons_found": weapons_found,
        "fire_found": fire_found,
        "context_detections": [],  # Always empty - context engine removed
        "annotated_image": annotated_image,
        "models_used": models_used,
    }


# ═══════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _draw_bounding_boxes(
    image: Image.Image, 
    detections: List[dict], 
    weapons: List[dict]
) -> Image.Image:
    """
    Draw bounding boxes and labels on the image.
    
    Color coding:
    - Green (80%+): High confidence detection
    - Yellow (55-79%): Medium confidence
    - Gray (<55%): Low confidence
    - Red (weapons): Weapons detected by YOLO
    
    Args:
        image: PIL Image to draw on
        detections: List of all detections
        weapons: List of weapon detections
        
    Returns:
        Annotated PIL Image
    """
    draw = ImageDraw.Draw(image)
    
    # Try to load a nice font, fall back to default
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
            14
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 
            11
        )
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw weapon boxes first (so they appear on top)
    weapon_labels = {w["label"] for w in weapons}
    
    # Draw regular detections
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        label = det["label"]
        
        # Skip if this is a weapon (will be drawn separately)
        if label in weapon_labels:
            continue
        
        # Determine color based on confidence
        if conf >= 80:
            color = "#00e676"  # Bright green
        elif conf >= 55:
            color = "#ffab00"  # Amber yellow
        else:
            color = "#94a3b8"  # Gray
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label background and text
        label_text = f"{label} {conf:.0f}%"
        
        # Get text bounding box
        text_bbox = draw.textbbox((x1, y1 - 18), label_text, font=font_small)
        
        # Draw label background
        draw.rectangle(text_bbox, fill=color)
        
        # Draw label text
        draw.text((x1, y1 - 18), label_text, fill="#000000", font=font_small)
    
    # Draw weapons in red with special highlighting
    for weapon in weapons:
        x1, y1, x2, y2 = weapon["bbox"]
        conf = weapon["confidence"]
        label = weapon["label"]
        
        # Draw thick red bounding box
        draw.rectangle([x1, y1, x2, y2], outline="#ff3d3d", width=4)
        
        # Draw warning label
        warning_text = f"⚠️ {label.upper()} {conf:.0f}%"
        
        # Get text bounding box
        text_bbox = draw.textbbox((x1, y1 - 22), warning_text, font=font)
        
        # Draw warning background
        draw.rectangle(text_bbox, fill="#ff3d3d")
        
        # Draw warning text in white
        draw.text((x1, y1 - 22), warning_text, fill="#ffffff", font=font)
    
    return image


def _no_yolo_result() -> dict:
    """
    Return empty result structure when YOLO is not available.
    Creates a placeholder image with error message.
    
    Returns:
        Empty detection result dictionary
    """
    blank_image = Image.new("RGB", (640, 480), color="#1e293b")
    draw = ImageDraw.Draw(blank_image)
    
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
            20
        )
    except Exception:
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
        "context_detections": [],
        "annotated_image": blank_image,
        "models_used": [],
    }


# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_detection_summary(detection_result: dict) -> str:
    """
    Generate a human-readable summary of detection results.
    
    Args:
        detection_result: Output from detect_objects()
        
    Returns:
        Formatted summary string
    """
    total = detection_result.get("total_objects", 0)
    categories = detection_result.get("category_counts", {})
    weapons = detection_result.get("weapons_found", [])
    
    summary = f"Detected {total} objects across {len(categories)} categories.\n"
    
    if weapons:
        summary += f"⚠️ WARNING: {len(weapons)} weapon(s) detected!\n"
    
    if categories:
        summary += "Categories: " + ", ".join(
            f"{cat}({count})" for cat, count in sorted(categories.items())
        )
    
    return summary


def filter_by_confidence(
    detections: List[dict], 
    min_confidence: float
) -> List[dict]:
    """
    Filter detections by minimum confidence threshold.
    
    Args:
        detections: List of detection dictionaries
        min_confidence: Minimum confidence (0-100)
        
    Returns:
        Filtered list of detections
    """
    return [
        det for det in detections 
        if det.get("confidence", 0) >= min_confidence
    ]


def get_objects_by_category(
    detection_result: dict, 
    category: str
) -> List[dict]:
    """
    Get all detected objects belonging to a specific category.
    
    Args:
        detection_result: Output from detect_objects()
        category: Category name (e.g., "PERSON", "VEHICLE")
        
    Returns:
        List of detection dictionaries in that category
    """
    if category not in COCO_CATEGORIES:
        return []
    
    category_keywords = COCO_CATEGORIES[category]
    detections = detection_result.get("detections", [])
    
    return [
        det for det in detections 
        if det.get("label", "").lower() in category_keywords
    ]