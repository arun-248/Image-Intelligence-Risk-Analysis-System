"""
detection.py — Smart Dual Detection + Context Analysis Engine
Fixed version with confidence thresholds and improved rules
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

_model_coco = None
_model_oiv7 = None

def get_coco_model():
    global _model_coco
    if _model_coco is None and YOLO_AVAILABLE:
        _model_coco = YOLO("yolov8n.pt")
    return _model_coco

def get_oiv7_model():
    global _model_oiv7
    if _model_oiv7 is None and YOLO_AVAILABLE:
        try:
            _model_oiv7 = YOLO("yolov8n-oiv7.pt")
        except Exception:
            _model_oiv7 = None
    return _model_oiv7

WEAPON_CLASSES = {
    "handgun","gun","rifle","shotgun","weapon","pistol","firearm","revolver",
    "machine gun","knife","sword","dagger","axe","baseball bat","blade"
}
VEHICLE_CLASSES = {
    "ambulance","fire truck","police car","car","truck","bus","motorcycle",
    "bicycle","traffic light","stop sign","vehicle","tank","van"
}
PERSON_CLASSES  = {"person","man","woman","boy","girl","human","people"}
SAFETY_CLASSES  = {"helmet","crash helmet","safety vest","face mask","seat belt"}
FIRE_CLASSES    = {"fire","flame","smoke","explosion"}

def _cat(label):
    l = label.lower()
    if any(w in l for w in WEAPON_CLASSES):  return "WEAPON"
    if any(w in l for w in FIRE_CLASSES):    return "FIRE_HAZARD"
    if any(w in l for w in VEHICLE_CLASSES): return "VEHICLE"
    if any(w in l for w in PERSON_CLASSES):  return "PERSON"
    if any(w in l for w in SAFETY_CLASSES):  return "SAFETY_EQUIPMENT"
    return "GENERAL"

COLORS = {
    "WEAPON":           (255, 30,  30),
    "INFERRED_WEAPON":  (255, 80,  80),
    "FIRE_HAZARD":      (255, 140,  0),
    "VEHICLE":          (30,  144, 255),
    "PERSON":           (0,   220,  80),
    "SAFETY_EQUIPMENT": (255, 215,  0),
    "GENERAL":          (150, 150, 150),
    "CONTEXT":          (200,  50, 255),
}


# ─────────────────────────────────────────────────────────────────
# CONTEXT ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────

def context_analysis(image: Image.Image, yolo_detections: list) -> list:
    """Analyzes image context to infer weapons/threats with CONFIDENCE THRESHOLDS"""
    img_array = np.array(image.convert("RGB"))
    h, w = img_array.shape[:2]
    inferred = []
    
    # Only consider high-confidence person detections
    persons = [d for d in yolo_detections 
               if d["category"] == "PERSON" and d["confidence"] >= 60]
    n_persons = len(persons)

    # Check 1: People lying on ground (requires 2+ people for context)
    lying_count = 0
    if n_persons >= 2:
        for p in persons:
            x1,y1,x2,y2 = p["bbox"]
            bw = x2 - x1
            bh = y2 - y1
            if bw > bh * 1.4:
                lying_count += 1

    # Check 2: Weapon proxy detection (only if people present)
    weapon_proxy_count = 0
    if n_persons >= 1:
        weapon_proxy_count = _find_elongated_dark_objects_numpy(img_array, persons)

    # Check 3: Extended arms (weapon pointing posture)
    extended_arms = 0
    if n_persons >= 2:
        for p in persons:
            x1,y1,x2,y2 = p["bbox"]
            bw = x2 - x1
            bh = y2 - y1
            if bh > 0 and bw / bh > 0.6:
                extended_arms += 1

    # Check 4: Indoor scene
    indoor_scene = _is_indoor_scene(img_array)

    # Check 5: Blood color signature
    has_blood_color = _detect_blood_color(img_array)

    # Check 6: Raised hands
    standing = n_persons - lying_count
    has_raised_hands = _detect_raised_hands(img_array, persons)

    # ═══ STRICTER INFERENCE RULES WITH HIGHER THRESHOLDS ═══
    
    # Rule 1: Armed robbery (needs 3+ people + lying victim + indoor)
    if n_persons >= 3 and lying_count >= 1 and standing >= 2 and indoor_scene:
        inferred.append({
            "label": "⚠ Inferred: Armed Robbery Posture",
            "confidence": 72.0,  # Lowered from 78
            "bbox": (10, 10, w-10, h-10),
            "area": w * h,
            "category": "INFERRED_WEAPON",
            "source": "CONTEXT_ENGINE",
            "reason": f"{n_persons} people, {lying_count} lying (victim), {standing} standing — robbery pattern"
        })

    # Rule 2: Weapon pointing (needs 3+ people + extended arms + indoor)
    if extended_arms >= 2 and indoor_scene and n_persons >= 3:
        inferred.append({
            "label": "⚠ Inferred: Weapon Pointing Posture",
            "confidence": 65.0,  # Lowered from 72
            "bbox": (20, 20, w-20, h-20),
            "area": w * h,
            "category": "INFERRED_WEAPON",
            "source": "CONTEXT_ENGINE",
            "reason": f"{extended_arms} people in weapon-pointing posture"
        })

    # Rule 3: Weapon-shaped object (needs 2+ people minimum)
    if weapon_proxy_count >= 1 and n_persons >= 2:
        inferred.append({
            "label": "⚠ Inferred: Weapon-Shaped Object",
            "confidence": 58.0,  # Lowered from 65
            "bbox": (30, 30, w//2, h//2),
            "area": (w//2) * (h//2),
            "category": "INFERRED_WEAPON",
            "source": "CONTEXT_ENGINE",
            "reason": "Dark elongated object near person — possible firearm/weapon"
        })

    # Rule 4: Violence/injury (needs lying + blood + 2+ people)
    if has_blood_color and lying_count >= 1 and n_persons >= 2:
        inferred.append({
            "label": "⚠ Inferred: Violence/Injury Scene",
            "confidence": 68.0,  # Lowered from 70
            "bbox": (5, 5, w-5, h-5),
            "area": w * h,
            "category": "INFERRED_WEAPON",
            "source": "CONTEXT_ENGINE",
            "reason": "Blood-color signature + person lying — violence/injury"
        })

    # Rule 5: Hands-up victim (needs 4+ people + indoor)
    if has_raised_hands and n_persons >= 4 and indoor_scene:
        inferred.append({
            "label": "⚠ Inferred: Hands-Up (Victim Posture)",
            "confidence": 70.0,  # Lowered from 75
            "bbox": (w//3, 0, 2*w//3, h//2),
            "area": (w//3) * (h//2),
            "category": "INFERRED_WEAPON",
            "source": "CONTEXT_ENGINE",
            "reason": "Person with raised hands — possible robbery victim posture"
        })

    return inferred


def _find_elongated_dark_objects_numpy(img, persons, min_aspect=3.0):
    """Pure numpy weapon proxy detection"""
    count = 0
    try:
        gray = np.mean(img, axis=2)
        dark = (gray < 60).astype(np.uint8)
        h, w = gray.shape
        step = 20
        for y in range(0, h - step, step):
            for x in range(0, w - step, step):
                patch = dark[y:y+step, x:x+step]
                ph, pw = patch.shape
                row_filled = (patch.sum(axis=1) > pw * 0.6).sum()
                col_filled = (patch.sum(axis=0) > ph * 0.6).sum()
                aspect = max(row_filled, col_filled) / (min(row_filled, col_filled) + 1)
                if aspect >= min_aspect:
                    for p in persons:
                        px1,py1,px2,py2 = p["bbox"]
                        cx, cy = x + step//2, y + step//2
                        if px1-30 < cx < px2+30 and py1 < cy < py1+(py2-py1)*0.7:
                            count += 1
                            break
    except Exception:
        pass
    return count


def _is_indoor_scene(img):
    try:
        top = img[:int(img.shape[0]*0.15), :, :]
        return float(top.std()) < 45
    except Exception:
        return False


def _detect_blood_color(img):
    """FIXED: More strict blood color detection to reduce false positives"""
    try:
        arr = img.astype(np.float32)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        # Stricter: needs dark red, not bright red like sunsets
        mask = (r > 100) & (r < 160) & (r > g*2.0) & (r > b*2.0) & ((r+g+b)/3 < 120)
        ratio = float(mask.sum()) / (img.shape[0] * img.shape[1])
        return ratio > 0.03  # Raised threshold from 0.02
    except Exception:
        return False


def _detect_raised_hands(img, persons):
    try:
        h_img = img.shape[0]
        for p in persons:
            x1,y1,x2,y2 = p["bbox"]
            bh = y2 - y1
            if y1 < h_img * 0.15 and bh > h_img * 0.4:
                return True
    except Exception:
        pass
    return False


# ─────────────────────────────────────────────────────────────────
# MAIN DETECTION FUNCTION
# ─────────────────────────────────────────────────────────────────

def detect_objects(image: Image.Image, confidence_threshold: float = 0.25):
    """Main detection with confidence tracking"""
    if not YOLO_AVAILABLE:
        return _fallback(image)

    img_array = np.array(image.convert("RGB"))
    all_dets = []

    # Layer 1: COCO model
    try:
        for result in get_coco_model()(img_array, conf=confidence_threshold, verbose=False):
            if result.boxes:
                for box in result.boxes:
                    label = result.names[int(box.cls[0])]
                    conf  = float(box.conf[0])
                    x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
                    all_dets.append({
                        "label": label,
                        "confidence": round(conf*100,1),
                        "bbox": (x1,y1,x2,y2),
                        "area": (x2-x1)*(y2-y1),
                        "category": _cat(label),
                        "source": "COCO"
                    })
    except Exception:
        pass

    # Layer 2: OIV7 model (weapon-focused)
    try:
        oiv7 = get_oiv7_model()
        if oiv7:
            for result in oiv7(img_array, conf=max(0.15, confidence_threshold-0.1), verbose=False):
                if result.boxes:
                    for box in result.boxes:
                        label = result.names[int(box.cls[0])]
                        conf  = float(box.conf[0])
                        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
                        if not _is_dup((x1,y1,x2,y2), all_dets):
                            all_dets.append({
                                "label": label,
                                "confidence": round(conf*100,1),
                                "bbox": (x1,y1,x2,y2),
                                "area": (x2-x1)*(y2-y1),
                                "category": _cat(label),
                                "source": "OIV7"
                            })
    except Exception:
        pass

    # Layer 3: Context Analysis (only if high-confidence people detected)
    context_dets = context_analysis(image, all_dets)
    all_dets.extend(context_dets)
    all_dets.sort(key=lambda x: x["confidence"], reverse=True)

    # Build structured object counts with confidence tracking
    obj_counts = {}
    cat_counts = {}
    for d in all_dets:
        label = d["label"]
        cat = d["category"]
        conf = d["confidence"]
        
        if label not in obj_counts:
            obj_counts[label] = {
                "count": 0,
                "max_confidence": 0.0,
                "avg_confidence": 0.0,
                "confidences": []
            }
        
        obj_counts[label]["count"] += 1
        obj_counts[label]["confidences"].append(conf)
        obj_counts[label]["max_confidence"] = max(obj_counts[label]["max_confidence"], conf)
        
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    # Calculate average confidences
    for obj in obj_counts.values():
        obj["avg_confidence"] = sum(obj["confidences"]) / len(obj["confidences"])

    weapons = [d for d in all_dets if d["category"] in ("WEAPON","INFERRED_WEAPON")]
    fires = [d for d in all_dets if d["category"] == "FIRE_HAZARD"]
    annotated = _draw(image.copy(), all_dets)

    if all_dets:
        # Show top 6 objects with counts
        parts = [f"{v['count']}x {k}" for k,v in list(obj_counts.items())[:6]]
        summary = "Detected: " + ", ".join(parts)
        if weapons:
            wnames = ", ".join(d["label"] for d in weapons[:3])
            summary = f"⚠️ THREAT: {wnames} | " + summary
    else:
        summary = "No objects detected. Try lowering confidence threshold."

    return {
        "detections": all_dets,
        "annotated_image": annotated,
        "object_counts": obj_counts,
        "category_counts": cat_counts,
        "weapons_found": weapons,
        "fire_found": fires,
        "total_objects": len(all_dets),
        "summary": summary,
        "models_used": ["YOLOv8n-COCO","YOLOv8n-OIV7","Context-Engine"],
        "context_detections": context_dets,
    }


# ─────────────────────────────────────────────────────────────────
# PIL-ONLY DRAWING
# ─────────────────────────────────────────────────────────────────

def _draw(image: Image.Image, detections: list) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    # Regular detections
    for det in [d for d in detections if d.get("source") != "CONTEXT_ENGINE"]:
        label = det["label"]
        conf  = det["confidence"]
        x1,y1,x2,y2 = det["bbox"]
        color = COLORS.get(det["category"], (150,150,150))
        thick = 3 if det["category"] in ("WEAPON","INFERRED_WEAPON") else 2
        for t in range(thick):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=color)
        text = f"{label} {conf}%"
        try:
            bb = draw.textbbox((x1, y1-18), text, font=font)
            draw.rectangle(bb, fill=color)
            draw.text((bb[0]+2, bb[1]), text, fill=(0,0,0), font=font)
        except Exception:
            draw.text((x1+2, y1+2), text, fill=color, font=font)

    # Context engine — dashed purple border
    for det in [d for d in detections if d.get("source") == "CONTEXT_ENGINE"]:
        x1,y1,x2,y2 = det["bbox"]
        color = (200, 50, 255)
        dash = 20
        for i in range(x1, x2, dash*2):
            draw.line([(i,y1),(min(i+dash,x2),y1)], fill=color, width=2)
            draw.line([(i,y2),(min(i+dash,x2),y2)], fill=color, width=2)
        for i in range(y1, y2, dash*2):
            draw.line([(x1,i),(x1,min(i+dash,y2))], fill=color, width=2)
            draw.line([(x2,i),(x2,min(i+dash,y2))], fill=color, width=2)
        text = f"{det['label'][:30]} {det['confidence']}%"
        try:
            draw.text((x1+2, y2+2), text, fill=color, font=font)
        except Exception:
            pass

    return img


def _is_dup(box, existing, thresh=0.5):
    x1,y1,x2,y2 = box
    for d in existing:
        ex1,ey1,ex2,ey2 = d["bbox"]
        ix1,iy1 = max(x1,ex1), max(y1,ey1)
        ix2,iy2 = min(x2,ex2), min(y2,ey2)
        if ix2<=ix1 or iy2<=iy1: continue
        inter = (ix2-ix1)*(iy2-iy1)
        union = (x2-x1)*(y2-y1)+(ex2-ex1)*(ey2-ey1)-inter
        if union>0 and inter/union>thresh: return True
    return False


def _fallback(image):
    return {
        "detections":[], "annotated_image":image, "object_counts":{},
        "category_counts":{}, "weapons_found":[], "fire_found":[],
        "total_objects":0, "summary":"Install: pip install ultralytics",
        "models_used":[], "context_detections":[]
    }