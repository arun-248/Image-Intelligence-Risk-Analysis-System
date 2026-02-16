"""
detection.py — Smart Dual Detection + Context Analysis Engine
PIL-only drawing — no cv2 required — works on Streamlit Cloud
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

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
# CONTEXT ANALYSIS ENGINE — pure numpy/PIL, no cv2 needed
# ─────────────────────────────────────────────────────────────────

def context_analysis(image: Image.Image, yolo_detections: list) -> list:
    img_array = np.array(image.convert("RGB"))
    h, w = img_array.shape[:2]
    inferred = []
    persons = [d for d in yolo_detections if d["category"] == "PERSON"]
    n_persons = len(persons)

    # Check 1: People lying on ground
    lying_count = 0
    for p in persons:
        x1,y1,x2,y2 = p["bbox"]
        bw = x2 - x1
        bh = y2 - y1
        if bw > bh * 1.4:
            lying_count += 1

    # Check 2: Weapon proxy — dark elongated shapes (numpy only)
    weapon_proxy_count = _find_elongated_dark_objects_numpy(img_array, persons)

    # Check 3: Extended arms
    extended_arms = 0
    for p in persons:
        x1,y1,x2,y2 = p["bbox"]
        bw = x2 - x1
        bh = y2 - y1
        if bh > 0 and bw / bh > 0.6:
            extended_arms += 1

    # Check 4: Indoor scene
    indoor_scene = _is_indoor_scene(img_array)

    # Check 5: Blood color
    has_blood_color = _detect_blood_color(img_array)

    # Check 6: Raised hands
    standing = n_persons - lying_count
    has_raised_hands = _detect_raised_hands(img_array, persons)

    # Inference Rules
    if n_persons >= 3 and lying_count >= 1 and standing >= 2 and indoor_scene:
        inferred.append({
            "label": "⚠ Inferred: Armed Robbery Posture",
            "confidence": 78.0, "bbox": (10, 10, w-10, h-10),
            "area": w * h, "category": "INFERRED_WEAPON",
            "source": "CONTEXT_ENGINE",
            "reason": f"{n_persons} people, {lying_count} lying (victim), {standing} standing — robbery pattern"
        })

    if extended_arms >= 2 and indoor_scene and n_persons >= 2:
        inferred.append({
            "label": "⚠ Inferred: Weapon Pointing Posture",
            "confidence": 72.0, "bbox": (20, 20, w-20, h-20),
            "area": w * h, "category": "INFERRED_WEAPON",
            "source": "CONTEXT_ENGINE",
            "reason": f"{extended_arms} people in weapon-pointing posture"
        })

    if weapon_proxy_count >= 1 and n_persons >= 1:
        inferred.append({
            "label": "⚠ Inferred: Weapon-Shaped Object",
            "confidence": 65.0, "bbox": (30, 30, w//2, h//2),
            "area": (w//2) * (h//2), "category": "INFERRED_WEAPON",
            "source": "CONTEXT_ENGINE",
            "reason": "Dark elongated object near person — possible firearm/weapon"
        })

    if has_blood_color and lying_count >= 1:
        inferred.append({
            "label": "⚠ Inferred: Violence/Injury Scene",
            "confidence": 70.0, "bbox": (5, 5, w-5, h-5),
            "area": w * h, "category": "INFERRED_WEAPON",
            "source": "CONTEXT_ENGINE",
            "reason": "Blood-color signature + person lying — violence/injury"
        })

    if has_raised_hands and n_persons >= 3 and indoor_scene:
        inferred.append({
            "label": "⚠ Inferred: Hands-Up (Victim Posture)",
            "confidence": 75.0, "bbox": (w//3, 0, 2*w//3, h//2),
            "area": (w//3) * (h//2), "category": "INFERRED_WEAPON",
            "source": "CONTEXT_ENGINE",
            "reason": "Person with raised hands — possible robbery victim posture"
        })

    return inferred


def _find_elongated_dark_objects_numpy(img, persons, min_aspect=3.0):
    """Pure numpy weapon proxy detection — no cv2."""
    count = 0
    try:
        gray = np.mean(img, axis=2)
        dark = (gray < 60).astype(np.uint8)
        row_sums = dark.sum(axis=1)
        col_sums = dark.sum(axis=0)
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
    try:
        arr = img.astype(np.float32)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        mask = (r > 120) & (r > g*1.8) & (r > b*1.8) & ((r+g+b)/3 < 140)
        return float(mask.sum()) / (img.shape[0] * img.shape[1]) > 0.02
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
                        "label": label, "confidence": round(conf*100,1),
                        "bbox": (x1,y1,x2,y2), "area": (x2-x1)*(y2-y1),
                        "category": _cat(label), "source": "COCO"
                    })
    except Exception:
        pass

    # Layer 2: OIV7 model
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
                                "label": label, "confidence": round(conf*100,1),
                                "bbox": (x1,y1,x2,y2), "area": (x2-x1)*(y2-y1),
                                "category": _cat(label), "source": "OIV7"
                            })
    except Exception:
        pass

    # Layer 3: Context Analysis
    context_dets = context_analysis(image, all_dets)
    all_dets.extend(context_dets)
    all_dets.sort(key=lambda x: x["confidence"], reverse=True)

    obj_counts = {}
    cat_counts = {}
    for d in all_dets:
        obj_counts[d["label"]] = obj_counts.get(d["label"],0)+1
        cat_counts[d["category"]] = cat_counts.get(d["category"],0)+1

    weapons   = [d for d in all_dets if d["category"] in ("WEAPON","INFERRED_WEAPON")]
    fires     = [d for d in all_dets if d["category"] == "FIRE_HAZARD"]
    annotated = _draw(image.copy(), all_dets)

    if all_dets:
        parts = [f"{v}x {k}" for k,v in list(obj_counts.items())[:6]]
        summary = "Detected: " + ", ".join(parts)
        if weapons:
            wnames = ", ".join(d["label"] for d in weapons[:3])
            summary = f"⚠️ THREAT: {wnames} | " + summary
    else:
        summary = "No objects detected. Try lowering confidence threshold."

    return {
        "detections": all_dets, "annotated_image": annotated,
        "object_counts": obj_counts, "category_counts": cat_counts,
        "weapons_found": weapons, "fire_found": fires,
        "total_objects": len(all_dets), "summary": summary,
        "models_used": ["YOLOv8n-COCO","YOLOv8n-OIV7","Context-Engine"],
        "context_detections": context_dets,
    }


# ─────────────────────────────────────────────────────────────────
# PIL-ONLY DRAWING — zero cv2 dependency
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