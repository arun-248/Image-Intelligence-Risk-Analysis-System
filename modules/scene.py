"""
scene.py — Scene Classifier WITHOUT color-based violence detection
Fixed version - removed red color = violence rule
"""

import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

_model = None

def get_model():
    global _model
    if _model is None and TF_AVAILABLE:
        _model = MobileNetV2(weights="imagenet", include_top=True)
    return _model

SCENE_DEFINITIONS = {
    "violence":       ("🚨", 60, "Violence or physical altercation detected in scene"),
    "accident":       ("🚗💥", 55, "Vehicle accident or collision scene detected"),
    "fire_emergency": ("🔥", 65, "Fire or explosion emergency scene detected"),
    "robbery":        ("🔫", 70, "Robbery or armed threat scene detected"),
    "weapon_threat":  ("⚔️", 70, "Weapons present — potential threat situation"),
    "road":           ("🛣️", 35, "Traffic or road environment"),
    "crowded_area":   ("👥", 40, "High crowd density — many people present"),
    "hospital":       ("🏥", 15, "Medical or healthcare environment"),
    "office":         ("🏢",  5, "Professional office or workspace"),
    "classroom":      ("🎓",  5, "Educational environment"),
    "outdoor":        ("🌳", 10, "Open outdoor environment"),
    "kitchen":        ("🍳", 20, "Kitchen or food preparation area"),
    "parking":        ("🅿️", 20, "Parking lot or vehicle storage area"),
    "warehouse":      ("🏭", 25, "Industrial or storage facility"),
    "indoor":         ("🏠", 10, "Indoor residential or commercial space"),
    "night_scene":    ("🌙", 25, "Low-light or night-time environment"),
    "unknown":        ("❓", 10, "Scene type could not be determined"),
}

KEYWORD_SCENE_MAP = {
    "road":        ["traffic","street","highway","crosswalk","road","lane","cab"],
    "office":      ["desk","monitor","keyboard","computer","laptop","chair","office"],
    "hospital":    ["stretcher","stethoscope","syringe","hospital","ambulance","medical"],
    "classroom":   ["classroom","blackboard","school","lecture","whiteboard"],
    "outdoor":     ["tree","grass","sky","mountain","park","garden","forest","beach"],
    "kitchen":     ["kitchen","stove","refrigerator","oven","microwave","sink"],
    "crowded_area":["crowd","people","mall","market","stadium","concert","festival"],
    "parking":     ["parking","garage","lot","car park"],
    "warehouse":   ["warehouse","factory","industrial","storage","shelf","forklift"],
    "indoor":      ["room","living","bedroom","hallway","interior","sofa","couch"],
}


def classify_scene(image: Image.Image, detection_result: dict = None):
    """Classify scene using objects + ImageNet + brightness (NO color-based violence)"""
    signals = []

    # Signal 1: Object + Context based (HIGHEST priority)
    if detection_result:
        s = _from_objects(detection_result)
        if s: signals.append(s)

    # Signal 2: ImageNet deep learning
    if TF_AVAILABLE:
        s = _imagenet(image)
        if s: signals.append(s)

    # Signal 3: REMOVED COLOR ANALYSIS - was causing false positives
    # Old code checked if red > 120 = violence, which flagged sunsets/red rooms

    # Signal 4: Brightness
    s = _brightness(image)
    if s: signals.append(s)

    final_scene, final_conf, source = _combine(signals)
    info = SCENE_DEFINITIONS.get(final_scene, SCENE_DEFINITIONS["unknown"])
    emoji, base_risk, description = info

    return {
        "scene":           final_scene,
        "confidence":      final_conf,
        "scene_emoji":     emoji,
        "description":     description,
        "base_risk_score": base_risk,
        "top_predictions": _top_imagenet(image) if TF_AVAILABLE else [],
        "is_dangerous":    final_scene in ["violence","robbery","weapon_threat","fire_emergency","accident"],
        "source":          source,
    }


def _from_objects(det):
    """Detect scene from objects and context engine (with confidence thresholds)"""
    cats    = det.get("category_counts", {})
    objs    = det.get("object_counts",   {})
    obj_keys = [k.lower() for k in objs.keys()]
    weapons  = det.get("weapons_found",  [])
    fires    = det.get("fire_found",     [])
    context  = det.get("context_detections", [])

    # Real weapon detected by YOLO (check confidence)
    if weapons:
        real_weapons = [w for w in weapons 
                       if w.get("source") != "CONTEXT_ENGINE" 
                       and w.get("confidence", 0) >= 60]  # 60% minimum
        inferred = [w for w in weapons if w.get("source") == "CONTEXT_ENGINE"]

        if real_weapons:
            wnames = " ".join(w["label"].lower() for w in real_weapons)
            if any(g in wnames for g in ["gun","rifle","handgun","shotgun","pistol","firearm"]):
                return ("robbery", 88, "🔫 Firearm confirmed by YOLO")  # Lowered from 92
            return ("weapon_threat", 82, "⚔️ Weapon confirmed by YOLO")  # Lowered from 88

        # Context engine inferences (already have lower confidence)
        if inferred:
            reasons = " ".join(w.get("reason","") for w in inferred).lower()
            if "robbery" in reasons or "lying" in reasons:
                return ("robbery", 75, f"🔫 Context: {inferred[0].get('reason','robbery pattern')}")
            if "weapon" in reasons or "pointing" in reasons:
                return ("weapon_threat", 70, f"⚔️ Context: {inferred[0].get('reason','weapon posture')}")
            if "violence" in reasons or "injury" in reasons:
                return ("violence", 68, f"🚨 Context: {inferred[0].get('reason','violence')}")

    if fires:
        return ("fire_emergency", 85, "🔥 Fire/smoke detected")  # Lowered from 88

    if any(v in obj_keys for v in ["ambulance","fire truck"]):
        return ("accident", 75, "Emergency vehicle → accident scene")  # Lowered from 80

    # Get actual counts with confidence check
    person_n = sum(1 for obj, data in objs.items() 
                   if "person" in obj.lower() 
                   and data.get("max_confidence", 0) >= 50)
    
    vehicle_n = cats.get("VEHICLE", 0)

    if vehicle_n >= 2 and person_n >= 1:
        return ("road", 70, "Multiple vehicles + people")
    if vehicle_n >= 1 and any(t in obj_keys for t in ["traffic light","stop sign"]):
        return ("road", 75, "Vehicle + traffic sign")
    if person_n >= 5:
        return ("crowded_area", 72, f"{person_n} people detected")
    if any(o in obj_keys for o in ["laptop","keyboard","monitor","mouse","tie"]):
        return ("office", 70, "Office equipment/attire detected")
    if any(k in obj_keys for k in ["microwave","oven","refrigerator","toaster"]):
        return ("kitchen", 70, "Kitchen appliances detected")

    return None


def _imagenet(image):
    """ImageNet-based scene classification"""
    try:
        model = get_model()
        img = image.convert("RGB").resize((224,224))
        arr = np.expand_dims(preprocess_input(np.array(img, dtype=np.float32)), 0)
        decoded = decode_predictions(model.predict(arr, verbose=0), top=10)[0]
        scores = {s:0.0 for s in KEYWORD_SCENE_MAP}
        for _,name,score in decoded:
            nl = name.lower()
            for scene,kws in KEYWORD_SCENE_MAP.items():
                if any(k in nl for k in kws):
                    scores[scene] += float(score)
        best = max(scores, key=scores.get)
        bv   = scores[best]
        if bv < 0.05:
            return ("indoor", 40, "ImageNet: no strong match")
        return (best, round(bv*100,1), f"ImageNet: {best}")
    except Exception:
        return None


def _brightness(image):
    """Brightness-based scene detection (night scenes only)"""
    try:
        avg = np.array(image.convert("L").resize((100,100)),dtype=np.float32).mean()
        if avg < 50:
            return ("night_scene", 55, "Brightness: night/dark scene")
        return None
    except Exception:
        return None


def _combine(signals):
    """Combine multiple signals using weighted voting"""
    if not signals:
        return ("unknown", 0, "No signals")
    votes = {}
    for scene,conf,src in signals:
        if scene not in votes:
            votes[scene] = {"conf":0,"n":0,"srcs":[]}
        votes[scene]["conf"] += conf
        votes[scene]["n"]    += 1
        votes[scene]["srcs"].append(src)
    best, best_score, best_src = None, -1, ""
    for scene,v in votes.items():
        score = (v["conf"]/v["n"]) * (1 + 0.2*(v["n"]-1))
        if score > best_score:
            best_score = score
            best = scene
            best_src = " + ".join(v["srcs"])
    return (best, round(min(best_score,99),1), best_src)


def _top_imagenet(image):
    """Get top ImageNet predictions for display"""
    try:
        model = get_model()
        img = image.convert("RGB").resize((224,224))
        arr = np.expand_dims(preprocess_input(np.array(img,dtype=np.float32)),0)
        decoded = decode_predictions(model.predict(arr,verbose=0),top=8)[0]
        return [(n.replace("_"," "), round(float(s)*100,1)) for _,n,s in decoded]
    except Exception:
        return []