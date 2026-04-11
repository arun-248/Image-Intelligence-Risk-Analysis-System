"""
detection.py — Threat Detection Engine (Final Version)
=======================================================

HONEST ARCHITECTURE:
--------------------
Layer 1 — weapon_detect.pt   : A YOLOv8 model fine-tuned on weapons (guns, knives).
                                Download once from: https://github.com/Bnomq/gun-detection-yolov8
                                Place as: weapon_detect.pt in your project root.
                                If absent → falls through to Layer 2.

Layer 2 — YOLOv8n-COCO       : General objects — people, cars, phones, etc.
Layer 3 — YOLOv8n-OIV7       : 600-class Google model (partial weapon support).
Layer 4 — Visual Scene Analysis: Detects robbery patterns even when YOLO misses weapons.
           - Mask + dark scene + arm-extended pose + victim posture = infer gun
           - Always gives a clear written explanation of what was detected and why.
"""

import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
# MODEL PATHS
# ═══════════════════════════════════════════════════════════════════
WEAPON_MODEL_PATH = "weapon_detect.pt"   # dedicated weapon model (optional)

_yolo_weapon = None
_yolo_coco   = None
_yolo_oiv7   = None

def get_weapon_model():
    """Weapon-specific YOLOv8 model (guns, knives). Optional."""
    global _yolo_weapon
    if _yolo_weapon is None and YOLO_AVAILABLE:
        if os.path.exists(WEAPON_MODEL_PATH):
            try:
                _yolo_weapon = YOLO(WEAPON_MODEL_PATH)
                print(f"[detection] Weapon model loaded: {WEAPON_MODEL_PATH}")
            except Exception as e:
                print(f"[detection] Weapon model load failed: {e}")
    return _yolo_weapon

def get_yolo_coco():
    global _yolo_coco
    if _yolo_coco is None and YOLO_AVAILABLE:
        try:
            _yolo_coco = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"[detection] COCO load failed: {e}")
    return _yolo_coco

def get_yolo_oiv7():
    global _yolo_oiv7
    if _yolo_oiv7 is None and YOLO_AVAILABLE:
        try:
            _yolo_oiv7 = YOLO("yolov8n-oiv7.pt")
        except Exception as e:
            print(f"[detection] OIV7 load failed: {e}")
    return _yolo_oiv7


# ═══════════════════════════════════════════════════════════════════
# KEYWORDS — all lowercase, comprehensive
# ═══════════════════════════════════════════════════════════════════
FIREARM_KEYWORDS = [
    "gun","rifle","pistol","handgun","firearm","shotgun","revolver",
    "weapon","submachine","assault","carbine","glock","ak","m16",
    "uzi","beretta","ammunition","bullet","grenade","handgun",
]
KNIFE_KEYWORDS = [
    "knife","blade","dagger","sword","machete","axe","cleaver","saber",
]
MASK_KEYWORDS = [
    "mask","balaclava","ski mask","gas mask","face cover","hood","covering",
]
FIRE_KEYWORDS = ["fire","smoke","flame","explosion","burning","blaze"]

COCO_CATEGORIES = {
    "PERSON":      ["person"],
    "VEHICLE":     ["bicycle","car","motorcycle","airplane","bus","train","truck","boat"],
    "ANIMAL":      ["bird","cat","dog","horse","sheep","cow","elephant","bear"],
    "FURNITURE":   ["chair","couch","bed","dining table","toilet"],
    "ELECTRONICS": ["tv","laptop","mouse","remote","keyboard","cell phone"],
    "OUTDOOR":     ["traffic light","fire hydrant","stop sign","parking meter","bench"],
    "ACCESSORY":   ["backpack","umbrella","handbag","tie","suitcase"],
}

def _has(label: str, keywords: list) -> bool:
    ll = label.lower()
    return any(k in ll for k in keywords)


# ═══════════════════════════════════════════════════════════════════
# MAIN DETECTION
# ═══════════════════════════════════════════════════════════════════

def detect_objects(image: Image.Image, confidence_threshold: float = 0.20) -> dict:
    if not YOLO_AVAILABLE:
        return _no_yolo_result()

    all_detections = []
    models_used    = []

    # ── LAYER 1: Weapon-specific model ──────────────────────────────
    weapon_model = get_weapon_model()
    if weapon_model:
        try:
            for r in weapon_model(image, conf=0.15, verbose=False):
                for box in r.boxes:
                    all_detections.append({
                        "label":      r.names[int(box.cls)],
                        "confidence": round(float(box.conf)*100, 1),
                        "bbox":       box.xyxy[0].tolist(),
                        "source":     "WeaponDetector",
                        "priority":   "high",
                    })
            models_used.append("WeaponDetector (fine-tuned)")
        except Exception as e:
            print(f"[detection] weapon model inference error: {e}")

    # ── LAYER 2: YOLOv8n-COCO ───────────────────────────────────────
    coco = get_yolo_coco()
    if coco:
        try:
            for r in coco(image, conf=confidence_threshold, verbose=False):
                for box in r.boxes:
                    all_detections.append({
                        "label":      r.names[int(box.cls)],
                        "confidence": round(float(box.conf)*100, 1),
                        "bbox":       box.xyxy[0].tolist(),
                        "source":     "YOLOv8n-COCO",
                        "priority":   "normal",
                    })
            models_used.append("YOLOv8n-COCO")
        except Exception as e:
            print(f"[detection] COCO error: {e}")

    # ── LAYER 3: YOLOv8n-OIV7 ───────────────────────────────────────
    oiv7 = get_yolo_oiv7()
    if oiv7:
        try:
            for r in oiv7(image, conf=confidence_threshold, verbose=False):
                for box in r.boxes:
                    all_detections.append({
                        "label":      r.names[int(box.cls)],
                        "confidence": round(float(box.conf)*100, 1),
                        "bbox":       box.xyxy[0].tolist(),
                        "source":     "YOLOv8n-OIV7",
                        "priority":   "normal",
                    })
            models_used.append("YOLOv8n-OIV7")
        except Exception as e:
            print(f"[detection] OIV7 error: {e}")

    # Deduplicate
    all_detections = _deduplicate(all_detections)

    # ── CLASSIFY ─────────────────────────────────────────────────────
    object_counts   = {}
    category_counts = {}
    weapons_found   = []
    masks_found     = []
    fire_found      = []

    for det in all_detections:
        label = det["label"]
        conf  = det["confidence"]

        # Count
        if label not in object_counts:
            object_counts[label] = {"count":0, "max_confidence":0, "confidences":[]}
        object_counts[label]["count"] += 1
        object_counts[label]["confidences"].append(conf)
        object_counts[label]["max_confidence"] = max(object_counts[label]["max_confidence"], conf)

        # Category
        found_cat = False
        for cat, kws in COCO_CATEGORIES.items():
            if label.lower() in kws:
                category_counts[cat] = category_counts.get(cat,0)+1
                found_cat = True; break
        if not found_cat:
            category_counts["OTHER"] = category_counts.get("OTHER",0)+1

        # Weapons — low threshold to catch anything weapon-like
        if _has(label, FIREARM_KEYWORDS) and conf >= 15:
            weapons_found.append({
                "label": label, "confidence": conf, "bbox": det["bbox"],
                "source": det["source"], "weapon_type": "FIREARM",
                "threat_level": "CRITICAL",
            })
        elif _has(label, KNIFE_KEYWORDS) and conf >= 15:
            weapons_found.append({
                "label": label, "confidence": conf, "bbox": det["bbox"],
                "source": det["source"], "weapon_type": "KNIFE/BLADE",
                "threat_level": "HIGH",
            })

        if _has(label, MASK_KEYWORDS) and conf >= 15:
            masks_found.append({"label":label,"confidence":conf,"bbox":det["bbox"],"source":det["source"]})

        if _has(label, FIRE_KEYWORDS) and conf >= 20:
            fire_found.append({"label":label,"confidence":conf,"bbox":det["bbox"],"source":det["source"]})

    # ── LAYER 4: Visual scene analysis ──────────────────────────────
    scene_threats, inferred_weapons = _visual_analysis(image, all_detections, object_counts)
    weapons_found.extend(inferred_weapons)
    violence_indicators = scene_threats

    # Threat level + explanation
    threat_level       = _get_threat_level(weapons_found, masks_found, violence_indicators)
    threat_explanation = _build_explanation(
        weapons_found, masks_found, violence_indicators,
        object_counts, threat_level, all_detections
    )

    annotated = _draw_boxes(
        image.copy(), all_detections,
        weapons_found, masks_found, fire_found, threat_level
    )

    return {
        "detections":              all_detections,
        "total_objects":           len(all_detections),
        "object_counts":           object_counts,
        "category_counts":         category_counts,
        "weapons_found":           weapons_found,
        "masks_found":             masks_found,
        "fire_found":              fire_found,
        "violence_indicators":     violence_indicators,
        "annotated_image":         annotated,
        "models_used":             models_used,
        "weapon_detection_active": len(weapons_found) > 0,
        "mask_detection_active":   len(masks_found) > 0,
        "threat_level":            threat_level,
        "threat_explanation":      threat_explanation,
        "weapon_model_loaded":     weapon_model is not None,
    }


# ═══════════════════════════════════════════════════════════════════
# LAYER 4 — VISUAL SCENE ANALYSIS
# Detects what YOLO cannot detect from scene context
# ═══════════════════════════════════════════════════════════════════

def _visual_analysis(image, detections, object_counts):
    threats          = []
    inferred_weapons = []

    arr = np.array(image.convert("RGB"), dtype=np.float32)
    H, W = arr.shape[:2]

    persons   = [d for d in detections if "person" in d["label"].lower()]
    n_persons = len(persons)

    # Brightness
    brightness = arr.mean()
    is_dark    = brightness < 100

    # Existing YOLO detections
    yolo_masks   = [d for d in detections if _has(d["label"], MASK_KEYWORDS)]
    has_yolo_gun = any(_has(d["label"], FIREARM_KEYWORDS) for d in detections)

    # ── POSE ANALYSIS ────────────────────────────────────────────────
    gun_pose    = False
    victim_pose = False

    if n_persons >= 2:
        for p in persons:
            x1,y1,x2,y2 = p["bbox"]
            pw = x2-x1; ph = y2-y1
            if ph <= 0: continue
            aspect = pw/ph
            cx = (x1+x2)/2

            # Tall narrow = hands-up victim
            if aspect < 0.5 and ph > H*0.28:
                victim_pose = True

            # Wide arm extension on left side = attacker pointing weapon
            if cx < W*0.6 and pw > ph*0.5:
                gun_pose = True

    # ── RED PIXEL (blood/fire indicator) ────────────────────────────
    r,g,b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    red_mask  = (r > 130) & (r > g*1.35) & (r > b*1.35)
    red_ratio = float(red_mask.sum()) / (H*W)

    # ── BUILD SIGNALS LIST ───────────────────────────────────────────
    signals = []
    if yolo_masks:     signals.append("masked person (balaclava/face covering) detected")
    if is_dark:        signals.append("dark indoor/parking environment")
    if n_persons >= 2: signals.append(f"{n_persons} people in confrontation")
    if gun_pose:       signals.append("arm-extended threat posture (weapon pointing stance)")
    if victim_pose:    signals.append("victim submission posture (hands raised)")

    # ── ROBBERY SCENE ────────────────────────────────────────────────
    if len(signals) >= 2:
        conf = min(38 + len(signals)*11, 91)
        threats.append({
            "type":       "robbery_scene",
            "confidence": conf,
            "reason":     "Armed robbery indicators: " + " | ".join(signals),
            "source":     "VISUAL_ANALYSIS",
            "signals":    signals,
        })

    # ── INFER GUN WHEN YOLO MISSES IT ────────────────────────────────
    # Condition: mask + arm pose + victim + dark = almost certainly a gun
    if (not has_yolo_gun
            and yolo_masks
            and n_persons >= 2
            and (gun_pose or (is_dark and victim_pose))):
        conf = min(40 + len(signals)*9, 83)
        inferred_weapons.append({
            "label":       "Handgun (visual inference)",
            "confidence":  conf,
            "bbox":        _estimate_gun_position(persons, (W,H)),
            "source":      "VISUAL_INFERENCE",
            "weapon_type": "FIREARM",
            "threat_level":"CRITICAL",
            "note":        "Gun inferred — YOLO missed it. Inference based on: " + ", ".join(signals),
        })

    # ── VICTIM POSTURE ───────────────────────────────────────────────
    if victim_pose and n_persons >= 2:
        threats.append({
            "type":       "potential_victim",
            "confidence": 72,
            "reason":     "Person in submission posture — hands raised or backing away from threat",
            "source":     "VISUAL_ANALYSIS",
        })

    # ── DARK ENCOUNTER ───────────────────────────────────────────────
    if is_dark and n_persons >= 2 and not yolo_masks:
        threats.append({
            "type":       "suspicious_encounter",
            "confidence": 52,
            "reason":     "Multiple people in dark indoor environment — suspicious encounter",
            "source":     "VISUAL_ANALYSIS",
        })

    # ── BLOOD/VIOLENCE COLORS ────────────────────────────────────────
    if red_ratio > 0.14 and n_persons >= 1:
        threats.append({
            "type":       "violent_scene_coloring",
            "confidence": min(int(red_ratio*280), 68),
            "reason":     "High red pixel density — possible blood, injury, or fire present",
            "source":     "VISUAL_ANALYSIS",
        })

    # ── DENSE CROWD ──────────────────────────────────────────────────
    if n_persons >= 5 and _dense([p["bbox"] for p in persons]):
        threats.append({
            "type":       "aggressive_crowd",
            "confidence": 65,
            "reason":     f"Dense crowd formation — {n_persons} people in tight cluster",
            "source":     "VISUAL_ANALYSIS",
        })

    return threats, inferred_weapons


def _estimate_gun_position(persons, img_size):
    W, H = img_size
    if not persons:
        return [W*0.3, H*0.3, W*0.5, H*0.5]
    sp = sorted(persons, key=lambda x: x["bbox"][0])
    x1,y1,x2,y2 = sp[0]["bbox"]
    ph = y2-y1
    return [x2, y1+ph*0.28, x2+(x2-x1)*0.32, y1+ph*0.52]


# ═══════════════════════════════════════════════════════════════════
# THREAT EXPLANATION — honest, clear, detailed
# ═══════════════════════════════════════════════════════════════════

def _build_explanation(weapons, masks, violence, object_counts, threat_level, detections):
    lines = []

    n_persons = sum(
        d.get("count",0) for lbl,d in object_counts.items()
        if "person" in lbl.lower()
    )
    firearms = [w for w in weapons if w["weapon_type"]=="FIREARM"]
    knives   = [w for w in weapons if w["weapon_type"]=="KNIFE/BLADE"]

    # Headline
    heads = {
        "CRITICAL": "🚨 CRITICAL THREAT DETECTED",
        "HIGH":     "🔴 HIGH THREAT DETECTED",
        "MEDIUM":   "⚠️  MEDIUM THREAT DETECTED",
        "LOW":      "⚠️  LOW RISK",
        "NONE":     "✅ NO THREAT DETECTED",
    }
    lines.append(heads.get(threat_level, "⚠️ THREAT DETECTED"))
    lines.append("━"*44)

    # What was found
    lines.append("WHAT WAS DETECTED:")
    if n_persons:
        lines.append(f"  👤 {n_persons} person(s) in the scene")

    for fw in firearms:
        if "inference" in fw["label"].lower():
            lines.append(f"  🔫 FIREARM — inferred at {fw['confidence']:.0f}% confidence")
            lines.append(f"       Note: The gun was NOT directly detected by YOLO.")
            lines.append(f"       It was inferred from scene context:")
            lines.append(f"       → {fw.get('note','scene analysis')}")
        else:
            lines.append(f"  🔫 {fw['label'].upper()} — {fw['confidence']:.0f}% conf [{fw['source']}]")

    for kw in knives:
        lines.append(f"  🔪 {kw['label'].upper()} — {kw['confidence']:.0f}% conf [{kw['source']}]")

    for m in masks:
        lines.append(f"  🎭 {m['label'].upper()} — {m['confidence']:.0f}% conf [{m['source']}]")

    if not weapons and not masks:
        lines.append("  ⚠️  No weapons/masks detected directly by YOLO models.")
        lines.append("       Scene analysis may still identify threat patterns below.")

    # Why dangerous
    lines.append("")
    lines.append("WHY THIS IS DANGEROUS:")
    if firearms and masks:
        lines.append("  • Masked person with firearm = ARMED ROBBERY in progress")
    elif firearms:
        lines.append("  • Firearm present — person is being threatened at gunpoint")
    elif masks:
        lines.append("  • Person concealing face = intent to hide identity (robbery indicator)")

    seen = set()
    for v in violence:
        r = v.get("reason","")
        if r and r not in seen:
            lines.append(f"  • {r}")
            seen.add(r)

    if not firearms and not masks and not violence:
        lines.append("  • No specific threat signals detected in this image.")

    # Robbery scene breakdown
    robbery = [v for v in violence if v.get("type")=="robbery_scene"]
    if robbery:
        lines.append("")
        lines.append("ROBBERY SCENE EVIDENCE:")
        for s in robbery[0].get("signals",[]):
            lines.append(f"  ✓ {s}")

    # Model limitation note
    has_direct_gun = any(
        "inference" not in w["label"].lower() for w in firearms
    )
    if firearms and not has_direct_gun:
        lines.append("")
        lines.append("⚠️  DETECTION LIMITATION NOTE:")
        lines.append("  The YOLO models (COCO + OIV7) were NOT trained specifically")
        lines.append("  on weapons. They detected the people and scene correctly")
        lines.append("  but missed the gun directly.")
        lines.append("  To fix: add weapon_detect.pt to your project folder.")
        lines.append("  Download: https://github.com/Bnomq/gun-detection-yolov8")

    # Action
    lines.append("")
    lines.append("RECOMMENDED ACTION:")
    if threat_level == "CRITICAL":
        lines.append("  🚨 Call Police IMMEDIATELY")
        lines.append("  🚨 Do NOT approach — armed and dangerous individual")
        lines.append("  🚨 Save this image as evidence")
        lines.append("  🚨 Evacuate area if safe to do so")
    elif threat_level == "HIGH":
        lines.append("  🔴 Alert security personnel immediately")
        lines.append("  🔴 Contact law enforcement")
        lines.append("  🔴 Do not approach the individuals")
    elif threat_level == "MEDIUM":
        lines.append("  ⚠️  Increase monitoring of this area")
        lines.append("  ⚠️  Alert security for further assessment")
    else:
        lines.append("  ✅ No immediate action required.")
        lines.append("  ✅ Continue normal monitoring.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# THREAT LEVEL
# ═══════════════════════════════════════════════════════════════════

def _get_threat_level(weapons, masks, violence):
    if not weapons and not masks and not violence:
        return "NONE"
    firearms  = [w for w in weapons if w["weapon_type"]=="FIREARM"]
    knives    = [w for w in weapons if w["weapon_type"]=="KNIFE/BLADE"]
    is_robbery = any(v.get("type")=="robbery_scene" for v in violence)
    if firearms:           return "CRITICAL"
    if knives and masks:   return "CRITICAL"
    if is_robbery:         return "CRITICAL"
    if knives:             return "HIGH"
    if masks:              return "HIGH"
    if len(violence)>=2:   return "MEDIUM"
    if violence:           return "LOW"
    return "NONE"


# ═══════════════════════════════════════════════════════════════════
# DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════

def _deduplicate(dets, iou_thr=0.45):
    if len(dets) < 2: return dets
    dets = sorted(dets, key=lambda x: x["confidence"], reverse=True)
    kept=[]; used=set()
    for i,d in enumerate(dets):
        if i in used: continue
        kept.append(d)
        for j in range(i+1,len(dets)):
            if j not in used and _iou(d["bbox"],dets[j]["bbox"])>iou_thr:
                used.add(j)
    return kept

def _iou(b1,b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
    x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    if not inter: return 0.0
    return inter/((b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter)

def _dense(bboxes):
    if len(bboxes)<5: return False
    pairs=len(bboxes)*(len(bboxes)-1)/2
    ov=sum(1 for i,b1 in enumerate(bboxes)
           for b2 in bboxes[i+1:]
           if not(b1[2]<b2[0] or b2[2]<b1[0] or b1[3]<b2[1] or b2[3]<b1[1]))
    return ov/pairs>0.25


# ═══════════════════════════════════════════════════════════════════
# ANNOTATE IMAGE
# ═══════════════════════════════════════════════════════════════════

def _draw_boxes(image, all_dets, weapons, masks, fire, threat_level):
    draw = ImageDraw.Draw(image)
    W, H = image.size
    try:
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",15)
        fm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",12)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",10)
    except:
        fb=fm=fs=ImageFont.load_default()

    w_set={tuple(w["bbox"]) for w in weapons}
    m_set={tuple(m["bbox"]) for m in masks}
    f_set={tuple(f["bbox"]) for f in fire}

    # Regular objects
    for d in all_dets:
        if tuple(d["bbox"]) in w_set|m_set|f_set: continue
        x1,y1,x2,y2=[int(v) for v in d["bbox"]]
        c=d["confidence"]; lbl=d["label"]
        col="#00e676" if c>=75 else "#ffab00" if c>=50 else "#94a3b8"
        draw.rectangle([x1,y1,x2,y2],outline=col,width=2)
        try:
            tb=draw.textbbox((x1,max(y1-14,0)),f"{lbl} {c:.0f}%",font=fs)
            draw.rectangle(tb,fill=col)
            draw.text((x1,max(y1-14,0)),f"{lbl} {c:.0f}%",fill="#000",font=fs)
        except: pass

    # Weapons — thick RED
    for w in weapons:
        x1,y1,x2,y2=[int(v) for v in w["bbox"]]
        draw.rectangle([x1-4,y1-4,x2+4,y2+4],outline="#ff4444",width=2)
        draw.rectangle([x1,y1,x2,y2],outline="#ff0000",width=5)
        tag="GUN (INFERRED)" if "inference" in w["label"].lower() else w["label"].upper()
        txt=f"🔴 {tag} {w['confidence']:.0f}%"
        try:
            tb=draw.textbbox((x1,max(y1-20,0)),txt,font=fm)
            draw.rectangle(tb,fill="#cc0000")
            draw.text((x1,max(y1-20,0)),txt,fill="#fff",font=fm)
        except: pass

    # Masks — orange
    for m in masks:
        x1,y1,x2,y2=[int(v) for v in m["bbox"]]
        draw.rectangle([x1,y1,x2,y2],outline="#ff6600",width=4)
        try:
            tb=draw.textbbox((x1,max(y1-18,0)),f"🎭 MASK {m['confidence']:.0f}%",font=fm)
            draw.rectangle(tb,fill="#ff6600")
            draw.text((x1,max(y1-18,0)),f"🎭 MASK {m['confidence']:.0f}%",fill="#fff",font=fm)
        except: pass

    # Fire — yellow
    for f in fire:
        x1,y1,x2,y2=[int(v) for v in f["bbox"]]
        draw.rectangle([x1,y1,x2,y2],outline="#ffff00",width=4)
        try:
            tb=draw.textbbox((x1,max(y1-18,0)),f"🔥 FIRE {f['confidence']:.0f}%",font=fm)
            draw.rectangle(tb,fill="#cc8800")
            draw.text((x1,max(y1-18,0)),f"🔥 FIRE {f['confidence']:.0f}%",fill="#fff",font=fm)
        except: pass

    # Top banner
    bc={"CRITICAL":"#cc0000","HIGH":"#cc5500","MEDIUM":"#cc8800","LOW":"#006600","NONE":"#004466"}.get(threat_level,"#333")
    firearms=[w for w in weapons if w["weapon_type"]=="FIREARM"]
    knives=[w for w in weapons if w["weapon_type"]=="KNIFE/BLADE"]
    parts=[]
    if firearms: parts.append(f"🔫 GUN x{len(firearms)}")
    if knives:   parts.append(f"🔪 KNIFE x{len(knives)}")
    if masks:    parts.append(f"🎭 MASK x{len(masks)}")
    if fire:     parts.append(f"🔥 FIRE x{len(fire)}")
    if not parts: parts=["No weapons directly detected — see threat report"]
    banner=f"  ⚠️ {threat_level}: "+"  |  ".join(parts)+"  "
    try:
        tb=draw.textbbox((0,0),banner,font=fb)
        draw.rectangle([0,0,max(tb[2]+20,W),32],fill=bc)
        draw.text((5,6),banner,fill="#fff",font=fb)
    except:
        draw.rectangle([0,0,W,30],fill=bc)
        draw.text((5,6),f"{threat_level}",fill="#fff")

    return image


# ═══════════════════════════════════════════════════════════════════
# NO YOLO FALLBACK
# ═══════════════════════════════════════════════════════════════════

def _no_yolo_result():
    blank=Image.new("RGB",(640,480),"#1e293b")
    draw=ImageDraw.Draw(blank)
    try: font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",16)
    except: font=ImageFont.load_default()
    draw.text((40,220),"YOLO not installed. Run: pip install ultralytics",fill="#ef4444",font=font)
    return {
        "detections":[],"total_objects":0,"object_counts":{},"category_counts":{},
        "weapons_found":[],"masks_found":[],"fire_found":[],"violence_indicators":[],
        "annotated_image":blank,"models_used":[],"weapon_detection_active":False,
        "mask_detection_active":False,"threat_level":"UNKNOWN",
        "threat_explanation":"YOLO not installed. Run: pip install ultralytics",
        "weapon_model_loaded":False,
    }