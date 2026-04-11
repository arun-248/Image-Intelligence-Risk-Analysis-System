"""
ml_risk_engine.py — ML-Powered Risk Analysis Engine
=====================================================
UPGRADE 1: RandomForest trained on synthetic data (replaces if-else rules)
UPGRADE 2: SHAP-style feature importance (built-in sklearn, no extra install)
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ═══════════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "has_firearm", "has_knife_non_kitchen", "has_weapon_generic",
    "has_fire_or_smoke", "has_ambulance", "has_fire_truck",
    "has_motorcycle", "has_ladder", "has_phone",
    "person_count", "vehicle_count",
    "scene_violence", "scene_robbery", "scene_weapon_threat",
    "scene_fire_emergency", "scene_accident", "scene_road",
    "scene_crowded_area", "scene_night_scene", "scene_warehouse",
    "scene_hospital", "scene_office", "scene_military",
    "scene_outdoor", "scene_indoor", "scene_kitchen", "scene_other",
    "has_potential_victim", "has_aggressive_crowd", "has_violent_coloring",
    "weapon_in_indoor", "weapon_with_crowd", "weapon_with_violence",
    "people_at_accident", "night_with_vehicle", "night_with_crowd",
]

RISK_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

SCENE_LIST = [
    "violence", "robbery", "weapon_threat", "fire_emergency", "accident",
    "road", "crowded_area", "night_scene", "warehouse", "hospital",
    "office", "military", "outdoor", "indoor", "kitchen"
]

# ═══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_features(detection_result: Dict, scene_result: Dict) -> np.ndarray:
    objects = detection_result.get("object_counts", {})
    categories = detection_result.get("category_counts", {})
    scene = scene_result.get("scene", "unknown")
    violence = detection_result.get("violence_indicators", [])

    def has_obj(keyword: str, min_conf: int = 50) -> float:
        for name, data in objects.items():
            if keyword.lower() in name.lower():
                if data.get("max_confidence", 0) >= min_conf:
                    return 1.0
        return 0.0

    def has_weapon_type(keywords: list, min_conf: int = 50) -> float:
        for name, data in objects.items():
            n = name.lower()
            if any(k in n for k in keywords):
                if data.get("max_confidence", 0) >= min_conf:
                    return 1.0
        return 0.0

    def person_count_norm() -> float:
        for name, data in objects.items():
            if "person" in name.lower():
                if data.get("max_confidence", 0) >= 50:
                    return min(data.get("count", 0), 30) / 30.0
        return 0.0

    def vehicle_count_norm() -> float:
        return min(categories.get("VEHICLE", 0), 10) / 10.0

    def has_violence_type(vtype: str) -> float:
        return 1.0 if any(v.get("type") == vtype for v in violence) else 0.0

    firearm = has_weapon_type(["gun", "rifle", "pistol", "handgun", "firearm", "shotgun"])
    weapon_any = max(firearm, has_weapon_type(["knife", "blade", "sword", "axe", "weapon"]))
    in_indoor = 1.0 if scene in ["indoor", "office", "hospital", "classroom"] else 0.0
    pcount_raw = min(sum(
        d.get("count", 0) for n, d in objects.items()
        if "person" in n.lower() and d.get("max_confidence", 0) >= 50
    ), 30)
    has_crowd = 1.0 if pcount_raw >= 8 else 0.0
    has_victim = has_violence_type("potential_victim")
    has_aggcrowd = has_violence_type("aggressive_crowd")
    has_vcolour = has_violence_type("violent_scene_coloring")
    any_violence = max(has_victim, has_aggcrowd, has_vcolour)
    has_vehicle = 1.0 if categories.get("VEHICLE", 0) >= 1 else 0.0

    scene_vec = [1.0 if scene == s else 0.0 for s in SCENE_LIST]
    scene_other = 1.0 if scene not in SCENE_LIST else 0.0

    features = [
        firearm,
        1.0 if (has_obj("knife") and scene != "kitchen") else 0.0,
        weapon_any,
        1.0 if detection_result.get("fire_found") else 0.0,
        has_obj("ambulance"),
        has_obj("fire truck"),
        has_obj("motorcycle"),
        has_obj("ladder"),
        has_obj("phone", 55),
        person_count_norm(),
        vehicle_count_norm(),
    ] + scene_vec + [scene_other] + [
        has_victim, has_aggcrowd, has_vcolour,
        weapon_any * in_indoor,
        weapon_any * has_crowd,
        weapon_any * any_violence,
        1.0 if (scene == "accident" and pcount_raw >= 1) else 0.0,
        1.0 if (scene == "night_scene" and has_vehicle) else 0.0,
        1.0 if (scene == "night_scene" and pcount_raw >= 10) else 0.0,
    ]

    return np.array(features, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATION
# ═══════════════════════════════════════════════════════════════════

def _generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, len(FEATURE_NAMES)), dtype=np.float32)
    y = []

    scenes = SCENE_LIST + ["unknown"]
    scene_base_risk = {
        "violence": 45, "robbery": 55, "weapon_threat": 50,
        "fire_emergency": 50, "accident": 40, "road": 18,
        "crowded_area": 25, "night_scene": 20, "warehouse": 20,
        "hospital": 12, "office": 8, "military": 35,
        "outdoor": 10, "indoor": 8, "kitchen": 12, "unknown": 8,
    }

    for i in range(n_samples):
        scene = rng.choice(scenes)
        has_firearm = rng.random() < 0.08
        has_knife = rng.random() < 0.06
        has_weapon = has_firearm or has_knife or (rng.random() < 0.04)
        has_fire = rng.random() < 0.05
        has_ambulance = rng.random() < 0.04
        has_fire_truck = rng.random() < 0.03
        has_moto = rng.random() < 0.08
        has_ladder = rng.random() < 0.07
        has_phone = rng.random() < 0.12
        person_n = int(rng.integers(0, 30))
        vehicle_n = int(rng.integers(0, 8))
        has_victim = rng.random() < 0.05
        has_aggcrowd = rng.random() < 0.06
        has_vcolour = rng.random() < 0.07

        in_indoor = scene in ["indoor", "office", "hospital", "classroom"]
        has_crowd = person_n >= 8
        any_violence = has_victim or has_aggcrowd or has_vcolour
        has_veh = vehicle_n >= 1

        score = scene_base_risk.get(scene, 8)
        if has_firearm:                           score += 55
        if has_knife and scene != "kitchen":      score += 40
        if has_weapon:                            score = max(score, score + 35)
        if scene == "violence":                   score += 45
        if has_victim:                            score += 40
        if has_aggcrowd:                          score += 35
        if has_vcolour:                           score += 25
        if has_weapon and in_indoor:              score += 50
        if has_weapon and has_crowd:              score += 50
        if has_weapon and any_violence:           score += 55
        if has_fire:                              score += 45
        if has_fire and person_n >= 1:            score += 10
        if scene == "accident":                   score += 38
        if has_ambulance:                         score += 30
        if scene == "road" and person_n >= 1 and vehicle_n >= 2: score += 25
        if person_n >= 25:                        score += 22
        elif person_n >= 15:                      score += 18
        elif person_n >= 8:                       score += 15
        if scene == "night_scene" and person_n >= 10: score += 20
        score = min(score, 100)

        if score >= 60:   label = "CRITICAL"
        elif score >= 35: label = "HIGH"
        elif score >= 18: label = "MEDIUM"
        else:             label = "LOW"

        scene_vec = [1.0 if scene == s else 0.0 for s in SCENE_LIST]
        scene_other = 1.0 if scene not in SCENE_LIST else 0.0

        row = [
            float(has_firearm),
            float(has_knife and scene != "kitchen"),
            float(has_weapon),
            float(has_fire),
            float(has_ambulance),
            float(has_fire_truck),
            float(has_moto),
            float(has_ladder),
            float(has_phone),
            min(person_n, 30) / 30.0,
            min(vehicle_n, 10) / 10.0,
        ] + scene_vec + [scene_other] + [
            float(has_victim),
            float(has_aggcrowd),
            float(has_vcolour),
            float(has_weapon and in_indoor),
            float(has_weapon and has_crowd),
            float(has_weapon and any_violence),
            float(scene == "accident" and person_n >= 1),
            float(scene == "night_scene" and has_veh),
            float(scene == "night_scene" and person_n >= 10),
        ]

        X[i] = row
        y.append(label)

    return X, np.array(y)


# ═══════════════════════════════════════════════════════════════════
# MODEL TRAINING & CACHING
# ═══════════════════════════════════════════════════════════════════

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "risk_rf_model.pkl")
_cached_model = None
_cached_importances = None


def _train_model() -> Tuple[RandomForestClassifier, np.ndarray, str]:
    print("[ML Risk Engine] Generating 5000 synthetic training samples...")
    X, y = _generate_synthetic_data(n_samples=5000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("[ML Risk Engine] Training RandomForestClassifier (200 trees)...")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=RISK_LABELS, zero_division=0)
    print("[ML Risk Engine] Training complete.\n", report)

    with open(_MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"[ML Risk Engine] Model saved to {_MODEL_PATH}")
    return clf, clf.feature_importances_, report


def get_model() -> Tuple[RandomForestClassifier, np.ndarray]:
    global _cached_model, _cached_importances
    if _cached_model is not None:
        return _cached_model, _cached_importances
    if os.path.exists(_MODEL_PATH):
        print("[ML Risk Engine] Loading saved model...")
        with open(_MODEL_PATH, "rb") as f:
            clf = pickle.load(f)
        _cached_model = clf
        _cached_importances = clf.feature_importances_
        return _cached_model, _cached_importances
    clf, importances, _ = _train_model()
    _cached_model = clf
    _cached_importances = importances
    return _cached_model, _cached_importances


# ═══════════════════════════════════════════════════════════════════
# SHAP-STYLE FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════

def compute_shap_scores(feature_vector: np.ndarray) -> List[Dict]:
    model, global_importances = get_model()
    local_importance = global_importances * feature_vector
    total = local_importance.sum()
    contributions = local_importance / total * 100 if total > 0 else local_importance

    results = []
    for idx, (name, value, contrib) in enumerate(zip(FEATURE_NAMES, feature_vector, contributions)):
        if value > 0 or contrib > 0.5:
            results.append({
                "feature": name,
                "activation": float(value),
                "global_importance": float(global_importances[idx]),
                "local_contribution": float(contrib),
                "feature_label": _prettify_feature(name),
            })
    results.sort(key=lambda x: x["local_contribution"], reverse=True)
    return results[:10]


def _prettify_feature(name: str) -> str:
    labels = {
        "has_firearm": "🔫 Firearm Detected",
        "has_knife_non_kitchen": "🔪 Knife (Non-Kitchen)",
        "has_weapon_generic": "⚔️ Generic Weapon",
        "has_fire_or_smoke": "🔥 Fire / Smoke",
        "has_ambulance": "🚑 Ambulance",
        "has_fire_truck": "🚒 Fire Truck",
        "has_motorcycle": "🏍️ Motorcycle",
        "has_ladder": "🪜 Ladder (Fall Risk)",
        "has_phone": "📱 Phone",
        "person_count": "👥 Person Count",
        "vehicle_count": "🚗 Vehicle Count",
        "has_potential_victim": "🚨 Potential Victim",
        "has_aggressive_crowd": "😤 Aggressive Crowd",
        "has_violent_coloring": "🎨 Violence Indicators",
        "weapon_in_indoor": "🔫🏢 Weapon Indoors",
        "weapon_with_crowd": "🔫👥 Weapon + Crowd",
        "weapon_with_violence": "🔫🚨 Weapon + Violence",
        "people_at_accident": "🚗👤 People at Accident",
        "night_with_vehicle": "🌙🚗 Night + Vehicle",
        "night_with_crowd": "🌙👥 Night + Crowd",
    }
    for scene in SCENE_LIST:
        labels[f"scene_{scene}"] = f"🗺️ Scene: {scene.replace('_', ' ').title()}"
    labels["scene_other"] = "🗺️ Scene: Other"
    return labels.get(name, name.replace("_", " ").title())


# ═══════════════════════════════════════════════════════════════════
# RISK SCORE ESTIMATION
# ═══════════════════════════════════════════════════════════════════

def _estimate_risk_score(feature_vector: np.ndarray, predicted_label: str) -> int:
    model, _ = get_model()
    probs = model.predict_proba(feature_vector.reshape(1, -1))[0]
    classes = list(model.classes_)
    class_midpoints = {"LOW": 9, "MEDIUM": 28, "HIGH": 52, "CRITICAL": 80}
    score = sum(class_midpoints.get(cls, 9) * prob for cls, prob in zip(classes, probs))
    return int(round(score))


# ═══════════════════════════════════════════════════════════════════
# MAIN FUNCTION (drop-in replacement for analyze_risk)
# ═══════════════════════════════════════════════════════════════════

RISK_COLORS = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444", "CRITICAL": "#aa00ff"}
RISK_EMOJIS = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🔴", "CRITICAL": "🚨"}


def analyze_risk_ml(detection_result: Dict, scene_result: Dict) -> Dict:
    model, _ = get_model()
    features = extract_features(detection_result, scene_result)

    label = model.predict(features.reshape(1, -1))[0]
    probs = model.predict_proba(features.reshape(1, -1))[0]
    classes = list(model.classes_)

    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
    risk_score = _estimate_risk_score(features, label)
    shap_scores = compute_shap_scores(features)

    triggered_rules = []
    for item in shap_scores:
        if item["activation"] > 0 and item["local_contribution"] > 0.5:
            triggered_rules.append({
                "name": item["feature"],
                "score_added": round(item["local_contribution"]),
                "explanation": item["feature_label"],
                "category": _feature_to_category(item["feature"]),
            })

    scene = scene_result.get("scene", "unknown")
    weapons = detection_result.get("weapons_found", [])
    violence = detection_result.get("violence_indicators", [])

    return {
        "risk_score": risk_score,
        "risk_level": label,
        "risk_color": RISK_COLORS[label],
        "risk_emoji": RISK_EMOJIS[label],
        "triggered_rules": triggered_rules,
        "total_rules_triggered": len(triggered_rules),
        "category_scores": _build_category_scores(shap_scores),
        "scene_base_risk": scene_result.get("base_risk_score", 8),
        "explanation": _build_explanation(label, risk_score, shap_scores, scene, prob_dict),
        "recommendations": _build_recommendations(label, shap_scores),
        "weapons_detected": len(weapons) > 0,
        "weapon_details": ", ".join(w["label"] for w in weapons) if weapons else "No weapons detected",
        "violence_detected": len(violence) > 0,
        "ml_powered": True,
        "model_type": "RandomForest (200 trees, trained on 5000 synthetic samples)",
        "class_probabilities": prob_dict,
        "shap_scores": shap_scores,
        "feature_vector": features.tolist(),
    }


def _feature_to_category(feature: str) -> str:
    mapping = {
        "has_firearm": "security", "has_knife_non_kitchen": "security",
        "has_weapon_generic": "security", "weapon_in_indoor": "crime",
        "weapon_with_crowd": "violence", "weapon_with_violence": "violence",
        "has_fire_or_smoke": "fire_safety", "has_ambulance": "emergency",
        "has_fire_truck": "emergency", "people_at_accident": "emergency",
        "has_potential_victim": "violence", "has_aggressive_crowd": "violence",
        "has_violent_coloring": "violence", "night_with_vehicle": "traffic_safety",
        "night_with_crowd": "crowd_safety", "person_count": "crowd_safety",
        "has_ladder": "workplace_safety", "has_phone": "traffic_safety",
    }
    if feature.startswith("scene_"):
        return "scene"
    return mapping.get(feature, "general")


def _build_category_scores(shap_scores: List[Dict]) -> Dict:
    cats = {}
    for item in shap_scores:
        cat = _feature_to_category(item["feature"])
        cats[cat] = cats.get(cat, 0) + item["local_contribution"]
    return {k: round(v) for k, v in cats.items() if v > 0}


def _build_explanation(label, score, shap_scores, scene, probs) -> str:
    lines = [
        f"**🤖 ML Risk Assessment: {label} ({score}/100)**",
        f"**Model:** RandomForest (200 trees, trained on 5,000 samples)",
        f"**Scene:** {scene.replace('_', ' ').title()}", "",
        "**Class Probabilities:**"
    ]
    for cls in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        p = probs.get(cls, 0)
        lines.append(f"  {cls:8s} {'█' * int(p * 20)} {p*100:.1f}%")
    active = [s for s in shap_scores if s["activation"] > 0]
    if active:
        lines += ["", "**🔍 XAI — Feature Contributions (SHAP-style):**"]
        for item in active[:6]:
            bar = "▓" * max(1, int(item["local_contribution"] / 5))
            lines.append(f"  {item['feature_label']}: {bar} {item['local_contribution']:.1f}%")
    else:
        lines.append("\n✅ No significant risk features activated.")
    return "\n".join(lines)


def _build_recommendations(label, shap_scores) -> List[str]:
    base = {
        "LOW":      ["✅ Scene appears safe. Continue normal monitoring."],
        "MEDIUM":   ["⚠️ Elevated risk detected.", "Increase monitoring level.", "Document situation."],
        "HIGH":     ["🔴 HIGH RISK — Alert security personnel immediately.", "Document and report.", "Prepare protective action."],
        "CRITICAL": ["🚨 CRITICAL — Contact emergency services immediately.", "Do NOT approach threat.", "Evacuate if safe."],
    }
    recs = list(base.get(label, []))
    active = {s["feature"] for s in shap_scores if s["activation"] > 0}
    if "has_firearm" in active or "has_weapon_generic" in active:
        recs += ["🔫 WEAPON DETECTED — Contact law enforcement immediately.", "Do NOT confront armed individuals."]
    if "has_fire_or_smoke" in active:
        recs.append("🔥 Fire/Smoke — Activate alarm and evacuate.")
    if "has_potential_victim" in active:
        recs.append("🚨 Potential victim — Request medical assistance.")
    if "has_aggressive_crowd" in active:
        recs.append("👥 Aggressive crowd — Deploy crowd control measures.")
    if "people_at_accident" in active:
        recs.append("🚑 Accident — Clear access routes for emergency vehicles.")
    return recs


def retrain_model():
    global _cached_model, _cached_importances
    clf, importances, report = _train_model()
    _cached_model = clf
    _cached_importances = importances
    return report