"""
risk_engine.py — Advanced Risk Analysis Engine
UPGRADED: Better weapon detection, threat patterns, detailed explanations
Realistic scoring with lowered thresholds for actual threat detection
"""

from typing import Dict, List, Tuple, Callable, Optional


# ═══════════════════════════════════════════════════════════════════
# RISK LEVEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

RISK_LOW = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH = "HIGH"
RISK_CRITICAL = "CRITICAL"


def score_to_level(score: int) -> str:
    """
    Convert numeric risk score to risk level.
    
    UPDATED: Lowered thresholds to catch real threats
    - CRITICAL: 60+ (was 65+)
    - HIGH: 35+ (was 40+)
    - MEDIUM: 18+ (was 20+)
    - LOW: <18
    """
    if score >= 60:
        return RISK_CRITICAL
    if score >= 35:
        return RISK_HIGH
    if score >= 18:
        return RISK_MEDIUM
    return RISK_LOW


RISK_COLORS = {
    RISK_LOW: "#22c55e",
    RISK_MEDIUM: "#f59e0b",
    RISK_HIGH: "#ef4444",
    RISK_CRITICAL: "#aa00ff"
}

RISK_EMOJIS = {
    RISK_LOW: "✅",
    RISK_MEDIUM: "⚠️",
    RISK_HIGH: "🔴",
    RISK_CRITICAL: "🚨"
}


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (LOWERED CONFIDENCE THRESHOLDS)
# ═══════════════════════════════════════════════════════════════════

def _has_firearm(objects: Dict[str, dict], min_confidence: int = 50) -> bool:
    """
    Check for firearms (LOWERED from 80% to 50%)
    """
    firearm_keywords = ["gun", "rifle", "pistol", "handgun", "firearm", "shotgun", "revolver"]
    
    for obj_name, obj_data in objects.items():
        obj_name_lower = obj_name.lower()
        if any(keyword in obj_name_lower for keyword in firearm_keywords):
            max_conf = obj_data.get("max_confidence", 0)
            if max_conf >= min_confidence:
                return True
    
    return False


def _has_knife(objects: Dict[str, dict], min_confidence: int = 50) -> bool:
    """
    Check for knives (LOWERED from 75% to 50%)
    """
    for obj_name, obj_data in objects.items():
        if "knife" in obj_name.lower():
            max_conf = obj_data.get("max_confidence", 0)
            if max_conf >= min_confidence:
                return True
    
    return False


def _has_weapon(objects: Dict[str, dict], min_confidence: int = 50) -> bool:
    """
    Check for any weapon (LOWERED from 75% to 50%)
    """
    weapon_keywords = [
        "gun", "rifle", "pistol", "handgun", "firearm", "shotgun",
        "knife", "blade", "sword", "dagger", "machete", "axe", "weapon"
    ]
    
    for obj_name, obj_data in objects.items():
        obj_name_lower = obj_name.lower()
        if any(keyword in obj_name_lower for keyword in weapon_keywords):
            max_conf = obj_data.get("max_confidence", 0)
            if max_conf >= min_confidence:
                return True
    
    return False


def _get_person_count(objects: Dict[str, dict], min_confidence: int = 50) -> int:
    """
    Get person count (LOWERED from 60% to 50%)
    """
    for obj_name, obj_data in objects.items():
        if "person" in obj_name.lower():
            max_conf = obj_data.get("max_confidence", 0)
            if max_conf >= min_confidence:
                return obj_data.get("count", 0)
    
    return 0


def _has_object(
    objects: Dict[str, dict],
    object_keyword: str,
    min_confidence: int = 45
) -> bool:
    """
    Check for specific object (LOWERED from 55% to 45%)
    """
    for obj_name, obj_data in objects.items():
        if object_keyword.lower() in obj_name.lower():
            max_conf = obj_data.get("max_confidence", 0)
            if max_conf >= min_confidence:
                return True
    
    return False


def _get_vehicle_count(categories: Dict[str, int]) -> int:
    """Get vehicle count"""
    return categories.get("VEHICLE", 0)


def _has_violence_indicators(violence_list: List[dict]) -> bool:
    """Check if violence indicators present"""
    return len(violence_list) > 0


def _get_weapon_details(weapons_list: List[dict]) -> Tuple[int, float, List[str]]:
    """
    Get weapon details: count, max confidence, and types
    """
    if not weapons_list:
        return (0, 0.0, [])
    
    count = len(weapons_list)
    max_conf = max(w.get("confidence", 0) for w in weapons_list)
    types = list(set(w.get("label", "unknown") for w in weapons_list))
    
    return (count, max_conf, types)


# ═══════════════════════════════════════════════════════════════════
# RISK RULES (UPDATED WITH WEAPON DETECTION)
# ═══════════════════════════════════════════════════════════════════

RULES: List[Tuple[str, Callable, int, str, str]] = [

    # ═══════════════════════════════════════════════════════════════
    # CRITICAL THREATS (Weapons, Violence, Emergencies)
    # ═══════════════════════════════════════════════════════════════
    
    (
        "firearm_detected",
        lambda o, c, s, d: _has_firearm(o, 50),
        55,
        "🔫 FIREARM DETECTED by weapon detection system (50%+ confidence)",
        "security"
    ),
    
    (
        "multiple_firearms",
        lambda o, c, s, d: len([k for k, v in o.items() 
                               if any(gun in k.lower() for gun in ["gun", "rifle", "pistol"]) 
                               and v.get("max_confidence", 0) >= 50]) >= 2,
        65,
        "🔫🔫 MULTIPLE FIREARMS detected — extreme threat",
        "security"
    ),
    
    (
        "knife_non_kitchen",
        lambda o, c, s, d: _has_knife(o, 50) and s not in ["kitchen"],
        40,
        "🔪 Knife detected in non-kitchen environment",
        "security"
    ),
    
    (
        "weapon_generic",
        lambda o, c, s, d: _has_weapon(o, 50),
        35,
        "⚔️ Weapon detected by detection system",
        "security"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # VIOLENCE INDICATORS
    # ═══════════════════════════════════════════════════════════════
    
    (
        "violence_scene_detected",
        lambda o, c, s, d: s == "violence",
        45,
        "🚨 Violence scene detected by scene classifier",
        "violence"
    ),
    
    (
        "potential_victim_detected",
        lambda o, c, s, d: any(v.get("type") == "potential_victim" 
                              for v in d.get("violence_indicators", [])),
        40,
        "🚨 Potential victim detected (person lying down with others present)",
        "violence"
    ),
    
    (
        "aggressive_crowd_pattern",
        lambda o, c, s, d: any(v.get("type") == "aggressive_crowd" 
                              for v in d.get("violence_indicators", [])),
        35,
        "👥 Aggressive crowd formation detected",
        "violence"
    ),
    
    (
        "violent_scene_coloring",
        lambda o, c, s, d: any(v.get("type") == "violent_scene_coloring" 
                              for v in d.get("violence_indicators", [])),
        25,
        "🎨 Violence indicators in scene coloring/atmosphere",
        "violence"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # ARMED THREAT COMBINATIONS
    # ═══════════════════════════════════════════════════════════════
    
    (
        "robbery_scene",
        lambda o, c, s, d: s == "robbery",
        50,
        "🔫 Armed robbery scene detected",
        "crime"
    ),
    
    (
        "weapon_threat_scene",
        lambda o, c, s, d: s == "weapon_threat",
        45,
        "⚔️ Weapon threat scene detected",
        "crime"
    ),
    
    (
        "armed_civilian_indoor",
        lambda o, c, s, d: _has_weapon(o, 50) and s in ["indoor", "office", "hospital", "classroom"],
        50,
        "🔫 Weapon detected in civilian indoor space — immediate threat",
        "crime"
    ),
    
    (
        "weapon_with_crowd",
        lambda o, c, s, d: _has_weapon(o, 50) and _get_person_count(o) >= 8,
        50,
        "🚨 Weapon detected with crowd (8+ people) — mass casualty risk",
        "violence"
    ),
    
    (
        "armed_night_scene",
        lambda o, c, s, d: _has_weapon(o, 50) and s == "night_scene",
        42,
        "🌙🔫 Weapon detected in night scene — high danger",
        "crime"
    ),
    
    (
        "weapon_with_violence",
        lambda o, c, s, d: _has_weapon(o, 50) and _has_violence_indicators(d.get("violence_indicators", [])),
        55,
        "🚨⚔️ Weapon + violence indicators — CRITICAL THREAT",
        "violence"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # FIRE AND EMERGENCIES
    # ═══════════════════════════════════════════════════════════════
    
    (
        "fire_emergency",
        lambda o, c, s, d: s == "fire_emergency",
        45,
        "🔥 Fire or smoke emergency detected",
        "fire_safety"
    ),
    
    (
        "fire_with_people",
        lambda o, c, s, d: s == "fire_emergency" and _get_person_count(o) >= 1,
        55,
        "🔥👤 People in fire emergency — life-threatening",
        "fire_safety"
    ),
    
    (
        "fire_truck_present",
        lambda o, c, s, d: _has_object(o, "fire truck", 50),
        32,
        "🚒 Fire truck detected — active emergency response",
        "emergency"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # ACCIDENT SCENES
    # ═══════════════════════════════════════════════════════════════
    
    (
        "accident_scene",
        lambda o, c, s, d: s == "accident",
        38,
        "🚗💥 Vehicle accident scene",
        "emergency"
    ),
    
    (
        "ambulance_present",
        lambda o, c, s, d: _has_object(o, "ambulance", 50),
        30,
        "🚑 Ambulance detected — medical emergency",
        "emergency"
    ),
    
    (
        "people_at_accident",
        lambda o, c, s, d: s == "accident" and _get_person_count(o) >= 1,
        40,
        "🚗💥👤 People at accident scene — possible casualties",
        "emergency"
    ),
    
    (
        "multi_vehicle_accident",
        lambda o, c, s, d: s == "accident" and _get_vehicle_count(c) >= 3,
        42,
        "🚗🚗🚗 Multiple vehicles in accident — major collision",
        "emergency"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # TRAFFIC SAFETY
    # ═══════════════════════════════════════════════════════════════
    
    (
        "phone_while_driving",
        lambda o, c, s, d: _has_object(o, "phone", 55) and s == "road" and _get_person_count(o) >= 1,
        28,
        "📱🚗 Phone in road scene — distracted driving risk",
        "traffic_safety"
    ),
    
    (
        "pedestrian_in_traffic",
        lambda o, c, s, d: s == "road" and _get_person_count(o) >= 1 and _get_vehicle_count(c) >= 2,
        25,
        "🚶🚗 Pedestrian in active traffic with multiple vehicles",
        "traffic_safety"
    ),
    
    (
        "vehicle_in_crowd",
        lambda o, c, s, d: _get_vehicle_count(c) >= 1 and _get_person_count(o) >= 15,
        30,
        "🚗👥 Vehicle in crowded area (15+ people) — pedestrian danger",
        "traffic_safety"
    ),
    
    (
        "night_driving",
        lambda o, c, s, d: s == "night_scene" and _get_vehicle_count(c) >= 1,
        18,
        "🌙🚗 Vehicle in night scene — reduced visibility",
        "traffic_safety"
    ),
    
    (
        "motorcycle_no_helmet",
        lambda o, c, s, d: _has_object(o, "motorcycle", 55) and s in ["road", "outdoor"],
        20,
        "🏍️ Motorcycle detected — verify helmet use",
        "traffic_safety"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # CROWD SAFETY (LOWERED THRESHOLDS)
    # ═══════════════════════════════════════════════════════════════
    
    (
        "extreme_crowd",
        lambda o, c, s, d: _get_person_count(o) >= 25,
        22,
        "👥👥👥 Extreme crowd (25+ people) — serious crowd management needed",
        "crowd_safety"
    ),
    
    (
        "very_large_crowd",
        lambda o, c, s, d: _get_person_count(o) >= 15 and _get_person_count(o) < 25,
        18,
        "👥👥 Very large crowd (15-24 people) — monitor for safety",
        "crowd_safety"
    ),
    
    (
        "large_crowd",
        lambda o, c, s, d: _get_person_count(o) >= 8 or s == "crowded_area",
        15,
        "👥 Large crowd (8+ people) detected",
        "crowd_safety"
    ),
    
    (
        "night_crowd",
        lambda o, c, s, d: s == "night_scene" and _get_person_count(o) >= 10,
        20,
        "🌙👥 Large crowd in night scene — safety monitoring required",
        "crowd_safety"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # WORKPLACE SAFETY
    # ═══════════════════════════════════════════════════════════════
    
    (
        "ladder_hazard",
        lambda o, c, s, d: _has_object(o, "ladder", 50),
        12,
        "🪜 Ladder detected — fall hazard",
        "workplace_safety"
    ),
    
    (
        "workers_industrial",
        lambda o, c, s, d: s == "warehouse" and _get_person_count(o) >= 1,
        12,
        "🏭 Workers in industrial area — verify PPE compliance",
        "workplace_safety"
    ),
    
    (
        "height_work",
        lambda o, c, s, d: _has_object(o, "ladder", 50) and _get_person_count(o) >= 1,
        18,
        "🪜👤 Person working at height — fall protection required",
        "workplace_safety"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # HEALTH AND MEDICAL
    # ═══════════════════════════════════════════════════════════════
    
    (
        "phone_in_hospital",
        lambda o, c, s, d: _has_object(o, "phone", 55) and s == "hospital",
        10,
        "📱🏥 Phone in hospital — equipment interference risk",
        "health_safety"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # MILITARY CONTEXT (Lower risk - legitimate use)
    # ═══════════════════════════════════════════════════════════════
    
    (
        "military_scene",
        lambda o, c, s, d: s == "military",
        0,  # No additional risk - already in base score
        "🪖 Military/law enforcement context detected",
        "security"
    ),
    
]


# ═══════════════════════════════════════════════════════════════════
# CATEGORY CAPS
# ═══════════════════════════════════════════════════════════════════

CATEGORY_CAPS = {
    "security": 40,          # Weapons (increased from 35)
    "crime": 40,             # Robbery, threats
    "violence": 40,          # Violence, aggression
    "emergency": 35,         # Accidents, medical
    "fire_safety": 35,       # Fire, smoke
    "traffic_safety": 28,    # Road safety
    "crowd_safety": 22,      # Crowd management
    "workplace_safety": 18,  # Industrial hazards
    "health_safety": 12,     # Medical facility
}


# ═══════════════════════════════════════════════════════════════════
# MAIN RISK ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════

def analyze_risk(detection_result: Dict, scene_result: Dict) -> Dict:
    """
    Advanced risk analysis with weapon detection and detailed explanations.
    
    UPGRADED:
    - Detects actual weapons from YOLO-OIV7
    - Violence pattern detection
    - Detailed explanations with evidence
    - Lowered thresholds for real threat detection
    
    Args:
        detection_result: Output from detect_objects()
        scene_result: Output from classify_scene()
        
    Returns:
        Complete risk analysis with detailed breakdown
    """
    # Extract data
    objects = detection_result.get("object_counts", {})
    categories = detection_result.get("category_counts", {})
    scene = scene_result.get("scene", "unknown")
    base_risk = scene_result.get("base_risk_score", 8)
    is_dangerous = scene_result.get("is_dangerous", False)
    
    # NEW: Extract weapon and violence data
    weapons_found = detection_result.get("weapons_found", [])
    violence_indicators = detection_result.get("violence_indicators", [])
    
    # Initialize
    category_scores = {category: 0 for category in CATEGORY_CAPS.keys()}
    triggered_rules = []
    
    # ───────────────────────────────────────────────────────────────
    # Evaluate all rules
    # ───────────────────────────────────────────────────────────────
    for rule_name, condition_func, score, explanation, category in RULES:
        try:
            # Pass full detection result for violence/weapon checks
            if condition_func(objects, categories, scene, detection_result):
                
                current_score = category_scores[category]
                category_cap = CATEGORY_CAPS[category]
                
                new_score = min(current_score + score, category_cap)
                points_added = new_score - current_score
                
                category_scores[category] = new_score
                
                if points_added > 0:
                    triggered_rules.append({
                        "name": rule_name,
                        "score_added": points_added,
                        "explanation": explanation,
                        "category": category,
                    })
        
        except Exception as e:
            print(f"Rule '{rule_name}' error: {e}")
            continue
    
    # ───────────────────────────────────────────────────────────────
    # Calculate total score
    # ───────────────────────────────────────────────────────────────
    total_score = base_risk + sum(category_scores.values())
    total_score = min(total_score, 100)
    
    risk_level = score_to_level(total_score)
    
    # ───────────────────────────────────────────────────────────────
    # Generate detailed explanation and recommendations
    # ───────────────────────────────────────────────────────────────
    explanation = _generate_detailed_explanation(
        risk_level,
        total_score,
        triggered_rules,
        scene,
        objects,
        weapons_found,
        violence_indicators
    )
    
    recommendations = _generate_recommendations(risk_level, triggered_rules, weapons_found)
    
    # ───────────────────────────────────────────────────────────────
    # Return complete analysis
    # ───────────────────────────────────────────────────────────────
    return {
        "risk_score": total_score,
        "risk_level": risk_level,
        "risk_color": RISK_COLORS[risk_level],
        "risk_emoji": RISK_EMOJIS[risk_level],
        "triggered_rules": triggered_rules,
        "total_rules_triggered": len(triggered_rules),
        "category_scores": {k: v for k, v in category_scores.items() if v > 0},
        "scene_base_risk": base_risk,
        "explanation": explanation,
        "recommendations": recommendations,
        "weapons_detected": len(weapons_found) > 0,
        "weapon_details": _format_weapon_details(weapons_found),
        "violence_detected": len(violence_indicators) > 0,
    }


# ═══════════════════════════════════════════════════════════════════
# DETAILED EXPLANATION GENERATOR
# ═══════════════════════════════════════════════════════════════════

def _generate_detailed_explanation(
    level: str,
    score: int,
    triggered: List[Dict],
    scene: str,
    objects: Dict[str, dict],
    weapons: List[dict],
    violence: List[dict]
) -> str:
    """Generate detailed, evidence-based explanation"""
    lines = []
    
    # Header
    lines.append(f"**Risk Assessment: {level} ({score}/100)**")
    lines.append(f"**Scene Type:** {scene.replace('_', ' ').title()}")
    
    # Weapon detection (if any)
    if weapons:
        lines.append("\n**⚠️ WEAPONS DETECTED:**")
        for w in weapons:
            lines.append(f"• {w['label']} ({w['confidence']:.0f}% confidence) - Detected by {w['source']}")
    
    # Violence indicators (if any)
    if violence:
        lines.append("\n**🚨 VIOLENCE INDICATORS:**")
        for v in violence:
            reason = v.get("reason", v.get("type", "Unknown"))
            conf = v.get("confidence", 0)
            lines.append(f"• {reason} ({conf:.0f}% confidence)")
    
    # Object summary
    if objects:
        object_list = [(name, data.get("count", 0)) for name, data in objects.items()]
        top_objects = sorted(object_list, key=lambda x: x[1], reverse=True)[:6]
        obj_str = ", ".join(f"{count}×{name}" for name, count in top_objects)
        lines.append(f"\n**Detected Objects:** {obj_str}")
    
    # Risk factors
    if triggered:
        lines.append("\n**Risk Factors Identified:**")
        for i, rule in enumerate(triggered, 1):
            lines.append(f"{i}. {rule['explanation']} (+{rule['score_added']} points)")
    else:
        lines.append("\n✅ No significant risk factors detected")
    
    return "\n".join(lines)


def _format_weapon_details(weapons: List[dict]) -> str:
    """Format weapon details for display"""
    if not weapons:
        return "No weapons detected"
    
    details = []
    for w in weapons:
        details.append(f"{w['label']} ({w['confidence']:.0f}% conf)")
    
    return ", ".join(details)


# ═══════════════════════════════════════════════════════════════════
# RECOMMENDATIONS GENERATOR
# ═══════════════════════════════════════════════════════════════════

def _generate_recommendations(
    level: str,
    triggered: List[Dict],
    weapons: List[dict]
) -> List[str]:
    """Generate actionable recommendations"""
    recommendations = []
    
    # Base recommendations by level
    base_recs = {
        RISK_LOW: [
            "Scene appears safe. Continue normal monitoring."
        ],
        RISK_MEDIUM: [
            "Increase monitoring level.",
            "Document the situation for records."
        ],
        RISK_HIGH: [
            "⚠️ IMMEDIATE ATTENTION REQUIRED",
            "Alert security personnel immediately.",
            "Document and report the situation.",
            "Prepare to take protective action."
        ],
        RISK_CRITICAL: [
            "🚨 CRITICAL THREAT - IMMEDIATE ACTION REQUIRED",
            "Contact emergency services (911) immediately.",
            "Do not approach the threat.",
            "Evacuate the area if safe to do so.",
            "Secure all personnel."
        ]
    }
    
    recommendations.extend(base_recs.get(level, []))
    
    # Weapon-specific recommendations
    if weapons:
        recommendations.append("🔫 WEAPONS DETECTED - Contact law enforcement immediately")
        recommendations.append("Do NOT confront armed individuals")
        recommendations.append("Follow active shooter protocols if applicable")
    
    # Category-specific recommendations
    triggered_categories = {rule["category"] for rule in triggered}
    
    if "violence" in triggered_categories:
        recommendations.append("🚨 Potential violence - Do not approach without backup")
    
    if "fire_safety" in triggered_categories:
        recommendations.append("🔥 Fire emergency - Activate fire alarm and evacuate")
    
    if "emergency" in triggered_categories:
        recommendations.append("🚑 Medical emergency - Clear access routes for ambulances")
    
    if "traffic_safety" in triggered_categories:
        recommendations.append("🚗 Traffic safety concern - Alert traffic management")
    
    if "crowd_safety" in triggered_categories:
        recommendations.append("👥 Crowd management - Deploy crowd control measures")
    
    return recommendations


# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_risk_summary(risk_result: Dict) -> str:
    """Generate one-line risk summary"""
    level = risk_result.get("risk_level", "UNKNOWN")
    score = risk_result.get("risk_score", 0)
    emoji = risk_result.get("risk_emoji", "❓")
    triggered = risk_result.get("total_rules_triggered", 0)
    weapons = risk_result.get("weapons_detected", False)
    
    weapon_str = " | WEAPONS DETECTED" if weapons else ""
    
    return f"{emoji} {level} ({score}/100) — {triggered} risk factor(s){weapon_str}"