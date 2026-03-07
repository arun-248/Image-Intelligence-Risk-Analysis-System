"""
risk_engine.py — Risk Analysis and Scoring Engine
PRODUCTION VERSION: Conservative, realistic risk assessment
FIX #3: Significantly raised thresholds (5 people ≠ extreme crowd)
FIX #4: All rules check confidence levels before triggering
FIX #5: Category caps prevent score stacking
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
    Convert numeric risk score to risk level category.
    
    FIX #5: Thresholds adjusted for more realistic distribution
    
    Args:
        score: Risk score (0-100)
        
    Returns:
        Risk level string
    """
    if score >= 65:
        return RISK_CRITICAL
    if score >= 40:
        return RISK_HIGH
    if score >= 20:
        return RISK_MEDIUM
    return RISK_LOW


# Risk level color coding for UI display
RISK_COLORS = {
    RISK_LOW: "#22c55e",      # Green
    RISK_MEDIUM: "#f59e0b",   # Amber
    RISK_HIGH: "#ef4444",     # Red
    RISK_CRITICAL: "#aa00ff"  # Purple
}

# Risk level emoji indicators
RISK_EMOJIS = {
    RISK_LOW: "✅",
    RISK_MEDIUM: "⚠️",
    RISK_HIGH: "🔴",
    RISK_CRITICAL: "🚨"
}


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR RULE CONDITIONS
# FIX #4: All helpers check confidence thresholds
# ═══════════════════════════════════════════════════════════════════

def _has_firearm(
    objects: Dict[str, dict], 
    min_confidence: int = 80
) -> bool:
    """
    Check if a firearm was detected with high confidence.
    
    FIX #4: Only returns True if confidence >= min_confidence
    
    Args:
        objects: Object counts dictionary from detection
        min_confidence: Minimum confidence percentage (default 80%)
        
    Returns:
        True if high-confidence firearm detected
    """
    firearm_keywords = ["gun", "rifle", "pistol", "handgun", "firearm", "shotgun", "revolver"]
    
    for obj_name, obj_data in objects.items():
        obj_name_lower = obj_name.lower()
        
        # Check if this is a firearm
        if any(keyword in obj_name_lower for keyword in firearm_keywords):
            # Check confidence level
            max_conf = obj_data.get("max_confidence", 0)
            if max_conf >= min_confidence:
                return True
    
    return False


def _has_knife(
    objects: Dict[str, dict], 
    min_confidence: int = 75
) -> bool:
    """
    Check if a knife was detected with high confidence.
    
    FIX #4: Only returns True if confidence >= min_confidence
    
    Args:
        objects: Object counts dictionary from detection
        min_confidence: Minimum confidence percentage (default 75%)
        
    Returns:
        True if high-confidence knife detected
    """
    for obj_name, obj_data in objects.items():
        if "knife" in obj_name.lower():
            max_conf = obj_data.get("max_confidence", 0)
            if max_conf >= min_confidence:
                return True
    
    return False


def _has_weapon(
    objects: Dict[str, dict], 
    min_confidence: int = 75
) -> bool:
    """
    Check if any weapon was detected with high confidence.
    
    FIX #4: Only returns True if confidence >= min_confidence
    
    Args:
        objects: Object counts dictionary from detection
        min_confidence: Minimum confidence percentage (default 75%)
        
    Returns:
        True if high-confidence weapon detected
    """
    weapon_keywords = [
        "gun", "rifle", "pistol", "handgun", "firearm", "shotgun", 
        "knife", "blade", "sword", "dagger", "machete", "axe"
    ]
    
    for obj_name, obj_data in objects.items():
        obj_name_lower = obj_name.lower()
        
        if any(keyword in obj_name_lower for keyword in weapon_keywords):
            max_conf = obj_data.get("max_confidence", 0)
            if max_conf >= min_confidence:
                return True
    
    return False


def _get_person_count(
    objects: Dict[str, dict], 
    min_confidence: int = 60
) -> int:
    """
    Get count of people detected with sufficient confidence.
    
    FIX #4: Only counts persons with confidence >= min_confidence
    
    Args:
        objects: Object counts dictionary from detection
        min_confidence: Minimum confidence percentage (default 60%)
        
    Returns:
        Number of high-confidence people detected
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
    min_confidence: int = 55
) -> bool:
    """
    Check if a specific object type was detected with sufficient confidence.
    
    FIX #4: Only returns True if confidence >= min_confidence
    
    Args:
        objects: Object counts dictionary from detection
        object_keyword: Keyword to search for (e.g., "phone", "car")
        min_confidence: Minimum confidence percentage (default 55%)
        
    Returns:
        True if high-confidence object detected
    """
    for obj_name, obj_data in objects.items():
        if object_keyword.lower() in obj_name.lower():
            max_conf = obj_data.get("max_confidence", 0)
            if max_conf >= min_confidence:
                return True
    
    return False


def _get_vehicle_count(categories: Dict[str, int]) -> int:
    """
    Get total count of vehicles detected.
    
    Args:
        categories: Category counts from detection
        
    Returns:
        Number of vehicles
    """
    return categories.get("VEHICLE", 0)


# ═══════════════════════════════════════════════════════════════════
# RISK RULES
# FIX #3: Significantly raised thresholds for crowd-based rules
# FIX #4: All rules use helper functions that check confidence
# ═══════════════════════════════════════════════════════════════════

# Rule format: (name, condition_function, score, explanation, category)
RULES: List[Tuple[str, Callable, int, str, str]] = [

    # ═══════════════════════════════════════════════════════════════
    # CRITICAL SECURITY THREATS (Highest Priority)
    # ═══════════════════════════════════════════════════════════════
    
    (
        "firearm_confirmed",
        lambda o, c, s, d: _has_firearm(o, min_confidence=80),
        50,
        "🔫 Firearm CONFIRMED by YOLO with 80%+ confidence — CRITICAL security alert.",
        "security"
    ),
    
    (
        "knife_non_kitchen",
        lambda o, c, s, d: _has_knife(o, 75) and s not in ["kitchen"],
        35,
        "🔪 Knife detected in non-kitchen environment with 75%+ confidence.",
        "security"
    ),
    
    (
        "weapon_generic",
        lambda o, c, s, d: _has_weapon(o, 75) and not _has_firearm(o, 80) and not _has_knife(o, 75),
        30,
        "⚔️ Weapon detected with 75%+ confidence.",
        "security"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # DANGEROUS SCENE CLASSIFICATIONS
    # ═══════════════════════════════════════════════════════════════
    
    (
        "robbery_scene",
        lambda o, c, s, d: s == "robbery",
        45,
        "🔫 Armed robbery scene detected by scene classifier.",
        "crime"
    ),
    
    (
        "weapon_threat_scene",
        lambda o, c, s, d: s == "weapon_threat",
        40,
        "⚔️ Weapon threat scene detected by scene classifier.",
        "crime"
    ),
    
    (
        "violence_scene",
        lambda o, c, s, d: s == "violence",
        35,
        "🚨 Violence or altercation scene detected by scene classifier.",
        "violence"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # WEAPON + CROWD COMBINATIONS
    # FIX #3: Raised threshold from 5 to 15 people
    # ═══════════════════════════════════════════════════════════════
    
    (
        "armed_crowded_area",
        lambda o, c, s, d: _has_weapon(o, 80) and _get_person_count(o) >= 15,
        45,
        "🚨 Weapon detected in crowded area (15+ people) — mass casualty risk.",
        "violence"
    ),
    
    (
        "armed_medium_crowd",
        lambda o, c, s, d: _has_weapon(o, 80) and _get_person_count(o) >= 8 and _get_person_count(o) < 15,
        35,
        "⚠️ Weapon detected with multiple people present (8-14 people).",
        "violence"
    ),
    
    (
        "armed_indoors",
        lambda o, c, s, d: _has_weapon(o, 80) and s in ["indoor", "office", "warehouse", "hospital"],
        40,
        "🔫 Weapon detected in indoor environment — immediate security concern.",
        "crime"
    ),
    
    (
        "armed_night",
        lambda o, c, s, d: _has_weapon(o, 80) and s == "night_scene",
        38,
        "🌙🔫 Weapon detected in night-time scene — increased danger.",
        "crime"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # FIRE AND EMERGENCY SITUATIONS
    # ═══════════════════════════════════════════════════════════════
    
    (
        "fire_emergency",
        lambda o, c, s, d: s == "fire_emergency",
        40,
        "🔥 Fire or smoke emergency scene detected.",
        "fire_safety"
    ),
    
    (
        "fire_with_people",
        lambda o, c, s, d: s == "fire_emergency" and _get_person_count(o) >= 1,
        50,
        "🔥👤 People present in fire emergency scene — life-threatening situation.",
        "fire_safety"
    ),
    
    (
        "fire_truck_present",
        lambda o, c, s, d: _has_object(o, "fire truck", 60),
        30,
        "🚒 Fire truck detected — active fire emergency response.",
        "emergency"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # ACCIDENT SCENES
    # ═══════════════════════════════════════════════════════════════
    
    (
        "accident_scene",
        lambda o, c, s, d: s == "accident",
        35,
        "🚗💥 Vehicle accident scene detected.",
        "emergency"
    ),
    
    (
        "ambulance_present",
        lambda o, c, s, d: _has_object(o, "ambulance", 60),
        28,
        "🚑 Ambulance detected — medical emergency in progress.",
        "emergency"
    ),
    
    (
        "people_at_accident",
        lambda o, c, s, d: s == "accident" and _get_person_count(o) >= 1,
        38,
        "🚗💥👤 People at accident scene — possible casualties.",
        "emergency"
    ),
    
    (
        "multi_vehicle_accident",
        lambda o, c, s, d: s == "accident" and _get_vehicle_count(c) >= 3,
        40,
        "🚗🚗🚗 Multiple vehicles in accident scene — major collision.",
        "emergency"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # TRAFFIC SAFETY VIOLATIONS
    # ═══════════════════════════════════════════════════════════════
    
    (
        "phone_while_driving",
        lambda o, c, s, d: _has_object(o, "phone", 60) and s == "road" and _get_person_count(o) >= 1,
        25,
        "📱🚗 Phone detected in road scene — distracted driving risk.",
        "traffic_safety"
    ),
    
    (
        "pedestrian_in_traffic",
        lambda o, c, s, d: s == "road" and _get_person_count(o) >= 1 and _get_vehicle_count(c) >= 2,
        22,
        "🚶🚗 Pedestrian in active traffic area with multiple vehicles.",
        "traffic_safety"
    ),
    
    (
        "vehicle_in_crowd",
        lambda o, c, s, d: _get_vehicle_count(c) >= 1 and _get_person_count(o) >= 20,
        28,
        "🚗👥 Vehicle in very crowded area (20+ people) — pedestrian safety concern.",
        "traffic_safety"
    ),
    
    (
        "night_driving",
        lambda o, c, s, d: s == "night_scene" and _get_vehicle_count(c) >= 1,
        15,
        "🌙🚗 Vehicle in night-time scene — reduced visibility risk.",
        "traffic_safety"
    ),
    
    (
        "motorcycle_no_helmet",
        lambda o, c, s, d: _has_object(o, "motorcycle", 60) and s in ["road", "outdoor"] and not _has_object(o, "helmet", 50),
        18,
        "🏍️ Motorcycle detected without confirmed helmet — safety violation.",
        "traffic_safety"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # CROWD SAFETY
    # FIX #3: Significantly raised thresholds (was 5, now 25 for "extreme")
    # ═══════════════════════════════════════════════════════════════
    
    (
        "extreme_crowd",
        lambda o, c, s, d: _get_person_count(o) >= 40,
        22,
        "👥👥👥 Extreme crowd density (40+ people) — serious crowd management needed.",
        "crowd_safety"
    ),
    
    (
        "very_large_crowd",
        lambda o, c, s, d: _get_person_count(o) >= 25 and _get_person_count(o) < 40,
        18,
        "👥👥 Very large crowd (25-39 people) — crowd control recommended.",
        "crowd_safety"
    ),
    
    (
        "large_crowd",
        lambda o, c, s, d: _get_person_count(o) >= 15 or s == "crowded_area",
        15,
        "👥 Large crowd (15+ people) detected — monitor for safety.",
        "crowd_safety"
    ),
    
    (
        "night_crowd",
        lambda o, c, s, d: s == "night_scene" and _get_person_count(o) >= 15,
        18,
        "🌙👥 Large crowd in night-time scene — safety monitoring required.",
        "crowd_safety"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # WORKPLACE SAFETY
    # ═══════════════════════════════════════════════════════════════
    
    (
        "ladder_hazard",
        lambda o, c, s, d: _has_object(o, "ladder", 55),
        12,
        "🪜 Ladder detected — fall hazard present.",
        "workplace_safety"
    ),
    
    (
        "workers_industrial",
        lambda o, c, s, d: s == "warehouse" and _get_person_count(o) >= 1,
        10,
        "🏭 Workers in industrial area — PPE compliance check recommended.",
        "workplace_safety"
    ),
    
    (
        "height_work",
        lambda o, c, s, d: _has_object(o, "ladder", 55) and _get_person_count(o) >= 1,
        15,
        "🪜👤 Person working at height — fall protection required.",
        "workplace_safety"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # HEALTH AND MEDICAL SAFETY
    # ═══════════════════════════════════════════════════════════════
    
    (
        "phone_in_hospital",
        lambda o, c, s, d: _has_object(o, "phone", 60) and s == "hospital",
        10,
        "📱🏥 Mobile phone in hospital — potential equipment interference.",
        "health_safety"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # SECURITY CONCERNS (Minor)
    # ═══════════════════════════════════════════════════════════════
    
    (
        "backpack_very_large_crowd",
        lambda o, c, s, d: _has_object(o, "backpack", 50) and _get_person_count(o) >= 30,
        12,
        "🎒 Backpack in very large crowd (30+ people) — security awareness.",
        "security"
    ),
    
    (
        "night_person_outdoor",
        lambda o, c, s, d: s == "night_scene" and _get_person_count(o) >= 1 and s in ["outdoor", "parking"],
        8,
        "🌙 Person in night-time outdoor scene — visibility concern.",
        "security"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # PROPERTY PROTECTION
    # ═══════════════════════════════════════════════════════════════
    
    (
        "electronics_outdoor",
        lambda o, c, s, d: (_has_object(o, "laptop", 55) or _has_object(o, "tv", 55)) and s == "outdoor",
        8,
        "💻 Electronics in outdoor scene — theft or weather damage risk.",
        "property"
    ),
]


# ═══════════════════════════════════════════════════════════════════
# CATEGORY CAPS (Prevents unbounded score stacking)
# FIX #5: Each category has a maximum contribution to total risk score
# ═══════════════════════════════════════════════════════════════════

CATEGORY_CAPS = {
    "security": 35,          # Weapons, threats
    "crime": 35,             # Robbery, criminal activity
    "violence": 35,          # Violence, aggression
    "emergency": 30,         # Accidents, medical emergencies
    "fire_safety": 30,       # Fire, smoke, explosions
    "traffic_safety": 25,    # Road safety violations
    "crowd_safety": 20,      # Crowd management
    "workplace_safety": 15,  # Industrial hazards
    "health_safety": 10,     # Medical facility concerns
    "property": 8,           # Property protection
}


# ═══════════════════════════════════════════════════════════════════
# MAIN RISK ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════

def analyze_risk(
    detection_result: Dict,
    scene_result: Dict
) -> Dict:
    """
    Perform comprehensive risk analysis on detected objects and scene.
    
    FIX #3: Uses realistic thresholds (not 5 people = extreme crowd)
    FIX #4: All rules check confidence before triggering
    FIX #5: Category caps prevent unbounded score accumulation
    
    Args:
        detection_result: Output from detect_objects()
        scene_result: Output from classify_scene()
        
    Returns:
        Dictionary containing:
        - risk_score: Total risk score (0-100)
        - risk_level: Risk level category (LOW/MEDIUM/HIGH/CRITICAL)
        - risk_color: Color code for UI display
        - risk_emoji: Emoji indicator
        - triggered_rules: List of rules that triggered
        - total_rules_triggered: Count of triggered rules
        - category_scores: Risk points per category
        - scene_base_risk: Base risk from scene type
        - explanation: Human-readable explanation
        - recommendations: List of recommended actions
    """
    # Extract data from results
    objects = detection_result.get("object_counts", {})
    categories = detection_result.get("category_counts", {})
    scene = scene_result.get("scene", "unknown")
    base_risk = scene_result.get("base_risk_score", 5)
    is_dangerous = scene_result.get("is_dangerous", False)
    
    # Initialize category scores
    category_scores = {category: 0 for category in CATEGORY_CAPS.keys()}
    triggered_rules = []
    
    # ───────────────────────────────────────────────────────────────
    # Evaluate all risk rules
    # ───────────────────────────────────────────────────────────────
    for rule_name, condition_func, score, explanation, category in RULES:
        try:
            # Check if rule condition is met
            if condition_func(objects, categories, scene, is_dangerous):
                
                # Calculate score to add (respecting category cap)
                current_category_score = category_scores[category]
                category_cap = CATEGORY_CAPS[category]
                
                # Add score but don't exceed category cap
                new_category_score = min(
                    current_category_score + score,
                    category_cap
                )
                
                # Calculate actual points added
                points_added = new_category_score - current_category_score
                
                # Update category score
                category_scores[category] = new_category_score
                
                # Record triggered rule (only if points were actually added)
                if points_added > 0:
                    triggered_rules.append({
                        "name": rule_name,
                        "score_added": points_added,
                        "explanation": explanation,
                        "category": category,
                    })
        
        except Exception as e:
            # Rule evaluation failed - skip this rule
            print(f"Rule '{rule_name}' evaluation error: {e}")
            continue
    
    # ───────────────────────────────────────────────────────────────
    # Calculate total risk score
    # ───────────────────────────────────────────────────────────────
    total_score = base_risk + sum(category_scores.values())
    total_score = min(total_score, 100)  # Cap at 100
    
    risk_level = score_to_level(total_score)
    
    # ───────────────────────────────────────────────────────────────
    # Generate explanation and recommendations
    # ───────────────────────────────────────────────────────────────
    explanation = _generate_explanation(
        risk_level,
        total_score,
        triggered_rules,
        scene,
        objects
    )
    
    recommendations = _generate_recommendations(
        risk_level,
        triggered_rules
    )
    
    # ───────────────────────────────────────────────────────────────
    # Return complete risk analysis
    # ───────────────────────────────────────────────────────────────
    return {
        "risk_score": total_score,
        "risk_level": risk_level,
        "risk_color": RISK_COLORS[risk_level],
        "risk_emoji": RISK_EMOJIS[risk_level],
        "triggered_rules": triggered_rules,
        "total_rules_triggered": len(triggered_rules),
        "category_scores": {
            k: v for k, v in category_scores.items() if v > 0
        },
        "scene_base_risk": base_risk,
        "explanation": explanation,
        "recommendations": recommendations,
    }


# ═══════════════════════════════════════════════════════════════════
# EXPLANATION GENERATION
# ═══════════════════════════════════════════════════════════════════

def _generate_explanation(
    level: str,
    score: int,
    triggered: List[Dict],
    scene: str,
    objects: Dict[str, dict]
) -> str:
    """
    Generate human-readable explanation of risk analysis.
    
    Args:
        level: Risk level (LOW/MEDIUM/HIGH/CRITICAL)
        score: Total risk score
        triggered: List of triggered rules
        scene: Scene type
        objects: Detected objects
        
    Returns:
        Multi-line explanation string
    """
    lines = []
    
    # Header
    lines.append(f"**Risk Level: {level} ({score}/100)**")
    lines.append(f"Scene: {scene.replace('_', ' ').title()}")
    
    # Top detected objects
    if objects:
        object_list = [
            (name, data.get("count", 0)) 
            for name, data in objects.items()
        ]
        top_objects = sorted(object_list, key=lambda x: x[1], reverse=True)[:6]
        
        obj_str = ", ".join(
            f"{count}×{name}" for name, count in top_objects
        )
        lines.append(f"Objects: {obj_str}")
    
    # Risk factors
    if triggered:
        lines.append("\n**Risk Factors:**")
        for rule in triggered:
            lines.append(
                f"• {rule['explanation']} (+{rule['score_added']})"
            )
    else:
        lines.append("\n✅ No risk factors detected.")
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# RECOMMENDATION GENERATION
# ═══════════════════════════════════════════════════════════════════

def _generate_recommendations(
    level: str,
    triggered: List[Dict]
) -> List[str]:
    """
    Generate actionable recommendations based on risk level and factors.
    
    Args:
        level: Risk level (LOW/MEDIUM/HIGH/CRITICAL)
        triggered: List of triggered rules
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Base recommendations by risk level
    base_recommendations = {
        RISK_LOW: [
            "Scene appears safe. Continue monitoring as needed."
        ],
        RISK_MEDIUM: [
            "Increase monitoring level. Alert relevant personnel.",
            "Document the situation for records."
        ],
        RISK_HIGH: [
            "Immediate attention required. Alert security team.",
            "Document and report the situation immediately.",
            "Prepare to take protective action if situation escalates."
        ],
        RISK_CRITICAL: [
            "⚠️ IMMEDIATE ACTION REQUIRED ⚠️",
            "Contact emergency services immediately.",
            "Secure the area and evacuate if necessary.",
            "Do not approach the threat directly."
        ]
    }
    
    recommendations.extend(base_recommendations.get(level, []))
    
    # Category-specific recommendations
    triggered_categories = {rule["category"] for rule in triggered}
    
    if "security" in triggered_categories or "crime" in triggered_categories:
        recommendations.append(
            "🔒 Alert law enforcement — security threat detected."
        )
    
    if "violence" in triggered_categories:
        recommendations.append(
            "🚨 Do not approach without proper security support."
        )
    
    if "fire_safety" in triggered_categories:
        recommendations.append(
            "🔥 Activate fire safety protocols immediately. Evacuate the area."
        )
    
    if "emergency" in triggered_categories:
        recommendations.append(
            "🚑 Emergency medical services may be needed. Clear access routes."
        )
    
    if "traffic_safety" in triggered_categories:
        recommendations.append(
            "🚗 Enforce traffic safety rules. Alert traffic management."
        )
    
    if "crowd_safety" in triggered_categories:
        recommendations.append(
            "👥 Deploy crowd management measures. Monitor for bottlenecks."
        )
    
    if "workplace_safety" in triggered_categories:
        recommendations.append(
            "🏭 Verify PPE compliance. Conduct safety briefing."
        )
    
    return recommendations


# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_risk_summary(risk_result: Dict) -> str:
    """
    Generate a one-line risk summary.
    
    Args:
        risk_result: Output from analyze_risk()
        
    Returns:
        Summary string
    """
    level = risk_result.get("risk_level", "UNKNOWN")
    score = risk_result.get("risk_score", 0)
    emoji = risk_result.get("risk_emoji", "❓")
    triggered = risk_result.get("total_rules_triggered", 0)
    
    return f"{emoji} {level} ({score}/100) — {triggered} risk factor(s) detected"


def filter_rules_by_category(
    triggered_rules: List[Dict],
    category: str
) -> List[Dict]:
    """
    Filter triggered rules by category.
    
    Args:
        triggered_rules: List of triggered rule dictionaries
        category: Category to filter by
        
    Returns:
        Filtered list of rules
    """
    return [
        rule for rule in triggered_rules 
        if rule.get("category") == category
    ]


def get_highest_risk_category(category_scores: Dict[str, int]) -> Optional[str]:
    """
    Get the category contributing most to risk score.
    
    Args:
        category_scores: Dictionary of category scores
        
    Returns:
        Category name with highest score, or None if empty
    """
    if not category_scores:
        return None
    
    return max(category_scores, key=category_scores.get)