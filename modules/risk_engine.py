"""
risk_engine.py — Risk Analysis Engine with CATEGORY CAPS and CONFIDENCE THRESHOLDS
Fixed version - no unbounded score stacking
"""

RISK_LOW      = "LOW"
RISK_MEDIUM   = "MEDIUM"
RISK_HIGH     = "HIGH"
RISK_CRITICAL = "CRITICAL"

def score_to_level(score):
    if score >= 75: return RISK_CRITICAL
    if score >= 50: return RISK_HIGH
    if score >= 25: return RISK_MEDIUM
    return RISK_LOW

RISK_COLORS = {
    RISK_LOW:"#22c55e", RISK_MEDIUM:"#f59e0b",
    RISK_HIGH:"#ef4444", RISK_CRITICAL:"#aa00ff"
}
RISK_EMOJIS = {
    RISK_LOW:"✅", RISK_MEDIUM:"⚠️", RISK_HIGH:"🔴", RISK_CRITICAL:"🚨"
}

# Helper functions with confidence thresholds
def _has_weapon(objs, min_conf=70):
    """Check if real weapon detected with minimum confidence"""
    for obj_name, obj_data in objs.items():
        if any(w in obj_name.lower() for w in ["gun","rifle","knife","pistol","firearm"]):
            if obj_data.get("max_confidence", 0) >= min_conf:
                return True
    return False

def _has_inferred_weapon(cats):
    """Check if context engine inferred a weapon"""
    return cats.get("INFERRED_WEAPON", 0) > 0

def _get_person_count(objs, min_conf=50):
    """Get person count with confidence threshold"""
    for obj_name, obj_data in objs.items():
        if "person" in obj_name.lower():
            if obj_data.get("max_confidence", 0) >= min_conf:
                return obj_data.get("count", 0)
    return 0


# ═══════════════════════════════════════════════════════════════
# RISK RULES WITH CONFIDENCE REQUIREMENTS
# ═══════════════════════════════════════════════════════════════

RULES = [

    # ══ CONFIRMED WEAPONS (high confidence only) ══════════════════
    ("firearm_confirmed",
     lambda o,c,s,d: _has_weapon(o, min_conf=70),
     60, "🔫 Firearm CONFIRMED by deep learning (70%+ confidence) — CRITICAL alert.",
     "security"),

    ("knife_public",
     lambda o,c,s,d: any("knife" in k.lower() for k, v in o.items() 
                         if v.get("max_confidence",0) >= 60) 
                     and s not in ["kitchen"],
     40, "🔪 Knife confirmed in non-kitchen environment — potential weapon threat.",
     "security"),

    # ══ CONTEXT ENGINE INFERENCES ═════════════════════════════════
    ("robbery_pattern_inferred",
     lambda o,c,s,d: s == "robbery",
     55, "🔫 Robbery pattern detected — threatening positions/victim postures.",
     "crime"),

    ("weapon_posture_inferred",
     lambda o,c,s,d: _has_inferred_weapon(c) and s in ["robbery","weapon_threat","indoor"],
     45, "⚠️ Weapon-like posture/object inferred by context analysis.",
     "crime"),

    ("violence_inferred",
     lambda o,c,s,d: s == "violence",
     45, "🚨 Violence scene detected by context analysis.",
     "violence"),

    # ══ ROBBERY / ARMED CRIME ══════════════════════════════════════
    ("weapon_threat_scene",
     lambda o,c,s,d: s == "weapon_threat",
     55, "⚔️ Weapon threat scene confirmed — security intervention required.",
     "crime"),

    ("armed_person_indoor",
     lambda o,c,s,d: (_has_weapon(o, 65) or _has_inferred_weapon(c)) 
                     and _get_person_count(o) >= 1
                     and s in ["indoor","office","warehouse","robbery"],
     50, "🔫 Armed person(s) detected indoors — possible robbery/hostage situation.",
     "crime"),

    ("gun_night_scene",
     lambda o,c,s,d: _has_weapon(o, 70) and s == "night_scene",
     50, "🌙🔫 Armed individual in night-time scene — high danger.",
     "crime"),

    # ══ VIOLENCE ════════════════════════════════════════════════════
    ("violence_scene",
     lambda o,c,s,d: s == "violence",
     50, "🚨 Violence scene detected — physical altercation in progress.",
     "violence"),

    ("armed_violence",
     lambda o,c,s,d: s == "violence" and (_has_weapon(o, 65) or _has_inferred_weapon(c)),
     65, "🚨⚔️ Armed violence — CRITICAL threat to life.",
     "violence"),

    ("mass_casualty_risk",
     lambda o,c,s,d: (_has_weapon(o, 70) or _has_inferred_weapon(c)) 
                     and (s == "crowded_area" or _get_person_count(o) >= 5),
     60, "🚨 Weapons in crowded area — mass casualty risk.",
     "violence"),

    # ══ ACCIDENT / EMERGENCY ════════════════════════════════════════
    ("accident_scene",
     lambda o,c,s,d: s == "accident",
     50, "🚗💥 Accident scene — emergency services may be required.",
     "emergency"),

    ("ambulance_detected",
     lambda o,c,s,d: any("ambulance" in k.lower() for k in o.keys()),
     35, "🚑 Ambulance detected — active emergency in progress.",
     "emergency"),

    ("fire_truck_detected",
     lambda o,c,s,d: any("fire truck" in k.lower() for k in o.keys()),
     40, "🚒 Fire truck detected — fire emergency in progress.",
     "emergency"),

    ("persons_at_accident",
     lambda o,c,s,d: s == "accident" and _get_person_count(o) >= 1,
     40, "🚗💥👤 Person at accident scene — possible casualties.",
     "emergency"),

    # ══ FIRE / EXPLOSION ════════════════════════════════════════════
    ("fire_scene",
     lambda o,c,s,d: s == "fire_emergency",
     60, "🔥 Fire/explosion scene — evacuate immediately.",
     "fire_safety"),

    ("fire_with_people",
     lambda o,c,s,d: s == "fire_emergency" and _get_person_count(o) >= 1,
     65, "🔥👤 People in fire emergency scene — life-threatening.",
     "fire_safety"),

    # ══ TRAFFIC SAFETY ══════════════════════════════════════════════
    ("phone_while_driving",
     lambda o,c,s,d: any("phone" in k.lower() for k in o.keys()) 
                     and s in ["road","parking"]
                     and _get_person_count(o) >= 1,
     35, "📱🚗 Phone use in road scene — distracted driving risk.",
     "traffic_safety"),

    ("no_helmet_bike",
     lambda o,c,s,d: any("motorcycle" in k.lower() for k in o.keys()) 
                     and s in ["road","outdoor","parking"]
                     and not any("helmet" in k.lower() for k in o.keys()),
     20, "🏍️ Motorcycle without confirmed helmet — safety violation.",
     "traffic_safety"),

    ("pedestrian_in_traffic",
     lambda o,c,s,d: _get_person_count(o) >= 1 and s == "road"
                     and c.get("VEHICLE",0) >= 1,
     25, "🚶🚗 Pedestrian in active traffic zone — collision risk.",
     "traffic_safety"),

    ("vehicle_in_crowd",
     lambda o,c,s,d: s == "crowded_area" and c.get("VEHICLE",0) >= 1,
     35, "🚗👥 Vehicle in crowded area — serious pedestrian danger.",
     "traffic_safety"),

    ("night_driving",
     lambda o,c,s,d: s == "night_scene" and c.get("VEHICLE",0) >= 1,
     20, "🌙🚗 Vehicle in night/dark scene — increased accident risk.",
     "traffic_safety"),

    ("multi_vehicle_collision",
     lambda o,c,s,d: c.get("VEHICLE",0) >= 3 and s in ["road","accident"],
     35, "🚗🚗 Multiple vehicles in accident scene — possible pile-up.",
     "traffic_safety"),

    # ══ CROWD SAFETY ════════════════════════════════════════════════
    ("high_crowd",
     lambda o,c,s,d: _get_person_count(o) >= 5 or s == "crowded_area",
     20, "👥 High crowd density — stampede/health risk.",
     "crowd_safety"),

    ("extreme_crowd",
     lambda o,c,s,d: _get_person_count(o) >= 10,
     25, "👥👥 Extreme crowd — emergency crowd management required.",
     "crowd_safety"),

    ("night_crowd",
     lambda o,c,s,d: s == "night_scene" and _get_person_count(o) >= 5,
     20, "🌙👥 Large night-time crowd — safety monitoring required.",
     "crowd_safety"),

    # ══ WORKPLACE SAFETY ════════════════════════════════════════════
    ("ladder_hazard",
     lambda o,c,s,d: any("ladder" in k.lower() for k in o.keys()),
     15, "🪜 Ladder detected — fall hazard present.",
     "workplace_safety"),

    ("warehouse_people",
     lambda o,c,s,d: s == "warehouse" and _get_person_count(o) >= 1,
     15, "🏭 People in industrial area — PPE compliance check needed.",
     "workplace_safety"),

    # ══ HEALTH / MEDICAL ════════════════════════════════════════════
    ("phone_hospital",
     lambda o,c,s,d: any("phone" in k.lower() for k in o.keys()) and s == "hospital",
     15, "📱🏥 Mobile phone in hospital — equipment interference risk.",
     "health_safety"),

    # ══ SECURITY (FIXED: backpack now requires 20+ people) ══════════
    ("backpack_crowd",
     lambda o,c,s,d: any("backpack" in k.lower() for k in o.keys())
                     and _get_person_count(o) >= 20,  # FIXED: was 3
     15, "🎒 Backpack in very large crowd (20+ people) — security concern.",
     "security"),

    ("night_person",
     lambda o,c,s,d: s == "night_scene" and _get_person_count(o) >= 1,
     10, "🌙 Person in night-time scene — visibility risk.",
     "security"),

    # ══ PROPERTY ════════════════════════════════════════════════════
    ("electronics_outdoor",
     lambda o,c,s,d: any(e in " ".join(o.keys()).lower() 
                        for e in ["laptop","tv","monitor"]) 
                     and s == "outdoor",
     8, "💻 Electronics in outdoor scene — theft/weather risk.",
     "property"),
]


# ═══════════════════════════════════════════════════════════════
# CATEGORY CAPS (prevents unbounded stacking)
# ═══════════════════════════════════════════════════════════════

CATEGORY_CAPS = {
    "security":         35,
    "crime":            40,
    "violence":         45,
    "emergency":        35,
    "fire_safety":      40,
    "traffic_safety":   30,
    "crowd_safety":     25,
    "workplace_safety": 20,
    "health_safety":    15,
    "property":         10,
}


# ═══════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════

def analyze_risk(detection_result: dict, scene_result: dict) -> dict:
    """Analyze risk with category caps and confidence thresholds"""
    objs      = detection_result.get("object_counts",   {})
    cats      = detection_result.get("category_counts", {})
    scene     = scene_result.get("scene",               "unknown")
    base_risk = scene_result.get("base_risk_score",     10)
    is_danger = scene_result.get("is_dangerous",        False)

    # Initialize category scores
    category_scores = {cat: 0 for cat in CATEGORY_CAPS.keys()}
    triggered = []

    # Apply rules with category capping
    for name, check_fn, score, explanation, category in RULES:
        try:
            if check_fn(objs, cats, scene, is_danger):
                # Add score to category, but respect the cap
                old_score = category_scores[category]
                category_scores[category] = min(
                    old_score + score,
                    CATEGORY_CAPS[category]
                )
                actual_added = category_scores[category] - old_score
                
                if actual_added > 0:
                    triggered.append({
                        "name":        name,
                        "score_added": actual_added,
                        "explanation": explanation,
                        "category":    category,
                    })
        except Exception:
            pass

    # Total score = base risk + sum of capped categories
    total = base_risk + sum(category_scores.values())
    total = min(total, 100)
    level = score_to_level(total)

    return {
        "risk_score":            total,
        "risk_level":            level,
        "risk_color":            RISK_COLORS[level],
        "risk_emoji":            RISK_EMOJIS[level],
        "triggered_rules":       triggered,
        "total_rules_triggered": len(triggered),
        "category_scores":       {k:v for k,v in category_scores.items() if v > 0},
        "scene_base_risk":       base_risk,
        "explanation":           _explanation(level, total, triggered, scene, objs),
        "recommendations":       _recommendations(level, triggered),
    }


def _explanation(level, score, triggered, scene, objs):
    """Generate explanation text"""
    lines = [f"**Risk Level: {level} ({score}/100)**",
             f"Scene: {scene.replace('_',' ').title()}"]
    
    # Show top 6 objects
    obj_list = [(k, v.get("count", 0)) for k, v in objs.items()]
    if obj_list:
        top_objs = sorted(obj_list, key=lambda x: x[1], reverse=True)[:6]
        lines.append("Objects: " + ", ".join(f"{cnt}×{name}" for name, cnt in top_objs))
    
    if triggered:
        lines.append("\n**Risk Factors:**")
        for r in triggered:
            lines.append(f"• {r['explanation']} (+{r['score_added']})")
    else:
        lines.append("\n✅ No risk factors detected.")
    
    return "\n".join(lines)


def _recommendations(level, triggered):
    """Generate recommendations"""
    base = {
        RISK_LOW:      ["Scene appears safe. Continue monitoring."],
        RISK_MEDIUM:   ["Increase monitoring. Alert relevant personnel. Document."],
        RISK_HIGH:     ["Immediate attention required. Alert security. Document and report."],
        RISK_CRITICAL: ["IMMEDIATE ACTION REQUIRED. Contact emergency services. Secure the area."],
    }
    recs = list(base.get(level, []))
    cats = {r["category"] for r in triggered}
    
    if "security" in cats or "crime" in cats:
        recs.append("🔒 Alert law enforcement — security threat detected.")
    if "traffic_safety" in cats:
        recs.append("🚗 Enforce road safety rules.")
    if "fire_safety" in cats:
        recs.append("🔥 Activate fire safety protocols immediately.")
    if "violence" in cats:
        recs.append("🚨 Do not approach without security support.")
    if "crowd_safety" in cats:
        recs.append("👥 Deploy crowd management measures.")
    if "emergency" in cats:
        recs.append("🚑 Emergency services may be needed — clear the area.")
    
    return recs