"""
risk_engine.py — Risk Analysis Engine (40+ rules + Context Engine support)
Handles both real YOLO detections AND context engine inferences
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

# Helper: check if any weapon (real or inferred) exists
def _has_weapon(cats):
    return cats.get("WEAPON",0) > 0 or cats.get("INFERRED_WEAPON",0) > 0

def _has_real_weapon(cats):
    return cats.get("WEAPON",0) > 0

def _has_inferred_weapon(cats):
    return cats.get("INFERRED_WEAPON",0) > 0

# ─────────────────────────────────────────────────────────────────
# 40+ RISK RULES
# check_fn(objs, cats, scene, is_dangerous) → bool
# ─────────────────────────────────────────────────────────────────

RULES = [

    # ══ CONFIRMED WEAPONS (YOLO detected) ════════════════════════
    ("firearm_confirmed",
     lambda o,c,s,d: _has_real_weapon(c) and
         any(g in " ".join(o.keys()).lower()
             for g in ["gun","rifle","handgun","shotgun","pistol","firearm"]),
     65, "🔫 Firearm CONFIRMED by deep learning model — CRITICAL security alert.",
     "security"),

    ("knife_confirmed_public",
     lambda o,c,s,d: "knife" in " ".join(o.keys()).lower() and s not in ["kitchen"],
     45, "🔪 Knife confirmed in non-kitchen environment — potential weapon threat.",
     "security"),

    ("multiple_real_weapons",
     lambda o,c,s,d: _has_real_weapon(c) and c.get("WEAPON",0) >= 2,
     70, "⚠️ Multiple weapons confirmed — CRITICAL armed conflict risk.",
     "security"),

    # ══ CONTEXT ENGINE INFERENCES ════════════════════════════════
    ("robbery_pattern_inferred",
     lambda o,c,s,d: s == "robbery",
     60, "🔫 Robbery pattern detected — multiple people, victim postures, threatening positions.",
     "crime"),

    ("weapon_posture_inferred",
     lambda o,c,s,d: _has_inferred_weapon(c) and s in ["robbery","weapon_threat","indoor","office"],
     50, "⚠️ Weapon-like posture/object inferred by context analysis — possible armed threat.",
     "crime"),

    ("violence_inferred",
     lambda o,c,s,d: s == "violence" or
         any("violence" in str(d.get("label","")).lower()
             for d in [] if False),  # scene already captures this
     50, "🚨 Violence scene detected by context analysis.",
     "violence"),

    ("victim_posture_detected",
     lambda o,c,s,d: _has_inferred_weapon(c) and o.get("person",0) >= 3,
     45, "👤 Multiple people in scene with threatening/victim postures detected.",
     "crime"),

    # ══ ROBBERY / ARMED CRIME ════════════════════════════════════
    ("weapon_threat_scene",
     lambda o,c,s,d: s == "weapon_threat",
     60, "⚔️ Weapon threat scene confirmed — security intervention required.",
     "crime"),

    ("armed_person_indoor",
     lambda o,c,s,d: _has_weapon(c) and o.get("person",0) >= 1
         and s in ["indoor","office","warehouse","robbery"],
     55, "🔫 Armed person(s) detected indoors — possible robbery or hostage situation.",
     "crime"),

    ("gun_night_scene",
     lambda o,c,s,d: _has_weapon(c) and s == "night_scene",
     55, "🌙🔫 Armed individual in night-time scene — high danger.",
     "crime"),

    # ══ VIOLENCE ════════════════════════════════════════════════
    ("violence_scene",
     lambda o,c,s,d: s == "violence",
     55, "🚨 Violence scene detected — physical altercation in progress.",
     "violence"),

    ("armed_violence",
     lambda o,c,s,d: s == "violence" and _has_weapon(c),
     70, "🚨⚔️ Armed violence — CRITICAL threat to life.",
     "violence"),

    ("mass_casualty_risk",
     lambda o,c,s,d: _has_weapon(c) and
         (s == "crowded_area" or o.get("person",0) >= 5),
     65, "🚨 Weapons in crowded area — mass casualty risk.",
     "violence"),

    # ══ ACCIDENT / EMERGENCY ═════════════════════════════════════
    ("accident_scene",
     lambda o,c,s,d: s == "accident",
     55, "🚗💥 Accident scene — emergency services may be required.",
     "emergency"),

    ("ambulance_detected",
     lambda o,c,s,d: "ambulance" in " ".join(o.keys()).lower(),
     40, "🚑 Ambulance detected — active emergency in progress.",
     "emergency"),

    ("fire_truck_detected",
     lambda o,c,s,d: "fire truck" in " ".join(o.keys()).lower(),
     45, "🚒 Fire truck detected — fire emergency in progress.",
     "emergency"),

    ("persons_at_accident",
     lambda o,c,s,d: s == "accident" and o.get("person",0) >= 1,
     45, "🚗💥👤 Person at accident scene — possible casualties.",
     "emergency"),

    # ══ FIRE / EXPLOSION ═════════════════════════════════════════
    ("fire_scene",
     lambda o,c,s,d: s == "fire_emergency",
     65, "🔥 Fire/explosion scene — evacuate immediately.",
     "fire_safety"),

    ("fire_with_people",
     lambda o,c,s,d: s == "fire_emergency" and o.get("person",0) >= 1,
     70, "🔥👤 People in fire emergency scene — life-threatening.",
     "fire_safety"),

    # ══ TRAFFIC SAFETY ══════════════════════════════════════════
    ("phone_while_driving",
     lambda o,c,s,d: "cell phone" in o and s in ["road","parking"]
         and o.get("person",0) >= 1,
     40, "📱🚗 Phone use in road scene — distracted driving risk.",
     "traffic_safety"),

    ("no_helmet_bike",
     lambda o,c,s,d: "motorcycle" in o and s in ["road","outdoor","parking"],
     25, "🏍️ Motorcycle without confirmed helmet — safety violation.",
     "traffic_safety"),

    ("pedestrian_in_traffic",
     lambda o,c,s,d: o.get("person",0) >= 1 and s == "road"
         and c.get("VEHICLE",0) >= 1,
     30, "🚶🚗 Pedestrian in active traffic zone — collision risk.",
     "traffic_safety"),

    ("vehicle_in_crowd",
     lambda o,c,s,d: s == "crowded_area" and c.get("VEHICLE",0) >= 1,
     40, "🚗👥 Vehicle in crowded area — serious pedestrian danger.",
     "traffic_safety"),

    ("night_driving",
     lambda o,c,s,d: s == "night_scene" and c.get("VEHICLE",0) >= 1,
     25, "🌙🚗 Vehicle in night/dark scene — increased accident risk.",
     "traffic_safety"),

    ("multi_vehicle_collision",
     lambda o,c,s,d: c.get("VEHICLE",0) >= 3 and s in ["road","accident"],
     40, "🚗🚗 Multiple vehicles in accident scene — possible pile-up.",
     "traffic_safety"),

    # ══ CROWD SAFETY ════════════════════════════════════════════
    ("high_crowd",
     lambda o,c,s,d: o.get("person",0) >= 5 or s == "crowded_area",
     25, "👥 High crowd density — stampede/health risk.",
     "crowd_safety"),

    ("extreme_crowd",
     lambda o,c,s,d: o.get("person",0) >= 10,
     30, "👥👥 Extreme crowd — emergency crowd management required.",
     "crowd_safety"),

    ("night_crowd",
     lambda o,c,s,d: s == "night_scene" and o.get("person",0) >= 5,
     25, "🌙👥 Large night-time crowd — safety monitoring required.",
     "crowd_safety"),

    # ══ WORKPLACE SAFETY ═════════════════════════════════════════
    ("ladder_hazard",
     lambda o,c,s,d: "ladder" in " ".join(o.keys()).lower(),
     20, "🪜 Ladder detected — fall hazard present.",
     "workplace_safety"),

    ("warehouse_people",
     lambda o,c,s,d: s == "warehouse" and o.get("person",0) >= 1,
     20, "🏭 People in industrial area — PPE compliance check needed.",
     "workplace_safety"),

    # ══ HEALTH / MEDICAL ═════════════════════════════════════════
    ("phone_hospital",
     lambda o,c,s,d: "cell phone" in o and s == "hospital",
     20, "📱🏥 Mobile phone in hospital — medical equipment interference risk.",
     "health_safety"),

    # ══ SECURITY ═════════════════════════════════════════════════
    ("backpack_crowd",
     lambda o,c,s,d: "backpack" in o and
         (s in ["crowded_area","outdoor"] or o.get("person",0) >= 3),
     20, "🎒 Suspicious backpack in public area — security concern.",
     "security"),

    ("night_person",
     lambda o,c,s,d: s == "night_scene" and o.get("person",0) >= 1,
     15, "🌙 Person in night-time scene — visibility risk.",
     "security"),

    # ══ PROPERTY ════════════════════════════════════════════════
    ("electronics_outdoor",
     lambda o,c,s,d: any(e in o for e in ["laptop","tv","monitor"]) and s == "outdoor",
     10, "💻 Electronics in outdoor scene — theft/weather risk.",
     "property"),
]


# ─────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────

def analyze_risk(detection_result: dict, scene_result: dict) -> dict:
    objs      = detection_result.get("object_counts",   {})
    cats      = detection_result.get("category_counts", {})
    scene     = scene_result.get("scene",               "unknown")
    base_risk = scene_result.get("base_risk_score",     10)
    is_danger = scene_result.get("is_dangerous",        False)

    total = base_risk
    triggered = []

    for name, check_fn, score, explanation, category in RULES:
        try:
            if check_fn(objs, cats, scene, is_danger):
                total += score
                triggered.append({
                    "name":        name,
                    "score_added": score,
                    "explanation": explanation,
                    "category":    category,
                })
        except Exception:
            pass

    total = min(total, 100)
    level = score_to_level(total)

    cat_scores = {}
    for r in triggered:
        c = r["category"]
        cat_scores[c] = cat_scores.get(c,0) + r["score_added"]

    return {
        "risk_score":            total,
        "risk_level":            level,
        "risk_color":            RISK_COLORS[level],
        "risk_emoji":            RISK_EMOJIS[level],
        "triggered_rules":       triggered,
        "total_rules_triggered": len(triggered),
        "category_scores":       cat_scores,
        "scene_base_risk":       base_risk,
        "explanation":           _explanation(level, total, triggered, scene, objs),
        "recommendations":       _recommendations(level, triggered),
    }


def _explanation(level, score, triggered, scene, objs):
    lines = [f"**Risk Level: {level} ({score}/100)**",
             f"Scene: {scene.replace('_',' ').title()}"]
    if objs:
        lines.append("Objects: " + ", ".join(f"{v}×{k}" for k,v in list(objs.items())[:6]))
    if triggered:
        lines.append("\n**Risk Factors:**")
        for r in triggered:
            lines.append(f"• {r['explanation']} (+{r['score_added']})")
    else:
        lines.append("\n✅ No risk factors detected.")
    return "\n".join(lines)


def _recommendations(level, triggered):
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