"""
app.py — VisionIQ: AI Multi-Image Intelligence & Risk Analysis System
Run: streamlit run app.py
"""

import streamlit as st
from PIL import Image
import time

st.set_page_config(
    page_title="VisionIQ — AI Risk Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from modules.detection import detect_objects
from modules.scene import classify_scene
from modules.risk_engine import analyze_risk
from modules.similarity import compare_images
from modules.utils import (
    make_confidence_bar_chart, make_risk_gauge,
    make_object_count_pie, make_similarity_heatmap,
    make_risk_category_bar, generate_ai_report,
    get_image_info, resize_for_display,
)

# ═══════════════════════════════════════════════════════
# CSS — Military/Tactical Intelligence Dashboard
# Orbitron headings + Source Code Pro data
# Deep navy + electric cyan + amber alerts
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Source+Code+Pro:wght@300;400;600&family=Exo+2:wght@300;400;600;700&display=swap');

:root {
    --bg-deep:    #020b18;
    --bg-panel:   #041428;
    --bg-card:    #071e38;
    --bg-card2:   #0a2540;
    --cyan:       #00e5ff;
    --cyan-dim:   #00b4cc;
    --amber:      #ffab00;
    --red:        #ff3d3d;
    --green:      #00e676;
    --purple:     #aa00ff;
    --text:       #cdd9e5;
    --text-dim:   #607d8b;
    --border:     #0d3558;
    --border-hi:  #1a5276;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-deep) !important;
    color: var(--text) !important;
    font-family: 'Exo 2', sans-serif !important;
}

/* scanlines */
.stApp::before {
    content: '';
    position: fixed; top:0; left:0; right:0; bottom:0;
    background: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,229,255,0.012) 2px,rgba(0,229,255,0.012) 4px);
    pointer-events: none; z-index: 0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#020b18 0%,#041428 50%,#020b18 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

h1 { font-family: 'Orbitron',monospace !important; letter-spacing:4px !important; font-weight:900 !important; }
h2,h3 { font-family: 'Orbitron',monospace !important; letter-spacing:2px !important; }

[data-baseweb="tab-list"] {
    background: var(--bg-panel) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 4px !important; padding: 4px 4px 0 !important;
}
[data-baseweb="tab"] {
    font-family: 'Orbitron',monospace !important; font-size:10px !important;
    letter-spacing:1.5px !important; color: var(--text-dim) !important;
    background: var(--bg-card) !important;
    border-radius: 4px 4px 0 0 !important; padding: 8px 14px !important;
    border: 1px solid var(--border) !important; border-bottom:none !important;
}
[aria-selected="true"] {
    color: var(--cyan) !important; background: var(--bg-card2) !important;
    border-color: var(--cyan) !important; box-shadow: 0 -2px 12px #00e5ff22 !important;
}
[data-baseweb="tab-panel"] {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-top: 1px solid var(--cyan) !important;
    border-radius: 0 4px 4px 4px !important; padding: 24px !important;
}

.stButton > button {
    background: linear-gradient(135deg,#041428,#071e38) !important;
    border: 1px solid var(--cyan) !important; color: var(--cyan) !important;
    font-family: 'Orbitron',monospace !important; font-weight:700 !important;
    font-size:11px !important; letter-spacing:3px !important;
    border-radius:4px !important; padding:12px 32px !important;
    text-transform:uppercase !important; transition:all 0.3s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#071e38,#0a2540) !important;
    box-shadow: 0 0 20px #00e5ff44 !important; transform:translateY(-1px) !important;
}

[data-testid="stFileUploader"] {
    border: 2px dashed var(--border-hi) !important;
    border-radius:8px !important; background: var(--bg-panel) !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg,var(--cyan),var(--cyan-dim)) !important;
    box-shadow: 0 0 8px var(--cyan) !important;
}
.stProgress > div > div { background: var(--bg-card2) !important; }
hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius:2px; }

/* ── Component classes ── */
.ph { font-family:'Source Code Pro',monospace; font-size:10px; color:var(--cyan-dim);
      letter-spacing:3px; text-transform:uppercase; border-bottom:1px solid var(--border);
      padding-bottom:8px; margin-bottom:16px; }

.sb { background:var(--bg-card); border:1px solid var(--border); border-radius:6px;
      padding:16px 20px; position:relative; overflow:hidden; margin-bottom:12px; }
.sb::after { content:''; position:absolute; left:0;top:0;bottom:0; width:3px; background:var(--cyan); }
.sb.am::after{background:var(--amber);} .sb.rd::after{background:var(--red);} .sb.gn::after{background:var(--green);}
.sv { font-family:'Orbitron',monospace; font-size:30px; font-weight:700; color:var(--cyan); line-height:1; }
.sv.am{color:var(--amber);} .sv.rd{color:var(--red);} .sv.gn{color:var(--green);}
.sl { font-family:'Source Code Pro',monospace; font-size:10px; color:var(--text-dim);
      letter-spacing:2px; text-transform:uppercase; margin-top:4px; }
.ss { font-size:13px; color:var(--text); margin-top:6px; line-height:1.5; }

.pill { display:inline-block; font-family:'Orbitron',monospace; font-size:11px; font-weight:700;
        letter-spacing:2px; padding:4px 14px; border-radius:2px; margin:2px; }
.pc { background:#1a003d; color:#d580ff; border:1px solid #aa00ff; }
.ph2 { background:#1a0000; color:#ff8080; border:1px solid #ff3d3d; }
.pm { background:#1a0d00; color:#ffd080; border:1px solid #ffab00; }
.pl { background:#001a08; color:#80ffb3; border:1px solid #00e676; }

.rc { background:var(--bg-card); border:1px solid var(--border);
      border-left:3px solid var(--red); border-radius:4px; padding:12px 16px; margin:8px 0; }
.rc.am{border-left-color:var(--amber);} .rc.gn{border-left-color:var(--green);}
.rt { font-family:'Source Code Pro',monospace; font-size:12px; color:var(--text); font-weight:600; }
.rd2 { font-size:13px; color:var(--text-dim); margin-top:4px; line-height:1.6; }
.rs { font-family:'Orbitron',monospace; font-size:11px; color:var(--amber); margin-top:4px; }

.or { display:flex; align-items:center; justify-content:space-between;
      background:var(--bg-card); border:1px solid var(--border);
      border-radius:4px; padding:10px 14px; margin:6px 0; }
.on { font-family:'Source Code Pro',monospace; font-size:13px; color:var(--text);
      font-weight:600; text-transform:uppercase; letter-spacing:1px; }
.oc { font-family:'Orbitron',monospace; font-size:13px; font-weight:700; }

.sd { background:linear-gradient(135deg,var(--bg-card),var(--bg-card2));
      border:1px solid var(--border-hi); border-radius:8px; padding:28px;
      text-align:center; position:relative; overflow:hidden; }

.eb { background:var(--bg-panel); border:1px solid var(--border);
      border-radius:6px; padding:16px 20px; margin:10px 0; }
.et { font-family:'Orbitron',monospace; font-size:11px; color:var(--cyan-dim);
      letter-spacing:2px; text-transform:uppercase; margin-bottom:8px; }
.ex { font-size:14px; color:var(--text); line-height:1.8; }

.ri { display:flex; align-items:flex-start; gap:10px; padding:10px 14px;
      background:var(--bg-card); border:1px solid var(--border);
      border-radius:4px; margin:6px 0; font-size:14px; color:var(--text); line-height:1.6; }

.sp { background:var(--bg-card); border:1px solid var(--border);
      border-radius:6px; padding:14px 18px; margin:8px 0;
      display:flex; align-items:center; justify-content:space-between; }
.sc2 { font-family:'Orbitron',monospace; font-size:18px; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 24px 0;'>
        <div style='font-family:Source Code Pro,monospace;font-size:9px;
                    color:#607d8b;letter-spacing:4px;margin-bottom:8px;'>SYSTEM ONLINE ●</div>
        <div style='font-family:Orbitron,monospace;font-size:22px;font-weight:900;
                    color:#00e5ff;letter-spacing:3px;'>
            VISION<span style='color:#ffab00;'>IQ</span></div>
        <div style='font-family:Source Code Pro,monospace;font-size:10px;
                    color:#607d8b;letter-spacing:2px;margin-top:4px;'>AI RISK INTELLIGENCE SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="ph">Analysis Mode</div>', unsafe_allow_html=True)
    mode = st.selectbox("mode", [
        "🔬  Single Image — Full Analysis",
        "🖼️  Multi-Image — Similarity & Comparison"
    ], label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="ph">Detection Sensitivity</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("conf", 0.1, 0.9, 0.3, 0.05, label_visibility="collapsed")
    st.markdown(f"""
    <div style='font-family:Source Code Pro,monospace;font-size:11px;color:#607d8b;'>
    Threshold: <span style='color:#00e5ff;'>{conf_threshold:.0%}</span> —
    {"High sensitivity" if conf_threshold<0.4 else "Balanced" if conf_threshold<0.6 else "High precision"}
    </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div class="ph">AI Modules Active</div>
    <div style='font-family:Source Code Pro,monospace;font-size:11px;line-height:2;'>
        <span style='color:#00e676;'>●</span> <span style='color:#cdd9e5;'>YOLOv8 Object Detection</span><br>
        <span style='color:#00e676;'>●</span> <span style='color:#cdd9e5;'>MobileNetV2 Scene AI</span><br>
        <span style='color:#00e676;'>●</span> <span style='color:#cdd9e5;'>Risk Analysis Engine</span><br>
        <span style='color:#00e676;'>●</span> <span style='color:#cdd9e5;'>Image Similarity Engine</span><br>
        <span style='color:#00e676;'>●</span> <span style='color:#cdd9e5;'>Explainable AI Dashboard</span>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# MAIN HEADER
# ═══════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:20px 0 32px 0;'>
    <div style='font-family:Source Code Pro,monospace;font-size:10px;
                color:#607d8b;letter-spacing:5px;margin-bottom:12px;'>
        ─────── AI · COMPUTER VISION · DEEP LEARNING ───────
    </div>
    <div style='font-family:Orbitron,monospace;font-size:38px;font-weight:900;
                background:linear-gradient(90deg,#00e5ff 0%,#00b4cc 40%,#ffab00 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                letter-spacing:6px;line-height:1.2;'>VISIONIQ</div>
    <div style='font-family:Orbitron,monospace;font-size:12px;font-weight:400;
                color:#607d8b;letter-spacing:4px;margin-top:8px;'>
        MULTI-IMAGE INTELLIGENCE &amp; RISK ANALYSIS SYSTEM</div>
    <div style='font-family:Source Code Pro,monospace;font-size:11px;
                color:#1a3a5c;margin-top:16px;letter-spacing:2px;'>
        OBJECT DETECTION · SCENE UNDERSTANDING · RISK SCORING · IMAGE FORENSICS
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# HOW IT WORKS — always visible
# ═══════════════════════════════════════════════════════
with st.expander("📖  WHAT IS THIS SYSTEM? — Click to understand everything", expanded=False):
    st.markdown("""
    <div style='font-family:Exo 2,sans-serif;line-height:1.9;color:#cdd9e5;'>

    <div style='font-family:Orbitron,monospace;font-size:13px;color:#00e5ff;
                letter-spacing:2px;margin-bottom:16px;'>WHAT IS VISIONIQ?</div>

    <b style='color:#ffab00;'>Simple explanation:</b> Think of it as a smart security camera with a brain.
    You upload a photo → the AI looks at every pixel → tells you what objects are present,
    where the scene is taking place, and whether any danger or risk exists.

    <br><br>
    <b style='color:#ffab00;'>Real world use cases:</b><br>
    🏙️ <b>Smart city surveillance</b> — Auto-detect accidents, wrong-way drivers, overcrowded areas<br>
    🏭 <b>Factory/workplace safety</b> — Detect workers without helmets, blocked exits<br>
    🚔 <b>Law enforcement / forensics</b> — Compare images to find duplicates, detect weapons<br>
    🏫 <b>School/campus security</b> — Detect suspicious bags, overcrowding<br>
    🏥 <b>Hospital monitoring</b> — Detect phone use in restricted zones<br>

    <br>
    <div style='font-family:Orbitron,monospace;font-size:11px;color:#00e5ff;
                letter-spacing:2px;margin:12px 0 8px 0;'>THE AI PIPELINE</div>
    <div style='background:#020b18;border:1px solid #0d3558;border-radius:6px;
                padding:16px;font-family:Source Code Pro,monospace;font-size:12px;line-height:2;'>
    <span style='color:#00e5ff;'>STEP 1</span><span style='color:#607d8b;'> ── </span>
    <span>Image uploaded → Preprocessed (resize to 640px, normalize pixel values 0–1)</span><br>
    <span style='color:#00e5ff;'>STEP 2</span><span style='color:#607d8b;'> ── </span>
    <span>YOLOv8 neural network scans image → finds objects + draws bounding boxes + confidence %</span><br>
    <span style='color:#00e5ff;'>STEP 3</span><span style='color:#607d8b;'> ── </span>
    <span>MobileNetV2 deep learning model classifies scene type (road, office, kitchen etc.)</span><br>
    <span style='color:#00e5ff;'>STEP 4</span><span style='color:#607d8b;'> ── </span>
    <span>Risk Engine checks 15+ rules using objects + scene → calculates risk score 0–100</span><br>
    <span style='color:#ffab00;'>OUTPUT</span><span style='color:#607d8b;'> ── </span>
    <span>Dashboard shows annotated image, charts, risk score, recommendations, full report</span>
    </div>

    <br>
    <b style='color:#ffab00;'>What is a confidence score?</b> AI models don't say "this IS a car" —
    they say "I am 87% sure this is a car." Higher % = more reliable. You can adjust the minimum
    threshold in the sidebar. Lower it to see more detections, raise it for higher accuracy only.

    <br><br>
    <b style='color:#ffab00;'>What is a risk score?</b> The risk engine checks specific dangerous
    combinations. Example: detecting a <i>person</i> + <i>cell phone</i> + <i>road scene</i>
    together = distracted driving risk (+40 points). Multiple rules can trigger at once.
    Total risk score = sum of all triggered rules. 0–24 = Low, 25–49 = Medium, 50–74 = High, 75+ = Critical.

    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# SINGLE IMAGE MODE
# ═══════════════════════════════════════════════════════
if "Single Image" in mode:

    st.markdown('<div class="ph" style="margin-top:8px;">Upload Image for Analysis</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "upload", type=["jpg","jpeg","png","webp","bmp"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_info = get_image_info(image)

        # Info bar
        c1,c2,c3,c4,c5 = st.columns(5)
        data = [
            ("FILE", uploaded_file.name[:18], ""),
            ("RESOLUTION", f"{img_info['width']}×{img_info['height']}", ""),
            ("FILE SIZE", f"{round(len(uploaded_file.getvalue())/1024,1)} KB", "am"),
            ("MEGAPIXELS", f"{img_info['megapixels']} MP", ""),
            ("COLOR MODE", img_info['mode'], ""),
        ]
        for col, (label, val, cls) in zip([c1,c2,c3,c4,c5], data):
            with col:
                st.markdown(f"""
                <div class='sb {cls}'>
                    <div class='sv {cls}' style='font-size:15px;'>{val}</div>
                    <div class='sl'>{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button("🛰️   INITIATE FULL AI ANALYSIS", width="stretch")

        if analyze:
            prog = st.progress(0)
            status = st.empty()
            det_result = sce_result = ris_result = report = None

            for pct, msg in [
                (10,  "Preprocessing image..."),
                (30,  "Running YOLOv8 object detection neural network..."),
                (55,  "Running MobileNetV2 scene classification..."),
                (75,  "Running risk analysis engine..."),
                (90,  "Generating explainable AI report..."),
                (100, "Analysis complete."),
            ]:
                status.markdown(
                    f'<div style="font-family:Source Code Pro,monospace;font-size:12px;'
                    f'color:#00e5ff;padding:4px 0;">› {msg}</div>',
                    unsafe_allow_html=True
                )
                prog.progress(pct)
                if pct == 30:
                    det_result = detect_objects(image, confidence_threshold=conf_threshold)
                elif pct == 55:
                    sce_result = classify_scene(image, detection_result=det_result)
                elif pct == 75:
                    ris_result = analyze_risk(det_result, sce_result)
                elif pct == 90:
                    report = generate_ai_report(uploaded_file.name, det_result, sce_result, ris_result)
                else:
                    time.sleep(0.2)

            prog.empty(); status.empty()

            # ── Danger alert banner ──────────────────────────────
            weapons   = det_result.get("weapons_found", [])
            is_danger = sce_result.get("is_dangerous",  False)
            risk_sc   = ris_result.get("risk_score",     0)

            if weapons:
                wnames = ", ".join(d["label"] for d in weapons)
                st.markdown(
                    f"<div style='background:#2d0000;border:2px solid #ff3d3d;"
                    f"border-radius:8px;padding:16px 20px;margin:12px 0;'>"
                    f"<div style='font-family:Orbitron,monospace;font-size:14px;"
                    f"color:#ff3d3d;letter-spacing:2px;font-weight:700;'>"
                    f"🔫 CRITICAL ALERT — WEAPON DETECTED: {wnames.upper()}</div>"
                    f"<div style='font-size:13px;color:#fca5a5;margin-top:6px;'>"
                    f"Detected by YOLOv8 Open Images V7 (600-class model). "
                    f"Immediate security review recommended.</div></div>",
                    unsafe_allow_html=True
                )
            elif is_danger:
                scene_label = sce_result.get("scene","").replace("_"," ").upper()
                st.error(f"⚠️ DANGER SCENE DETECTED: {scene_label}  |  Risk Score: {risk_sc}/100 — See Risk Analysis tab")
            elif risk_sc >= 50:
                st.warning(f"⚠️ HIGH RISK SCORE: {risk_sc}/100 — Check Risk Analysis tab for details")

            t1,t2,t3,t4,t5 = st.tabs([
                "🔍  DETECTION", "🌍  SCENE", "⚠️  RISK ANALYSIS", "📊  CHARTS", "📄  REPORT"
            ])

            # ──────────────────────────────────────────
            # TAB 1 — DETECTION
            # ──────────────────────────────────────────
            with t1:
                st.markdown("""
                <div class='eb'><div class='et'>What is Object Detection?</div>
                <div class='ex'>
                <b>YOLOv8</b> (You Only Look Once, version 8) is a deep learning neural network
                trained on 80 object categories using millions of images. It scans every region
                of your image and identifies objects, drawing a <b>bounding box</b> around each one.
                The <b>confidence score</b> tells you how certain the AI is — 90%+ is very reliable,
                50–70% means the AI is less sure. You can adjust the threshold in the sidebar.
                </div></div>
                """, unsafe_allow_html=True)

                col_img, col_det = st.columns([3,2])
                with col_img:
                    st.markdown('<div class="ph">Annotated Output — Objects with Bounding Boxes</div>',
                                unsafe_allow_html=True)
                    st.image(resize_for_display(det_result["annotated_image"]), width="stretch")
                    st.markdown("""<div style='font-family:Source Code Pro,monospace;font-size:11px;
                        color:#607d8b;text-align:center;margin-top:6px;'>
                        Each colored box = one detected object · Label = name + confidence %
                    </div>""", unsafe_allow_html=True)

                with col_det:
                    st.markdown('<div class="ph">Detection Summary</div>', unsafe_allow_html=True)
                    total = det_result["total_objects"]
                    avg_conf = (sum(d["confidence"] for d in det_result["detections"]) / total
                               if total > 0 else 0)
                    ca, cb = st.columns(2)
                    with ca:
                        st.markdown(f"""<div class='sb'><div class='sv'>{total}</div>
                        <div class='sl'>Objects Found</div>
                        <div class='ss'>{len(det_result["object_counts"])} categories</div></div>""",
                        unsafe_allow_html=True)
                    with cb:
                        st.markdown(f"""<div class='sb am'><div class='sv am'>{avg_conf:.0f}%</div>
                        <div class='sl'>Avg Confidence</div>
                        <div class='ss'>AI certainty level</div></div>""",
                        unsafe_allow_html=True)

                    if det_result["detections"]:
                        st.markdown('<div class="ph" style="margin-top:14px;">Detected Objects</div>',
                                    unsafe_allow_html=True)
                        for label, count in det_result["object_counts"].items():
                            best_conf = max(d["confidence"] for d in det_result["detections"]
                                           if d["label"] == label)
                            color = ("#00e676" if best_conf>=80 else
                                     "#ffab00" if best_conf>=55 else "#ff3d3d")
                            st.markdown(f"""
                            <div class='or'>
                                <div>
                                    <div class='on'>🏷 {label}</div>
                                    <div style='font-family:Source Code Pro,monospace;
                                                font-size:10px;color:#607d8b;margin-top:2px;'>
                                        {count} instance{"s" if count>1 else ""} detected</div>
                                    <div style='background:#0d3558;border-radius:2px;height:3px;
                                                width:100%;margin-top:6px;'>
                                        <div style='background:{color};height:3px;border-radius:2px;
                                                    width:{int(best_conf)}%;'></div></div>
                                </div>
                                <div class='oc' style='color:{color};'>{best_conf}%</div>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""<div class='eb' style='text-align:center;padding:24px;'>
                        <div style='font-size:32px;color:#607d8b;'>🔍</div>
                        <div class='et' style='text-align:center;margin-top:8px;'>No Objects Detected</div>
                        <div class='ex' style='text-align:center;'>
                        Lower the confidence threshold in the sidebar (try 0.20),
                        or upload an image with people, vehicles, or everyday objects.
                        </div></div>""", unsafe_allow_html=True)

            # ──────────────────────────────────────────
            # TAB 2 — SCENE
            # ──────────────────────────────────────────
            with t2:
                st.markdown("""
                <div class='eb'><div class='et'>What is Scene Understanding?</div>
                <div class='ex'>
                The system doesn't just detect objects — it also tries to understand the
                <b>environment</b> where the photo was taken. Is this a road? An office?
                A hospital? This uses <b>MobileNetV2</b>, a deep learning model trained on
                ImageNet (14 million labeled images from 1000 categories). <br><br>
                Knowing the scene <b>dramatically improves risk analysis</b> — a knife in a
                kitchen scene is normal, but in a crowded street scene it triggers a security
                alert. Scene type also contributes a base risk score before any objects are considered.
                </div></div>
                """, unsafe_allow_html=True)

                scene = sce_result.get("scene","unknown")
                conf  = sce_result.get("confidence",0)
                emoji = sce_result.get("scene_emoji","📍")
                desc  = sce_result.get("description","")
                base_risk = sce_result.get("base_risk_score",10)
                top_preds = sce_result.get("top_predictions",[])[:8]

                col_sc1, col_sc2 = st.columns([2,3])
                with col_sc1:
                    st.markdown(f"""
                    <div class='sd'>
                        <div style='font-size:64px;margin-bottom:12px;'>{emoji}</div>
                        <div style='font-family:Orbitron,monospace;font-size:22px;font-weight:700;
                                    color:#00e5ff;letter-spacing:4px;text-transform:uppercase;'>
                            {scene.replace('_',' ')}</div>
                        <div style='font-family:Source Code Pro,monospace;font-size:13px;
                                    color:#607d8b;margin-top:8px;'>
                            CONFIDENCE: <span style='color:#ffab00;'>{conf}%</span></div>
                        <div style='font-size:14px;color:#cdd9e5;margin-top:12px;line-height:1.7;'>
                            {desc}</div>
                        <div style='margin-top:16px;padding:10px;background:#020b18;
                                    border-radius:4px;border:1px solid #0d3558;'>
                            <div style='font-family:Source Code Pro,monospace;font-size:10px;
                                        color:#607d8b;letter-spacing:2px;'>INHERENT SCENE RISK</div>
                            <div style='font-family:Orbitron,monospace;font-size:22px;
                                        color:#ffab00;font-weight:700;margin-top:4px;'>
                                {base_risk}/100</div>
                            <div style='font-size:11px;color:#607d8b;margin-top:4px;'>
                                Base risk score before object analysis</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                with col_sc2:
                    st.markdown('<div class="ph">What the AI Detected (Top Predictions from ImageNet)</div>',
                                unsafe_allow_html=True)
                    st.markdown("""<div class='ex' style='font-size:12px;color:#607d8b;margin-bottom:12px;'>
                    MobileNetV2 outputs its top ImageNet class predictions. We map these to scene
                    types using keyword matching. Each % shows how strongly that concept appeared.
                    </div>""", unsafe_allow_html=True)
                    if top_preds:
                        for pname, pconf in top_preds:
                            bar = min(int(pconf * 4), 100)
                            st.markdown(f"""
                            <div style='display:flex;align-items:center;justify-content:space-between;
                                        margin:5px 0;font-family:Source Code Pro,monospace;font-size:12px;'>
                                <span style='color:#cdd9e5;text-transform:capitalize;flex:1;'>{pname}</span>
                                <div style='flex:2;margin:0 12px;background:#0d3558;height:4px;border-radius:2px;'>
                                    <div style='background:#00e5ff;height:4px;border-radius:2px;width:{bar}%;'></div>
                                </div>
                                <span style='color:#ffab00;min-width:42px;text-align:right;'>{pconf}%</span>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.warning("TensorFlow not installed. Run: `pip install tensorflow-cpu` then restart the app.")

            # ──────────────────────────────────────────
            # TAB 3 — RISK ANALYSIS
            # ──────────────────────────────────────────
            with t3:
                st.markdown("""
                <div class='eb'><div class='et'>How Does Risk Analysis Work?</div>
                <div class='ex'>
                The Risk Engine is the <b>most unique and intelligent part</b> of this system.
                It combines the detected objects AND scene type, then checks <b>15+ predefined
                risk rules</b>. Each rule looks for specific dangerous combinations.<br><br>
                Example rule: <i>"Is there a PERSON + CELL PHONE + ROAD scene?"</i>
                → triggers distracted driving alert (+40 points).<br><br>
                The total score is the sum of all triggered rules + the scene's base risk.
                The system then tells you <b>exactly why</b> the score is what it is —
                this is called <b>Explainable AI (XAI)</b>: you can understand and verify
                every decision the AI made, unlike a black-box system.
                </div></div>
                """, unsafe_allow_html=True)

                r_score = ris_result.get("risk_score",0)
                r_level = ris_result.get("risk_level","LOW")
                r_emoji = ris_result.get("risk_emoji","✅")
                r_rules = ris_result.get("triggered_rules",[])
                r_recs  = ris_result.get("recommendations",[])
                r_cats  = ris_result.get("category_scores",{})
                s_base  = ris_result.get("scene_base_risk",10)

                pill_map = {"CRITICAL":"pc","HIGH":"ph2","MEDIUM":"pm","LOW":"pl"}
                pcls = pill_map.get(r_level,"pl")

                col_g, col_r = st.columns([2,3])
                with col_g:
                    st.plotly_chart(make_risk_gauge(r_score, r_level), width="stretch")
                    # Score breakdown
                    breakdown_rows = f"""
                    <div style='display:flex;justify-content:space-between;'>
                        <span style='color:#607d8b;'>Scene base risk</span>
                        <span style='color:#ffab00;'>+{s_base}</span></div>"""
                    for rule in r_rules:
                        breakdown_rows += f"""
                        <div style='display:flex;justify-content:space-between;'>
                            <span style='color:#607d8b;font-size:11px;'>
                                {rule["name"].replace("_"," ")}</span>
                            <span style='color:#ff3d3d;'>+{rule["score_added"]}</span></div>"""
                    breakdown_rows += f"""
                    <div style='border-top:1px solid #0d3558;margin-top:6px;padding-top:6px;
                                display:flex;justify-content:space-between;'>
                        <span style='color:#cdd9e5;font-weight:600;'>TOTAL</span>
                        <span style='color:#00e5ff;font-family:Orbitron,monospace;
                                     font-weight:700;'>{r_score}/100</span></div>"""
                    st.markdown(f"""
                    <div class='eb' style='margin-top:8px;'>
                        <div class='et'>Score Breakdown</div>
                        <div style='font-family:Source Code Pro,monospace;font-size:12px;line-height:2;'>
                        {breakdown_rows}</div>
                    </div>""", unsafe_allow_html=True)

                with col_r:
                    sb_cls = "rd" if r_level in ["HIGH","CRITICAL"] else "am" if r_level=="MEDIUM" else "gn"
                    verdict = ("⚡ Immediate attention required." if r_level in ["HIGH","CRITICAL"]
                               else "⚠️ Monitor this situation carefully." if r_level=="MEDIUM"
                               else "✅ Scene appears safe.")
                    st.markdown(f"""
                    <div class='sb {sb_cls}'>
                        <div style='margin-bottom:10px;'>
                            <span class='pill {pcls}'>{r_emoji} {r_level} RISK — {r_score}/100</span>
                        </div>
                        <div class='ss'><b>{len(r_rules)}</b> risk factor{"s" if len(r_rules)!=1 else ""} detected. {verdict}</div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown('<div class="ph" style="margin-top:16px;">Risk Factors Identified</div>',
                                unsafe_allow_html=True)
                    if r_rules:
                        for rule in r_rules:
                            sc = rule['score_added']
                            sev = "rd" if sc>=30 else "am" if sc>=15 else "gn"
                            cat = rule['category'].replace('_',' ').title()
                            st.markdown(f"""
                            <div class='rc {sev}'>
                                <div class='rt'>{rule['explanation']}</div>
                                <div class='rd2'>
                                    This object-scene combination matches a known risk pattern.
                                    Category: <b>{cat}</b>.
                                </div>
                                <div class='rs'>RISK CONTRIBUTION: +{sc} points added to total</div>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""<div class='rc gn'>
                        <div class='rt'>✅ No Risk Factors Triggered</div>
                        <div class='rd2'>The AI checked all 15+ risk rules against the detected objects
                        and scene type but found no dangerous combinations. This scene appears safe
                        according to current risk rules.</div></div>""", unsafe_allow_html=True)

                    if r_recs:
                        st.markdown('<div class="ph" style="margin-top:16px;">AI Recommendations</div>',
                                    unsafe_allow_html=True)
                        for rec in r_recs:
                            st.markdown(f"""
                            <div class='ri'>
                                <span style='color:#ffab00;font-size:16px;'>›</span>
                                <span>{rec}</span>
                            </div>""", unsafe_allow_html=True)

            # ──────────────────────────────────────────
            # TAB 4 — CHARTS
            # ──────────────────────────────────────────
            with t4:
                st.markdown("""
                <div class='eb'><div class='et'>Visual Analytics</div>
                <div class='ex'>
                These charts visualize the analysis data. The <b>Confidence Bar Chart</b> shows
                how certain the AI is about each object detection — green bars (80%+) are very
                reliable, yellow is moderate, red is uncertain. The <b>Object Distribution Chart</b>
                shows how many of each type were found. The <b>Risk Category Chart</b> shows
                which risk categories contributed most to the overall score.
                </div></div>
                """, unsafe_allow_html=True)

                cc1, cc2 = st.columns(2)
                with cc1:
                    st.plotly_chart(make_confidence_bar_chart(det_result.get("detections",[])),
                                    width="stretch")
                with cc2:
                    st.plotly_chart(make_object_count_pie(det_result.get("object_counts",{})),
                                    width="stretch")
                if ris_result.get("category_scores"):
                    st.plotly_chart(make_risk_category_bar(ris_result.get("category_scores",{})),
                                    width="stretch")
                else:
                    st.markdown("""<div style='text-align:center;padding:30px;
                        font-family:Source Code Pro,monospace;font-size:12px;color:#607d8b;'>
                        No risk categories were triggered for this image.
                        Risk category chart will appear when risk factors are detected.
                    </div>""", unsafe_allow_html=True)

            # ──────────────────────────────────────────
            # TAB 5 — REPORT
            # ──────────────────────────────────────────
            with t5:
                st.markdown("""
                <div class='eb'><div class='et'>Complete AI Analysis Report</div>
                <div class='ex'>
                This is the full written summary of everything the system found. It covers
                the scene, all detected objects, the risk analysis with explanations, and
                recommendations. Download it to include in your project documentation or
                presentation. This demonstrates the <b>Explainable AI (XAI)</b> capability
                of the system — every conclusion has a clear, human-readable reason.
                </div></div>
                """, unsafe_allow_html=True)
                st.markdown(report)
                st.download_button("📥  Download Full Report (.md)", data=report,
                    file_name=f"visioniq_{uploaded_file.name}.md",
                    mime="text/markdown", width="stretch")

    else:
        st.markdown("""
        <div style='text-align:center;padding:100px 0;'>
            <div style='font-size:72px;color:#0d3558;'>🛰️</div>
            <div style='font-family:Orbitron,monospace;font-size:16px;
                        letter-spacing:4px;color:#1a3a5c;margin-top:20px;'>
                AWAITING IMAGE INPUT</div>
            <div style='font-family:Source Code Pro,monospace;font-size:12px;
                        color:#0d3558;margin-top:10px;letter-spacing:2px;'>
                Upload JPG · PNG · WEBP · BMP to begin analysis</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# MULTI-IMAGE MODE
# ═══════════════════════════════════════════════════════
elif "Multi-Image" in mode:

    st.markdown("""
    <div class='eb'><div class='et'>What is Image Similarity / Forensic Comparison?</div>
    <div class='ex'>
    This module compares multiple images to find how visually similar they are.
    It works by converting each image into a <b>feature vector</b> (a list of 1280 numbers)
    using MobileNetV2's deep learning layers — this captures the visual "essence" of the image.
    Then <b>cosine similarity</b> (a mathematical formula) measures how close two vectors are.
    <br><br>
    <b>100%</b> = exact duplicate (same file) · <b>80–99%</b> = near duplicate ·
    <b>60–80%</b> = very similar content · <b>below 45%</b> = different images.<br><br>
    <b>Use cases:</b> Detecting duplicate evidence photos, checking if surveillance frames
    are related, finding similar images in a dataset, content-based image retrieval.
    </div></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ph" style="margin-top:16px;">Upload 2–10 Images to Compare</div>',
                unsafe_allow_html=True)

    uploaded_files = st.file_uploader("multi", type=["jpg","jpeg","png","webp"],
        accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_files and len(uploaded_files) >= 2:
        images = [Image.open(f).convert("RGB") for f in uploaded_files]
        names  = [f.name for f in uploaded_files]

        thumb_cols = st.columns(min(len(images), 5))
        for i, (img, name) in enumerate(zip(images, names)):
            with thumb_cols[i % 5]:
                t = img.copy(); t.thumbnail((180,180))
                st.image(t, caption=name[:16], width="stretch")

        run_sim = st.button("🛰️   RUN SIMILARITY ANALYSIS", width="stretch")

        if run_sim:
            with st.spinner("Extracting deep learning embeddings and computing cosine similarity..."):
                sim_result = compare_images(images, names)

            pairs = sim_result["pairs"]
            dups  = sim_result["duplicates"]
            best  = pairs[0] if pairs else None

            c1,c2,c3,c4 = st.columns(4)
            for col, (label, val, cls) in zip([c1,c2,c3,c4], [
                ("Images Analyzed", sim_result["total_images"], ""),
                ("Pairs Compared", len(pairs), "am"),
                ("Duplicates Found", len(dups), "rd" if dups else "gn"),
                ("Highest Similarity", f"{best['similarity']}%" if best else "N/A", ""),
            ]):
                with col:
                    st.markdown(f"""<div class='sb {cls}'>
                    <div class='sv {cls}'>{val}</div>
                    <div class='sl'>{label}</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="ph">Similarity Heatmap</div>', unsafe_allow_html=True)
            st.markdown("""<div style='font-family:Source Code Pro,monospace;font-size:11px;
                color:#607d8b;margin-bottom:8px;'>
                Green = very similar · Blue/dark = very different ·
                Diagonal is always 100% (image vs itself)
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(make_similarity_heatmap(
                sim_result["similarity_matrix"], sim_result["names"]),
                width="stretch")

            if dups:
                st.markdown("""<div style='background:#1a0000;border:1px solid #ff3d3d;
                    border-radius:6px;padding:14px 18px;margin:12px 0;'>
                    <div style='font-family:Orbitron,monospace;font-size:12px;
                                color:#ff3d3d;letter-spacing:2px;'>
                    ⚠️ DUPLICATE IMAGES DETECTED</div></div>""", unsafe_allow_html=True)
                for dup in dups:
                    st.error(f"🔴 **{dup['img1']}** ↔ **{dup['img2']}** — {dup['relationship']} ({dup['similarity']}% similar)")

            st.markdown('<div class="ph" style="margin-top:20px;">All Pair Results (Sorted by Similarity)</div>',
                        unsafe_allow_html=True)
            for pair in pairs:
                sim  = pair["similarity"]
                rel  = pair["relationship"]
                col  = ("#00e676" if sim>=80 else "#ffab00" if sim>=50 else "#607d8b")
                st.markdown(f"""
                <div class='sp'>
                    <div style='flex:3;font-family:Source Code Pro,monospace;font-size:12px;'>
                        <span style='color:#cdd9e5;'>{pair["img1"]}</span>
                        <span style='color:#607d8b;margin:0 10px;'>↔</span>
                        <span style='color:#cdd9e5;'>{pair["img2"]}</span>
                    </div>
                    <div style='flex:2;margin:0 16px;'>
                        <div style='background:#0d3558;height:4px;border-radius:2px;'>
                            <div style='background:{col};height:4px;border-radius:2px;width:{int(sim)}%;'></div>
                        </div>
                    </div>
                    <div style='flex:1;text-align:right;'>
                        <div class='sc2' style='color:{col};'>{sim}%</div>
                        <div style='font-family:Source Code Pro,monospace;font-size:10px;
                                    color:#607d8b;margin-top:2px;'>{rel}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

    elif uploaded_files and len(uploaded_files) == 1:
        st.warning("⚠️ Upload at least 2 images to run comparison.")
    else:
        st.markdown("""
        <div style='text-align:center;padding:80px 0;'>
            <div style='font-size:60px;color:#0d3558;'>🖼️🖼️</div>
            <div style='font-family:Orbitron,monospace;font-size:14px;
                        letter-spacing:3px;color:#1a3a5c;margin-top:16px;'>
                UPLOAD 2+ IMAGES TO BEGIN COMPARISON</div>
        </div>""", unsafe_allow_html=True)