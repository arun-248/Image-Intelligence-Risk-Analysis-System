"""
utils.py — Helper functions for charts, reports, and image processing
Complete version with all features
"""

import io
import base64
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict


# ═══════════════════════════════════════════════════════
# IMAGE HELPERS
# ═══════════════════════════════════════════════════════

def pil_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Convert bytes to PIL Image."""
    return Image.open(io.BytesIO(image_bytes))


def resize_for_display(image: Image.Image, max_width: int = 800) -> Image.Image:
    """Resize image for display while maintaining aspect ratio."""
    w, h = image.size
    if w > max_width:
        ratio = max_width / w
        new_size = (max_width, int(h * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def get_image_info(image: Image.Image) -> dict:
    """Get basic image metadata."""
    return {
        "width": image.size[0],
        "height": image.size[1],
        "mode": image.mode,
        "format": getattr(image, "format", "Unknown"),
        "megapixels": round((image.size[0] * image.size[1]) / 1_000_000, 2)
    }


# ═══════════════════════════════════════════════════════
# CHART GENERATORS (Plotly)
# ═══════════════════════════════════════════════════════

def make_confidence_bar_chart(detections: list) -> go.Figure:
    """
    Create a horizontal bar chart of object detection confidence scores.
    Handles both old structure (flat list) and new structure (with confidence tracking)
    """
    if not detections:
        return _empty_chart("No objects detected")

    # Sort by confidence
    sorted_dets = sorted(detections, key=lambda x: x.get("confidence", 0), reverse=True)[:15]

    labels = [f"{d.get('label', 'Unknown')}" for d in sorted_dets]
    scores = [d.get("confidence", 0) for d in sorted_dets]

    # Color coding: Green (80+), Yellow (55+), Red (<55)
    colors = []
    for s in scores:
        if s >= 80:
            colors.append("#00e676")  # Green
        elif s >= 55:
            colors.append("#ffab00")  # Yellow
        else:
            colors.append("#ff3d3d")  # Red

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.0f}%" for s in scores],
        textposition="outside",
    ))

    fig.update_layout(
        title="Object Detection Confidence Scores",
        xaxis_title="Confidence (%)",
        xaxis_range=[0, 110],
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#020b18",
        paper_bgcolor="#020b18",
        font=dict(color="#cdd9e5", family="Source Code Pro, monospace"),
        title_font=dict(color="#00e5ff", size=14, family="Orbitron, monospace"),
        height=max(250, len(sorted_dets) * 35 + 80),
        margin=dict(l=150, r=60, t=50, b=30),
    )
    return fig


def make_risk_gauge(risk_score: int, risk_level: str) -> go.Figure:
    """
    Create a gauge chart showing the risk score (0-100).
    """
    color_map = {
        "LOW": "#22c55e",
        "MEDIUM": "#f59e0b",
        "HIGH": "#ef4444",
        "CRITICAL": "#aa00ff",
    }
    color = color_map.get(risk_level, "#94a3b8")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={
            "text": f"Risk Level: {risk_level}",
            "font": {"color": color, "size": 16, "family": "Orbitron, monospace"}
        },
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#607d8b"},
            "bar": {"color": color, "thickness": 0.6},
            "bgcolor": "#041428",
            "bordercolor": "#0d3558",
            "borderwidth": 2,
            "steps": [
                {"range": [0, 25],   "color": "#0a2818"},  # Dark green
                {"range": [25, 50],  "color": "#2a1a08"},  # Dark amber
                {"range": [50, 75],  "color": "#2a0808"},  # Dark red
                {"range": [75, 100], "color": "#1a0a2a"},  # Dark purple
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 3},
                "thickness": 0.8,
                "value": risk_score
            }
        },
        number={
            "font": {"color": color, "size": 36, "family": "Orbitron, monospace"},
            "suffix": "/100"
        },
    ))

    fig.update_layout(
        plot_bgcolor="#020b18",
        paper_bgcolor="#020b18",
        font=dict(color="#cdd9e5"),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def make_object_count_pie(object_counts: dict) -> go.Figure:
    """
    Create a pie/donut chart of detected object types.
    Handles both old structure (int counts) and new structure (dict with 'count' key)
    """
    if not object_counts:
        return _empty_chart("No objects to display")

    # Extract labels and values (handle both formats)
    labels = []
    values = []
    
    for name, data in object_counts.items():
        labels.append(name)
        if isinstance(data, dict):
            values.append(data.get("count", 0))
        else:
            values.append(data)

    # Custom color palette (dark theme friendly)
    colors = [
        "#00e5ff", "#ffab00", "#ff3d3d", "#00e676", "#aa00ff",
        "#ff6b9d", "#ffd700", "#00bcd4", "#8bc34a", "#ff5722",
        "#9c27b0", "#cddc39", "#03a9f4", "#ffc107", "#e91e63"
    ]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(
            colors=colors[:len(labels)],
            line=dict(color="#020b18", width=2)
        ),
        textfont=dict(size=12, color="#ffffff", family="Source Code Pro, monospace"),
        textposition="auto",
    ))

    fig.update_layout(
        title="Detected Object Distribution",
        plot_bgcolor="#020b18",
        paper_bgcolor="#020b18",
        font=dict(color="#cdd9e5", family="Source Code Pro, monospace"),
        title_font=dict(color="#00e5ff", size=14, family="Orbitron, monospace"),
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(
            font=dict(color="#cdd9e5", size=10),
            bgcolor="#041428",
            bordercolor="#0d3558",
            borderwidth=1
        ),
    )
    return fig


def make_similarity_heatmap(similarity_matrix: list, names: list) -> go.Figure:
    """
    Create a heatmap of image similarity scores.
    """
    if not similarity_matrix:
        return _empty_chart("No similarity data")

    # Convert to percentages for display
    z_percent = [[v * 100 for v in row] for row in similarity_matrix]

    fig = go.Figure(go.Heatmap(
        z=z_percent,
        x=names,
        y=names,
        colorscale=[
            [0, "#0a1929"],      # Very dark blue (different)
            [0.3, "#1a3a5c"],    # Dark blue
            [0.5, "#f59e0b"],    # Amber (similar)
            [0.7, "#00e676"],    # Green
            [1, "#00e5ff"]       # Cyan (identical)
        ],
        zmin=0,
        zmax=100,
        text=[[f"{v:.0f}%" for v in row] for row in z_percent],
        texttemplate="%{text}",
        textfont={"size": 11, "color": "white", "family": "Source Code Pro, monospace"},
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Similarity: %{z:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title="Image Similarity Matrix (%)",
        plot_bgcolor="#020b18",
        paper_bgcolor="#020b18",
        font=dict(color="#cdd9e5", family="Source Code Pro, monospace"),
        title_font=dict(color="#00e5ff", size=14, family="Orbitron, monospace"),
        height=max(350, len(names) * 60 + 100),
        margin=dict(l=120, r=10, t=60, b=100),
        xaxis=dict(
            tickfont=dict(color="#607d8b", size=10),
            tickangle=-45
        ),
        yaxis=dict(
            tickfont=dict(color="#607d8b", size=10)
        ),
        coloraxis_colorbar=dict(
            title="Similarity %",
            titlefont=dict(color="#00e5ff"),
            tickfont=dict(color="#cdd9e5"),
            bgcolor="#041428",
            bordercolor="#0d3558",
            borderwidth=1
        ),
    )
    return fig


def make_risk_category_bar(category_scores: dict) -> go.Figure:
    """
    Bar chart for risk category breakdown.
    Shows which categories contributed most to the total risk score.
    """
    if not category_scores:
        return _empty_chart("No risk categories triggered")

    # Format category names and get scores
    categories = [c.replace("_", " ").title() for c in category_scores.keys()]
    scores = list(category_scores.values())

    # Color code by severity
    colors = []
    for s in scores:
        if s >= 30:
            colors.append("#ff3d3d")    # Red (high)
        elif s >= 15:
            colors.append("#ffab00")    # Amber (medium)
        else:
            colors.append("#00e676")    # Green (low)

    fig = go.Figure(go.Bar(
        x=categories,
        y=scores,
        marker_color=colors,
        text=[f"+{s}" for s in scores],
        textposition="outside",
        textfont=dict(color="#cdd9e5", family="Orbitron, monospace"),
    ))

    fig.update_layout(
        title="Risk Score Contribution by Category",
        yaxis_title="Points Added to Total Score",
        plot_bgcolor="#020b18",
        paper_bgcolor="#020b18",
        font=dict(color="#cdd9e5", family="Source Code Pro, monospace"),
        title_font=dict(color="#00e5ff", size=14, family="Orbitron, monospace"),
        height=320,
        margin=dict(l=60, r=20, t=60, b=100),
        xaxis=dict(
            tickfont=dict(color="#607d8b"),
            tickangle=-30
        ),
        yaxis=dict(
            tickfont=dict(color="#607d8b"),
            gridcolor="#0d3558"
        ),
    )
    return fig


def _empty_chart(message: str) -> go.Figure:
    """Create an empty chart with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#607d8b", family="Source Code Pro, monospace")
    )
    fig.update_layout(
        plot_bgcolor="#020b18",
        paper_bgcolor="#020b18",
        height=250,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ═══════════════════════════════════════════════════════
# AI REPORT GENERATOR
# ═══════════════════════════════════════════════════════

def generate_ai_report(
    image_name: str,
    detection_result: dict,
    scene_result: dict,
    risk_result: dict
) -> str:
    """
    Generate a complete AI analysis report in markdown format.
    Now handles both old and new object_counts structure.
    """
    import datetime
    
    # Extract scene info
    scene = scene_result.get("scene", "unknown").replace("_", " ").title()
    scene_conf = scene_result.get("confidence", 0)
    scene_desc = scene_result.get("description", "N/A")
    base_risk = scene_result.get("base_risk_score", 10)
    
    # Extract risk info
    risk_level = risk_result.get("risk_level", "UNKNOWN")
    risk_score = risk_result.get("risk_score", 0)
    triggered = risk_result.get("triggered_rules", [])
    recs = risk_result.get("recommendations", [])
    cat_scores = risk_result.get("category_scores", {})
    
    # Extract detection info (handle both formats)
    objects = detection_result.get("object_counts", {})
    total_objects = detection_result.get("total_objects", 0)
    models_used = detection_result.get("models_used", [])
    
    # Format object list (handle both dict and int values)
    obj_list = []
    for name, data in objects.items():
        if isinstance(data, dict):
            count = data.get("count", 0)
            conf = data.get("max_confidence", 0)
            obj_list.append(f"{count}× {name} (max conf: {conf:.0f}%)")
        else:
            obj_list.append(f"{data}× {name}")
    
    obj_str = ", ".join(obj_list) if obj_list else "None detected"
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Build report
    report = f"""# 🛰️ VisionIQ AI Analysis Report

**File:** `{image_name}`  
**Analysis Date:** {timestamp}  
**Models Used:** {', '.join(models_used) if models_used else 'N/A'}

---

## 🌍 Scene Analysis

**Detected Scene Type:** {scene}  
**Confidence:** {scene_conf}%  
**Description:** {scene_desc}  
**Inherent Scene Risk:** {base_risk}/100 points

---

## 🔍 Object Detection Results

**Total Objects Detected:** {total_objects}  
**Unique Object Types:** {len(objects)}

### Detected Objects:
{chr(10).join([f'- {obj}' for obj in obj_list]) if obj_list else '- No objects detected'}

---

## ⚠️ Risk Analysis

**Final Risk Score:** **{risk_score}/100**  
**Risk Level:** **{risk_level}** {risk_result.get('risk_emoji', '')}  
**Risk Factors Triggered:** {len(triggered)}

### Category Breakdown:
{chr(10).join([f'- **{cat.replace("_", " ").title()}:** +{score} points' for cat, score in cat_scores.items()]) if cat_scores else '- No categories triggered'}

### Detailed Risk Factors:
{chr(10).join([f'{i+1}. **{r["explanation"]}**  \n   ↳ Category: {r["category"].replace("_", " ").title()} | Score Added: +{r["score_added"]}' for i, r in enumerate(triggered)]) if triggered else '✅ No significant risk factors detected.'}

---

## 💡 AI Recommendations

{chr(10).join([f'{i+1}. {rec}' for i, rec in enumerate(recs)]) if recs else 'No specific recommendations at this time.'}

---

## 📊 Analysis Summary

This image was analyzed using a multi-stage AI pipeline:

1. **YOLOv8 Object Detection** — Identified {total_objects} objects across {len(objects)} categories
2. **MobileNetV2 Scene Classification** — Classified scene as "{scene}" with {scene_conf}% confidence
3. **Risk Engine Analysis** — Evaluated {len(triggered)} risk factors using 30+ predefined rules with category caps
4. **Explainable AI Output** — Generated this comprehensive report with clear reasoning for all conclusions

### Key Findings:
- Scene base risk contributed **+{base_risk}** points
- Object-scene combinations triggered **{len(triggered)}** risk rules
- Category caps prevented unbounded score stacking
- Final risk assessment: **{risk_level}** ({risk_score}/100)

---

*Generated by VisionIQ AI Multi-Image Intelligence & Risk Analysis System*  
*All conclusions are explainable and traceable to specific detection results and rule triggers.*  
*For questions about this analysis, review the triggered rules and their explanations above.*
"""
    
    return report