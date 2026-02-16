"""
utils.py — Utility & Helper Functions
Shared across all modules
"""

import io
import base64
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict


# -------------------------------------------------------------------
# IMAGE HELPERS
# -------------------------------------------------------------------

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
        return image.resize(new_size, Image.LANCZOS)
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


# -------------------------------------------------------------------
# CHART GENERATORS (Plotly)
# -------------------------------------------------------------------

def make_confidence_bar_chart(detections: list) -> go.Figure:
    """
    Create a horizontal bar chart of object detection confidence scores.
    """
    if not detections:
        return _empty_chart("No objects detected")

    # Sort by confidence
    sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)[:15]

    labels = [f"{d['label']}" for d in sorted_dets]
    scores = [d["confidence"] for d in sorted_dets]

    colors = [
        "#ef4444" if s >= 80 else "#f59e0b" if s >= 60 else "#22c55e"
        for s in scores
    ]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s}%" for s in scores],
        textposition="outside",
    ))

    fig.update_layout(
        title="Object Detection Confidence Scores",
        xaxis_title="Confidence (%)",
        xaxis_range=[0, 110],
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0", family="monospace"),
        title_font=dict(color="#38bdf8", size=14),
        height=max(250, len(sorted_dets) * 35 + 80),
        margin=dict(l=10, r=60, t=50, b=30),
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
        "CRITICAL": "#7c3aed",
    }
    color = color_map.get(risk_level, "#94a3b8")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        title={"text": f"Risk Score — {risk_level}", "font": {"color": color, "size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar": {"color": color},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 25],   "color": "#052e16"},
                {"range": [25, 50],  "color": "#1c1917"},
                {"range": [50, 75],  "color": "#1c0a00"},
                {"range": [75, 100], "color": "#1a0030"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": risk_score
            }
        },
        number={"font": {"color": color, "size": 40}, "suffix": "/100"},
    ))

    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def make_object_count_pie(object_counts: dict) -> go.Figure:
    """
    Create a pie/donut chart of detected object types.
    """
    if not object_counts:
        return _empty_chart("No objects to display")

    labels = list(object_counts.keys())
    values = list(object_counts.values())

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(
            colors=px.colors.qualitative.Set3[:len(labels)],
            line=dict(color="#0f172a", width=2)
        ),
        textfont=dict(size=12, color="#e2e8f0"),
    ))

    fig.update_layout(
        title="Detected Object Distribution",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0", family="monospace"),
        title_font=dict(color="#38bdf8", size=14),
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(font=dict(color="#e2e8f0")),
    )
    return fig


def make_similarity_heatmap(similarity_matrix: list, names: list) -> go.Figure:
    """
    Create a heatmap of image similarity scores.
    """
    if not similarity_matrix:
        return _empty_chart("No similarity data")

    # Convert percentages for display
    z_percent = [[v * 100 for v in row] for row in similarity_matrix]

    fig = go.Figure(go.Heatmap(
        z=z_percent,
        x=names,
        y=names,
        colorscale=[
            [0, "#1e3a5f"],
            [0.5, "#f59e0b"],
            [1, "#22c55e"]
        ],
        zmin=0,
        zmax=100,
        text=[[f"{v:.0f}%" for v in row] for row in z_percent],
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"},
        hoverongaps=False,
    ))

    fig.update_layout(
        title="Image Similarity Matrix (%)",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0", family="monospace"),
        title_font=dict(color="#38bdf8", size=14),
        height=max(300, len(names) * 60 + 100),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(tickfont=dict(color="#94a3b8")),
        yaxis=dict(tickfont=dict(color="#94a3b8")),
        coloraxis_colorbar=dict(tickfont=dict(color="#e2e8f0")),
    )
    return fig


def make_risk_category_bar(category_scores: dict) -> go.Figure:
    """Bar chart for risk category breakdown."""
    if not category_scores:
        return _empty_chart("No risk categories triggered")

    categories = [c.replace("_", " ").title() for c in category_scores.keys()]
    scores = list(category_scores.values())

    fig = go.Figure(go.Bar(
        x=categories,
        y=scores,
        marker_color=["#ef4444" if s >= 30 else "#f59e0b" if s >= 15 else "#22c55e"
                      for s in scores],
        text=[f"+{s}" for s in scores],
        textposition="outside",
    ))

    fig.update_layout(
        title="Risk Score by Category",
        yaxis_title="Score Added",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0", family="monospace"),
        title_font=dict(color="#38bdf8", size=14),
        height=280,
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def _empty_chart(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#94a3b8")
    )
    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        height=250,
    )
    return fig


# -------------------------------------------------------------------
# AI REPORT GENERATOR
# -------------------------------------------------------------------

def generate_ai_report(
    image_name: str,
    detection_result: dict,
    scene_result: dict,
    risk_result: dict
) -> str:
    """
    Generate a complete AI analysis report in markdown format.
    """
    scene = scene_result.get("scene", "unknown").replace("_", " ").title()
    risk_level = risk_result.get("risk_level", "UNKNOWN")
    risk_score = risk_result.get("risk_score", 0)
    objects = detection_result.get("object_counts", {})
    triggered = risk_result.get("triggered_rules", [])
    recs = risk_result.get("recommendations", [])

    obj_str = ", ".join([f"{v}× {k}" for k, v in objects.items()]) if objects else "None detected"

    report = f"""
# 🤖 AI Analysis Report
**Image:** {image_name}

---

## 📷 Scene Analysis
- **Detected Scene:** {scene}
- **Confidence:** {scene_result.get('confidence', 0)}%
- **Description:** {scene_result.get('description', 'N/A')}

---

## 🔍 Object Detection
- **Total Objects Found:** {detection_result.get('total_objects', 0)}
- **Objects:** {obj_str}

---

## ⚠️ Risk Analysis
- **Risk Score:** {risk_score}/100
- **Risk Level:** {risk_level}
- **Factors Detected:** {len(triggered)}

### Risk Factors:
{"".join([f'- {r["explanation"]}' + chr(10) for r in triggered]) if triggered else '- No significant risk factors detected.'}

---

## 💡 Recommendations
{"".join([f'- {r}' + chr(10) for r in recs])}

---
*Generated by AI Multi-Image Intelligence & Risk Analysis System*
"""
    return report