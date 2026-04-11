"""
gradcam.py — Grad-CAM Heatmap Visualization
============================================
UPGRADE 3: CNN interpretability using TensorFlow GradientTape (already in requirements)
"""

import numpy as np
from PIL import Image
from typing import Optional, Tuple

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def compute_gradcam(image: Image.Image, model=None, target_class_idx=None,
                    last_conv_layer_name: str = "Conv_1"):
    if not TF_AVAILABLE:
        return None, "TensorFlow not available."
    try:
        if model is None:
            from modules.scene import get_model as get_scene_model
            model = get_scene_model()
        if model is None:
            return None, "MobileNetV2 model not loaded."

        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

        img_rgb = image.convert("RGB").resize((224, 224))
        img_array = np.array(img_rgb, dtype=np.float32)
        img_tensor = tf.constant(np.expand_dims(preprocess_input(img_array.copy()), axis=0))

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            if target_class_idx is None:
                target_class_idx = int(tf.argmax(predictions[0]))
            class_score = predictions[:, target_class_idx]

        grads = tape.gradient(class_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        heatmap = tf.nn.relu(heatmap).numpy()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        try:
            decoded = decode_predictions(predictions.numpy(), top=1)[0][0]
            info = f"Grad-CAM for '{decoded[1].replace('_',' ')}' ({decoded[2]*100:.1f}% conf) — Layer: {last_conv_layer_name}"
        except Exception:
            info = f"Grad-CAM for class {target_class_idx} — Layer: {last_conv_layer_name}"

        return heatmap, info
    except Exception as e:
        return None, f"Grad-CAM failed: {str(e)}"


def overlay_heatmap_on_image(original_image: Image.Image, heatmap: np.ndarray,
                              alpha: float = 0.45) -> Image.Image:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orig_w, orig_h = original_image.size
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize((orig_w, orig_h), Image.BILINEAR)
    heatmap_arr = np.array(heatmap_img) / 255.0

    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap_arr)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(colored_rgb)

    orig_rgb = original_image.convert("RGB")
    return Image.blend(orig_rgb, heatmap_pil, alpha=alpha)


def generate_gradcam_visualization(image: Image.Image, model=None, alpha: float = 0.45):
    heatmap, info = compute_gradcam(image, model=model)
    if heatmap is None:
        return None, info
    overlay = overlay_heatmap_on_image(image, heatmap, alpha=alpha)
    return overlay, info


def display_gradcam_in_streamlit(image: Image.Image, model=None):
    try:
        import streamlit as st
        with st.spinner("🔥 Computing Grad-CAM heatmap..."):
            overlay, info = generate_gradcam_visualization(image, model=model)

        if overlay is None:
            st.warning(f"Grad-CAM unavailable: {info}")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)

        st.caption(f"🔬 {info}")
        st.info(
            "**Grad-CAM Interpretation:** Red/yellow = highest influence on scene classification. "
            "Blue/green = low influence. This shows WHERE in the image the model was 'looking'."
        )
    except Exception as e:
        import streamlit as st
        st.error(f"Grad-CAM error: {e}")