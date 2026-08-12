"""
🛡️ DeepGuard AI — Deepfake Detection Web Application
Premium Gradio interface with Grad-CAM heatmap visualization.
"""
import gradio as gr
import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.face_detector import FaceDetector
from inference.predictor import DeepfakePredictor

# ============================================================
# INITIALIZE AI COMPONENTS
# ============================================================
print("[DeepGuard] Initializing...")
face_detector = FaceDetector()

# === ENSEMBLE: Load BOTH models for combined prediction ===
import tensorflow as tf

# Primary model (new V2 380x380 EfficientNetB4)
PRIMARY_PATHS = [
    'deepfake_model_best.keras',
    'streamlit/deepfake_model_best.keras',
    'deepfake_model.keras',
    'streamlit/deepfake_model.keras',
]

predictor = None

for path in PRIMARY_PATHS:
    if os.path.exists(path):
        predictor = DeepfakePredictor(model_path=path)
        print(f"[DeepGuard] Primary model loaded: {path} ({os.path.getsize(path)/1024/1024:.1f} MB)")
        break

if predictor is None or predictor.model is None:
    raise FileNotFoundError("Could not find deepfake_model_best.keras! Please place it in project root or streamlit/ folder.")

def ensemble_predict(image_rgb):
    """
    Prediction using V2 EfficientNetB4 model.
    Returns (raw_score, confidence, label).
    raw_score: 0.0=Fake, 1.0=Real
    """
    if predictor and predictor.model:
        img = cv2.resize(image_rgb, (predictor.img_size, predictor.img_size)).astype('float32')
        raw = float(predictor.model.predict(np.expand_dims(img, 0), verbose=0)[0][0])
    else:
        return 0.5, 0.5, "Error"

    if raw >= 0.5:
        return raw, raw, "Real"
    else:
        return raw, 1.0 - raw, "Fake"

print("[DeepGuard] Prediction engine ready!\n")


# ============================================================
# IMAGE ANALYSIS
# ============================================================
def analyze_image(image):
    """Analyze a single image using V2 EfficientNetB4 model."""
    if image is None:
        return None, None, "Please upload an image."

    # 1. Detect face for bounding box display
    face_img, box = face_detector.extract_face(image)

    # 2. Predict on full image (matches training dataset preprocessing where full square portraits were used)
    raw_score, confidence, label = ensemble_predict(image)

    # If face crop exists, run secondary check for face region score
    if face_img is not None:
        raw_face, _, _ = ensemble_predict(face_img)
    else:
        raw_face = raw_score

    # Create annotated image
    display_img = image.copy()
    if box is not None:
        x, y, w, h = box
        color = (0, 200, 80) if label == "Real" else (230, 60, 60)
        thickness = max(2, min(display_img.shape[:2]) // 150)
        cv2.rectangle(display_img, (x, y), (x + w, y + h), color, thickness)

        font_scale = max(0.6, min(display_img.shape[:2]) / 800)
        text = f"{label}: {confidence * 100:.1f}%"
        text_y = max(30, y - 10)
        cv2.putText(display_img, text, (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    # Generate Grad-CAM heatmap on full image
    target_grad_img = cv2.resize(image, (predictor.img_size, predictor.img_size))
    heatmap = predictor.generate_gradcam(target_grad_img)
    heatmap_overlay = predictor.create_heatmap_overlay(target_grad_img, heatmap, alpha=0.4)

    # Build result markdown
    emoji = "🟢" if label == "Real" else "🔴"
    verdict = "AUTHENTIC" if label == "Real" else "MANIPULATED"

    face_score_str = f"{raw_face * 100:.1f}% real" if raw_face is not None else "N/A"
    full_score_str = f"{raw_score * 100:.1f}% real"

    result_md = f"""
## {emoji} Verdict: **{verdict}**

| Metric | Value |
|:-------|:------|
| **Classification** | {label} |
| **Confidence** | {confidence * 100:.2f}% |
| **Full Image Score** | {full_score_str} |
| **Face Crop Score** | {face_score_str} |
| **Face Detected** | {'Yes' if box is not None else 'No'} |

{"> The Grad-CAM heatmap (right) shows the regions the model focused on for its decision." if heatmap.max() > 0 else ''}
"""

    return display_img, heatmap_overlay, result_md


# ============================================================
# VIDEO ANALYSIS
# ============================================================
def analyze_video(video_path):
    """Analyze a video for deepfake detection frame-by-frame."""
    if not video_path:
        return "Please upload a video."

    # Extract faces from video frames
    faces = face_detector.extract_faces_from_video(
        video_path, frame_skip=5, max_frames=50
    )

    if not faces:
        return "No faces detected in the video. Please try another video with visible faces."

    # Predict
    confidence, label, per_frame = predictor.predict_video_faces(faces)

    fake_count = sum(1 for lbl, _ in per_frame if lbl == "Fake")
    real_count = sum(1 for lbl, _ in per_frame if lbl == "Real")
    total = len(per_frame)
    fake_pct = (fake_count / total) * 100

    emoji = "🟢" if label == "Real" else "🔴"
    verdict = "AUTHENTIC" if label == "Real" else "MANIPULATED"

    result_md = f"""
## {emoji} Video Verdict: **{verdict}**

| Metric | Value |
|:-------|:------|
| **Classification** | {label} |
| **Overall Confidence** | {confidence * 100:.2f}% |
| **Frames Analyzed** | {total} |
| **Fake Frames** | {fake_count} ({fake_pct:.1f}%) |
| **Real Frames** | {real_count} ({100 - fake_pct:.1f}%) |

### Frame-by-Frame Breakdown:
"""

    # Show first 20 frame results
    for i, (lbl, conf) in enumerate(per_frame[:20]):
        icon = "✅" if lbl == "Real" else "❌"
        result_md += f"- Frame {i + 1}: {icon} {lbl} ({conf * 100:.1f}%)\n"

    if total > 20:
        result_md += f"\n*...and {total - 20} more frames*\n"

    return result_md


# ============================================================
# GRADIO UI
# ============================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

body, .gradio-container {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #0b0f19 !important;
    color: #e2e8f0 !important;
    max-width: 1250px !important;
    margin: auto !important;
}

.hero-header {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    border-radius: 16px;
    padding: 2.2rem;
    text-align: center;
    margin-bottom: 1.8rem;
    backdrop-filter: blur(12px);
}

.hero-header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 1rem;
}

.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #38bdf8;
}

.verdict-card-real {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 95, 70, 0.25) 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 12px;
    padding: 1.2rem;
    margin-top: 1rem;
}

.verdict-card-fake {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(153, 27, 27, 0.25) 100%);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 12px;
    padding: 1.2rem;
    margin-top: 1rem;
}

/* Custom table styling */
table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin-top: 0.8rem;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

th {
    background: rgba(30, 41, 59, 0.8) !important;
    color: #cbd5e1 !important;
    padding: 10px 14px !important;
    font-weight: 600 !important;
}

td {
    padding: 10px 14px !important;
    border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
    color: #f1f5f9 !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4) !important;
    transition: all 0.2s ease !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(2, 132, 199, 0.6) !important;
}
"""

theme_soft = gr.themes.Soft(primary_hue="cyan", secondary_hue="indigo", neutral_hue="slate")

with gr.Blocks(
    title="DeepGuard AI — Deepfake Detection",
    theme=theme_soft,
    css=CUSTOM_CSS
) as demo:

    gr.HTML("""
        <div class="hero-header">
            <h1>🛡️ DeepGuard AI</h1>
            <div class="hero-subtitle">
                Enterprise-Grade Deepfake &amp; Synthetic Media Detection Engine
            </div>
            <div class="model-badge">
                <span>⚡ Model V2 Active</span>
                <span>•</span>
                <span>EfficientNetB4 (380x380)</span>
                <span>•</span>
                <span>95.5% Val Accuracy</span>
            </div>
        </div>
    """)

    with gr.Tabs():
        # ---- Image Tab ----
        with gr.TabItem("🖼️ Image Detection", id="image"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        label="Upload Target Image",
                        type="numpy",
                        height=350
                    )
                    img_btn = gr.Button(
                        "🔍 Analyze Image",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    img_output = gr.Image(
                        label="Detection Overlay",
                        height=350
                    )

                with gr.Column(scale=1):
                    heatmap_output = gr.Image(
                        label="Grad-CAM Focus Map",
                        height=350
                    )

            img_result = gr.Markdown(label="Analysis Results")

            img_btn.click(
                analyze_image,
                inputs=img_input,
                outputs=[img_output, heatmap_output, img_result]
            )

        # ---- Video Tab ----
        with gr.TabItem("🎬 Video Detection", id="video"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_input = gr.Video(
                        label="Upload Target Video",
                        height=350
                    )
                    vid_btn = gr.Button(
                        "🔍 Analyze Video",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    vid_result = gr.Markdown(label="Video Frame Diagnostics")

            vid_btn.click(
                analyze_video,
                inputs=vid_input,
                outputs=vid_result
            )

        # ---- About Tab ----
        with gr.TabItem("ℹ️ Architecture & Methodology", id="about"):
            gr.Markdown("""
### 🛡️ DeepGuard AI Technical Overview

DeepGuard AI is a production-grade machine learning system engineered to detect deepfakes, AI-generated synthetic faces, and facial manipulations with high confidence.

#### 📐 Core System Architecture
- **Backbone Network:** EfficientNetB4 (19.3M parameters, ImageNet pretrained)
- **Input Resolution:** 380×380 px RGB
- **Face Localization:** MTCNN (Multi-Task Cascaded Convolutional Neural Networks)
- **Interpretability:** Grad-CAM (Gradient-Weighted Class Activation Mapping)
- **Optimizer:** AdamW with Weight Decay (`1e-4`) & Mixed Precision (`fp16`)

#### 🔄 V2 Cyclic Training Strategy
To eliminate overfitting and prevent memorization of noise:
- Data was split into **6 non-overlapping cyclic chunks** (30,000+ images per cycle).
- MD5 Hash Deduplication removed exact duplicate images across Kaggle & HuggingFace source datasets.
- Each cycle trained for 3 epochs with learning rate `1e-5`, preventing overconfidence while maintaining high generalization.

#### 📊 Benchmark Metrics (V2 Model)
| Metric | Score |
|:-------|:------|
| **Validation Accuracy** | **95.48%** |
| **Validation AUC** | **0.9913** |
| **Precision** | **95.66%** |
| **Recall** | **94.40%** |
            """)

    gr.HTML("""
        <div style="text-align: center; padding: 1.5rem; color: #64748b; font-size: 0.85rem;">
            DeepGuard AI — Built with EfficientNetB4, Grad-CAM &amp; Gradio
        </div>
    """)


# ============================================================
# LAUNCH
# ============================================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

