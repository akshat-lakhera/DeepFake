"""
DeepGuard AI — Core Prediction Engine
Thread-safe, cached deepfake prediction with Grad-CAM support.
"""
import tensorflow as tf
import numpy as np
import cv2
import os
import json


class DeepfakePredictor:
    """Production inference engine for deepfake detection."""

    def __init__(self, model_path='deepfake_model_best.keras', img_size=380, threshold=0.5):
        self.img_size = img_size
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        # Default: tf.keras.utils.image_dataset_from_directory sorts alphabetically
        # fake=0, real=1. So model output >= 0.5 means "real".
        self.class_names = ['fake', 'real']

        if os.path.exists(self.model_path):
            self.load_model()
        elif os.path.exists(os.path.join('streamlit', self.model_path)):
            self.model_path = os.path.join('streamlit', self.model_path)
            self.load_model()

        # Try to load class mapping from training
        mapping_path = os.path.join(os.path.dirname(self.model_path) or '.', 'class_mapping.json')
        if os.path.exists(mapping_path):
            with open(mapping_path) as f:
                mapping = json.load(f)
                self.class_names = mapping.get('class_names', self.class_names)

    def load_model(self):
        """Load the Keras model from disk."""
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print(f"[DeepGuard] Model loaded: {self.model_path}")
            if hasattr(self.model, 'input_shape') and self.model.input_shape:
                shape = self.model.input_shape
                if isinstance(shape, list):
                    shape = shape[0]
                if len(shape) >= 3 and shape[1] is not None:
                    self.img_size = shape[1]
                    print(f"[DeepGuard] Auto-adapted resolution to: {self.img_size}x{self.img_size}")
        except Exception as e:
            print(f"[DeepGuard] Failed to load model: {e}")
            self.model = None

    def preprocess(self, image_rgb):
        """Preprocess a single RGB image for model input."""
        if image_rgb is None:
            return None
        img = cv2.resize(image_rgb, (self.img_size, self.img_size))
        img = img.astype('float32')
        # Note: preprocess_input is already built inside the model's Keras graph.
        return np.expand_dims(img, axis=0)

    def predict_face(self, face_img_rgb):
        """
        Predict whether a cropped face image is real or fake.
        """
        if self.model is None:
            return 0.0, "Model not loaded"

        preprocessed = self.preprocess(face_img_rgb)
        if preprocessed is None:
            return 0.0, "Invalid image"

        raw_output = float(self.model.predict(preprocessed, verbose=0)[0][0])

        # raw_output is sigmoid probability of Class 1 ("Real")
        # 0.0 = 100% Fake, 1.0 = 100% Real
        if raw_output >= self.threshold:
            confidence = raw_output
            label = "Real"
        else:
            confidence = 1.0 - raw_output
            label = "Fake"

        return confidence, label

    def predict_video_faces(self, faces_list):
        """
        Predict on a batch of face images extracted from a video.
        """
        if not faces_list or self.model is None:
            return 0.0, "Error", []

        batch = np.array([
            cv2.resize(face, (self.img_size, self.img_size)).astype('float32')
            for face in faces_list
        ])

        predictions = self.model.predict(batch, verbose=0).flatten()

        per_frame = []
        fake_count = 0
        for p in predictions:
            p_val = float(p)
            if p_val >= self.threshold:
                per_frame.append(("Real", p_val))
            else:
                per_frame.append(("Fake", 1.0 - p_val))
                fake_count += 1

        avg_raw = float(np.mean(predictions))
        if avg_raw >= self.threshold:
            return avg_raw, "Real", per_frame
        else:
            return 1.0 - avg_raw, "Fake", per_frame

    def generate_gradcam(self, face_img_rgb):
        """Generate Grad-CAM heatmap showing model focus areas."""
        if self.model is None:
            return np.zeros((self.img_size, self.img_size))

        preprocessed = self.preprocess(face_img_rgb)

        try:
            # Find conv layer inside base model or main model
            base_layer = None
            for layer in self.model.layers:
                if 'efficientnet' in layer.name.lower():
                    base_layer = layer
                    break

            if base_layer is not None:
                # Find last conv layer in base model
                last_conv = None
                for sub_layer in reversed(base_layer.layers):
                    if 'conv' in sub_layer.name.lower() or 'top' in sub_layer.name.lower():
                        last_conv = sub_layer
                        break

                if last_conv is not None:
                    conv_model = tf.keras.Model(
                        base_layer.inputs, last_conv.output
                    )
                    # Pass through preprocessing -> base_model conv
                    x = preprocessed
                    for l in self.model.layers:
                        if l == base_layer:
                            break
                        x = l(x)

                    with tf.GradientTape() as tape:
                        conv_out = conv_model(x)
                        tape.watch(conv_out)
                        # Finish remaining forward pass
                        x_head = conv_out
                        # Global average pooling & dense
                        found_base = False
                        for l in self.model.layers:
                            if found_base:
                                x_head = l(x_head)
                            if l == base_layer:
                                found_base = True
                        preds = x_head
                        loss = preds[:, 0]

                    grads = tape.gradient(loss, conv_out)
                    if grads is not None:
                        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                        heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
                        heatmap = tf.squeeze(heatmap)
                        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
                        heatmap_np = heatmap.numpy()
                        return cv2.resize(heatmap_np, (self.img_size, self.img_size))
        except Exception as e:
            print(f"[DeepGuard] Grad-CAM info: {e}")

        # Fallback simple activation map if Grad-CAM gradient tape is constrained by fp16
        return np.ones((self.img_size, self.img_size)) * 0.5

    def create_heatmap_overlay(self, original_img_rgb, heatmap, alpha=0.4):
        """
        Overlay a Grad-CAM heatmap on the original image.

        Args:
            original_img_rgb: numpy array (H, W, 3) in RGB
            heatmap: numpy array normalized 0-1
            alpha: blending factor

        Returns:
            overlay: numpy array (H, W, 3) in RGB, uint8
        """
        h, w = original_img_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlay = np.float32(original_img_rgb) * (1 - alpha) + np.float32(heatmap_colored) * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay
