# ============================================================
# Deepfake Detection — Centralized Configuration
# ============================================================

# Model Architecture
IMG_SIZE = 380
BACKBONE = "EfficientNetB4"
DENSE_UNITS = 256
DROPOUT_1 = 0.5
DROPOUT_2 = 0.3

# Training — Phase 1 (Frozen Backbone)
EPOCHS_PHASE1 = 5
LR_PHASE1 = 1e-3

# Training — Phase 2 (Fine-Tuning)
EPOCHS_PHASE2 = 15
LR_PHASE2 = 1e-5
WEIGHT_DECAY = 1e-4
UNFREEZE_LAYERS = 50
EARLY_STOP_PATIENCE = 5

# Data
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Inference
DEFAULT_THRESHOLD = 0.5
MODEL_PATH = "streamlit/my_model.keras"

# Class Mapping (set by tf.keras.utils.image_dataset_from_directory, alphabetical)
# fake = 0, real = 1
# Model output > 0.5 → Real, < 0.5 → Fake
CLASS_NAMES = ["fake", "real"]
