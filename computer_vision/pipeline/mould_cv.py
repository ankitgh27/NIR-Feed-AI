
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "mould_efficientnetb0_best.keras"
)
SCREENING_THRESHOLD = 0.60

_model = None


def load_model():
    """Load the saved mould classifier once."""
    global _model

    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)

    return _model


def predict_mould_from_image(image_path):
    """Predict visible mould probability from one image."""

    model = load_model()

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    probability = float(
        model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    classification = (
        "MOLDY"
        if probability >= SCREENING_THRESHOLD
        else "NORMAL"
    )

    return {
        "mould_probability": round(probability, 4),
        "classification": classification,
        "threshold": SCREENING_THRESHOLD
    }


def analyze_mould(image_path):
    """Complete API-ready mould analysis."""

    result = predict_mould_from_image(image_path)

    probability = result["mould_probability"]

    if probability >= 0.60:
        visual_status = "HIGH_VISIBLE_MOULD_RISK"
        advisory = (
            "Visible mould-like characteristics detected. "
            "Further inspection is recommended before using the feed."
        )

    elif probability >= 0.35:
        visual_status = "POSSIBLE_VISIBLE_MOULD"
        advisory = (
            "Some mould-like visual characteristics detected. "
            "Manual inspection is recommended."
        )

    else:
        visual_status = "LOW_VISIBLE_MOULD_INDICATION"
        advisory = (
            "No strong visible mould indication detected "
            "from the image."
        )

    return {
        "module": "computer_vision_mould",
        "model": "EfficientNetB0",
        "mould_probability": probability,
        "threshold": SCREENING_THRESHOLD,
        "classification": result["classification"],
        "visual_status": visual_status,
        "advisory": advisory,
        "scientific_note": (
            "Visual screening only. This does not confirm "
            "fungal species or aflatoxin/mycotoxin concentration."
        )
    }
