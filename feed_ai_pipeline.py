
import sys
from pathlib import Path

import joblib
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_ROOT = PROJECT_ROOT / "models"

MOULD_MODULE_DIR = (
    PROJECT_ROOT
    / "computer_vision"
    / "pipeline"
)


# ============================================================
# NIR MODELS
# ============================================================

CP_MODEL = joblib.load(
    MODEL_ROOT / "CP" / "cp_pls_model.pkl"
)

NDF_MODEL = joblib.load(
    MODEL_ROOT / "NDF" / "ndf_pls_model.pkl"
)

ADF_MODEL = joblib.load(
    MODEL_ROOT / "ADF" / "adf_pls_model.pkl"
)

IVDMD_MODEL = joblib.load(
    MODEL_ROOT / "IVDMD" / "ivdmd_pls_model.pkl"
)

NUTRITION_WAVELENGTHS = joblib.load(
    MODEL_ROOT / "CP" / "cp_wavelength_cols.pkl"
)


# ============================================================
# MOISTURE MODEL
# ============================================================

MOISTURE_MODEL = joblib.load(
    MODEL_ROOT
    / "Moisture"
    / "moisture_pls_model.pkl"
)

MOISTURE_WAVELENGTHS = joblib.load(
    MODEL_ROOT
    / "Moisture"
    / "moisture_wavelength_cols.pkl"
)


# ============================================================
# SPECTRAL ANOMALY SYSTEM
# ============================================================

ANOMALY_DIR = MODEL_ROOT / "Anomaly"

ANOMALY_SCALER = joblib.load(
    ANOMALY_DIR / "anomaly_scaler.pkl"
)

ANOMALY_PCA = joblib.load(
    ANOMALY_DIR / "anomaly_pca.pkl"
)

ISOLATION_FOREST = joblib.load(
    ANOMALY_DIR / "isolation_forest.pkl"
)

PCA_TRAINING_CENTER = joblib.load(
    ANOMALY_DIR / "pca_training_center.pkl"
)

PCA_DISTANCE_THRESHOLD = float(
    joblib.load(
        ANOMALY_DIR / "pca_distance_threshold.pkl"
    )
)


# ============================================================
# MOULD CV
# ============================================================

if str(MOULD_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MOULD_MODULE_DIR))

import mould_cv


# ============================================================
# HELPERS
# ============================================================

def _prepare_spectrum(
    spectrum,
    expected_features,
    name
):
    """Validate one spectrum."""

    X = np.asarray(
        spectrum,
        dtype=float
    ).reshape(1, -1)

    if X.shape[1] != expected_features:
        raise ValueError(
            f"{name} requires {expected_features} values, "
            f"received {X.shape[1]}."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return X


# ============================================================
# NUTRITION PREDICTION
# ============================================================

def predict_nutrition_from_nir(
    nutrition_nir_spectrum
):
    """Predict CP, NDF, ADF and IVDMD."""

    X = _prepare_spectrum(
        nutrition_nir_spectrum,
        len(NUTRITION_WAVELENGTHS),
        "Nutrition NIR spectrum"
    )

    return {
        "CP_percent": round(
            float(
                CP_MODEL.predict(X).ravel()[0]
            ),
            4
        ),

        "NDF_percent": round(
            float(
                NDF_MODEL.predict(X).ravel()[0]
            ),
            4
        ),

        "ADF_percent": round(
            float(
                ADF_MODEL.predict(X).ravel()[0]
            ),
            4
        ),

        "IVDMD_percent": round(
            float(
                IVDMD_MODEL.predict(X).ravel()[0]
            ),
            4
        )
    }


# ============================================================
# MOISTURE PREDICTION
# ============================================================

def predict_moisture_from_nir(
    moisture_nir_spectrum
):
    """Predict moisture from the 141-feature spectrum."""

    X = _prepare_spectrum(
        moisture_nir_spectrum,
        len(MOISTURE_WAVELENGTHS),
        "Moisture NIR spectrum"
    )

    value = float(
        MOISTURE_MODEL.predict(X).ravel()[0]
    )

    return {
        "Moisture_percent": round(
            value,
            4
        )
    }


# ============================================================
# SPECTRAL ANOMALY DETECTION
# ============================================================

def detect_spectral_anomaly(
    nutrition_nir_spectrum
):
    """
    Detect unusual spectral patterns.

    IMPORTANT:
    This does NOT identify a specific contaminant.
    """

    X = _prepare_spectrum(
        nutrition_nir_spectrum,
        len(NUTRITION_WAVELENGTHS),
        "Nutrition NIR spectrum"
    )

    X_scaled = ANOMALY_SCALER.transform(X)

    X_pca = ANOMALY_PCA.transform(X_scaled)

    # Isolation Forest
    if_prediction = int(
        ISOLATION_FOREST.predict(X_pca)[0]
    )

    if_score = float(
        ISOLATION_FOREST.decision_function(X_pca)[0]
    )

    if_anomalous = (
        if_prediction == -1
    )

    # PCA distance
    pca_distance = float(
        np.linalg.norm(
            X_pca[0] - PCA_TRAINING_CENTER
        )
    )

    pca_anomalous = (
        pca_distance >
        PCA_DISTANCE_THRESHOLD
    )

    # Strong warning only when both systems agree
    strong_warning = (
        if_anomalous
        and pca_anomalous
    )

    return {
        "isolation_forest_anomalous":
            bool(if_anomalous),

        "isolation_forest_score":
            round(if_score, 4),

        "pca_distance":
            round(pca_distance, 4),

        "pca_distance_threshold":
            round(
                PCA_DISTANCE_THRESHOLD,
                4
            ),

        "pca_distance_anomalous":
            bool(pca_anomalous),

        "combined_status":
            (
                "STRONG_SPECTRAL_WARNING"
                if strong_warning
                else "NO_STRONG_SPECTRAL_WARNING"
            ),

        "specific_contaminant_detection":
            False
    }


# ============================================================
# QUALITY ASSESSMENT
# ============================================================

def generate_quality_assessment(
    nutrition,
    moisture,
    spectral_analysis=None,
    mould_analysis=None
):
    """
    Prototype rule-based interpretation.

    These thresholds are NOT validated veterinary/feed
    safety standards.
    """

    cp = nutrition["CP_percent"]
    ndf = nutrition["NDF_percent"]
    adf = nutrition["ADF_percent"]
    ivdmd = nutrition["IVDMD_percent"]
    moisture_value = moisture["Moisture_percent"]

    risks = []
    recommendations = []

    # CP
    if cp < 7:
        cp_status = "LOW"
        risks.append("LOW_CRUDE_PROTEIN")
    elif cp <= 12:
        cp_status = "ADEQUATE"
    else:
        cp_status = "HIGH"

    # NDF
    if ndf > 60:
        ndf_status = "HIGH"
        risks.append("HIGH_NDF")
    elif ndf >= 40:
        ndf_status = "MODERATE"
    else:
        ndf_status = "LOW"

    # ADF
    if adf > 40:
        adf_status = "HIGH"
        risks.append("HIGH_ADF")
    elif adf >= 30:
        adf_status = "MODERATE"
    else:
        adf_status = "LOW"

    # IVDMD
    if ivdmd < 55:
        ivdmd_status = "LOW"
        risks.append("LOW_DIGESTIBILITY")
    elif ivdmd <= 65:
        ivdmd_status = "MODERATE"
    else:
        ivdmd_status = "GOOD"

    # Moisture
    if moisture_value > 70:
        moisture_status = "VERY_HIGH"
        risks.append("VERY_HIGH_MOISTURE")
    elif moisture_value > 50:
        moisture_status = "HIGH"
        risks.append("HIGH_MOISTURE")
    elif moisture_value >= 20:
        moisture_status = "MODERATE"
    else:
        moisture_status = "LOW"

    # Spectral anomaly
    if spectral_analysis is not None:

        if (
            spectral_analysis["combined_status"]
            == "STRONG_SPECTRAL_WARNING"
        ):
            risks.append(
                "UNUSUAL_SPECTRAL_PATTERN"
            )

            recommendations.append(
                "Unusual NIR spectral pattern detected. "
                "Manual review is recommended."
            )

    # Visible mould
    if mould_analysis is not None:

        mould_status = mould_analysis.get(
            "visual_status"
        )

        if mould_status == "HIGH_VISIBLE_MOULD_RISK":

            risks.append(
                "VISIBLE_MOULD_RISK"
            )

            recommendations.append(
                "Visible mould-like characteristics detected. "
                "Further inspection is recommended before feed use."
            )

        elif mould_status == "POSSIBLE_VISIBLE_MOULD":

            risks.append(
                "POSSIBLE_VISIBLE_MOULD"
            )

            recommendations.append(
                "Possible visible mould characteristics detected. "
                "Manual inspection is recommended."
            )

    # Basic recommendations
    if cp_status == "LOW":
        recommendations.append(
            "Review protein adequacy in the feed formulation."
        )

    if ndf_status == "HIGH":
        recommendations.append(
            "Review fibre level and ration formulation."
        )

    if adf_status == "HIGH":
        recommendations.append(
            "Review fibre quality and digestibility implications."
        )

    if ivdmd_status == "LOW":
        recommendations.append(
            "Low predicted digestibility; formulation review is recommended."
        )

    if moisture_status in [
        "HIGH",
        "VERY_HIGH"
    ]:
        recommendations.append(
            "Review moisture and storage conditions because elevated "
            "moisture can increase spoilage risk."
        )

    # Overall result
    if len(risks) >= 2:
        overall = "NEEDS_ATTENTION"

    elif len(risks) == 1:
        overall = "MODERATE"

    else:
        overall = "GOOD"

    return {
        "overall_status":
            overall,

        "individual_status": {
            "CP": cp_status,
            "NDF": ndf_status,
            "ADF": adf_status,
            "IVDMD": ivdmd_status,
            "Moisture": moisture_status
        },

        "risk_flags":
            risks,

        "recommendations":
            recommendations,

        "prototype_rule_note":
            "These interpretation thresholds are prototype rules "
            "and are not validated veterinary or regulatory standards."
    }


# ============================================================
# COMPLETE MULTIMODAL ANALYSIS
# ============================================================

def final_feed_analysis(
    nutrition_nir_spectrum,
    moisture_nir_spectrum=None,
    image_path=None,
    sample_id=None
):
    """
    Complete multimodal feed analysis.

    Nutrition spectrum:
        1050 values, 400–2498 nm

    Moisture spectrum:
        141 values, 950–1650 nm

    Image:
        Optional visible-mould screening image.
    """

    nutrition = predict_nutrition_from_nir(
        nutrition_nir_spectrum
    )

    moisture = None

    if moisture_nir_spectrum is not None:
        moisture = predict_moisture_from_nir(
            moisture_nir_spectrum
        )

    spectral_analysis = detect_spectral_anomaly(
        nutrition_nir_spectrum
    )

    mould_analysis = None

    if image_path is not None:
        mould_analysis = mould_cv.analyze_mould(
            image_path
        )

    quality = None

    if moisture is not None:
        quality = generate_quality_assessment(
            nutrition=nutrition,
            moisture=moisture,
            spectral_analysis=spectral_analysis,
            mould_analysis=mould_analysis
        )

    return {
        "sample_id":
            sample_id,

        "nutritional_analysis":
            nutrition,

        "moisture_analysis":
            moisture,

        "spectral_analysis":
            spectral_analysis,

        "computer_vision_analysis":
            mould_analysis,

        "quality_assessment":
            quality,

        "limitations": [
            "NIR nutritional models require local validation "
            "against laboratory reference measurements.",

            "Moisture uses a separate 950–1650 nm spectral domain.",

            "Visible mould screening was developed using "
            "cereal grain images and requires validation on "
            "actual cattle-feed/silage images.",

            "Visible mould screening does not confirm fungal "
            "species or aflatoxin/mycotoxin concentration.",

            "Spectral anomaly detection identifies unusual "
            "spectra but does not identify a specific contaminant.",

            "Sand/silica, urea adulteration and mineral estimation "
            "are not currently validated modules."
        ]
    }
