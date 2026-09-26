
from pathlib import Path
import sys
import importlib

# =========================================================
# NIR FEED AI — PERMANENT RUNTIME BOOTSTRAP
# =========================================================

# ---------------------------------------------------------
# 1. Mount Google Drive when necessary
# ---------------------------------------------------------
try:
    from google.colab import drive

    if not Path("/content/gdrive/MyDrive").exists():
        drive.mount("/content/gdrive")

except Exception as e:
    print("⚠️ Drive mount check:", e)

# ---------------------------------------------------------
# 2. Permanent project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path("/content/gdrive/MyDrive/NIR_Feed_AI_Project")

MODEL_ROOT = PROJECT_ROOT / "models"
DATASET_ROOT = PROJECT_ROOT / "datasets"
PIPELINE_ROOT = PROJECT_ROOT / "pipeline"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CV_ROOT = PROJECT_ROOT / "Computer_Vision"

# ---------------------------------------------------------
# 3. Add project directories to Python path
# ---------------------------------------------------------
for path in [
    PIPELINE_ROOT,
    CV_ROOT,
    CV_ROOT / "pipeline",
]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# ---------------------------------------------------------
# 4. Verify permanent assets
# ---------------------------------------------------------
required_assets = {
    "CP": MODEL_ROOT / "CP/cp_pls_model.pkl",
    "NDF": MODEL_ROOT / "NDF/ndf_pls_model.pkl",
    "ADF": MODEL_ROOT / "ADF/adf_pls_model.pkl",
    "IVDMD": MODEL_ROOT / "IVDMD/ivdmd_pls_model.pkl",
    "Moisture": MODEL_ROOT / "Moisture/moisture_pls_model.pkl",
    "Anomaly scaler": MODEL_ROOT / "Anomaly/anomaly_scaler.pkl",
    "Anomaly PCA": MODEL_ROOT / "Anomaly/anomaly_pca.pkl",
    "Isolation Forest": MODEL_ROOT / "Anomaly/isolation_forest.pkl",
    "CV model": CV_ROOT / "models/mould_efficientnetb0_best.keras",
    "Unified pipeline": PIPELINE_ROOT / "feed_ai_pipeline.py",
    "Nutrition wavelengths": MODEL_ROOT / "CP/cp_wavelength_cols.pkl",
}

print("=" * 70)
print("NIR FEED AI — BOOTSTRAP")
print("=" * 70)

missing = []

for name, path in required_assets.items():
    if path.exists():
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: {path}")
        missing.append(name)

if missing:
    raise FileNotFoundError(
        "Missing permanent assets: " + ", ".join(missing)
    )

# ---------------------------------------------------------
# 5. Load the unified pipeline fresh
# ---------------------------------------------------------
if "feed_ai_pipeline" in sys.modules:
    del sys.modules["feed_ai_pipeline"]

import feed_ai_pipeline

print("\n✅ Unified pipeline loaded")
print("Pipeline:", Path(feed_ai_pipeline.__file__).resolve())

# ---------------------------------------------------------
# 6. Verify key functions
# ---------------------------------------------------------
functions = [
    "predict_nutrition_from_nir",
    "predict_moisture_from_nir",
    "detect_spectral_anomaly",
    "generate_quality_assessment",
    "final_feed_analysis",
]

for fn in functions:
    if hasattr(feed_ai_pipeline, fn):
        print(f"✅ {fn}")
    else:
        raise AttributeError(f"Missing pipeline function: {fn}")

# ---------------------------------------------------------
# 7. Final runtime status
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("✅ NIR FEED AI RUNTIME READY")
print("=" * 70)
print("Permanent source:", PROJECT_ROOT)
print("Models:", MODEL_ROOT)
print("Datasets:", DATASET_ROOT)
print("Pipeline:", PIPELINE_ROOT)
print("Outputs:", OUTPUT_ROOT)
print("Computer Vision:", CV_ROOT)
print("=" * 70)
