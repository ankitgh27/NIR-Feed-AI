from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile


# =========================================================
# NIR FEED AI API
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PIPELINE_DIR = PROJECT_ROOT

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


import feed_ai_pipeline


# =========================================================
# DEMO DATA CACHE
# =========================================================
# The demo endpoint caches the real datasets after the first
# request. This prevents scanning thousands of cells on every
# F001/F002/F003 request.

_demo_nutrition_spectra = None
_demo_nutrition_rows = None

_demo_moisture_spectra = None
_demo_moisture_rows = None

_demo_normal_images = None
_demo_moldy_images = None


app = FastAPI(
    title="NIR Feed AI API",
    version="1.0",
    description=(
        "Multimodal cattle-feed quality assessment API using "
        "NIR models, spectral anomaly screening and visible "
        "mould image screening."
    )
)


# =========================================================
# API RESPONSE SCHEMA
# =========================================================

class AnalyzeResponse(BaseModel):
    """
    Top-level response contract for /analyze.

    The multimodal pipeline may evolve by adding analysis
    sections, so unknown top-level fields are preserved.
    """

    sample_id: Optional[str] = None

    model_config = ConfigDict(extra="allow")


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "service": "NIR Feed AI API",
        "version": "1.0",
        "status": "ready"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "pipeline": "loaded",
        "project_root": str(PROJECT_ROOT)
    }


# =========================================================
# DEMO ANALYSIS ENDPOINT
# =========================================================
#
# This endpoint is for the Flutter Android Emulator.
#
# Flutter sends:
#     sample_id = F001
#
# FastAPI then loads the real demo data:
#
#     1050 nutrition NIR values
#     141 moisture NIR values
#     real mould image
#
# and sends them through the REAL trained AI pipeline.
#
# Android Emulator accesses the Windows PC using:
#
#     http://10.0.2.2:8000
#
# =========================================================

@app.post("/demo-analyze")
async def demo_analyze(
    sample_id: str = Form("F001")
):
    global _demo_nutrition_spectra
    global _demo_nutrition_rows
    global _demo_moisture_spectra
    global _demo_moisture_rows
    global _demo_normal_images
    global _demo_moldy_images

    try:

        # =====================================================
        # 1. CONVERT SAMPLE ID TO SAMPLE NUMBER
        # =====================================================

        try:
            sample_number = int(
                sample_id.upper().replace("F", "")
            )
        except ValueError:
            raise ValueError(
                "Invalid sample_id. Use values such as F001, F002, F003."
            )

        if sample_number < 1:
            raise ValueError(
                "Sample number must be >= 1."
            )

        # =====================================================
        # 2. LOAD/CACHE REAL NUTRITION SPECTRA
        # =====================================================

        if _demo_nutrition_spectra is None:

            nutrition_file = (
                PROJECT_ROOT
                / "dataverse_files"
                / "02a. Fulldata.tab"
            )

            if not nutrition_file.exists():
                raise FileNotFoundError(
                    f"Nutrition dataset not found: {nutrition_file}"
                )

            nutrition_df = pd.read_csv(
                nutrition_file,
                sep="\t",
                low_memory=False
            )

            # Expected wavelengths:
            # 400, 402, 404, ..., 2498 = 1050 values

            nutrition_wavelength_columns = []

            for column in nutrition_df.columns:
                try:
                    wavelength = int(str(column).strip())

                    if 400 <= wavelength <= 2498:
                        nutrition_wavelength_columns.append(
                            (wavelength, column)
                        )

                except (ValueError, TypeError):
                    continue

            nutrition_wavelength_columns.sort(
                key=lambda item: item[0]
            )

            if len(nutrition_wavelength_columns) != 1050:
                raise ValueError(
                    "Expected 1050 nutrition wavelength columns, "
                    f"found {len(nutrition_wavelength_columns)}."
                )

            nutrition_columns = [
                column
                for _, column in nutrition_wavelength_columns
            ]

            # Vectorized conversion is much faster than looping
            # through every cell individually.
            nutrition_numeric = nutrition_df[
                nutrition_columns
            ].apply(
                pd.to_numeric,
                errors="coerce"
            )

            valid_mask = (
                nutrition_numeric.notna().all(axis=1)
                & np.isfinite(
                    nutrition_numeric.to_numpy(
                        dtype=float
                    )
                ).all(axis=1)
            )

            valid_rows = np.flatnonzero(
                valid_mask.to_numpy()
            )

            if len(valid_rows) == 0:
                raise ValueError(
                    "No complete nutrition NIR spectra were found."
                )

            _demo_nutrition_rows = valid_rows.tolist()

            _demo_nutrition_spectra = (
                nutrition_numeric.iloc[
                    valid_rows
                ].to_numpy(
                    dtype=float
                )
            )

        nutrition_position = (
            sample_number - 1
        ) % len(_demo_nutrition_rows)

        nutrition_row_index = _demo_nutrition_rows[
            nutrition_position
        ]

        nutrition_spectrum = (
            _demo_nutrition_spectra[
                nutrition_position
            ].tolist()
        )

        if len(nutrition_spectrum) != 1050:
            raise ValueError(
                "Nutrition spectrum must contain exactly "
                f"1050 values; got {len(nutrition_spectrum)}."
            )

        # =====================================================
        # 3. LOAD/CACHE REAL MOISTURE SPECTRA
        # =====================================================

        if _demo_moisture_spectra is None:

            moisture_file = (
                PROJECT_ROOT
                / "datasets"
                / "sensAIfood_Perten"
                / "Corn_sensAIfood_Perten.csv"
            )

            if not moisture_file.exists():
                raise FileNotFoundError(
                    f"Moisture dataset not found: {moisture_file}"
                )

            moisture_df = pd.read_csv(
                moisture_file,
                low_memory=False
            )

            # Expected:
            # 950, 955, 960, ..., 1650 = 141 values

            moisture_wavelength_columns = []

            for column in moisture_df.columns:
                try:
                    wavelength = int(str(column).strip())

                    if (
                        950 <= wavelength <= 1650
                        and (wavelength - 950) % 5 == 0
                    ):
                        moisture_wavelength_columns.append(
                            (wavelength, column)
                        )

                except (ValueError, TypeError):
                    continue

            moisture_wavelength_columns.sort(
                key=lambda item: item[0]
            )

            if len(moisture_wavelength_columns) != 141:
                raise ValueError(
                    "Expected 141 moisture wavelength columns, "
                    f"found {len(moisture_wavelength_columns)}."
                )

            moisture_columns = [
                column
                for _, column in moisture_wavelength_columns
            ]

            moisture_numeric = moisture_df[
                moisture_columns
            ].apply(
                pd.to_numeric,
                errors="coerce"
            )

            valid_mask = (
                moisture_numeric.notna().all(axis=1)
                & np.isfinite(
                    moisture_numeric.to_numpy(
                        dtype=float
                    )
                ).all(axis=1)
            )

            valid_rows = np.flatnonzero(
                valid_mask.to_numpy()
            )

            if len(valid_rows) == 0:
                raise ValueError(
                    "No complete moisture NIR spectra were found."
                )

            _demo_moisture_rows = valid_rows.tolist()

            _demo_moisture_spectra = (
                moisture_numeric.iloc[
                    valid_rows
                ].to_numpy(
                    dtype=float
                )
            )

        moisture_position = (
            sample_number - 1
        ) % len(_demo_moisture_rows)

        moisture_row_index = _demo_moisture_rows[
            moisture_position
        ]

        moisture_spectrum = (
            _demo_moisture_spectra[
                moisture_position
            ].tolist()
        )

        if len(moisture_spectrum) != 141:
            raise ValueError(
                "Moisture spectrum must contain exactly "
                f"141 values; got {len(moisture_spectrum)}."
            )

        # =====================================================
        # 4. LOAD/CACHE REAL MOULD IMAGES
        # =====================================================

        if (
            _demo_normal_images is None
            or _demo_moldy_images is None
        ):

            normal_dir = (
                PROJECT_ROOT
                / "Computer_Vision"
                / "runtime_backup"
                / "Mould_CV_Dataset"
                / "train"
                / "0_Normal"
            )

            moldy_dir = (
                PROJECT_ROOT
                / "Computer_Vision"
                / "runtime_backup"
                / "Mould_CV_Dataset"
                / "train"
                / "1_Moldy"
            )

            _demo_normal_images = sorted(
                list(normal_dir.glob("*.png"))
                + list(normal_dir.glob("*.jpg"))
                + list(normal_dir.glob("*.jpeg"))
            )

            _demo_moldy_images = sorted(
                list(moldy_dir.glob("*.png"))
                + list(moldy_dir.glob("*.jpg"))
                + list(moldy_dir.glob("*.jpeg"))
            )

            if not _demo_normal_images:
                raise FileNotFoundError(
                    f"No normal mould images found in {normal_dir}"
                )

            if not _demo_moldy_images:
                raise FileNotFoundError(
                    f"No moldy images found in {moldy_dir}"
                )

        # Odd samples -> moldy
        # Even samples -> normal

        if sample_number % 2 == 1:
            image_list = _demo_moldy_images
        else:
            image_list = _demo_normal_images

        image_index = (
            (sample_number - 1) // 2
        ) % len(image_list)

        mould_image = image_list[
            image_index
        ]

        # =====================================================
        # 5. RUN THE REAL UNIFIED AI PIPELINE
        # =====================================================

        result = feed_ai_pipeline.final_feed_analysis(
            nutrition_nir_spectrum=nutrition_spectrum,
            moisture_nir_spectrum=moisture_spectrum,
            image_path=str(mould_image),
            sample_id=sample_id
        )

        # =====================================================
        # 6. RETURN DEMO SOURCE INFORMATION
        # =====================================================

        result["demo_input"] = {
            "sample_id": sample_id,
            "nutrition_source_row": int(
                nutrition_row_index
            ),
            "moisture_source_row": int(
                moisture_row_index
            ),
            "mould_image": mould_image.name,
        }

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Feed analysis failed: {e}"
        )


# =========================================================
# REAL /ANALYZE ENDPOINT
# =========================================================

@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
)
async def analyze_feed(
    nutrition_nir: str = Form(...),
    moisture_nir: Optional[str] = Form(None),
    sample_id: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    """
    Run complete multimodal feed analysis.

    nutrition_nir:
        1050 comma-separated NIR values.

    moisture_nir:
        Optional 141 comma-separated NIR values.

    image:
        Optional JPG/JPEG/PNG feed image.
    """

    # -----------------------------------------------------
    # Parse nutrition spectrum
    # -----------------------------------------------------

    try:
        nutrition_values = np.array(
            [
                float(x.strip())
                for x in nutrition_nir.split(",")
                if x.strip() != ""
            ],
            dtype=float
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid nutrition_nir data: {e}"
        )

    if len(nutrition_values) != 1050:

        raise HTTPException(
            status_code=400,
            detail=(
                "nutrition_nir must contain exactly "
                f"1050 values; received {len(nutrition_values)}."
            )
        )

    if not np.all(
        np.isfinite(nutrition_values)
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "nutrition_nir contains NaN "
                "or infinite values."
            )
        )


    # -----------------------------------------------------
    # Parse moisture spectrum when supplied
    # -----------------------------------------------------

    moisture_values = None

    if moisture_nir is not None:

        try:
            moisture_values = np.array(
                [
                    float(x.strip())
                    for x in moisture_nir.split(",")
                    if x.strip() != ""
                ],
                dtype=float
            )

        except Exception as e:

            raise HTTPException(
                status_code=400,
                detail=f"Invalid moisture_nir data: {e}"
            )

        if len(moisture_values) != 141:

            raise HTTPException(
                status_code=400,
                detail=(
                    "moisture_nir must contain exactly "
                    f"141 values; received {len(moisture_values)}."
                )
            )

        if not np.all(
            np.isfinite(moisture_values)
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "moisture_nir contains NaN "
                    "or infinite values."
                )
            )


    # -----------------------------------------------------
    # Handle optional image
    # -----------------------------------------------------

    temp_image_path = None

    if image is not None:

        allowed_types = {
            "image/jpeg",
            "image/png"
        }

        if image.content_type not in allowed_types:

            raise HTTPException(
                status_code=400,
                detail="Image must be JPEG or PNG."
            )

        suffix = (
            ".png"
            if image.content_type == "image/png"
            else ".jpg"
        )

        try:

            contents = await image.read()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(contents)
                temp_image_path = temp_file.name

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=f"Could not process image: {e}"
            )


    # -----------------------------------------------------
    # Run existing unified pipeline
    # -----------------------------------------------------

    try:

        result = feed_ai_pipeline.final_feed_analysis(
            nutrition_nir_spectrum=nutrition_values,
            moisture_nir_spectrum=moisture_values,
            image_path=temp_image_path,
            sample_id=sample_id
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Feed analysis failed: {e}"
        )

    finally:

        # Remove temporary image
        if temp_image_path is not None:

            try:
                Path(temp_image_path).unlink(
                    missing_ok=True
                )

            except Exception:
                pass
