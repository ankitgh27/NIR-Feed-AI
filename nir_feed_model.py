"""
NIR Feed AI Model
Reusable training utilities for NIR nutritional prediction.

Targets supported by the selected CIAT/Urochloa dataset:
CP, NDF, ADF, IVDMD.

This is a research/prototype calibration model, not a clinical/veterinary
or production feeding decision system.
"""

from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline


TARGET_ALIASES = {
    "CP": ["CP", "Crude Protein", "Crude_Protein", "crude_protein"],
    "NDF": ["NDF", "Neutral Detergent Fiber", "Neutral_Detergent_Fiber"],
    "ADF": ["ADF", "Acid Detergent Fiber", "Acid_Detergent_Fiber"],
    "IVDMD": ["IVDMD", "In Vitro Dry Matter", "In_Vitro_Dry_Matter"],
}


class SNVTransformer(BaseEstimator, TransformerMixin):
    """Standard Normal Variate: center and scale each spectrum independently."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        return (X - means) / stds


class SavitzkyGolayTransformer(BaseEstimator, TransformerMixin):
    """Optional Savitzky-Golay spectral smoothing/derivative transform."""

    def __init__(self, window_length=11, polyorder=2, deriv=1):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return savgol_filter(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            axis=1,
        )


def find_column(df, aliases):
    normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias.lower())
        if key in normalized:
            return normalized[key]
    return None


def detect_targets(df):
    found = {}
    for target, aliases in TARGET_ALIASES.items():
        col = find_column(df, aliases)
        if col is not None:
            found[target] = col
    return found


def detect_wavelength_columns(df):
    """Detect numeric spectral columns whose names look like wavelengths."""
    candidates = []
    for c in df.columns:
        s = str(c).strip()
        try:
            value = float(s)
            if 350 <= value <= 2600:
                candidates.append((value, c))
        except Exception:
            pass

    candidates.sort()
    return [c for _, c in candidates]


def load_tab_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, sep="\t")
    return df


def prepare_target(df, target_name):
    target_col = detect_targets(df).get(target_name)
    if target_col is None:
        raise ValueError(
            f"Could not find target {target_name}. Available columns include: "
            + ", ".join(map(str, df.columns[:40]))
        )

    wavelength_cols = detect_wavelength_columns(df)
    if len(wavelength_cols) < 100:
        raise ValueError(
            f"Only {len(wavelength_cols)} wavelength-like columns detected. "
            "Check the dataset format."
        )

    subset = df[wavelength_cols + [target_col]].copy()
    subset = subset.apply(pd.to_numeric, errors="coerce").dropna()

    X = subset[wavelength_cols].to_numpy(dtype=float)
    y = subset[target_col].to_numpy(dtype=float)
    wavelengths = np.array([float(c) for c in wavelength_cols], dtype=float)
    return X, y, wavelengths, target_col


def build_pls_pipeline(n_components=10, preprocessing="snv"):
    steps = []
    if preprocessing == "snv":
        steps.append(("snv", SNVTransformer()))
    elif preprocessing == "sg1":
        steps.append(("sg1", SavitzkyGolayTransformer(11, 2, 1)))
    elif preprocessing == "sg2":
        steps.append(("sg2", SavitzkyGolayTransformer(11, 2, 2)))
    elif preprocessing not in ("raw", None):
        raise ValueError("preprocessing must be raw, snv, sg1 or sg2")

    steps.append(("pls", PLSRegression(n_components=n_components, scale=True, max_iter=2000)))
    return Pipeline(steps)


def evaluate_regression(model, X_test, y_test):
    pred = np.asarray(model.predict(X_test)).ravel()
    return {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
        "R2": float(r2_score(y_test, pred)),
    }, pred


def train_target_model(
    df,
    target_name,
    n_components=10,
    preprocessing="snv",
    test_size=0.2,
    random_state=42,
):
    X, y, wavelengths, target_col = prepare_target(df, target_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = build_pls_pipeline(
        n_components=n_components,
        preprocessing=preprocessing,
    )
    model.fit(X_train, y_train)

    metrics, pred = evaluate_regression(model, X_test, y_test)

    return {
        "model": model,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "pred": pred,
        "wavelengths": wavelengths,
        "target_column": target_col,
        "n_samples": len(y),
    }


def tune_pls_components(X, y, preprocessing="snv", components=(2, 4, 6, 8, 10, 12, 15, 20)):
    """Cross-validate PLS component count using negative RMSE."""
    max_allowed = min(X.shape[0] - 1, X.shape[1])
    candidates = [c for c in components if c <= max_allowed]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for c in candidates:
        model = build_pls_pipeline(c, preprocessing)
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        rows.append({
            "n_components": c,
            "CV_RMSE_mean": float(-scores.mean()),
            "CV_RMSE_std": float(scores.std()),
        })

    return pd.DataFrame(rows).sort_values("CV_RMSE_mean").reset_index(drop=True)


def fit_anomaly_detector(X, contamination=0.03, random_state=42):
    """Fit a simple spectral anomaly detector after SNV preprocessing."""
    X_snv = SNVTransformer().fit_transform(X)
    detector = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=300,
    )
    detector.fit(X_snv)
    return detector


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
