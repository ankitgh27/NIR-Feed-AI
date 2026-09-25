
FEED QUALITY AI MODEL
=====================

Models:
- CP PLS regression
- NDF PLS regression
- ADF PLS regression
- IVDMD PLS regression

Confidence system:
- StandardScaler
- PCA
- Spectral distance threshold

Input:
- 1050 NIR wavelength values
- 400–2498 nm
- 2 nm spacing

Outputs:
- Crude Protein (%)
- NDF (%)
- ADF (%)
- IVDMD (%)
- Spectral confidence
- Quality status
- Risk flags
- Recommendations

Important:
These models were trained using a forage NIR dataset.
They should be locally validated against laboratory
reference measurements before real-world deployment.
