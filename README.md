# 🐄 NIR Feed AI

### AI-Assisted Cattle Feed Quality Assessment using NIR Spectroscopy & Machine Learning

NIR Feed AI is a research and prototype system designed to analyse cattle-feed quality using **Near-Infrared (NIR) spectroscopy, Machine Learning, spectral anomaly detection, moisture analysis and computer vision**.

The system is designed as a multimodal pipeline connecting AI models with a FastAPI backend and a Flutter-based mobile interface.

---

## 🎯 Project Objective

Traditional feed-quality assessment can require laboratory testing, specialised equipment and trained personnel.

NIR Feed AI explores an AI-assisted approach where spectral information from feed samples can be processed to estimate important nutritional and quality parameters.

---

## 🧠 What Does NIR Feed AI Analyse?

### 1. Nutritional Parameters

The NIR nutrition models predict:

| Parameter | Full Form |
|---|---|
| **CP** | Crude Protein |
| **NDF** | Neutral Detergent Fiber |
| **ADF** | Acid Detergent Fiber |
| **IVDMD** | In-Vitro Dry Matter Digestibility |

The initial nutrition model uses NIR spectra containing approximately **1050 wavelength features** over the **400–2498 nm** spectral range.

---

### 2. Moisture Analysis

A separate NIR model is used for moisture estimation.

**Spectral range:**

```text
950–1650 nm
