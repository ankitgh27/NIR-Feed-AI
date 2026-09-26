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
```

The moisture model uses approximately **141 wavelength features** and is trained separately from the nutritional models because it operates on a different spectral domain.

The model estimates the moisture content of a feed sample and helps identify potential quality issues associated with excessive moisture.

---

## 🔬 3. Spectral Anomaly Detection

NIR Feed AI also includes a spectral anomaly detection module.

The purpose of this module is to determine whether an input spectrum significantly differs from the spectral patterns observed in the training data.

Two complementary approaches are used:

* **Isolation Forest**
* **Principal Component Analysis (PCA)**

### Isolation Forest

Isolation Forest is used to identify unusual spectral patterns that may represent samples outside the normal distribution.

```text
Input Spectrum
      ↓
Feature Processing
      ↓
Isolation Forest
      ↓
Anomaly / Normal
```

### PCA-Based Distance Analysis

Principal Component Analysis is used to project high-dimensional spectral data into a lower-dimensional representation.

The system calculates the distance of an input spectrum from the learned spectral distribution and compares it with a predefined threshold.

This provides an additional layer of validation before interpreting the model predictions.

---

## 👁️ 4. Visible Mould Detection

The project also explores computer-vision-based detection of visible mould contamination.

A computer vision pipeline analyses feed/grain images and attempts to identify visual characteristics associated with mould.

The intended workflow is:

```text
Feed Image
    ↓
Image Preprocessing
    ↓
Computer Vision Model
    ↓
Mould Probability
    ↓
Quality Assessment
```

This module is designed as a complementary method to NIR-based analysis.

NIR spectroscopy provides information from the spectral characteristics of the sample, while computer vision can provide visual evidence of surface-level contamination.

> **Note:** The current mould-detection prototype requires further validation on real cattle-feed and silage images before being used as a production diagnostic system.

---

# 🧠 Multimodal AI Pipeline

NIR Feed AI combines multiple AI components into a unified feed-quality assessment pipeline.

```text
                    ┌──────────────────────┐
                    │    Feed Sample       │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       NIR Spectrum       Moisture NIR      Feed Image
              │                │                │
              ▼                ▼                ▼
       Nutrition Models   Moisture Model   CV Model
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                    Spectral Anomaly Check
                               │
                               ▼
                     Quality Assessment
                               │
                               ▼
                    ┌────────────────────┐
                    │   Final Analysis   │
                    └────────────────────┘
```

The system combines the outputs of different models to generate an interpretable feed-quality assessment.

---

# 📊 Quality Assessment

The system does not only return raw numerical predictions.

The predicted values are also interpreted against predefined quality ranges to generate understandable status indicators.

Example:

```text
Crude Protein       → LOW
NDF                 → HIGH
ADF                 → MODERATE
IVDMD               → GOOD
Moisture            → ATTENTION
Spectral Anomaly    → NOT DETECTED
```

The overall result can then be represented as:

```text
NEEDS_ATTENTION
```

This approach is intended to make complex ML outputs easier to understand for users who may not have expertise in spectroscopy or machine learning.

---

# 🏗️ System Architecture

NIR Feed AI follows a modular architecture consisting of:

### AI/ML Layer

Responsible for:

* NIR preprocessing
* Nutritional prediction
* Moisture prediction
* Spectral anomaly detection
* Computer vision analysis
* Quality classification

### Backend Layer

The backend is implemented using **FastAPI**.

It provides API endpoints for communication between the AI models and the mobile application.

Example workflow:

```text
Flutter Application
        ↓
     REST API
        ↓
     FastAPI
        ↓
   AI Pipeline
        ↓
ML Models / Analysis
        ↓
     JSON Result
        ↓
Flutter Application
```

### Mobile Application

A Flutter-based mobile application provides the user interface.

The application is designed to:

* Submit feed analysis requests
* Display nutritional predictions
* Display moisture estimation
* Show anomaly status
* Display mould analysis
* Present an overall quality assessment
* Provide results in a farmer-friendly format

---

# 📱 Flutter Mobile Application

The project includes a prototype mobile application named:

**Feed Quality AI**

The interface is designed with accessibility and simplicity in mind.

### Planned / Prototype Features

* 🌾 Feed quality analysis
* 🤖 AI-powered predictions
* 📊 Nutritional metrics dashboard
* 💧 Moisture estimation
* 🔬 Spectral anomaly detection
* 👁️ Visual mould screening
* 🌐 Multilingual interface
* 📱 Mobile-first workflow

The application is intended to support:

* English
* Hindi
* Bengali

The goal is to make AI-assisted feed analysis accessible beyond laboratory environments.

---

# ⚡ FastAPI Backend

FastAPI acts as the communication layer between the Flutter application and the machine-learning pipeline.

A typical request follows:

```text
POST /analyze
```

The backend receives the required analysis input, processes it through the appropriate models and returns a structured JSON response.

Example response structure:

```json
{
  "nutrition": {
    "CP_percent": 4.85,
    "NDF_percent": 68.69,
    "ADF_percent": 37.55,
    "IVDMD_percent": 66.90
  },
  "moisture": {
    "value": 17.70
  },
  "spectral_analysis": {
    "anomalous": false
  },
  "quality_assessment": {
    "overall_status": "NEEDS_ATTENTION"
  }
}
```

The modular API architecture makes it possible to independently update the AI models without redesigning the mobile application.

---

# 🗂️ Project Structure

The project is organized into separate components for models, datasets, backend services and mobile development.

A simplified structure is:

```text
NIR-Feed-AI/
│
├── backend/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   └── schemas/
│
├── models/
│   ├── Nutrition/
│   ├── Moisture/
│   ├── Anomaly/
│   └── Vision/
│
├── datasets/
│   ├── nutrition/
│   ├── moisture/
│   └── vision/
│
├── flutter_app/
│   ├── lib/
│   ├── assets/
│   └── pubspec.yaml
│
├── notebooks/
│   ├── preprocessing/
│   ├── model_training/
│   └── evaluation/
│
├── requirements.txt
└── README.md
```

The exact structure may vary depending on the current development branch.

---

# 🧪 Dataset & Model Information

### Nutrition Dataset

The nutrition dataset contains approximately:

```text
Samples:              1112
Valid samples:        416
Spectral features:    1050
Wavelength range:     400–2498 nm
```

The nutritional models are designed to predict:

```text
CP
NDF
ADF
IVDMD
```

### Moisture Dataset

The moisture dataset contains:

```text
Samples:              150
Spectral features:    141
Wavelength range:     950–1650 nm
```

The moisture model is implemented as a machine-learning pipeline and is trained independently from the nutrition models.

---

# 🔄 End-to-End Workflow

The complete system can be represented as:

```text
             Feed Sample
                  │
                  ▼
          ┌───────────────┐
          │ Data Capture  │
          └───────┬───────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
   NIR Spectrum          Image
        │                   │
        ▼                   ▼
 Preprocessing        Image Processing
        │                   │
   ┌────┴─────┐             ▼
   │          │       Mould Detection
   ▼          ▼
Nutrition   Moisture
Models      Model
   │          │
   └────┬─────┘
        │
        ▼
Spectral Anomaly Detection
        │
        ▼
 Quality Assessment Engine
        │
        ▼
   FastAPI Backend
        │
        ▼
 Flutter Mobile App
        │
        ▼
 Farmer / User
```

---

# 🎯 Example Analysis

An example analysis produced by the prototype:

```text
Crude Protein (CP)       : 4.85 %
Neutral Detergent Fiber  : 68.69 %
Acid Detergent Fiber     : 37.55 %
IVDMD                    : 66.90 %
Moisture                 : 17.70 %

Spectral Anomaly        : Not Detected

Overall Status           : NEEDS_ATTENTION
```

These values demonstrate how the different model outputs can be combined into a single feed-quality assessment.

> These example values are demonstration outputs from the prototype and should not be interpreted as laboratory-certified measurements.

---

# 🔬 Machine Learning Approach

The project follows a modular machine-learning workflow:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Spectral Preprocessing
     ↓
Feature Preparation
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Serialization
     ↓
FastAPI Integration
     ↓
Application Deployment
```

The architecture allows individual models to be retrained or replaced without modifying the complete system.

---

# 🛠️ Technology Stack

| Component         | Technology                       |
| ----------------- | -------------------------------- |
| Programming       | Python                           |
| Machine Learning  | Scikit-learn / ML pipelines      |
| Data Processing   | NumPy, Pandas                    |
| Visualization     | Matplotlib                       |
| Backend           | FastAPI                          |
| API               | REST                             |
| Mobile App        | Flutter / Dart                   |
| Computer Vision   | Python-based CV pipeline         |
| Anomaly Detection | Isolation Forest + PCA           |
| Model Storage     | Pickle / serialized ML pipelines |
| Development       | Google Colab / Local Environment |

---

# 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/ankitgh27/NIR-Feed-AI.git
cd NIR-Feed-AI
```

### 2. Create a Python environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI server

```bash
uvicorn backend.main:app --reload
```

The API can then be accessed locally through the FastAPI development server.

---

# 📡 API Testing

The API can be tested using:

* Swagger UI
* Postman
* cURL
* Flutter application

FastAPI automatically provides interactive API documentation through Swagger UI.

Typical workflow:

```text
Client
  ↓
POST /analyze
  ↓
FastAPI
  ↓
ML Pipeline
  ↓
Prediction
  ↓
JSON Response
```

---

# 🌱 Real-World Use Cases

NIR Feed AI is intended to explore applications in:

* 🐄 Dairy farming
* 🌾 Cattle-feed quality monitoring
* 🏭 Feed manufacturing
* 🧪 Agricultural laboratories
* 🔬 Research institutions
* 🚜 Smart agriculture
* 📱 Digital livestock management

The system could potentially support faster preliminary feed screening before detailed laboratory analysis.

---

# ⚠️ Limitations

NIR Feed AI is currently a **research prototype** and should not be considered a certified feed-testing system.

Important limitations include:

### NIR Model Validation

NIR models can be sensitive to:

* Feed composition
* Geographic variation
* Instrument characteristics
* Sample preparation
* Spectral preprocessing
* Calibration datasets

Therefore, further validation using locally representative cattle-feed samples and calibrated NIR instruments is required.

### Moisture Model

The moisture model operates on a separate spectral range from the nutrition models and therefore requires compatible spectral input.

### Mould Detection

The current computer-vision component requires additional validation on real cattle-feed and silage datasets.

### Hardware

The complete system is designed around access to NIR spectral measurements. A standard mobile phone camera cannot directly replace an NIR spectrometer.

### Regulatory Validation

The prototype has not been validated or certified for regulatory, veterinary, commercial or laboratory-grade feed testing.

---

# 🔮 Future Scope

Future development could include:

### Hardware Integration

Integration with portable NIR spectroscopy devices could enable on-site feed analysis.

```text
Portable NIR Sensor
        ↓
   ESP32 / Edge Device
        ↓
      API
        ↓
    AI Models
        ↓
 Mobile Application
```

### Edge AI

Models could potentially be optimized for deployment directly on:

* Edge devices
* Smartphones
* Embedded systems
* Portable spectroscopy hardware

### Improved Computer Vision

Future versions could use larger cattle-feed and silage image datasets for more robust mould and contamination detection.

### Model Improvement

Potential improvements include:

* Larger datasets
* Local calibration
* Cross-validation
* External validation
* Model explainability
* Uncertainty estimation
* Domain adaptation

### Multilingual AI

The application can be expanded with additional Indian languages to improve accessibility for farmers.

---

# 📈 Future Vision

The long-term vision of NIR Feed AI is to create a **portable, AI-assisted feed-quality assessment platform** that combines spectroscopy, computer vision and machine learning.

```text
              PORTABLE NIR DEVICE
                       │
                       ▼
                 Spectral Data
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   Nutrition Analysis        Moisture Analysis
          │                         │
          └────────────┬────────────┘
                       ▼
              Anomaly Detection
                       │
                       ▼
              Computer Vision
                       │
                       ▼
              Quality Assessment
                       │
                       ▼
              Mobile Application
                       │
                       ▼
                  End User
```

The objective is not to replace laboratory testing, but to explore how AI can make **preliminary feed-quality screening faster, more accessible and easier to interpret**.

---

# 👨‍💻 Project Status

**Status: 🚧 Research Prototype / Proof of Concept**

Current components include:

* ✅ NIR nutrition prediction
* ✅ Moisture prediction
* ✅ Spectral anomaly detection
* ✅ Quality assessment logic
* ✅ FastAPI backend
* ✅ Flutter mobile application prototype
* ✅ API integration
* 🚧 Computer vision mould detection
* 🚧 Portable NIR hardware integration
* 🚧 Large-scale field validation
* 🚧 Production deployment

---

# 📜 Disclaimer

NIR Feed AI is an academic/research prototype developed for experimentation with spectroscopy, machine learning and smart-agriculture technologies.

Predictions generated by the prototype should **not be treated as laboratory-certified feed analysis or veterinary advice**.

Further calibration, validation and field testing are required before deployment in real-world commercial or regulatory environments.

---

# 🤝 Contributing

Contributions, suggestions and research collaborations are welcome.

If you would like to contribute:

```bash
git fork https://github.com/ankitgh27/NIR-Feed-AI
```

Create a feature branch:

```bash
git checkout -b feature/your-feature
```

Commit your changes:

```bash
git commit -m "Add your feature"
```

Push the branch:

```bash
git push origin feature/your-feature
```

Then open a Pull Request.

---

# 📬 Contact

**Ankit Ghosh**

GitHub:
https://github.com/ankitgh27

Project Repository:
https://github.com/ankitgh27/NIR-Feed-AI

---

## ⭐ If You Find This Project Interesting

Consider giving the repository a ⭐ on GitHub.

This project is an exploration of how **NIR spectroscopy + Machine Learning + Computer Vision + Mobile Technology** can be combined to build intelligent tools for livestock and smart agriculture.


**Spectral range:**

```text
950–1650 nm
