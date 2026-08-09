# Skin Cancer Detection Using Deep Learning

An AI-powered web application that classifies dermoscopic skin lesion images into 7 disease categories using deep learning, providing confidence scores and risk level assessment.

---

## 2. Project Overview

**What it does:** Users upload or capture a skin lesion image. The system preprocesses it and runs it through a trained deep learning model, returning the predicted disease class, confidence percentage, and risk level (High / Medium / Low).

**Problem it addresses:** Manual diagnosis of skin lesions requires specialist dermatologists who are not universally available, especially in rural and resource-limited regions. The process is time-consuming, subjective, and expensive.

**Purpose:** To build a decision-support tool that assists healthcare professionals in preliminary skin cancer screening — reducing diagnostic delays and improving early detection rates.

---

## 3. Key Features

- Upload skin lesion image via file browser
- Real-time camera capture using browser MediaDevices API
- Automated hair removal preprocessing before prediction
- CLAHE contrast enhancement for better lesion visibility
- 7-class skin lesion classification with confidence scores
- Risk level assessment (High / Medium / Low) per prediction
- Probability distribution display for all 7 classes
- REST API backend for easy integration

---

## 4. Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.10, JavaScript, HTML5, CSS3 |
| Deep Learning | TensorFlow 2.19, Keras |
| Model | ResNet50V2 (pretrained on ImageNet) |
| Image Processing | OpenCV (cv2) 4.9, Pillow 10.0 |
| Data Processing | NumPy 1.26, Pandas |
| ML Utilities | scikit-learn |
| Backend Framework | Flask 3.0, Flask-CORS |
| Frontend | JavaScript, HTML5, CSS3 |
| Training Platform | Kaggle (Tesla P100 GPU, 16GB VRAM) |
| Database | None — model loaded into memory at runtime |

---

## 5. System Architecture / Project Workflow

```
User (Browser)
    |
    | Upload image / Camera capture
    ↓
Frontend (HTML / CSS / JS)
    |
    | POST /predict  (multipart/form-data)
    ↓
Flask Backend (app.py — localhost:5000)
    |
    ├── Resize image to 224×224
    ├── Hair Removal (blackhat filter + inpaint)
    ├── CLAHE contrast enhancement
    └── Normalize pixels → [-1, 1]
    |
    ↓
ResNet50V2 Model (skin_cancer_model.keras)
    |
    | 7 probability values (softmax output)
    ↓
Flask Backend
    |
    | Build JSON response (class, confidence, risk, description)
    ↓
Frontend (HTML / CSS / JS)
    |
    | Display results dashboard
    ↓
User sees: Disease name | Confidence % | Risk Level | All 7 probabilities
```

---

## 6. Project Structure

```
skin_cancer_app/
│
├── backend/
│   ├── app.py                      ← Flask API server (main entry point)
│   ├── requirements.txt            ← Python dependencies
│   ├── venv/                       ← Virtual environment (not committed)
│   └── model/
│       ├── skin_cancer_model.keras ← Trained model weights (download separately)
│       ├── class_info.json         ← Class names, codes, risk levels
│       └── model_config.json       ← Input shape, normalization config
│
├── frontend/
│   ├── templates/
│   │   └── index.html              ← Main UI page
│   └── static/
│       ├── css/styles.css          ← Styling
│       └── js/script.js            ← Upload, camera, fetch logic
│
└── README.md
```

---

## 7. Dataset

**Name:** HAM10000 (Human Against Machine with 10000 Training Images)

**Source:** Published by Tschandl et al. (2018) — *Scientific Data, 5, 180161*
DOI: https://doi.org/10.1038/sdata.2018.161

**Download:** https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection

**Total Images:** 10,015 dermoscopic images

**Classes:**

| Code | Disease | Count | % | Risk |
|---|---|---|---|---|
| NV | Melanocytic Nevi | 6,705 | 66.95% | Low |
| MEL | Melanoma | 1,113 | 11.11% | High |
| BKL | Benign Keratosis-like Lesions | 1,099 | 10.97% | Low |
| BCC | Basal Cell Carcinoma | 514 | 5.13% | High |
| AKIEC | Actinic Keratoses | 327 | 3.26% | High |
| VASC | Vascular Lesions | 142 | 1.42% | Medium |
| DF | Dermatofibroma | 115 | 1.15% | Low |

**Dataset Organization after download:**
```
HAM10000_images_part_1/    ← 5,000 images (.jpg)
HAM10000_images_part_2/    ← 5,015 images (.jpg)
HAM10000_metadata.csv      ← image_id, dx, age, sex, localization
```

---

## 8. Machine Learning / Technical Methodology

**Model:** ResNet50V2

**Why ResNet50V2:**
- No built-in preprocessing layer — allows explicit normalization control
- Pretrained on 1.2M ImageNet images — strong general feature extraction
- Residual (skip) connections solve vanishing gradient in deep networks
- Better gradient flow than original ResNet via pre-activation design

**Training Approach — Two Phase:**

| | Phase 1 | Phase 2 |
|---|---|---|
| Backbone | Frozen | Top 60% unfrozen |
| Epochs | 30 | 94 (early stopping) |
| Learning Rate | 1e-3 | 1e-5 |
| Optimizer | Adam | Adam + weight_decay=1e-4 |
| Goal | Train classification head | Fine-tune domain features |

**Classification Head added on top of ResNet50V2:**
```
GlobalAveragePooling2D
→ BatchNorm → Dropout(0.6) → Dense(512, ReLU, L2=2e-4)
→ BatchNorm → Dropout(0.5) → Dense(256, ReLU, L2=2e-4)
→ BatchNorm → Dropout(0.4)
→ Dense(7, Softmax)
```

**Preprocessing (applied identically during training and inference):**

| Step | Method |
|---|---|
| Resize | 224×224 pixels |
| Hair Removal | Morphological blackhat filter (9×9 kernel) + cv2.inpaint |
| CLAHE | L channel of LAB colorspace, clipLimit=2.0, tileGridSize=(8,8) |
| Normalize | (pixel / 127.5) - 1.0 → range [-1, +1] |

**Class Imbalance Handling:**
- Real image oversampling → minimum 1,000–2,000 samples per class
- Class weights (sklearn balanced) applied during training

**Data Augmentation (tf.data pipeline):**
- Random horizontal and vertical flips
- Random 90° rotations
- Random brightness (±0.15) and contrast (0.85–1.15)

**Key Parameters:**

| Parameter | Value |
|---|---|
| Input size | 224 × 224 × 3 |
| Batch size | 32 |
| Dropout | 0.6 / 0.5 / 0.4 |
| L2 regularization | 2e-4 |
| EarlyStopping patience | 15 epochs |

---

## 9. Installation / Setup

**Requirements:**
- Python 3.10 → https://www.python.org/downloads/
- pip (bundled with Python)
- ~2 GB free disk space

**Clone the repository:**
```bash
git clone https://github.com/your-username/skin_cancer_app.git
cd skin_cancer_app
```

**Create and activate virtual environment:**
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 10. Configuration

**Model files required** — download from your Kaggle notebook Output tab after training:

| File | Location | Purpose |
|---|---|---|
| `skin_cancer_model.keras` | `backend/model/` | Trained model weights |
| `class_info.json` | `backend/model/` | Class names and risk levels |
| `model_config.json` | `backend/model/` | Input shape configuration |

**No environment variables required** — all paths are relative within the project folder.

> If the model files are missing, the app runs in Demo Mode with random predictions and shows a warning banner.

---

## 11. How to Run

**Step 1 — Start the backend:**
```bash
cd backend
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

python app.py
```

Expected output:
```
Model loaded. Input shape: (None, 224, 224, 3)
Server running at http://localhost:5000
```

**Step 2 — Open the application:**
```
http://localhost:5000
```

**Verify the API is working:**
```
http://localhost:5000/health
```
Expected: `{ "status": "ok", "model_loaded": true }`

> Both terminal (backend) and browser must be open simultaneously. The backend must be running before using the frontend.

---

## 12. Results / Performance

**Final Model (v6):**

| Metric | Value |
|---|---|
| Test Accuracy | **88.54%** |
| Validation Accuracy | **87.85%** |
| Macro Avg F1-Score | **0.8865** |
| Weighted Avg F1-Score | **0.8878** |

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Melanocytic Nevi | 0.8889 | 0.9730 | 0.9290 |
| Melanoma | 0.9157 | 0.9157 | 0.9157 |
| Benign Keratosis | 0.7566 | 0.8099 | 0.7823 |
| Basal Cell Carcinoma | 0.9659 | 1.0000 | 0.9827 |
| Actinic Keratoses | 0.9357 | 0.8723 | 0.9029 |
| Vascular Lesions | 0.6544 | 0.7355 | 0.6926 |
| Dermatofibroma | 1.0000 | 1.0000 | 1.0000 |

**Overfitting Analysis:**
- Train accuracy (epoch 94): 98.86%
- Best validation accuracy (epoch 79): 88.62%
- Train-validation gap: 10.24% — mild overfitting
- Test accuracy (88.54%) close to validation (87.85%) — model generalizes well to unseen data

---

## 13. Limitations

**Dataset limitations:**
- Trained exclusively on HAM10000 — may not generalize to populations or skin tones not represented in the dataset
- Vascular lesions class has only 142 original images — insufficient real diversity even after oversampling

**Model limitations:**
- No "unknown" or "out-of-distribution" detection — always outputs a prediction even for non-lesion images
- Performance on standard smartphone camera photos is unknown — model trained only on dermoscopic images
- Vascular lesions F1-score of 0.69 remains the weakest class despite boosted oversampling

**Technical limitations:**
- Requires Flask server running locally — not accessible without a running backend
- No GPU acceleration during inference by default — prediction takes 0.3–1 second on CPU
- Model file is ~90MB — not suitable for on-device mobile inference without TFLite conversion

**Clinical limitations:**
- Not clinically validated against real patient outcomes
- Not a substitute for professional dermatological diagnosis
- Intended only as a preliminary screening and decision-support tool


