# 🛰️ TruthLens – Multilingual Fake News Detection System

### 🔍 **TruthLens** is a multilingual Fake News Detection System built using both **Classical NLP** and **Deep Learning** techniques.  
It identifies whether a given news article is **real** or **fake** by analyzing **linguistic patterns**, **word usage**, and **contextual semantics** — supporting **English**, **Hindi**, and **Gujarati** languages.

---

## 🚀 Project Overview

TruthLens aims to combat misinformation across multiple languages using explainable and adaptable AI models.  
It focuses on building **accurate**, **efficient**, and **interpretable** models that generalize across linguistic boundaries.

---

## 🎯 Objectives

- ✅ Preprocess multilingual textual datasets for ML applications.  
- ✅ Train and evaluate Fake News Detection models using both **NLP** and **Deep Learning** approaches.  
- ✅ Compare performance between **classical** ML models and **neural network-based** models.  
- ✅ Develop a **reproducible**, **modular**, and **language-independent** detection pipeline.

---

## 🧠 Techniques Used

| Technique | Description |
|------------|-------------|
| **Classical NLP** | TF-IDF Vectorization + Linear Support Vector Classifier (LinearSVC) |
| **Deep Learning** | Bidirectional LSTM with pre-trained embeddings (GloVe / fastText) |
| **Data Balancing** | Upsampling of minority classes to mitigate class imbalance |
| **Stratified Data Splitting** | Maintains equal label distribution across train/val/test |
| **Model Calibration** | CalibratedClassifierCV used for probability outputs from SVM |

---

## 📂 Project Directory Structure

TRUTHLENS/
├── .venv/ # Python virtual environment
├── .vscode/ # VS Code configuration files
│
├── Data/ # All data-related folders
│ ├── RAW_Data/ # Original unprocessed datasets
│ └── Preprocessed/ # Cleaned datasets for model input
│
├── embeddings/ # Pre-trained word embeddings
│ └── glove.6B.100d.txt # GloVe vectors for English
│
├── models/ # All trained models and outputs
│ ├── light/ # Classical (TF-IDF + SVM) models
│ │ ├── english_light_model/
│ │ ├── gujarati_light_model/
│ │ └── hindi_light_generalized/
│ └── pro/ # Deep Learning (LSTM) models
│ └── english_pro_lstm/
│ ├── best_model.h5
│ ├── english_pro_lstm.keras
│ ├── info.json
│ ├── meta.joblib
│ └── tokenizer.json
│
├── notebooks/ # Jupyter notebooks for experimentation
│ ├── English_Light.ipynb
│ ├── Gujarati_Light.ipynb
│ └── Hindi_Light.ipynb
│
├── app_light.py # Inference app for classical models
├── train_pro_english_lstm.py # Training script for English LSTM
└── train_pro_hindi_lstm.py # Training script for Hindi LSTM


---

## 🧩 Project Flow

### 1️⃣ **Data Collection**
- Raw news data for English, Hindi, and Gujarati stored in `RAW_Data/`.

### 2️⃣ **Data Preprocessing**
- Cleaning, tokenization, normalization → saved in `Preprocessed/`.

### 3️⃣ **Embedding Setup**
- Pre-trained embeddings (GloVe / fastText) are loaded for better semantic understanding.

### 4️⃣ **Model Training**
- **Light Models:** TF-IDF + LinearSVC  
- **Pro Models:** Bidirectional LSTM using pre-trained embeddings.

### 5️⃣ **Model Outputs**
- Trained model + tokenizer/vectorizer + metadata stored in `models/`.

### 6️⃣ **Evaluation & Metrics**
- Evaluated on validation and test data using Accuracy, Precision, Recall, F1, and ROC-AUC.

### 7️⃣ **Inference & Deployment**
- Models reloaded for real-time prediction via `app_light.py`.

---

## 📈 Textual Flow Chart

RAW_Data
↓
Preprocessing → Cleaned CSVs (Preprocessed folder)
↓
Embeddings Loaded (GloVe / fastText)
↓
Model Training
├── Light Models (TF-IDF + LinearSVC)
└── Pro Models (LSTM + Embeddings)
↓
Evaluation on Validation & Test Sets
↓
Model + Tokenizer + Metrics Saved (models/)
↓
Reload Model → Predict → Classify News as Real or Fake


---

## 📊 Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | Percentage of correctly predicted positive cases |
| **Recall** | Ability to identify all relevant instances |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Measures ability to discriminate between classes |

---

## 🧾 Results Summary (Typical)

| Model | Accuracy | F1-Score |
|--------|-----------|----------|
| English Light Model | 92% | 0.92 |
| English LSTM Model | 93% | 0.92 |
| Hindi LSTM Model | 96% | 0.96 |
| Gujarati LSTM Model | 95% | 0.95 |

---

## 🛠️ Technologies and Libraries

- **Programming Language:** Python 3.10+
- **Libraries:**
  - TensorFlow / Keras  
  - scikit-learn  
  - pandas  
  - numpy  
  - gensim  
  - joblib  

---

## ⚙️ Installation and Setup


---

## 📊 Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | Percentage of correctly predicted positive cases |
| **Recall** | Ability to identify all relevant instances |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Measures ability to discriminate between classes |

---

## 🧾 Results Summary (Typical)

| Model | Accuracy | F1-Score |
|--------|-----------|----------|
| English Light Model | 92% | 0.92 |
| English LSTM Model | 93% | 0.92 |
| Hindi LSTM Model | 96% | 0.96 |
| Gujarati LSTM Model | 95% | 0.95 |

---

## 🛠️ Technologies and Libraries

- **Programming Language:** Python 3.10+
- **Libraries:**
  - TensorFlow / Keras  
  - scikit-learn  
  - pandas  
  - numpy  
  - gensim  
  - joblib  

---

## ⚙️ Installation and Setup


---

## 📊 Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | Percentage of correctly predicted positive cases |
| **Recall** | Ability to identify all relevant instances |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Measures ability to discriminate between classes |

---

## 🧾 Results Summary (Typical)

| Model | Accuracy | F1-Score |
|--------|-----------|----------|
| English Light Model | 92% | 0.92 |
| English LSTM Model | 93% | 0.92 |
| Hindi LSTM Model | 96% | 0.96 |
| Gujarati LSTM Model | 95% | 0.95 |

---

## 🛠️ Technologies and Libraries

- **Programming Language:** Python 3.10+
- **Libraries:**
  - TensorFlow / Keras  
  - scikit-learn  
  - pandas  
  - numpy  
  - gensim  
  - joblib  

---

## ⚙️ Installation and Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/TruthLens.git
cd TruthLens

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

