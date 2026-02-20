TRUTHLENS – A multilingual Fake News Detection System

Project Overview:
TruthLens is a multilingual fake news detection system developed using both classical NLP and deep learning techniques.
It identifies whether a given news text is real or fake by analyzing linguistic patterns, word usage, and contextual semantics.
The project supports English, Hindi, and Gujarati languages.

The main objective is to design models that are both accurate and computationally efficient while being explainable and adaptable to multiple languages.
##############################################################################################
Objectives:

To preprocess multilingual textual datasets for machine learning applications.

To train and evaluate fake news detection models using both NLP and Deep Learning approaches.

To compare the performance of classical machine learning models and neural network-based models.

To develop a reproducible, modular, and language-independent fake news detection pipeline.

Techniques Used:

Classical NLP – TF-IDF Vectorization combined with Linear Support Vector Classifier (LinearSVC).

Deep Learning – Bidirectional LSTM model with pre-trained embeddings such as GloVe and fastText.

Data Balancing – Upsampling of minority classes to reduce bias.

Stratified Data Splitting – Ensures equal label distribution across training, validation, and testing sets.

Model Calibration – CalibratedClassifierCV used to obtain probability outputs from SVM.
##############################################################################################
Project Directory Structure:

TRUTHLENS
│
├── .venv → Python virtual environment
├── .vscode → VS Code configuration files
│
├── Data/ → All data-related folders
│ ├── embeddings/ → Pre-trained word embeddings
│ │ └── glove.6B.100d.txt → GloVe vectors for English
│ ├── Preprocessed/ → Cleaned datasets for model input
│ └── RAW_Data/ → Original unprocessed datasets
│
├── models/ → All trained models and outputs
│ ├── light/ → Classical (TF-IDF + SVM) models
│ │ ├── english_light_model/
│ │ ├── gujarati_light_model/
│ │ └── hindi_light_generalized/
│ └── pro/ → Deep Learning (LSTM) models
│ └── english_pro_lstm/
│ ├── best_model.h5
│ ├── english_pro_lstm.keras
│ ├── info.json
│ ├── meta.joblib
│ └── tokenizer.json
│
├── notebooks/ → Jupyter Notebooks for experimentation
│ ├── models/
│ ├── English_Light.ipynb
│ ├── Gujarati_Light.ipynb
│ └── Hindi_Light.ipynb
│
├── app_light.py → Application script for Light Models
├── train_pro_english_lstm.py → Training script for English LSTM model
└── train_pro_hindi_lstm.py → Training script for Hindi LSTM model
#############################################################################################################
Flow of the Project: 

Data Collection:
Raw news data is stored in the RAW_Data folder.
Each dataset corresponds to a specific language such as English, Hindi, or Gujarati.

Data Preprocessing:
Raw data is cleaned, tokenized, normalized, and saved in the Preprocessed folder.
The cleaned data is used for both Light and Pro model training.

Embeddings Setup:
Pre-trained word embeddings (GloVe for English, fastText for Indian languages) are placed inside the embeddings folder.
These embeddings are used in the LSTM models to represent words numerically.

Model Training:
Light models use classical NLP methods (TF-IDF + LinearSVC).
Pro models use deep learning methods (Bidirectional LSTM).

Light model scripts are located under the notebooks or app_light.py file.

Pro model training scripts are located as train_pro_english_lstm.py and train_pro_hindi_lstm.py.

Model Outputs:
After training, all models and related metadata are saved inside the models directory.
Each model folder contains:

The trained model file (.keras or .joblib)

Tokenizer or vectorizer

Information files (info.json, meta.joblib) storing hyperparameters and thresholds

Evaluation and Metrics:
Each script evaluates the trained model on validation and test data.
Metrics such as Accuracy, F1-Score, Precision, Recall, and ROC-AUC are saved in the respective model folder.

Inference and Deployment:
The trained models can be reloaded using the saved artifacts.
The app_light.py script is designed for real-time inference using the light model.

Flow Chart (Text Description):

Data (RAW_Data)
↓
Preprocessing → Cleaned CSVs (Preprocessed folder)
↓
Embeddings Loaded (GloVe / fastText)
↓
Model Training
├── Light Models (TF-IDF + LinearSVC)
└── Pro Models (LSTM + Embeddings)
↓
Evaluation on Validation and Test Sets
↓
Metrics + Model + Tokenizer Saved (models folder)
↓
Reload Model → Predict → Classify News as Real or Fake


Flow of Operations:
RAW_Data
→ Preprocessed
→ Embeddings Loaded
→ Feature Extraction / Tokenization
→ Model Training (Light or Pro)
→ Evaluation on Validation and Test Data
→ Model and Metrics Saved
→ Real-time Prediction

Evaluation Metrics:
Accuracy – Overall correctness of predictions.
Precision – Percentage of correctly predicted positive cases.
Recall – Ability to identify all relevant instances.
F1-Score – Harmonic mean of precision and recall.
ROC-AUC – Ability of the model to discriminate between classes.

Results Summary (Typical):
English Light Model: Accuracy 92%, F1 0.92
English LSTM Model: Accuracy 93%, F1 0.92
Hindi LSTM Model: Accuracy 96%, F1 0.96
Gujarati LSTM Model: Accuracy 95%, F1 0.95

Technologies and Libraries:
Python 3.10+
TensorFlow
scikit-learn
pandas
numpy
gensim
joblib

Future Enhancements:

Integrate transformer-based models such as BERT or mBERT for better contextual understanding.

Build a Flask or FastAPI web interface for real-time prediction.

Add explainability modules using LIME or SHAP to interpret model decisions.

Extend model support to other Indian languages.

#######################################################################################################

Developer Information:
Project Name: TruthLens
Developer: [Sanyam Kanwar]
Course: MCA – Minor Project (3rd Semester)
Guide: [Mr. Kishan Ahuja]
Institution: [Chandigarh University]
Year: 2025

