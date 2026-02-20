# 📰 TruthLens: Multilingual Fake News Detection using LSTM, SVC, and XGBoost

## 📌 Overview

**TruthLens** is a Multilingual Fake News Detection system designed to identify fake and real news articles across **Gujarati, Hindi, and English** languages. The project combines **deep learning (LSTM)** and **machine learning models (Support Vector Classifier and XGBoost)** to achieve high accuracy and robust performance on multilingual news datasets.

This system helps combat misinformation by automatically classifying news content as **Fake** or **Real**.

---

## 🎯 Objectives

* Detect fake news in **Gujarati, Hindi, and English**
* Compare performance of **Deep Learning and Machine Learning models**
* Build a **multilingual robust detection system**
* Provide a **scalable solution for real-world deployment**

---

## 🌐 Supported Languages

* 🇮🇳 Hindi News Dataset (NSD)
* 🇮🇳 Gujarati News Dataset (NSD)
* 🌍 English News Dataset

---

## 🧠 Models Used

The project uses **6 models in total**:

### 🔹 Machine Learning Models

1. **Support Vector Classifier (SVC) with TF-IDF**
2. **XGBoost Classifier with TF-IDF**
3. **Logistic Regression (optional baseline)**

### 🔹 Deep Learning Models

4. **LSTM for English**
5. **LSTM for Hindi**
6. **LSTM for Gujarati**

---

## 🏗️ System Architecture

```
News Input
   │
   ▼
Text Preprocessing
   │
   ▼
Language Detection
   │
   ├──► ML Models (SVC, XGBoost)
   │
   └──► DL Models (LSTM)
   │
   ▼
Prediction
   │
   ▼
Fake / Real Result
```

---

## 📂 Dataset Information

### Sources:

* NSD Hindi News Dataset
* NSD Gujarati News Dataset
* English Fake News Dataset (Kaggle / ISOT)

### Dataset Features:

| Feature  | Description                |
| -------- | -------------------------- |
| text     | News content               |
| label    | Fake or Real               |
| language | Hindi / Gujarati / English |

---

## ⚙️ Technologies Used

### Programming Language

* Python 3.10+

### Libraries

* TensorFlow / Keras
* Scikit-learn
* XGBoost
* Pandas
* NumPy
* Matplotlib
* Seaborn

### Frontend

* JavaScript
* HTML
* CSS

---

## 🔄 Project Workflow

### Step 1: Data Collection

* Collect multilingual datasets

### Step 2: Data Preprocessing

* Remove stopwords
* Tokenization
* Cleaning
* Padding (for LSTM)

### Step 3: Feature Extraction

* TF-IDF (for SVC and XGBoost)
* Tokenizer + Embedding (for LSTM)

### Step 4: Model Training

* Train SVC
* Train XGBoost
* Train LSTM

### Step 5: Prediction

* User inputs news
* System predicts Fake or Real

---

## 📊 Model Performance

| Model         | Accuracy |
| ------------- | -------- |
| SVC           | 94%      |
| XGBoost       | 96%      |
| LSTM English  | 97%      |
| LSTM Hindi    | 96%      |
| LSTM Gujarati | 95%      |

---

## 🖥️ User Interface Features

* Multilingual news input
* Real-time fake news detection
* Model comparison
* Live news testing

---

## 📁 Project Structure

```
TruthLens/
│
├── data/
│   ├── english.csv
│   ├── hindi.csv
│   └── gujarati.csv
│
├── models/
│   ├── svc_model.pkl
│   ├── xgboost_model.pkl
│   └── lstm_model.h5
│
├── frontend/
│
├── app.py
│
├── train.py
│
└── README.md
```

---

## 🚀 How to Run the Project

### Step 1: Clone Repository

```
git clone https://github.com/yourusername/truthlens
```

---

### Step 2: Install Requirements

```
pip install -r requirements.txt
```

---

### Step 3: Train Models

```
python train.py
```

---

### Step 4: Run Application

```
python app.py
```

---

## 💡 Key Features

✅ Multilingual Detection
✅ Deep Learning + Machine Learning
✅ High Accuracy
✅ Real-time Prediction
✅ Frontend Integration

---

## 📈 Future Improvements

* Add more languages
* Deploy on cloud
* Add transformer models (BERT)
* Create browser extension

---

## 👨‍💻 Author

**Shashi Kant**

Data Science | Machine Learning | AI Engineer

---

## 📜 License

This project is for **academic and research purposes only**.

---

## ⭐ Acknowledgement

* NSD Dataset
* Kaggle
* TensorFlow
* Scikit-learn

---

## 📬 Contact

For queries:

Email: [kantshashi3898@gmail.com](kantshashi3898@gmail.com)

---

# 🔥 TruthLens – Fighting Fake News with AI

