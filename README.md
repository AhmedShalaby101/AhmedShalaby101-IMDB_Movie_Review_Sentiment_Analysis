# IMDb Sentiment Analysis with Multinomial Naive Bayes

## 📌 Description

This project performs sentiment analysis on IMDb movie reviews using a **Multinomial Naive Bayes** classifier. The pipeline includes:

- Data loading and preprocessing (text cleaning, stemming, stopword removal)
- Feature extraction using **TF-IDF Vectorizer**
- Model training, evaluation, and visualization of learning curves

The goal is to classify reviews as **positive (1)** or **negative (0)** with high accuracy.

---

## 🚀 Features

### ✍️ Text Preprocessing

- Removes HTML tags, special characters, and converts text to lowercase
- Handles emojis and applies stemming (**PorterStemmer**)

### ⚖️ TF-IDF Vectorization

- Transforms text into numerical features

### 📊 Model Evaluation

- Cross-validation, learning curves, and performance metrics (accuracy, confusion matrix)

### 📆 Modular Workflow

```
Raw → Processed → Cleaned → Trained → Saved model
```

---

## ⚙️ Installation

### Clone the repository:

```bash
git clone https://github.com/your-username/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure `requirements.txt` includes:

```
cycler==0.12.1
joblib==1.5.1
matplotlib==3.10.3
numpy==2.3.0
pandas==2.3.0
scikit_learn==1.7.0
```

### Download NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## 📂 Dataset

**Source**: IMDb Reviews (raw text files)

**Structure:**

```
train/pos/
train/neg/
test/pos/
test/neg/
```

Combined into a single file: `preprocessed_data.csv`

**Preprocessing**:

- Shuffled and split into train/test sets (stratified)

---

## 📁 Project Structure

```
data/
├── raw/           # Original IMDb files
├── processed/     # preprocessed_data.csv
└── interim/       # cleaned_text.csv
models/            # Saved pipeline
reports/figures/   # Learning curve plot
scripts/           # .py files
```

---

## 🛠️ Usage

### 1. Data Preparation

Run the preprocessing scripts in order:

```bash
python 01_load_and_preprocess.py      # Converts raw text to CSV
python 02_clean_text.py               # Applies NLP cleaning
python 03_train_model.py              # Trains and evaluates the model
```

### 2. Model Training & Evaluation

The pipeline (`03_train_model.py`):

- Splits data (80% train, 20% test)
- Trains a `MultinomialNB` classifier with TF-IDF features
- Generates:
  - Learning curve plot: `reports/figures/learning_curve.png`
  - Saved model: `models/multinb_tfidf_pipeline.pkl`

**Example output:**

```
Model accuracy on train data = 0.92
Mean CV Accuracy: 0.89
Model accuracy on test data = 0.88
```

### 3. Load the Saved Model

```python
import joblib
model = joblib.load("../../models/multinb_tfidf_pipeline.pkl")
model.predict(["This movie was fantastic!"])  # Output: [1] (positive)
```

---

## 📊 Results

| Metric         | Score         |
| -------------- | ------------- |
| Train Accuracy | 0.8974%       |
| Test Accuracy  | 0.86%         |
| CV Accuracy    | 0.86% ± 0.004 |

### Confusion Matrix

|                | Predicted Neg | Predicted Pos |
| -------------- | ------------- | ------------- |
| **Actual Neg** | 43.49%        | 6.51%         |
| **Actual Pos** | 7.49%         | 42.51%        |

---

## 🔧 Future Improvements

- Experiment with deep learning (LSTM, BERT) for higher accuracy
- Add hyperparameter tuning (e.g., `alpha` for MultinomialNB)

---

