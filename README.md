# Language Detector

An intelligent text classification system that distinguishes between English and Dutch using character-level statistical features and linear models.

## Overview

This project develops machine learning classifiers to identify whether a text passage is written in English or Dutch. Using carefully engineered character-level features, the models achieve near-perfect classification accuracy on real-world text samples.

## Features

- **Character-Level Feature Engineering**: 48 statistical features including:
  - Letter frequency distributions (26 features)
  - Bigram frequencies (20 features)
  - Average word length
  - Vowel ratio analysis
- **Multi-Model Comparison**: Perceptron, Linear SVM, and Logistic Regression
- **Real-World Testing**: Evaluated on sentences from news websites and Wikipedia
- **Interpretability**: Feature importance analysis showing key linguistic differences

## Dataset

- **Training**: 140 sentences from Universal Declaration of Human Rights
  - 69 English sentences
  - 71 Dutch sentences
- **Development**: 40 collected sentences (news, Wikipedia)
- **Test**: 40 held-out sentences from diverse sources

## Feature Engineering

The system extracts linguistic patterns that distinguish the two languages:

**English Indicators:**
- High frequency of 'th', 'he', 'ed' bigrams
- Specific letter distributions

**Dutch Indicators:**
- Common 'ij', 'aa', 'de', 'en' bigrams
- Longer average word length
- Distinct vowel patterns

## Models & Results

| Model | Best Hyperparameters | Test Accuracy |
|-------|---------------------|---------------|
| Perceptron | max_iterations=10 | 97.50% |
| Linear SVC | C=1.0 | 100.00% |
| Logistic Regression | C=0.001, penalty=none | 100.00% |

## Key Insights

- Character-level features are highly effective for language identification
- Simple linear models achieve perfect separation with proper feature engineering
- Small training sets (140 samples) are sufficient with well-designed features
- The 'ij' bigram and specific letter frequencies are strong Dutch indicators

## Technologies

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Usage

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Extract features from text
features = extract_features("This is an example sentence")

# Load trained model
model = LogisticRegression(C=0.001, penalty='none', max_iter=10000)
model.fit(X_train, y_train)

# Predict language (0=English, 1=Dutch)
prediction = model.predict(features.reshape(1, -1))
language = "English" if prediction[0] == 0 else "Dutch"
