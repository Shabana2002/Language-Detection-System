# ------------------------------------------------------------
# HIGH-ACCURACY LANGUAGE DETECTION TRAINING SCRIPT
# ------------------------------------------------------------

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Paths to datasets
MAIN_DATA_PATH = r"C:\Users\hp\Downloads\language_detection_balanced_all.csv"
MALAYALAM_PATH = r"C:\Users\hp\Downloads\malayalam_augmented.csv"

# Load datasets
main_df = pd.read_csv(MAIN_DATA_PATH)
malayalam_df = pd.read_csv(MALAYALAM_PATH)

# Combine datasets
df = pd.concat([main_df, malayalam_df], ignore_index=True)

print("Combined dataset shape:", df.shape)
print("Languages in dataset:", df['language'].value_counts())


# ------------------------------------------------------------
# CLEAN & FILTER DATA (CRITICAL FOR ACCURACY)
# ------------------------------------------------------------

# Remove missing values
df = df.dropna(subset=["sentence", "language"])

# Convert text to lowercase (helps char n-grams)
df["sentence"] = df["sentence"].astype(str).str.lower()

# Remove extremely rare languages (noise reduction)
MIN_SAMPLES = 200
language_counts = df["language"].value_counts()
df = df[df["language"].isin(language_counts[language_counts >= MIN_SAMPLES].index)]

print("Languages kept:", df["language"].nunique())
print("Dataset size:", df.shape)

# ------------------------------------------------------------
# FEATURES & LABELS
# ------------------------------------------------------------
X = df["sentence"]
y = df["language"]

# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# TF-IDF VECTORIZATION (CHAR N-GRAMS) - NON-LATIN FRIENDLY
# ------------------------------------------------------------
vectorizer = TfidfVectorizer(
    analyzer='char',       # character-level
    ngram_range=(2,6),
    lowercase=False,       # DO NOT lowercase non-Latin scripts
    min_df=2,
    max_df=0.9,
    max_features=250_000,
    sublinear_tf=True,
    norm="l2"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ------------------------------------------------------------
# NAIVE BAYES (BASELINE)
# ------------------------------------------------------------
nb_model = MultinomialNB(alpha=0.01)
nb_model.fit(X_train_vec, y_train)

# ------------------------------------------------------------
# LINEAR SVM (BEST MODEL FOR LANGUAGE DETECTION)
# ------------------------------------------------------------
svm_model = LinearSVC(
    C=3.0,
    max_iter=5000,
    class_weight="balanced"
)

svm_model.fit(X_train_vec, y_train)

# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------
nb_preds = nb_model.predict(X_test_vec)
svm_preds = svm_model.predict(X_test_vec)

print("\nNaive Bayes Accuracy:", accuracy_score(y_test, nb_preds))
print("Linear SVM Accuracy:", accuracy_score(y_test, svm_preds))

print("\nLinear SVM Classification Report:\n")
print(classification_report(y_test, svm_preds, digits=4))

# ------------------------------------------------------------
# SAVE MODELS
# ------------------------------------------------------------
joblib.dump(nb_model, "models/nb_model.pkl")
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\n✅ Models trained and saved successfully!")




# CODE EXPLANATION -

#📥 Data Loading

# Imports required libraries for data processing and machine learning
# Loads the main balanced language dataset
# Loads an additional Malayalam dataset
# Combines both datasets into one
# Prints dataset size and language distribution

#🧹 Data Cleaning & Filtering

# Removes rows with missing sentence or language values
# Converts all text to lowercase for consistency
# Removes languages with fewer than 200 samples to reduce noise
# Prints remaining number of languages and dataset size

#🎯 Feature & Label Setup

# X → sentences (input text)
# y → language labels (output)

#🔀 Train–Test Split

# Splits data into 80% training and 20% testing
# Uses stratified sampling to keep language distribution balanced

#🔡 Text Vectorization (TF-IDF)

# Converts text into numeric features
# Uses character n-grams (2–6 characters)
# Works well for non-Latin languages
# Limits features to reduce memory usage

#🧠 Model Training

# Trains a Naive Bayes model as a baseline
# Trains a Linear SVM model for higher accuracy
# Uses balanced class weights to handle language imbalance

#📊 Model Evaluation

# Makes predictions on test data
# Prints accuracy for both models
# Prints a detailed classification repoMakes predictions on test data
# Prints accuracy for both models
# Prints a detailed classification report for the SVM

#💾 Saving Models

# Saves the trained Naive Bayes model
# Saves the trained Linear SVM model
# Saves the TF-IDF vectorizer
# Confirms successful training and saving

#✅ Purpose of This Script

# Trains high-accuracy language detection models
# Handles many languages including non-Latin scripts
# Produces ready-to-use models for deployment