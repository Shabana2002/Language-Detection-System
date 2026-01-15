# ------------------------------------------------------------
# 🌍 LANGUAGE DETECTION SYSTEM - STREAMLIT (GLOBAL TECH UI)
# ------------------------------------------------------------

import streamlit as st
import joblib
from PyPDF2 import PdfReader
import time

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Language Detection System",
    layout="centered",
    page_icon="🌐"
)

# -----------------------------
# GLOBAL TECH CSS
# -----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
}

.hero {
    background: linear-gradient(90deg, #3F51B5, #00BCD4);
    padding: 30px;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
}

.card {
    background: #111827;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0px 0px 15px rgba(0, 188, 212, 0.15);
}

.badge {
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: bold;
    color: white;
    display: inline-block;
}

.lang-en { background: #3F51B5; }
.lang-fr { background: #2196F3; }
.lang-es { background: #4CAF50; }
.lang-ar { background: #FF5722; }
.lang-de { background: #9C27B0; }
.lang-it { background: #009688; }
.lang-pt { background: #795548; }
.lang-hi { background: #E91E63; }
.lang-ur { background: #607D8B; }

hr {
    border: 1px solid #263238;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HERO SECTION
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🌍 Detect Any Language Instantly</h1>
    <p>Upload text or documents and identify languages line-by-line using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    nb_model = joblib.load("models/nb_model.pkl")
    svm_model = joblib.load("models/svm_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return nb_model, svm_model, vectorizer

nb_model, svm_model, vectorizer = load_models()

# -----------------------------
# SIDEBAR (IMPROVED)
# -----------------------------
st.sidebar.header("⚙️ Model Selection")

model_choice = st.sidebar.radio(
    "Choose a model:",
    ("Multinomial Naive Bayes", "Linear SVM")
)

st.sidebar.markdown("---")

if model_choice == "Multinomial Naive Bayes":
    st.sidebar.info(
        "🧮 **Naive Bayes**\n\n"
        "- Fast predictions\n"
        "- Best for short text\n"
        "- Lightweight model"
    )
else:
    st.sidebar.info(
        "⚡ **Linear SVM**\n\n"
        "- Higher accuracy\n"
        "- Better for long text\n"
        "- More robust predictions"
    )

st.sidebar.markdown("---")
st.sidebar.caption("Powered by TF-IDF + ML")

# -----------------------------
# LANGUAGE FLAGS & CLASSES
# -----------------------------
LANG_MAP = {
    "English": ("🇬🇧", "lang-en"),
    "French": ("🇫🇷", "lang-fr"),
    "Spanish": ("🇪🇸", "lang-es"),
    "Arabic": ("🇸🇦", "lang-ar"),
    "German": ("🇩🇪", "lang-de"),
    "Italian": ("🇮🇹", "lang-it"),
    "Portuguese": ("🇵🇹", "lang-pt"),
    "Hindi": ("🇮🇳", "lang-hi"),
    "Urdu": ("🇵🇰", "lang-ur"),
}

# -----------------------------
# TEXT INPUT CARD
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📝 Enter Text")
text_input = st.text_area(
    "Type or paste text here (use multiple lines for multiple languages):",
    height=160
)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# FILE UPLOADER CARD
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📁 Upload File")
uploaded_file = st.file_uploader("Upload a TXT or PDF file", type=["txt", "pdf"])
st.markdown("</div>", unsafe_allow_html=True)

file_lines = []

if uploaded_file:
    try:
        if uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")
        else:
            pdf = PdfReader(uploaded_file)
            content = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        file_lines = [line.strip() for line in content.split("\n") if line.strip()]
    except Exception as e:
        st.error(f"Error reading file: {e}")

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_language(text, model):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

# -----------------------------
# ACTION BUTTONS
# -----------------------------
col1, col2 = st.columns(2)

detect = col1.button("🔍 Detect Language")
clear = col2.button("🧹 Clear")

if clear:
    st.rerun()

# -----------------------------
# DETECTION LOGIC
# -----------------------------
if detect:
    model = nb_model if model_choice == "Multinomial Naive Bayes" else svm_model

    lines = []
    source = ""

    if text_input.strip():
        lines = [line.strip() for line in text_input.split("\n") if line.strip()]
        source = "Text Input"
    elif file_lines:
        lines = file_lines
        source = "Uploaded File"
    else:
        st.warning("Please enter text or upload a file.")
        st.stop()

    with st.spinner("🔄 Detecting languages..."):
        time.sleep(0.6)
        results = [(line, predict_language(line, model)) for line in lines]

    # -----------------------------
    # RESULTS CARD
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(f"📊 Detection Results ({source})")
    st.success(f"✅ {len(results)} lines detected successfully")

    for text, lang in results:
        flag, css = LANG_MAP.get(lang, ("🌐", "lang-en"))
        st.markdown(
            f"**{text}**  \n"
            f"<span class='badge {css}'>{flag} {lang}</span>",
            unsafe_allow_html=True
        )
        st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)



# CODE EXPLANATION -


#🔧 Setup & Configuration

# Imports required libraries (Streamlit, ML models, PDF reader)
# Sets page title, layout, and icon

#🎨 UI Styling (Global Tech Theme)

# Applies custom CSS for dark gradient background
# Creates styled cards, badges, and hero section
# Adds color-coded language labels

#🦸 Hero Section

# Displays the main title and short description at the top

#🧠 Model Loading

# Loads Naive Bayes, Linear SVM, and TF-IDF vectorizer
# Caches models for faster performance

#🧭 Sidebar

# Allows users to select the ML model
# Displays model strengths and usage tips

#🌍 Language Mapping

# Maps each language to a flag emoji and color style

#📝 Text Input

# Allows users to enter multiple lines of text
# Each line can be a different language

#📁 File Upload

# Accepts TXT and PDF files
# Extracts and splits text into lines

#🤖 Language Prediction

# Converts text into TF-IDF features
# Predicts language using the selected model

#🔘 Action Buttons

# Detect Language → starts prediction
# Clear → refreshes the app

#⏳ Detection Process

# Shows a spinner while predicting
# Processes text input or uploaded file

#📊 Results Display

# Shows detected language for each line
# Displays flag and color-coded badge
# Shows total number of detected lines

#🎯 Purpose

# Provides an interactive UI for language detection
# Supports multiple languages and input formats
# Designed for clarity, usability, and visual appeal