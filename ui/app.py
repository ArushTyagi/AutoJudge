import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack

from features.feature_engineering import (
    preprocess_text,
    extract_numeric_features
)



# LOAD MODELS & PREPROCESSORS

@st.cache_resource
def load_models():
    best_regressor = joblib.load("models/best_regressor.pkl")
    best_classifier = joblib.load("models/best_classifier.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")

    clf_easy = joblib.load("models/ordinal_easy.pkl")
    clf_hard = joblib.load("models/ordinal_hard.pkl")

    word_tfidf = joblib.load("models/word_tfidf.pkl")
    char_tfidf = joblib.load("models/char_tfidf.pkl")
    scaler = joblib.load("models/scaler.pkl")

    return (
        best_regressor,
        best_classifier,
        label_encoder,
        clf_easy,
        clf_hard,
        word_tfidf,
        char_tfidf,
        scaler
    )


(
    regressor,
    classifier,
    label_encoder,
    clf_easy,
    clf_hard,
    word_tfidf,
    char_tfidf,
    scaler
) = load_models()



# FEATURE BUILDER (INFERENCE)

def build_features_for_ui(text):
    text = preprocess_text(text)

    # Text features
    X_word = word_tfidf.transform([text])
    X_char = char_tfidf.transform([text])

    # Numeric features
    num_vec = extract_numeric_features(text)
    num_features = np.array([num_vec], dtype=np.float64)

    # 🔐 SAFETY CHECK (CRITICAL)
    expected = scaler.n_features_in_
    actual = num_features.shape[1]

    if actual != expected:
        raise ValueError(
            f"Numeric feature mismatch: expected {expected}, got {actual}. "
            "UI and training feature pipelines are inconsistent."
        )

    X_num = scaler.transform(num_features)

    return hstack([X_word, X_char, X_num]).tocsr()




# STREAMLIT UI

st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("🧠 AutoJudge")
st.write("Predict programming problem difficulty using AI")

st.markdown("---")

title = st.text_input("📌 Problem Title")

description = st.text_area(
    "📝 Problem Description",
    height=180
)

input_desc = st.text_area(
    "📥 Input Description",
    height=120
)

output_desc = st.text_area(
    "📤 Output Description",
    height=120
)

st.markdown("---")

if st.button("🔍 Predict Difficulty"):
    if not description.strip():
        st.warning("Please enter the problem description.")
    else:
        combined_text = " ".join([
            title,
            description,
            input_desc,
            output_desc
        ])

        X = build_features_for_ui(combined_text)

        # -------------------------------
        # REGRESSION
        # -------------------------------
        score = regressor.predict(X)[0]

        # -------------------------------
        # ORDINAL CLASSIFICATION
        # -------------------------------
        easy_prob = clf_easy.predict_proba(X)[0][1]
        hard_prob = clf_hard.predict_proba(X)[0][1]

        if easy_prob >= 0.6:
            difficulty = "Easy"
        elif hard_prob >= 0.4:
            difficulty = "Hard"
        else:
            difficulty = "Medium"

        # -------------------------------
        # DISPLAY RESULTS
        # -------------------------------
        st.success("✅ Prediction Complete")

        st.subheader("📊 Results")
        st.write(f"**Difficulty Class:** `{difficulty}`")
        st.write(f"**Difficulty Score:** `{score:.2f}`")

       
