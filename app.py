import streamlit as st
import numpy as np
import joblib
import time

st.set_page_config(
    page_title="AI Disease Predictor",
    page_icon="🩺",
    layout="wide"
)
st.markdown("""
<style>

/* =========================
   MAIN BACKGROUND
========================= */
.stApp {
    background: linear-gradient(
        135deg,
        #0B1120 0%,
        #172554 50%,
        #1E293B 100%
    );
}

/* =========================
   TITLE
========================= */
.main-title {
    font-size: 60px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(
        90deg,
        #00E5FF,
        #00FF99
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* =========================
   SUBTITLE
========================= */
.subtitle {
    text-align: center;
    color: #CBD5E1;
    font-size: 20px;
    margin-bottom: 30px;
}

/* =========================
   GLASS CARD
========================= */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    border-radius: 25px;
    padding: 30px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* =========================
   TEXT COLORS
========================= */
h1,h2,h3,h4,h5,h6,
p,label,span {
    color: #F8FAFC !important;
}

/* =========================
   INPUT BOXES
========================= */
.stTextInput input,
.stNumberInput input,
textarea {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}

/* =========================
   MULTISELECT BOX
========================= */
.stMultiSelect [data-baseweb="select"] {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
}

/* Selected symptom tags */
.stMultiSelect span {
    color: white !important;
}

/* =========================
   DROPDOWN FIX
========================= */

/* Dropdown background */
div[role="listbox"] {
    background: white !important;
}

/* Symptom names */
div[role="option"] {
    color: black !important;
    background: white !important;
}

/* All text inside options */
div[role="option"] * {
    color: black !important;
}

/* Select All */
div[role="listbox"] * {
    color: black !important;
}

/* Hover effect */
div[role="option"]:hover {
    background: #E5E7EB !important;
    color: black !important;
}

/* Search text */
.stMultiSelect input {
    color: black !important;
}

/* Placeholder */
.stMultiSelect input::placeholder {
    color: #6B7280 !important;
}

/* =========================
   BUTTON
========================= */
.stButton > button {
    width: 100%;
    background: linear-gradient(
        90deg,
        #00E5FF,
        #00FF99
    );
    color: #0F172A !important;
    font-size: 18px;
    font-weight: 700;
    border: none;
    border-radius: 15px;
    padding: 12px;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0px 5px 20px rgba(0,255,153,0.4);
}

/* =========================
   RESULT CARD
========================= */
.result-card {
    background: linear-gradient(
        135deg,
        rgba(0,255,153,0.15),
        rgba(0,229,255,0.12)
    );
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(0,255,153,0.3);
    margin-top: 25px;
    box-shadow: 0 0 25px rgba(0,255,153,0.15);
}

.prediction {
    font-size: 32px;
    font-weight: bold;
    color: #00FF99;
}

.confidence {
    color: #000000;
    font-size: 18px;
    font-weight: 600;
}

/* =========================
   SIDEBAR
========================= */
section[data-testid="stSidebar"] {
    background: #0F172A;
}

/* =========================
   FOOTER
========================= */
.footer {
    text-align: center;
    color: #94A3B8;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)
# Load models
model = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")
symptom_index = joblib.load("symptom_index.pkl")
description_dict = joblib.load("description_dict.pkl")
precaution_dict = joblib.load("precaution_dict.pkl")

st.markdown(
'<div class="main-title">🩺 AI Disease Prediction System</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">Predict diseases using symptoms with Machine Learning</div>',
unsafe_allow_html=True
)

st.markdown('<div class="glass">', unsafe_allow_html=True)

all_symptoms = sorted(symptom_index.keys())

selected_symptoms = st.multiselect(
"Select Symptoms",
all_symptoms
)

predict = st.button(
"🚀 Predict Disease",
use_container_width=True
)

st.markdown('</div>', unsafe_allow_html=True)

if predict:

    if len(selected_symptoms) == 0:
        st.warning("Please select symptoms.")
        st.stop()

    vector = np.zeros(len(symptom_index))

    for symptom in selected_symptoms:
        if symptom in symptom_index:
            vector[symptom_index[symptom]] = 1

    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    probs = model.predict_proba([vector])[0]

    top3 = np.argsort(probs)[-3:][::-1]

    best_idx = top3[0]

    disease = le.inverse_transform([best_idx])[0]

    confidence = probs[best_idx] * 100

    st.markdown(
    f"""
    <div class="result-card">
        <div class="prediction">
        🧬 Predicted Disease: {disease}
        </div>
        <br>
        Confidence: {confidence:.2f}%
    </div>
    """,
    unsafe_allow_html=True
    )

    st.progress(float(confidence/100))

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("📖 Description")

        st.write(
            description_dict.get(
                disease,
                "Description not available"
            )
        )

    with col2:

        st.subheader("🛡 Precautions")

        precautions = precaution_dict.get(
            disease,
            []
        )

        for p in precautions:
            st.success(p)

    st.subheader("🏆 Top 3 Predictions")

    for idx in top3:

        d = le.inverse_transform([idx])[0]

        c = probs[idx] * 100

        st.write(f"**{d}**")

        st.progress(float(c/100))

        st.write(f"{c:.2f}%")

st.markdown(
"""
<div class="footer">
Built with ❤️ using Streamlit + Machine Learning
</div>
""",
unsafe_allow_html=True
)