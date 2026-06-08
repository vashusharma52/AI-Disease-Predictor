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

/* Multiselect dropdown text */
.stMultiSelect [data-baseweb="select"] * {
    color: black !important;
}

/* Dropdown menu */
div[role="listbox"] {
    background-color: white !important;
}

/* Dropdown options */
div[role="option"] {
    color: black !important;
    background-color: white !important;
}

/* Hover effect */
div[role="option"]:hover {
    background-color: #e5e7eb !important;
    color: black !important;
}

/* Selected tags */
.stMultiSelect span {
    color: white !important;
}

/* Search box text */
.stMultiSelect input {
    color: black !important;
}

/* Placeholder text */
.stMultiSelect input::placeholder {
    color: #555 !important;
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