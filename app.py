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

.stApp{
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #334155
    );
}

/* Main Title */
.main-title{
    font-size:60px;
    font-weight:bold;
    text-align:center;
    background: linear-gradient(
        90deg,
        #00f5ff,
        #00ff88
    );
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    margin-bottom:10px;
}

/* Subtitle */
.subtitle{
    text-align:center;
    color:#f8fafc;   /* brighter white */
    font-size:20px;
    font-weight:500;
    margin-bottom:30px;
}

/* Glass Container */
.glass{
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
    padding:25px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.25);
    color:#ffffff;   /* all text inside becomes white */
}

/* Labels & General Text */
label, p, div, span{
    color:#f8fafc !important;
}

/* Input Fields */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"]{
    background-color: rgba(255,255,255,0.1) !important;
    color: white !important;
}

/* Result Card */
.result-card{
    background: rgba(0,255,150,0.15);
    padding:20px;
    border-radius:20px;
    margin-top:20px;
    border:1px solid rgba(0,255,150,0.4);
    color:white;
}

/* Prediction Text */
.prediction{
    font-size:30px;
    font-weight:bold;
    color:#00ff88;
}

/* Footer */
.footer{
    text-align:center;
    margin-top:40px;
    color:#e2e8f0;
}

/* Streamlit Markdown Text */
.stMarkdown{
    color:#ffffff !important;
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