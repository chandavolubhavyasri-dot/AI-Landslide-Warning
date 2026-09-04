import io
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from gtts import gTTS

st.set_page_config(
    page_title="AI Landslide Warning System",
    page_icon="🚨",
    layout="wide"
)

MODEL_FILE = Path("landslide_model.joblib")

LANGUAGES = {
    "Hindi": (
        "hi",
        "आपातकालीन चेतावनी! भूस्खलन का उच्च जोखिम। कृपया तुरंत सुरक्षित क्षेत्र में जाएं।"
    ),
    "Marathi": (
        "mr",
        "सावधान! दरड कोसळण्याचा धोका निर्माण झाला आहे. कृपया त्वरित सुरक्षित ठिकाणी स्थलांतर करा."
    ),
    "Malayalam": (
        "ml",
        "അടിയന്തര മുന്നറിയിപ്പ്! ഉരുൾപൊട്ടൽ സാധ്യതയുണ്ട്. ദയവായി ഉടൻ സുരക്ഷിത സ്ഥാനത്തേക്ക് മാറുക."
    ),
    "Kannada": (
        "kn",
        "ತುರ್ತು ಎಚ್ಚರಿಕೆ! ಭೂಕುಸಿತದ ಅಪಾಯವಿದೆ. ದಯವಿಟ್ಟು ತಕ್ಷಣ ಸುರಕ್ಷಿತ ಪ್ರದೇಶಕ್ಕೆ ತೆರಳಿ."
    ),
    "Bengali": (
        "bn",
        "জরুরি সতর্কতা! ভূমিধসের প্রবল ঝুঁকি রয়েছে। দয়া করে দ্রুত নিরাপদ স্থানে সরে যান।"
    ),
}

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


def make_voice(lang_code, message):
    audio = io.BytesIO()

    gTTS(
        text=message,
        lang=lang_code,
        slow=False
    ).write_to_fp(audio)

    audio.seek(0)
    return audio


st.title("🚨 AI Landslide Monitoring & Voice Warning System")

st.caption(
    "Machine-learning prototype for multilingual landslide early warning"
)

model = load_model()

if model is None:
    st.error(
        "ML model not found. Please run train_model.py first "
        "and upload landslide_model.joblib."
    )
    st.stop()


st.subheader("📡 Sensor Inputs")

col1, col2 = st.columns(2)

with col1:

    rainfall = st.number_input(
        "🌧️ Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=180.0,
        step=1.0
    )

    soil_moisture = st.number_input(
        "💧 Soil Moisture (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )


with col2:

    vibration = st.number_input(
        "📳 Ground Vibration",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.1
    )

    slope = st.number_input(
        "⛰️ Slope / Ground Tilt (degrees)",
        min_value=0.0,
        max_value=45.0,
        value=10.0,
        step=0.5
    )


if st.button("🔍 Predict Landslide Risk", type="primary"):

    input_data = pd.DataFrame([{
        "rainfall": rainfall,
        "soil_moisture": soil_moisture,
        "vibration": vibration,
        "slope": slope
    }])

    prediction = model.predict(input_data)[0]

    st.divider()

    if prediction == "CRITICAL":

        st.error("🔴 CRITICAL LANDSLIDE RISK")

        st.warning(
            "Immediate evacuation to a safe location is recommended. "
            "This is a prototype prediction and must not replace official warnings."
        )

        st.subheader("🔊 Multilingual Emergency Voice Warnings")

        for language, (lang_code, message) in LANGUAGES.items():

            st.markdown(f"**{language}**")

            st.write(message)

            try:

                audio = make_voice(
                    lang_code,
                    message
                )

                st.audio(
                    audio,
                    format="audio/mp3"
                )

            except Exception as exc:

                st.warning(
                    f"{language} voice could not be generated: {exc}"
                )

    elif prediction == "WARNING":

        st.warning(
            "🟠 WARNING — Elevated landslide risk detected. "
            "Continue monitoring and follow local authorities."
        )

    else:

        st.success(
            "🟢 SAFE — Current sensor pattern is classified as low risk."
        )

    st.subheader("📊 Model Input")

    st.dataframe(
        input_data,
        use_container_width=True
    )


st.divider()

st.info(
    "Prototype note: the included CSV is synthetic training data "
    "for demonstration. For real deployment, replace it with "
    "validated historical sensor/labeled landslide data."
)
