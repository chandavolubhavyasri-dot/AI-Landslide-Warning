import streamlit as st
import pandas as pd
import joblib
from gtts import gTTS
import io

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="AI Landslide Warning System",
    page_icon="⚠️",
    layout="centered"
)

st.title("⚠️ AI Landslide Warning System")
st.write("AI-based landslide risk prediction with multilingual voice warning.")

# -----------------------------
# Load trained ML model
# -----------------------------
model = joblib.load("landslide_model.joblib")

# -----------------------------
# User inputs
# -----------------------------
st.subheader("📊 Enter Environmental Data")

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    value=180.0
)

soil_moisture = st.number_input(
    "Soil Moisture (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

vibration = st.number_input(
    "Vibration",
    min_value=0.0,
    value=5.0
)

slope = st.number_input(
    "Slope (degrees)",
    min_value=0.0,
    max_value=90.0,
    value=10.0
)

# -----------------------------
# Language selection
# -----------------------------
language = st.selectbox(
    "🔊 Warning Voice Language",
    [
        "English",
        "Hindi",
        "Telugu",
        "Kannada",
        "Malayalam",
        "Marathi",
        "Bengali"
    ]
)

language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Bengali": "bn"
}

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Landslide Risk"):

    input_data = pd.DataFrame({
        "rainfall": [rainfall],
        "soil_moisture": [soil_moisture],
        "vibration": [vibration],
        "slope": [slope]
    })

    prediction = model.predict(input_data)[0]

    # Get probability if available
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]
    else:
        probability = float(prediction)

    st.subheader("🚨 Risk Assessment")

    # -----------------------------
    # Risk levels
    # -----------------------------
    if prediction == 1:
        risk_text = "HIGH LANDSLIDE RISK"
        warning_text = {
            "English": "Emergency warning. High landslide risk detected. Please move to a safe location and follow local authorities.",
            "Hindi": "आपातकालीन चेतावनी। भूस्खलन का उच्च जोखिम पाया गया है। कृपया सुरक्षित स्थान पर जाएं और स्थानीय अधिकारियों के निर्देशों का पालन करें।",
            "Telugu": "అత్యవసర హెచ్చరిక. కొండచరియలు విరిగిపడే ప్రమాదం ఎక్కువగా ఉంది. దయచేసి సురక్షితమైన ప్రదేశానికి వెళ్లి స్థానిక అధికారుల సూచనలను పాటించండి.",
            "Kannada": "ತುರ್ತು ಎಚ್ಚರಿಕೆ. ಭೂಕುಸಿತದ ಅಪಾಯ ಹೆಚ್ಚಾಗಿದೆ. ದಯವಿಟ್ಟು ಸುರಕ್ಷಿತ ಸ್ಥಳಕ್ಕೆ ತೆರಳಿ ಸ್ಥಳೀಯ ಅಧಿಕಾರಿಗಳ ಸೂಚನೆಗಳನ್ನು ಅನುಸರಿಸಿ.",
            "Malayalam": "അടിയന്തര മുന്നറിയിപ്പ്. മണ്ണിടിച്ചിലിന്റെ ഉയർന്ന അപകടസാധ്യത കണ്ടെത്തിയിട്ടുണ്ട്. ദയവായി സുരക്ഷിതമായ സ്ഥലത്തേക്ക് മാറി പ്രാദേശിക അധികാരികളുടെ നിർദ്ദേശങ്ങൾ പാലിക്കുക.",
            "Marathi": "आपत्कालीन इशारा. भूस्खलनाचा धोका जास्त आहे. कृपया सुरक्षित ठिकाणी जा आणि स्थानिक अधिकाऱ्यांच्या सूचनांचे पालन करा.",
            "Bengali": "জরুরি সতর্কতা। ভূমিধসের উচ্চ ঝুঁকি শনাক্ত হয়েছে। অনুগ্রহ করে নিরাপদ স্থানে যান এবং স্থানীয় কর্তৃপক্ষের নির্দেশ অনুসরণ করুন।"
        }

        st.error("🔴 " + risk_text)

    else:
        risk_text = "LOW LANDSLIDE RISK"
        warning_text = {
            "English": "Landslide risk is currently low. Continue monitoring environmental conditions.",
            "Hindi": "वर्तमान में भूस्खलन का जोखिम कम है। पर्यावरणीय परिस्थितियों की निगरानी जारी रखें।",
            "Telugu": "ప్రస్తుతం కొండచరియలు విరిగిపడే ప్రమాదం తక్కువగా ఉంది. పర్యావరణ పరిస్థితులను పర్యవేక్షిస్తూ ఉండండి.",
            "Kannada": "ಪ್ರಸ್ತುತ ಭೂಕುಸಿತದ ಅಪಾಯ ಕಡಿಮೆಯಾಗಿದೆ. ಪರಿಸರ ಪರಿಸ್ಥಿತಿಗಳನ್ನು ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡುತ್ತಿರಿ.",
            "Malayalam": "നിലവിൽ മണ്ണിടിച്ചിലിന്റെ അപകടസാധ്യത കുറവാണ്. പരിസ്ഥിതി സാഹചര്യങ്ങൾ നിരീക്ഷിക്കുന്നത് തുടരുക.",
            "Marathi": "सध्या भूस्खलनाचा धोका कमी आहे. पर्यावरणीय परिस्थितीचे निरीक्षण सुरू ठेवा.",
            "Bengali": "বর্তমানে ভূমিধসের ঝুঁকি কম। পরিবেশগত পরিস্থিতি পর্যবেক্ষণ করতে থাকুন।"
        }

        st.success("🟢 " + risk_text)

    st.write(
        f"**Estimated risk probability:** {probability * 100:.1f}%"
    )

    # -----------------------------
    # Generate voice warning
    # -----------------------------
    st.subheader("🔊 AI Voice Warning")

    selected_text = warning_text[language]

    st.info(selected_text)

    try:
        audio_buffer = io.BytesIO()

        tts = gTTS(
            text=selected_text,
            lang=language_codes[language],
            slow=False
        )

        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        st.audio(audio_buffer, format="audio/mp3")

        st.success(
            "🔊 Voice warning generated. Press the play button above to hear it."
        )

    except Exception as e:
        st.warning(
            "Voice generation could not be completed. "
            "Please check the internet connection."
        )

# -----------------------------
# Prototype note
# -----------------------------
st.divider()

st.info(
    "Prototype note: This system uses demonstration training data. "
    "For real-world emergency deployment, use validated historical sensor "
    "and landslide data and obtain appropriate expert validation."
)