import streamlit as st
import pandas as pd
import joblib
from gtts import gTTS
import io
import requests
from datetime import datetime
import pydeck as pdk


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NER Landslide Early Warning",
        page_icon="🚨",
            layout="wide"
            )


            # =========================================================
            # APPLICATION TITLE
            # =========================================================

            st.title("🚨 AI-BASED LANDSLIDE EARLY WARNING SYSTEM")

            st.markdown(
                """
                    **North Eastern Region Disaster Preparedness Platform**

                        🌦️ Real-time environmental data  
                            → 🤖 AI/ML prediction  
                                → 🗺️ GIS risk visualization  
                                    → 🚨 Early warning  
                                        → 📱 Field reporting  
                                            → 🔊 Local-language voice alerts
                                                """
                                                )
# =========================================================
# LOAD AI MODEL
# =========================================================

try:
    model = joblib.load("landslide_model.joblib")
    except Exception:
        st.error("❌ AI model could not be loaded.")
            st.stop()


            # =========================================================
            # LOCATION DATABASE
            # =========================================================

            LOCATION_OPTIONS = {
                "Darjeeling, West Bengal": (27.0410, 88.2663),
                    "Gangtok, Sikkim": (27.3389, 88.6065),
                        "Shillong, Meghalaya": (25.5788, 91.8933),
                            "Aizawl, Mizoram": (23.7271, 92.7176),
                                "Kohima, Nagaland": (25.6751, 94.1086),
                                    "Itanagar, Arunachal Pradesh": (27.0844, 93.6053),
                                        "Imphal, Manipur": (24.8170, 93.9368),
                                            "Dehradun, Uttarakhand": (30.3165, 78.0322),
                                                "Kurseong, West Bengal": (26.8820, 88.2770),
                                                    "Kalimpong, West Bengal": (27.0667, 88.4667)
                                                    }
                                          # =========================================================
                                          # LANGUAGES
                                          # =========================================================

                                          LANGUAGES = [
                                              "English",
                                                  "Hindi",
                                                      "Telugu",
                                                          "Kannada",
                                                              "Malayalam",
                                                                  "Marathi",
                                                                      "Bengali",
                                                                          "Assamese",
                                                                              "Manipuri",
                                                                                  "Nepali",
                                                                                      "Kashmiri",
                                                                                          "Tamil"
                                                                                          ]


                                                                                          LANGUAGE_CODES = {
                                                                                              "English": "en",
                                                                                                  "Hindi": "hi",
                                                                                                      "Telugu": "te",
                                                                                                          "Kannada": "kn",
                                                                                                              "Malayalam": "ml",
                                                                                                                  "Marathi": "mr",
                                                                                                                      "Bengali": "bn",
                                                                                                                          "Nepali": "ne",
                                                                                                                              "Tamil": "ta"
                                                                                                                              }


                                                                                                                              # =========================================================
                                                                                                                              # WARNING MESSAGES
                                                                                                                              # =========================================================

                                                                                                                              HIGH_WARNING = {

                                                                                                                                  "English":
                                                                                                                                          "Emergency warning. High landslide risk detected. "
                                                                                                                                                  "Please move to a safe location immediately.",

                                                                                                                                                      "Hindi":
                                                                                                                                                              "आपातकालीन चेतावनी। भूस्खलन का उच्च जोखिम पाया गया है। "
                                                                                                                                                                      "कृपया तुरंत सुरक्षित स्थान पर जाएं।",

                                                                                                                                                                          "Telugu":
                                                                                                                                                                                  "అత్యవసర హెచ్చరిక. కొండచరియలు విరిగిపడే ప్రమాదం ఎక్కువగా ఉంది. "
                                                                                                                                                                                          "దయచేసి వెంటనే సురక్షితమైన ప్రదేశానికి వెళ్లండి.",

                                                                                                                                                                                              "Kannada":
                                                                                                                                                                                                      "ತುರ್ತು ಎಚ್ಚರಿಕೆ. ಭೂಕುಸಿತದ ಅಪಾಯ ಹೆಚ್ಚಾಗಿದೆ. "
                                                                                                                                                                                                              "ದಯವಿಟ್ಟು ತಕ್ಷಣ ಸುರಕ್ಷಿತ ಸ್ಥಳಕ್ಕೆ ತೆರಳಿ.",

                                                                                                                                                                                                                  "Malayalam":
                                                                                                                                                                                                                          "അടിയന്തര മുന്നറിയിപ്പ്. മണ്ണിടിച്ചിലിന്റെ ഉയർന്ന അപകടസാധ്യത കണ്ടെത്തിയിട്ടുണ്ട്. "
                                                                                                                                                                                                                                  "ദയവായി ഉടൻ സുരക്ഷിതമായ സ്ഥലത്തേക്ക് മാറുക.",

                                                                                                                                                                                                                                      "Marathi":
                                                                                                                                                                                                                                              "आपत्कालीन इशारा. भूस्खलनाचा धोका जास्त आहे. "
                                                                                                                                                                                                                                                      "कृपया त्वरित सुरक्षित ठिकाणी जा.",

                                                                                                                                                                                                                                                          "Bengali":
                                                                                                                                                                                                                                                                  "জরুরি সতর্কতা। ভূমিধসের উচ্চ ঝুঁকি শনাক্ত হয়েছে। "
                                                                                                                                                                                                                                                                          "অনুগ্রহ করে অবিলম্বে নিরাপদ স্থানে যান।",

                                                                                                                                                                                                                                                                              "Assamese":
                                                                                                                                                                                                                                                                                      "জৰুৰীকালীন সতৰ্কবাণী। ভূমিস্খলনৰ উচ্চ আশংকা ধৰা পৰিছে। "
                                                                                                                                                                                                                                                                                              "অনুগ্ৰহ কৰি তৎক্ষণাত সুৰক্ষিত স্থানলৈ যাওক।",

                                                                                                                                                                                                                                                                                                  "Manipuri":
                                                                                                                                                                                                                                                                                                          "ꯅꯤꯡꯊꯤꯕ ꯁꯇꯔꯀꯕꯥ। "
                                                                                                                                                                                                                                                                                                                  "ꯃꯩꯅꯥ ꯂꯩꯕ ꯐꯥꯏꯗꯣꯛ ꯊꯣꯛꯄꯒꯤ ꯑꯁꯥꯡ ꯌꯥꯝ ꯂꯩ। "
                                                                                                                                                                                                                                                                                                                          "ꯁꯨꯔꯛꯁꯤꯇ ꯃꯐꯝꯗꯥ ꯆꯠꯂꯨ।",

                                                                                                                                                                                                                                                                                                                              "Nepali":
                                                                                                                                                                                                                                                                                                                                      "आपतकालीन चेतावनी। पहिरोको उच्च जोखिम पत्ता लागेको छ। "
                                                                                                                                                                                                                                                                                                                                              "कृपया तुरुन्त सुरक्षित स्थानमा जानुहोस्।",

                                                                                                                                                                                                                                                                                                                                                  "Kashmiri":
                                                                                                                                                                                                                                                                                                                                                          "ایمرجنسی انتباہ۔ زمین کھسکنے کا خطرہ زیادہ ہے۔ "
                                                                                                                                                                                                                                                                                                                                                                  "مہربانی کر کے فوراً محفوظ مقام پر منتقل ہوو۔",

                                                                                                                                                                                                                                                                                                                                                                      "Tamil":
                                                                                                                                                                                                                                                                                                                                                                              "அவசர எச்சரிக்கை. நிலச்சரிவு ஏற்படும் அபாயம் அதிகமாக உள்ளது. "
                                                                                                                                                                                                                                                                                                                                                                                      "தயவுசெய்து உடனடியாக பாதுகாப்பான இடத்திற்கு செல்லவும்."
                                                                                                                                                                                                                                                                                                                                                                                      }
                                                                                                                                                                                                                                                                                                                                                                    LOW_WARNING = {

                                                                                                                                                                                                                                                                                                                                                                            "English":
                                                                                                                                                                                                                                                                                                                                                                                    "Landslide risk is currently low. Please continue to stay alert.",

                                                                                                                                                                                                                                                                                                                                                                                        "Hindi":
                                                                                                                                                                                                                                                                                                                                                                                                "वर्तमान में भूस्खलन का जोखिम कम है। कृपया सतर्क रहें।",

                                                                                                                                                                                                                                                                                                                                                                                                    "Telugu":
                                                                                                                                                                                                                                                                                                                                                                                                            "ప్రస్తుతం కొండచరియలు విరిగిపడే ప్రమాదం తక్కువగా ఉంది. దయచేసి అప్రమత్తంగా ఉండండి.",

                                                                                                                                                                                                                                                                                                                                                                                                                "Kannada":
                                                                                                                                                                                                                                                                                                                                                                                                                        "ಪ್ರಸ್ತುತ ಭೂಕುಸಿತದ ಅಪಾಯ ಕಡಿಮೆಯಾಗಿದೆ. ದಯವಿಟ್ಟು ಎಚ್ಚರಿಕೆಯಿಂದಿರಿ.",

                                                                                                                                                                                                                                                                                                                                                                                                                            "Malayalam":
                                                                                                                                                                                                                                                                                                                                                                                                                                    "നിലവിൽ മണ്ണിടിച്ചിലിന്റെ അപകടസാധ്യത കുറവാണ്. ദയവായി ജാഗ്രത പാലിക്കുക.",

                                                                                                                                                                                                                                                                                                                                                                                                                                        "Marathi":
                                                                                                                                                                                                                                                                                                                                                                                                                                                "सध्या भूस्खलनाचा धोका कमी आहे. कृपया सतर्क रहा.",

                                                                                                                                                                                                                                                                                                                                                                                                                                                    "Bengali":
                                                                                                                                                                                                                                                                                                                                                                                                                                                            "বর্তমানে ভূমিধসের ঝুঁকি কম। দয়া করে সতর্ক থাকুন.",

                                                                                                                                                                                                                                                                                                                                                                                                                                                                "Assamese":
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        "বৰ্তমান ভূমিস্খলনৰ আশংকা কম। অনুগ্ৰহ কৰি সতৰ্ক হৈ থাকক।",

                                                                                                                                                                                                                                                                                                                                                                                                                                                                            "Manipuri":
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    "ꯃꯐꯝ ꯑꯁꯤꯗꯥ ꯐꯥꯏꯗꯣꯛ ꯊꯣꯛꯄꯒꯤ ꯑꯁꯥꯡ ꯅꯥꯈꯤ। ꯁꯇꯔꯀ ꯑꯣꯏꯅ ꯂꯩꯎ।",

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        "Nepali":
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                "हाल पहिरोको जोखिम कम छ। कृपया सतर्क रहनुहोस्।",

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    "Kashmiri":
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            "فی الحال زمین کھسکنے کا خطرہ کم چھ۔ مہربانی کر کے ہوشیار رہو۔",

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                "Tamil":
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        "தற்போது நிலச்சரிவு ஏற்படும் அபாயம் குறைவாக உள்ளது. தயவுசெய்து விழிப்புடன் இருங்கள்."
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        }


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # =========================================================
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # WEATHER DATA
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # =========================================================

                                                                                                                                                                                                                                                                                                                                                                    }                            
@st.cache_data(ttl=300)
def get_weather_data(latitude, longitude):

    url = "https://api.open-meteo.com/v1/forecast"

        params = {
                "latitude": latitude,
                        "longitude": longitude,
                                "current": "rain,precipitation,soil_moisture_0_to_1cm",
                                        "timezone": "auto"
                                            }

                                                try:

                                                        response = requests.get(
                                                                    url,
                                                                                params=params,
                                                                                            timeout=10
                                                                                                    )

                                                                                                            response.raise_for_status()

                                                                                                                    data = response.json()

                                                                                                                            current = data["current"]

                                                                                                                                    rainfall = float(
                                                                                                                                                current.get("rain", 0)
                                                                                                                                                        )

                                                                                                                                                                precipitation = float(
                                                                                                                                                                            current.get("precipitation", 0)
                                                                                                                                                                                    )

                                                                                                                                                                                            soil_moisture = float(
                                                                                                                                                                                                        current.get(
                                                                                                                                                                                                                        "soil_moisture_0_to_1cm",
                                                                                                                                                                                                                                        0
                                                                                                                                                                                                                                                    )
                                                                                                                                                                                                                                                            ) * 100

                                                                                                                                                                                                                                                                    return {
                                                                                                                                                                                                                                                                                "rainfall": rainfall,
                                                                                                                                                                                                                                                                                            "precipitation": precipitation,
                                                                                                                                                                                                                                                                                                        "soil_moisture": soil_moisture
                                                                                                                                                                                                                                                                                                                }

                                                                                                                                                                                                                                                                                                                    except Exception:

                                                                                                                                                                                                                                                                                                                            return None


                                                                                                                                                                                                                                                                                                                            # =========================================================
                                                                                                                                                                                                                                                                                                                            # AI RISK PREDICTION
                                                                                                                                                                                                                                                                                                                            # =========================================================

                                                                                                                                                                                                                                                                                                                            def predict_risk(rainfall, soil_moisture):

                                                                                                                                                                                                                                                                                                                                # Temporary sensor inputs.
                                                                                                                                                                                                                                                                                                                                    # Replace with real IoT sensor values later.

                                                                                                                                                                                                                                                                                                                                        vibration = 1.0
                                                                                                                                                                                                                                                                                                                                            slope = 20.0

                                                                                                                                                                                                                                                                                                                                                input_data = pd.DataFrame({
                                                                                                                                                                                                                                                                                                                                                        "rainfall": [rainfall],
                                                                                                                                                                                                                                                                                                                                                                "soil_moisture": [soil_moisture],
                                                                                                                                                                                                                                                                                                                                                                        "vibration": [vibration],
                                                                                                                                                                                                                                                                                                                                                                                "slope": [slope]
                                                                                                                                                                                                                                                                                                                                                                                    })

                                                                                                                                                                                                                                                                                                                                                                                        prediction = model.predict(input_data)[0]

                                                                                                                                                                                                                                                                                                                                                                                            if hasattr(model, "predict_proba"):

                                                                                                                                                                                                                                                                                                                                                                                                    probability = model.predict_proba(
                                                                                                                                                                                                                                                                                                                                                                                                                input_data
                                                                                                                                                                                                                                                                                                                                                                                                                        )[0][1]

                                                                                                                                                                                                                                                                                                                                                                                                                            else:

                                                                                                                                                                                                                                                                                                                                                                                                                                    probability = float(prediction)

                                                                                                                                                                                                                                                                                                                                                                                                                                        return int(prediction), float(probability)


                                                                                                                                                                                                                                                                                                                                                                                                                                        # =========================================================
                                                                                                                                                                                                                                                                                                                                                                                                                                        # VOICE GENERATION
                                                                                                                                                                                                                                                                                                                                                                                                                                        # =========================================================

                                                                                                                                                                                                                                                                                                                                                                                                                                        def generate_voice(text, selected_language):

                                                                                                                                                                                                                                                                                                                                                                                                                                            if selected_language not in LANGUAGE_CODES:
                                                                                                                                                                                                                                                                                                                                                                                                                                                    return None

                                                                                                                                                                                                                                                                                                                                                                                                                                                        try:

                                                                                                                                                                                                                                                                                                                                                                                                                                                                audio_buffer = io.BytesIO()

                                                                                                                                                                                                                                                                                                                                                                                                                                                                        tts = gTTS(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    text=text,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                lang=LANGUAGE_CODES[selected_language],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            slow=False
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            tts.write_to_fp(audio_buffer)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    audio_buffer.seek(0)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            return audio_buffer

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                except Exception:

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        return None


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # =========================================================
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # SESSION STATE
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # =========================================================

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        if "reports" not in st.session_state:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            st.session_state.reports = []


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            # =========================================================
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            # SIDEBAR
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            # =========================================================

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            st.sidebar.title("🧭 Control Panel")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            selected_language = st.sidebar.selectbox(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                "🗣️ Warning Language",
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    LANGUAGES
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    selected_location = st.sidebar.selectbox(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        "📍 Monitoring Location",
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            list(LOCATION_OPTIONS.keys())
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            )

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            latitude, longitude = LOCATION_OPTIONS[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                selected_location
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ]

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                st.sidebar.write(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    f"Latitude: {latitude}"
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    st.sidebar.write(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        f"Longitude: {longitude}"
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        )


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # =========================================================
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # DASHBOARD METRICS
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # =========================================================

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        st.subheader("📊 Disaster Monitoring Dashboard")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        col1, col2, col3, col4 = st.columns(4)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        with col1:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            st.metric(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    "📍 Monitoring Zone",
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            selected_location.split(",")[0]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                )

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                with col2:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    st.metric(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            "🗺️ GIS",
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    "ACTIVE"
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        )

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        with col3:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            st.metric(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    "🤖 AI Engine",
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            "ONLINE"
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                )

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                with col4:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    st.metric(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            "📡 Data Pipeline",
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    "LIVE"
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        )
   # =========================================================
   # CHECK SELECTED LOCATION
   # =========================================================

   st.divider()

   st.subheader("🚨 AI Risk Assessment")

   check_risk = st.button(
       "🔍 CHECK CURRENT RISK",
           use_container_width=True
           )


           if check_risk:

               with st.spinner(
                       "Collecting live environmental information..."
                           ):

                                   weather = get_weather_data(
                                               latitude,
                                                           longitude
                                                                   )

                                                                       if weather is None:

                                                                               st.error(
                                                                                           "Unable to obtain live weather data. "
                                                                                                       "Please check your internet connection."
                                                                                                               )

                                                                                                                       st.stop()

                                                                                                                           rainfall = weather["rainfall"]
                                                                                                                               precipitation = weather["precipitation"]
                                                                                                                                   soil_moisture = weather["soil_moisture"]

                                                                                                                                       prediction, probability = predict_risk(
                                                                                                                                               rainfall,
                                                                                                                                                       soil_moisture
                                                                                                                                                           )

                                                                                                                                                               st.session_state.last_result = {
                                                                                                                                                                       "location": selected_location,
                                                                                                                                                                               "latitude": latitude,
                                                                                                                                                                                       "longitude": longitude,
                                                                                                                                                                                               "rainfall": rainfall,
                                                                                                                                                                                                       "precipitation": precipitation,
                                                                                                                                                                                                               "soil_moisture": soil_moisture,
                                                                                                                                                                                                                       "prediction": prediction,
                                                                                                                                                                                                                               "probability": probability
                                                                                                                                                                                                                                   }
    # =========================================================
    # DISPLAY RESULT
    # =========================================================

    if "last_result" in st.session_state:

        result = st.session_state.last_result

            st.subheader(
                    f"📍 {result['location']}"
                        )

                            if result["prediction"] == 1:

                                    st.error(
                                                "🔴 HIGH LANDSLIDE RISK"
                                                        )

                                                                st.markdown(
                                                                            """
                                                                                        # 🚨 EMERGENCY WARNING

                                                                                                    ## MOVE TO A SAFE LOCATION IMMEDIATELY
                                                                                                                """
                                                                                                                        )

                                                                                                                                warning_text = HIGH_WARNING[
                                                                                                                                            selected_language
                                                                                                                                                    ]

                                                                                                                                                        else:

                                                                                                                                                                st.success(
                                                                                                                                                                            "🟢 LOW LANDSLIDE RISK"
                                                                                                                                                                                    )

                                                                                                                                                                                            st.markdown(
                                                                                                                                                                                                        """
                                                                                                                                                                                                                    # 🟢 NO IMMEDIATE WARNING

                                                                                                                                                                                                                                ## Continue to stay alert.
                                                                                                                                                                                                                                            """
                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                            warning_text = LOW_WARNING[
                                                                                                                                                                                                                                                                        selected_language
                                                                                                                                                                                                                                                                                ]

                                                                                                                                                                                                                                                                                    st.info(warning_text)
   # =====================================================
       # VOICE WARNING
           # =====================================================

               st.subheader("🔊 Local Language Voice Alert")

                   audio = generate_voice(
                           warning_text,
                                   selected_language
                                       )

                                           if audio is not None:

                                                   st.audio(
                                                               audio,
                                                                           format="audio/mp3",
                                                                                       autoplay=True
                                                                                               )

                                                                                                       st.success(
                                                                                                                   f"Voice warning generated in {selected_language}."
                                                                                                                           )

                                                                                                                               else:

                                                                                                                                       st.warning(
                                                                                                                                                   f"Text warning is available, but automatic "
                                                                                                                                                               f"voice generation is not currently supported "
                                                                                                                                                                           f"for {selected_language} by the selected TTS engine."
                                                                                                                                                                                   )


                                                                                                                                                                                       # =====================================================
                                                                                                                                                                                           # ENVIRONMENTAL DATA
                                                                                                                                                                                               # =====================================================

                                                                                                                                                                                                   st.subheader("🌦️ Environmental Conditions")

                                                                                                                                                                                                       c1, c2, c3 = st.columns(3)

                                                                                                                                                                                                           with c1:
                                                                                                                                                                                                                   st.metric(
                                                                                                                                                                                                                               "🌧️ Rainfall",
                                                                                                                                                                                                                                           f"{result['rainfall']:.1f} mm"
                                                                                                                                                                                                                                                   )

                                                                                                                                                                                                                                                       with c2:
                                                                                                                                                                                                                                                               st.metric(
                                                                                                                                                                                                                                                                           "🌧️ Precipitation",
                                                                                                                                                                                                                                                                                       f"{result['precipitation']:.1f} mm"
                                                                                                                                                                                                                                                                                               )

                                                                                                                                                                                                                                                                                                   with c3:
                                                                                                                                                                                                                                                                                                           st.metric(
                                                                                                                                                                                                                                                                                                                       "💧 Soil Moisture",
                                                                                                                                                                                                                                                                                                                                   f"{result['soil_moisture']:.1f}%"
                                                                                                                                                                                                                                                                                                                                           )

                                                                                                                                                                                                                                                                                                                                               st.metric(
                                                                                                                                                                                                                                                                                                                                                       "🤖 AI Risk Probability",
                                                                                                                                                                                                                                                                                                                                                               f"{result['probability'] * 100:.1f}%"
                                                                                                                                                                                                                                                                                                                                                                   )
 # =========================================================
 # GIS RISK HEATMAP
 # =========================================================

 st.divider()

 st.subheader("🗺️ GIS-Based Regional Risk Heatmap")

 st.caption(
        "Risk visualization across monitored locations."
 )

 map_records = []

 for name, coordinates in LOCATION_OPTIONS.items():

     lat, lon = coordinates

         weather = get_weather_data(
                    lat,
                            lon
         )

             if weather is not None:

                     try:

                                 prediction, probability = predict_risk(
                                                    weather["rainfall"],
                                                                    weather["soil_moisture"]
                                 )

                                             map_records.append({
                                                                "location": name,
                                                                                "latitude": lat,
                                                                                                "longitude": lon,
                                                                                                                "risk": probability,
                                                                                                                                "prediction": prediction
                                             })

                                                     except Exception:
                                                                 pass


                                                                 if map_records:

                                                                     map_df = pd.DataFrame(map_records)

                                                                         st.pydeck_chart(
                                                                                    pdk.Deck(
                                                                                                    map_style=None,
                                                                                                                initial_view_state=pdk.ViewState(
                                                                                                                                    latitude=25.5,
                                                                                                                                                    longitude=91.5,
                                                                                                                                                                    zoom=5.2,
                                                                                                                                                                                    pitch=0
                                                                                                                ),
                                                                                                                            layers=[
                                                                                                                                                pdk.Layer(
                                                                                                                                                                        "HeatmapLayer",
                                                                                                                                                                                            data=map_df,
                                                                                                                                                                                                                get_position="[longitude, latitude]",
                                                                                                                                                                                                                                    get_weight="risk",
                                                                                                                                                                                                                                                        radius_pixels=60,
                                                                                                                                                                                                                                                                            intensity=1,
                                                                                                                                                                                                                                                                                                threshold=0.05
                                                                                                                                                ),
                                                                                                                                                                pdk.Layer(
                                                                                                                                                                                        "ScatterplotLayer",
                                                                                                                                                                                                            data=map_df,
                                                                                                                                                                                                                                get_position="[longitude, latitude]",
                                                                                                                                                                                                                                                    get_radius=8000,
                                                                                                                                                                                                                                                                        get_fill_color="[255, 80, 80, 180]",
                                                                                                                                                                                                                                                                                            pickable=True
                                                                                                                                                                )
                                                                                                                            ],
                                                                                                                                        tooltip={
                                                                                                                                                            "html":
                                                                                                                                                                                "<b>{location}</b><br/>"
                                                                                                                                                                                                    "Risk: {risk}"
                                                                                                                                        }
                                                                                    ),
                                                                                            use_container_width=True
                                                                         )

                                                                             st.dataframe(
                                                                                        map_df[
                                                                                                        [
                                                                                                                            "location",
                                                                                                                                            "risk",
                                                                                                                                                            "prediction"
                                                                                                        ]
                                                                                        ],
                                                                                                use_container_width=True
                                                                             )

                                                                             else:

                                                                                 st.info(
                                                                                            "GIS data will appear when live environmental "
                                                                                                    "data is available."
                                                                                 )


                                                                                 )
                                                                                                        ]
                                                                                        ]
                                                                             )
                                                                                                                                        }
                                                                                                                                                                )
                                                                                                                                                )
                                                                                                                            ]
                                                                                                                )
                                                                                    )
                                                                         )
                                             })
                                 )
         )
 )  
 # =========================================================
 # FIELD REPORTING
 # =========================================================

 st.divider()

 st.subheader("📱 Field Reporting System")

 st.write(
     "Field workers, volunteers and citizens can "
         "report landslide observations."
         )

         with st.form("field_report_form"):

             report_name = st.text_input(
                     "👤 Reporter name"
                         )

                             report_location = st.selectbox(
                                     "📍 Report location",
                                             list(LOCATION_OPTIONS.keys())
                                                 )

                                                     report_type = st.selectbox(
                                                             "⚠️ Observation",
                                                                     [
                                                                                 "Landslide detected",
                                                                                             "Crack detected",
                                                                                                         "Rockfall detected",
                                                                                                                     "Heavy rainfall",
                                                                                                                                 "Road blocked",
                                                                                                                                             "Flooding",
                                                                                                                                                         "Other"
                                                                                                                                                                 ]
                                                                                                                                                                     )

                                                                                                                                                                         report_description = st.text_area(
                                                                                                                                                                                 "📝 Description"
                                                                                                                                                                                     )

                                                                                                                                                                                         submitted = st.form_submit_button(
                                                                                                                                                                                                 "📤 SUBMIT FIELD REPORT",
                                                                                                                                                                                                         use_container_width=True
                                                                                                                                                                                                             )


                                                                                                                                                                                                             if submitted:

                                                                                                                                                                                                                 report = {
                                                                                                                                                                                                                         "time": datetime.now().strftime(
                                                                                                                                                                                                                                     "%Y-%m-%d %H:%M:%S"
                                                                                                                                                                                                                                             ),
                                                                                                                                                                                                                                                     "reporter": report_name,
                                                                                                                                                                                                                                                             "location": report_location,
                                                                                                                                                                                                                                                                     "observation": report_type,
                                                                                                                                                                                                                                                                             "description": report_description
                                                                                                                                                                                                                                                                                 }

                                                                                                                                                                                                                                                                                     st.session_state.reports.append(report)

                                                                                                                                                                                                                                                                                         st.success(
                                                                                                                                                                                                                                                                                                 "✅ Field report saved successfully."
                                                                                                                                                                                                                                                                                                     )
  # =========================================================
  # OFFLINE REPORT QUEUE
  # =========================================================

  st.subheader("📥 Offline Sync Queue")

  if st.session_state.reports:

      reports_df = pd.DataFrame(
              st.session_state.reports
                  )

                      st.dataframe(
                              reports_df,
                                      use_container_width=True
                                          )

                                              csv_data = reports_df.to_csv(
                                                      index=False
                                                          )

                                                              st.download_button(
                                                                      "⬇️ DOWNLOAD REPORTS FOR SYNC",
                                                                              data=csv_data,
                                                                                      file_name="field_reports.csv",
                                                                                              mime="text/csv",
                                                                                                      use_container_width=True
                                                                                                          )

                                                                                                          else:

                                                                                                              st.info(
                                                                                                                      "No unsynchronized field reports."
                                                                                                                          )


                                                                                                                          # =========================================================
                                                                                                                          # AUTOMATED ALERT CENTER
                                                                                                                          # =========================================================

                                                                                                                          st.divider()

                                                                                                                          st.subheader("🚨 Automated Early Warning Center")

                                                                                                                          if "last_result" in st.session_state:

                                                                                                                              result = st.session_state.last_result

                                                                                                                                  if result["prediction"] == 1:

                                                                                                                                          st.error(
                                                                                                                                                      "🚨 HIGH-RISK ALERT GENERATED"
                                                                                                                                                              )

                                                                                                                                                                      st.write(
                                                                                                                                                                                  f"Warning zone: **{result['location']}**"
                                                                                                                                                                                          )

                                                                                                                                                                                                  st.write(
                                                                                                                                                                                                              f"Risk probability: "
                                                                                                                                                                                                                          f"**{result['probability'] * 100:.1f}%**"
                                                                                                                                                                                                                                  )

                                                                                                                                                                                                                                          st.write(
                                                                                                                                                                                                                                                      "Recommended action: Alert residents, "
                                                                                                                                                                                                                                                                  "field teams and disaster-management authorities."
                                                                                                                                                                                                                                                                          )

                                                                                                                                                                                                                                                                                  st.info(
                                                                                                                                                                                                                                                                                              "📲 SMS / mobile-app / voice-call delivery "
                                                                                                                                                                                                                                                                                                          "can be connected here using a telecom or "
                                                                                                                                                                                                                                                                                                                      "cloud notification provider."
                                                                                                                                                                                                                                                                                                                              )

                                                                                                                                                                                                                                                                                                                                  else:

                                                                                                                                                                                                                                                                                                                                          st.success(
                                                                                                                                                                                                                                                                                                                                                      "🟢 No high-risk automated alert required."
                                                                                                                                                                                                                                                                                                                                                              )
 # =========================================================
 # INTEGRATION STATUS
 # =========================================================

 st.divider()

 st.subheader("🔌 External Data & Service Integration")

 integration_data = pd.DataFrame({
     "Component": [
             "Weather API",
                     "IMD Weather API",
                             "Satellite Feed",
                                     "IoT Sensor Network",
                                             "GIS Dashboard",
                                                     "AI/ML Engine",
                                                             "SMS Gateway",
                                                                     "Mobile/Web Application",
                                                                             "Offline Sync"
                                                                                 ],

                                                                                     "Status": [
                                                                                             "LIVE DEMO",
                                                                                                     "READY FOR API",
                                                                                                             "READY FOR INTEGRATION",
                                                                                                                     "READY FOR INTEGRATION",
                                                                                                                             "ACTIVE",
                                                                                                                                     "ACTIVE",
                                                                                                                                             "READY FOR PROVIDER",
                                                                                                                                                     "ACTIVE",
                                                                                                                                                             "PROTOTYPE"
                                                                                                                                                                 ]
                                                                                                                                                                 })

                                                                                                                                                                 st.dataframe(
                                                                                                                                                                     integration_data,
                                                                                                                                                                         use_container_width=True,
                                                                                                                                                                             hide_index=True
                                                                                                                                                                             )


                                                                                                                                                                             # =========================================================
                                                                                                                                                                             # SYSTEM ARCHITECTURE
                                                                                                                                                                             # =========================================================

                                                                                                                                                                             st.divider()

                                                                                                                                                                             st.subheader("🏗️ Scalable System Architecture")

                                                                                                                                                                             st.markdown(
                                                                                                                                                                                 """
                                                                                                                                                                                     **🌦️ Data Sources**

                                                                                                                                                                                         IMD APIs • Weather APIs • Satellite Feeds • IoT Sensors  
                                                                                                                                                                                             ↓

                                                                                                                                                                                                 **☁️ Cloud Data Layer**

                                                                                                                                                                                                     Data collection • Validation • Storage • API Gateway  
                                                                                                                                                                                                         ↓

                                                                                                                                                                                                             **🤖 AI/ML Prediction Engine**

                                                                                                                                                                                                                 Rainfall + Soil Moisture + Slope + Vibration  
                                                                                                                                                                                                                     ↓

                                                                                                                                                                                                                         **🗺️ GIS Risk Engine**

                                                                                                                                                                                                                             Risk Zones • Heatmaps • Location Intelligence  
                                                                                                                                                                                                                                 ↓

                                                                                                                                                                                                                                     **🚨 Early Warning Engine**

                                                                                                                                                                                                                                         Risk classification • Alert generation • Evacuation recommendation  
                                                                                                                                                                                                                                             ↓

                                                                                                                                                                                                                                                 **📱 Citizen & Field Applications**

                                                                                                                                                                                                                                                     Mobile/Web App • Field Reports • Dashboard  
                                                                                                                                                                                                                                                         ↓

                                                                                                                                                                                                                                                             **🔊 Multilingual Alert Delivery**

                                                                                                                                                                                                                                                                 Voice • SMS • App Notification • Future Automated Calls
                                                                                                                                                                                                                                                                     """
                                                                                                                                                                                                                                                                     )


                                                                                                                                                                                                                                                                     # =========================================================
                                                                                                                                                                                                                                                                     # PROTOTYPE DISCLAIMER
                                                                                                                                                                                                                                                                     # =========================================================

                                                                                                                                                                                                                                                                     with st.expander("⚠️ Prototype & Safety Information"):

                                                                                                                                                                                                                                                                         st.write(
                                                                                                                                                                                                                                                                                 """
                                                                                                                                                                                                                                                                                         This SIH prototype demonstrates the architecture of
                                                                                                                                                                                                                                                                                                 an AI-based landslide early-warning platform.

                                                                                                                                                                                                                                                                                                         Live weather data is currently obtained from a weather
                                                                                                                                                                                                                                                                                                                 API for demonstration.

                                                                                                                                                                                                                                                                                                                         IMD APIs, satellite feeds, IoT vibration sensors,
                                                                                                                                                                                                                                                                                                                                 physical slope sensors and SMS/voice telecom services
                                                                                                                                                                                                                                                                                                                                         require their respective real-world integrations.

                                                                                                                                                                                                                                                                                                                                                 The current ML model is trained on demonstration data
                                                                                                                                                                                                                                                                                                                                                         and must not be used as the sole basis for real
                                                                                                                                                                                                                                                                                                                                                                 emergency decisions.

                                                                                                                                                                                                                                                                                                                                                                         A production system would require validated datasets,
                                                                                                                                                                                                                                                                                                                                                                                 domain-expert validation, sensor calibration,
                                                                                                                                                                                                                                                                                                                                                                                         government data integration, security, redundancy
                                                                                                                                                                                                                                                                                                                                                                                                 and disaster-management authority approval.
                                                                                                                                                                                                                                                                                                                                                                                                         """
                                                                                                                                                                                                                                                                                                                                                                                                             )


                                                                                                                                                                                                                                                                                                                                                                                                             st.caption(
                                                                                                                                                                                                                                                                                                                                                                                                                 "🌦️ Data → 🤖 AI → 🗺️ GIS → 🚨 Warning → 📱 Alert → 🌍 Resilience"
                                                                                                                                                                                                                                                                                                                                                                                                                 )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     