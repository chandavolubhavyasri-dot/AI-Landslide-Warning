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
