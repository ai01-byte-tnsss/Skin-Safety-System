import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.parse

# --- 1. إعدادات الصفحة والقاموس اللغوي ---
st.set_page_config(page_title="Global Skin Guard AI", layout="centered")

LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "🛡️ نظام الكشف عن سلامة الجلد", "upload": "📥 ارفع صورة الفحص", "camera": "📸 صورة فورية", "analyze": "🚀 بدء التحليل", "guide": "📚 الدليل الطبي الشامل", "malig": "الأورام الخبيثة", "benign": "الأورام الحميدة", "more": "تفاصيل وصور", "res_m": "🚨 اشتباه ورم خبيث", "res_b": "🔍 ورم حميد", "advice": "يرجى مراجعة المختص."},
    "English": {"dir": "ltr", "title": "🛡️ Skin Safety AI System", "upload": "📥 Upload Image", "camera": "📸 Take Photo", "analyze": "🚀 Analyze", "guide": "📚 Medical Guide", "malig": "Malignant", "benign": "Benign", "more": "Details & Images", "res_m": "🚨 Suspected Malignancy", "res_b": "🔍 Benign Tumor", "advice": "Please consult a specialist."},
    "Français": {"dir": "ltr", "title": "🛡️ Système IA de Sécurité Cutanée", "upload": "📥 Charger l'image", "camera": "📸 Caméra", "analyze": "🚀 Analyser", "guide": "📚 Guide Médical", "malig": "Malin", "benign": "Bénin", "more": "Détails", "res_m": "🚨 Suspect Malin", "res_b": "🔍 Bénin", "advice": "Consultez un médecin."},
    "Türkçe": {"dir": "ltr", "title": "🛡️ Cilt Güvenliği AI Sistemi", "upload": "📥 Resim Yükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz Et", "guide": "📚 Tıbbi Rehber", "malig": "Kötü Huylu", "benign": "İyi Huylu", "more": "Detaylar", "res_m": "🚨 Şüpheli Lezyon", "res_b": "🔍 İyi Huylu", "advice": "Doktora danışın."}
}

# اختيار اللغة
selected_lang = st.sidebar.selectbox("🌐 Choose Language / اختر اللغة", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

# --- 2. التنسيق البصري (CSS) ---
st.markdown(f"""
<style>
    div[dir='{t['dir']}'] {{ text-align: {'right' if t['dir']=='rtl' else 'left'}; }}
    .report-card {{ padding: 25px; border-radius: 15px; text-align: center; border: 4px solid; margin-top: 20px; }}
    .disease-card {{ border-right: 5px solid #0d47a1; border-left: 1px solid #eee; padding: 12px; background: #fdfdfd; margin-bottom: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
    .link-btn {{ display: inline-block; padding: 5px 12px; background-color: #1a73e8; color: white !important; text-decoration: none; border-radius: 5px; font-size: 13px; margin-top: 8px; }}
</style>
""", unsafe_allow_html=True)

# --- 3. محرك الذكاء الاصطناعي (AI Engine) ---
@st.cache_resource
def load_expert_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except: return None

interpreter = load_expert_model()

# --- 4. واجهة المستخدم الرئيسية ---
st.markdown(f"<div dir='{t['dir']}'>", unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align: center; color: #0d47a1;'>{t['title']}</h1>", unsafe_allow_html=True)

up_file = st.file_uploader(t['upload'], type=["jpg", "png", "jpeg"])
cam_file = st.camera_input(t['camera'])
active_img = up_file if up_file else cam_file

if active_img:
    img = Image.open(active_img)
    st.image(img, use_container_width=True)
    
    if st.button(t['analyze']):
        if interpreter:
            with st.spinner("Analyzing..."):
                # معالجة الصورة
                input_details = interpreter.get_input_details()
                target_size = input_details[0]['shape'][1:3]
                img_proc = img.convert("RGB").resize(target_size)
                img_array = np.array(img_proc).astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # التنبؤ الدقيق (Argmax)
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
                idx = np.argmax(output)

                # التفرقة بين الأنواع بناءً على مخرجات النموذج
                if idx in [1, 4, 17]: # أمثلة فئات خبيثة
                    res, col = t['res_m'], "#cf1322"
                else: # أمثلة فئات حميدة
                    res, col = t['res_b'], "#389e0d"

                st.markdown(f"<div class='report-card' style='border-color: {col}; color: {col};'><h2>{res}</h2><p>{t['advice']}</p></div>", unsafe_allow_html=True)

st.write("---")

# --- 5. الدليل الطبي الشامل (روابط دقيقة لكل نوع) ---
with st.expander(f"📖 {t['guide']}"):
    tab1, tab2 = st.tabs([t['malig'], t['benign']])
    
    with tab1: # 🔴 الأورام الخبيثة (8 أنواع بروابط دقيقة)
        m_diseases = [
            ("Basal Cell Carcinoma (BCC)", "https://www.mayoclinic.org/diseases-conditions/basal-cell-carcinoma/symptoms-causes/syc-20354487"),
            ("Squamous Cell Carcinoma (SCC)", "https://www.skincancer.org/skin-cancer-information/squamous-cell-carcinoma/"),
            ("Melanoma / الورم الميلانيني", "https://www.mayoclinic.org/diseases-conditions/melanoma/symptoms-causes/syc-20374884"),
            ("Merkel Cell Carcinoma", "https://www.mayoclinic.org/diseases-conditions/merkel-cell-carcinoma/symptoms-causes/syc-20351030"),
            ("Kaposi Sarcoma", "https://www.cancer.org/cancer/types/kaposi-sarcoma.html"),
            ("Sebaceous Gland Carcinoma", "https://www.mayoclinic.org/diseases-conditions/sebaceous-carcinoma/symptoms-causes/syc-20352957"),
            ("Dermatofibrosarcoma Protuberans (DFSP)", "https://www.cancer.gov/types/soft-tissue-sarcoma/patient/dfsp-treatment-pdq"),
            ("Cutaneous Lymphoma", "https://www.clfoundation.org/cutaneous-lymphoma")
        ]
        for name, link in m_diseases:
            st.markdown(f'<div class="disease-card"><strong>{name}</strong><br><a href="{link}" target="_blank" class="link-btn">{t["more"]}</a></div>', unsafe_allow_html=True)

    with tab2: # 🟢 الأورام الحميدة (8 أنواع بروابط دقيقة)
        b_diseases = [
            ("Nevi / Moles (الشامات)", "https://www.mayoclinic.org/diseases-conditions/moles/symptoms-causes/syc-20375200"),
            ("Seborrheic Keratosis", "https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878"),
            ("Lipomas (الأورام الشحمية)", "https://www.mayoclinic.org/diseases-conditions/lipoma/symptoms-causes/syc-20374470"),
            ("Hemangiomas (الأورام الوعائية)", "https://www.mayoclinic.org/diseases-conditions/infantile-hemangioma/symptoms-causes/syc-20353177"),
            ("Dermatofibromas", "https://my.clevelandclinic.org/health/diseases/22643-dermatofibroma"),
            ("Skin Cysts (الأكياس الجلدية)", "https://www.mayoclinic.org/diseases-conditions/sebaceous-cysts/symptoms-causes/syc-20352701"),
            ("Skin Tags (الزوائد الجلدية)", "https://www.healthline.com/health/skin-tag"),
            ("Actinic Keratosis (ما قبل السرطان)", "https://www.skincancer.org/skin-cancer-information/actinic-keratosis/")
        ]
        for name, link in b_diseases:
            st.markdown(f'<div class="disease-card" style="border-right-color:#389e0d;"><strong>{name}</strong><br><a href="{link}" target="_blank" class="link-btn">{t["more"]}</a></div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<br><p style='text-align: center; color: grey; font-size: 0.8em;'>Skin Safety Detection System © 2026</p>", unsafe_allow_html=True)
