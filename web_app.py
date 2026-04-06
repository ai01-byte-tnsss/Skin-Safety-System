import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.parse

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Global Skin Guard AI", layout="centered")

# --- 2. قاموس اللغات العالمي الشامل ---
LANG_DATA = {
    "English": {"dir": "ltr", "title": "🛡️ Skin Safety AI", "upload": "📥 Upload Image", "camera": "📸 Camera", "analyze": "🚀 Analyze", "guide": "📚 Medical Guide", "malig": "Malignant", "benign": "Benign", "more": "Details", "res_m": "🚨 Malignant Suspect", "res_b": "🔍 Benign", "res_g": "🩺 General", "advice": "Consult a doctor.", "share": "Share"},
    "Français": {"dir": "ltr", "title": "🛡️ IA de Sécurité Cutanée", "upload": "📥 Charger l'image", "camera": "📸 Caméra", "analyze": "🚀 Analyser", "guide": "📚 Guide Médical", "malig": "Malin", "benign": "Bénin", "more": "Détails", "res_m": "🚨 Suspect Malin", "res_b": "🔍 Bénin", "res_g": "🩺 Général", "advice": "Consultez un médecin.", "share": "Partager"},
    "العربية": {"dir": "rtl", "title": "🛡️ نظام الكشف عن سلامة الجلد", "upload": "📥 ارفع صورة الفحص", "camera": "📸 صورة فورية", "analyze": "🚀 بدء التحليل", "guide": "📚 الدليل الطبي", "malig": "الأورام الخبيثة", "benign": "الأورام الحميدة", "more": "تفاصيل وصور", "res_m": "🚨 اشتباه ورم خبيث", "res_b": "🔍 ورم حميد", "res_g": "🩺 حالة عامة", "advice": "يرجى مراجعة المختص.", "share": "مشاركة"},
    "Español": {"dir": "ltr", "title": "🛡️ IA de Seguridad Cutánea", "upload": "📥 Subir imagen", "camera": "📸 Cámara", "analyze": "🚀 Analizar", "guide": "📚 Guía Médica", "malig": "Maligno", "benign": "Benigno", "more": "Detalles", "res_m": "🚨 Sospecha Maligna", "res_b": "🔍 Benigno", "res_g": "🩺 General", "advice": "Consulte a un médico.", "share": "Compartir"},
    "Português": {"dir": "ltr", "title": "🛡️ IA de Segurança da Pele", "upload": "📥 Enviar imagem", "camera": "📸 Câmera", "analyze": "🚀 Analisar", "guide": "📚 Guia Médico", "malig": "Maligno", "benign": "Benigno", "more": "Detalhes", "res_m": "🚨 Suspeita Maligna", "res_b": "🔍 Benigno", "res_g": "🩺 Geral", "advice": "Consulte um médico.", "share": "Compartilhar"},
    "Deutsch": {"dir": "ltr", "title": "🛡️ Hautsicherheits-KI", "upload": "📥 Bild hochladen", "camera": "📸 Kamera", "analyze": "🚀 Analysieren", "guide": "📚 Med. Leitfaden", "malig": "Bösartig", "benign": "Gutartig", "more": "Details", "res_m": "🚨 Krebsverdacht", "res_b": "🔍 Gutartig", "res_g": "🩺 Allgemein", "advice": "Arzt aufsuchen.", "share": "Teilen"},
    "Русский": {"dir": "ltr", "title": "🛡️ ИИ Безопасности Кожи", "upload": "📥 Загрузить фото", "camera": "📸 Камера", "analyze": "🚀 Анализировать", "guide": "📚 Мед. справочник", "malig": "Злокачественные", "benign": "Доброкачественные", "more": "Подробнее", "res_m": "🚨 Подозрение на рак", "res_b": "🔍 Доброкачественное", "res_g": "🩺 Общее", "advice": "Обратитесь к врачу.", "share": "Поделиться"},
    "Türkçe": {"dir": "ltr", "title": "🛡️ Cilt Güvenliği AI", "upload": "📥 Resim Yükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz Et", "guide": "📚 Tıbbi Rehber", "malig": "Kötü Huylu", "benign": "İyi Huylu", "more": "Detaylar", "res_m": "🚨 Şüpheli Lezyon", "res_b": "🔍 İyi Huylu", "res_g": "🩺 Genel Durum", "advice": "Doktora danışın.", "share": "Paylaş"},
    # ... يمكن إضافة باقي القائمة بنفس التنسيق (اللغات الآسيوية والأفريقية)
}

# --- 3. اختيار اللغة وتطبيق التصميم ---
selected_lang = st.sidebar.selectbox("🌐 Choose Language / اختر اللغة", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

st.markdown(f"""
<style>
    div[dir='{t['dir']}'] {{ text-align: {'right' if t['dir']=='rtl' else 'left'}; }}
    .report-card {{ padding: 25px; border-radius: 15px; text-align: center; border: 4px solid; margin-top: 20px; }}
    .disease-card {{ border-right: 5px solid #0d47a1; padding: 10px; background: #fff; margin-bottom: 8px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .btn-link {{ display: inline-block; padding: 5px 10px; background: #1a73e8; color: white !important; text-decoration: none; border-radius: 4px; font-size: 12px; margin-top: 5px; }}
</style>
""", unsafe_allow_html=True)

# --- 4. معالجة النموذج (AI Engine) ---
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except: return None

interpreter = load_model()

# --- 5. واجهة المستخدم الرئيسية ---
st.markdown(f"<div dir='{t['dir']}'>", unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align: center; color: #0d47a1;'>{t['title']}</h1>", unsafe_allow_html=True)

# خيارات إدخال الصورة
input_mode = st.radio("", (t['upload'], t['camera']))
image_file = st.file_uploader(t['upload'], type=["jpg", "png", "jpeg"]) if input_mode == t['upload'] else st.camera_input(t['camera'])

if image_file:
    image = Image.open(image_file)
    st.image(image, use_container_width=True)
    
    if st.button(t['analyze']):
        if interpreter:
            with st.spinner("Processing..."):
                # معالجة الصورة للنموذج
                input_details = interpreter.get_input_details()
                target_size = input_details[0]['shape'][1:3]
                img_res = image.convert("RGB").resize(target_size)
                img_arr = np.array(img_res).astype(np.float32) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)

                # التنبؤ (Argmax لضمان التفرقة الدقيقة)
                interpreter.set_tensor(input_details[0]['index'], img_arr)
                interpreter.invoke()
                output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
                idx = np.argmax(output)

                # منطق التصنيف (التفرقة بين الأنواع بناءً على الفئات المعتمدة)
                if idx in [1, 4, 17]: # أمثلة للفئات الخبيثة في النموذج
                    res_txt, col = t['res_m'], "#cf1322"
                elif idx in [2, 5, 23]: # أمثلة للفئات الحميدة
                    res_txt, col = t['res_b'], "#389e0d"
                else:
                    res_txt, col = t['res_g'], "#096dd9"

                st.markdown(f"""<div class="report-card" style="border-color: {col}; color: {col};">
                    <h2>{res_txt}</h2><p>{t['advice']}</p></div>""", unsafe_allow_html=True)

                # زر المشاركة
                enc_msg = urllib.parse.quote(f"{res_txt} - {t['advice']}")
                st.markdown(f"""<div style='text-align:center; margin-top:15px;'>
                    <a href="https://wa.me/?text={enc_msg}" target="_blank" style="background:#25D366; color:white; padding:10px 20px; border-radius:10px; text-decoration:none;">WhatsApp 💬</a>
                </div>""", unsafe_allow_html=True)

st.write("---")

# --- 6. الدليل الطبي الشامل (كافة الأنواع) ---
with st.expander(f"📖 {t['guide']}"):
    col1, col2 = st.tabs([t['malig'], t['benign']])
    
    with col1: # الأورام الخبيثة
        m_list = [
            ("Basal Cell Carcinoma (BCC)", "https://www.mayoclinic.org/diseases-conditions/basal-cell-carcinoma/symptoms-causes/syc-20354487"),
            ("Squamous Cell Carcinoma (SCC)", "https://www.skincancer.org/skin-cancer-information/squamous-cell-carcinoma/"),
            ("Melanoma / الورم الميلانيني", "https://www.mayoclinic.org/diseases-conditions/melanoma/symptoms-causes/syc-20374884"),
            ("Merkel Cell Carcinoma", "https://www.mayoclinic.org/diseases-conditions/merkel-cell-carcinoma/symptoms-causes/syc-20351030"),
            ("Kaposi Sarcoma", "https://www.cancer.org/cancer/kaposi-sarcoma.html"),
            ("Sebaceous Gland Carcinoma", "https://www.mayoclinic.org/diseases-conditions/sebaceous-carcinoma/symptoms-causes/syc-20352957"),
            ("Dermatofibrosarcoma Protuberans", "https://www.cancer.gov/types/soft-tissue-sarcoma/patient/dfsp-treatment-pdq"),
            ("Cutaneous Lymphoma", "https://www.clfoundation.org/cutaneous-lymphoma")
        ]
        for name, link in m_list:
            st.markdown(f'<div class="disease-card"><strong>{name}</strong><br><a href="{link}" target="_blank" class="btn-link">{t["more"]}</a></div>', unsafe_allow_html=True)

    with col2: # الأورام الحميدة
        b_list = [
            ("Nevi / Moles (الشامات)", "https://www.mayoclinic.org/diseases-conditions/moles/symptoms-causes/syc-20375200"),
            ("Seborrheic Keratosis", "https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878"),
            ("Lipomas (الأورام الشحمية)", "https://www.mayoclinic.org/diseases-conditions/lipoma/symptoms-causes/syc-20374470"),
            ("Hemangiomas", "https://www.mayoclinic.org/diseases-conditions/infantile-hemangioma/symptoms-causes/syc-20353177"),
            ("Dermatofibromas", "https://www.healthline.com/health/dermatofibroma"),
            ("Skin Cysts (الأكياس الجلدية)", "https://www.mayoclinic.org/diseases-conditions/sebaceous-cysts/symptoms-causes/syc-20352701"),
            ("Skin Tags (الزوائد الجلدية)", "https://www.healthline.com/health/skin-tag"),
            ("Actinic Keratosis (Pre-cancer)", "https://www.skincancer.org/skin-cancer-information/actinic-keratosis/")
        ]
        for name, link in b_list:
            st.markdown(f'<div class="disease-card" style="border-right-color:#389e0d;"><strong>{name}</strong><br><a href="{link}" target="_blank" class="btn-link">{t["more"]}</a></div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<br><p style='text-align: center; color: grey; font-size: 0.8em;'>Skin Safety Detection System © 2026</p>", unsafe_allow_html=True)
