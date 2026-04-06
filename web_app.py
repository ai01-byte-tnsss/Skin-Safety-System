import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.parse

# --- 1. إعدادات الصفحة الأساسية ---
st.set_page_config(page_title="Skin Safety AI Expert", layout="centered")

# --- 2. القاموس اللغوي الشامل (بنية موحدة لمنع الأخطاء) ---
LANG_DATA = {
    "English": {"dir": "ltr", "title": "🛡️ Skin Safety AI", "upload": "📥 Upload Image", "camera": "📸 Camera", "analyze": "🚀 Analyze", "guide": "📚 Medical Guide", "malig": "Malignant", "benign": "Benign", "more": "Details", "res_m": "🚨 Malignant Suspect", "res_b": "🔍 Benign", "res_g": "🩺 General Condition", "advice": "Please consult a doctor.", "share": "Share Result"},
    "العربية": {"dir": "rtl", "title": "🛡️ نظام الكشف عن سلامة الجلد", "upload": "📥 ارفع صورة الفحص", "camera": "📸 صورة فورية", "analyze": "🚀 بدء التحليل", "guide": "📚 الدليل الطبي الشامل", "malig": "الأورام الخبيثة", "benign": "الأورام الحميدة", "more": "تفاصيل وصور", "res_m": "🚨 اشتباه ورم خبيث", "res_b": "🔍 ورم حميد", "res_g": "🩺 حالة عامة", "advice": "يرجى مراجعة المختص لضمان السلامة.", "share": "مشاركة"},
    "Français": {"dir": "ltr", "title": "🛡️ IA de Sécurité Cutanée", "upload": "📥 Charger l'image", "camera": "📸 Caméra", "analyze": "🚀 Analyser", "guide": "📚 Guide Médical", "malig": "Malin", "benign": "Bénin", "more": "Détails", "res_m": "🚨 Suspect Malin", "res_b": "🔍 Bénin", "res_g": "🩺 État Général", "advice": "Consultez un médecin.", "share": "Partager"},
    "Türkçe": {"dir": "ltr", "title": "🛡️ Cilt Güvenliği AI", "upload": "📥 Resim Yükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz Et", "guide": "📚 Tıbbi Rehber", "malig": "Kötü Huylu", "benign": "İyi Huylu", "more": "Detaylar", "res_m": "🚨 Kötü Huylu Şübhesi", "res_b": "🔍 İyi Huylu", "res_g": "🩺 Genel Durum", "advice": "Doktora danışın.", "share": "Paylaş"},
    "کوردی": {"dir": "rtl", "title": "🛡️ سیستەمی پشکنینی پێست", "upload": "📥 وێنە دابنێ", "camera": "📸 وێنە بگرە", "analyze": "🚀 شیکاری", "guide": "📚 ڕێبەری پزیشکی", "malig": "گرێی خراپ", "benign": "گرێی بێ زیان", "more": "زانیاری", "res_m": "🚨 گومانی گرێی خراپ", "res_b": "🔍 گرێی بێ زیان", "res_g": "🩺 باری گشتی", "advice": "سەردانی پزیشک بکە.", "share": "ناردن"},
    "فارسی": {"dir": "rtl", "title": "🛡️ هوش مصنوعی سلامت پوست", "upload": "📥 بارگذاری تصویر", "camera": "📸 دوربین", "analyze": "🚀 شروع آنالیز", "guide": "📚 راهنمای پزشکی", "malig": "بدخیم", "benign": "خوش‌خیم", "more": "جزئیات", "res_m": "🚨 مشکوک به بدخیم", "res_b": "🔍 خوش‌خیم", "res_g": "🩺 وضعیت عمومی", "advice": "به پزشک مراجعه کنید.", "share": "اشتراک‌گذاری"}
}

# --- 3. اختيار اللغة وتنسيق CSS ---
selected_lang = st.sidebar.selectbox("🌐 Choose Language / اختر اللغة", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

st.markdown(f"""
<style>
    div[dir='{t['dir']}'] {{ text-align: {'right' if t['dir']=='rtl' else 'left'}; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
    .report-card {{ padding: 25px; border-radius: 15px; text-align: center; border: 5px solid; margin-top: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
    .disease-item {{ border-right: 5px solid #0d47a1; border-left: 1px solid #eee; padding: 12px; background: #fff; margin-bottom: 8px; border-radius: 8px; }}
    .share-container {{ display: flex; justify-content: center; gap: 15px; margin-top: 20px; }}
    .link-btn {{ display: inline-block; padding: 6px 12px; background: #1a73e8; color: white !important; text-decoration: none; border-radius: 5px; font-size: 13px; margin-top: 5px; }}
</style>
""", unsafe_allow_html=True)

# --- 4. محرك الذكاء الاصطناعي (تحميل النموذج) ---
@st.cache_resource
def load_expert_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception:
        return None

interpreter = load_expert_model()

# --- 5. واجهة الفحص والتحليل ---
st.markdown(f"<div dir='{t['dir']}'>", unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align: center; color: #0d47a1;'>{t['title']}</h1>", unsafe_allow_html=True)

source = st.radio("", (t['upload'], t['camera']))
input_file = st.file_uploader(t['upload'], type=["jpg", "png", "jpeg"]) if source == t['upload'] else st.camera_input(t['camera'])

if input_file:
    img = Image.open(input_file)
    st.image(img, use_container_width=True)
    
    if st.button(t['analyze']):
        if interpreter:
            with st.spinner("AI is analyzing..."):
                # معالجة الصورة
                in_details = interpreter.get_input_details()
                target_size = in_details[0]['shape'][1:3]
                img_proc = img.convert("RGB").resize(target_size)
                img_arr = np.array(img_proc).astype(np.float32) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)

                # التنبؤ الدقيق
                interpreter.set_tensor(in_details[0]['index'], img_arr)
                interpreter.invoke()
                output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
                idx = np.argmax(output)

                # منطق التصنيف (خبيث: 1,4,17 | حميد: 2,5,23)
                if idx in [1, 4, 17]:
                    res_msg, color = t['res_m'], "#cf1322"
                elif idx in [2, 5, 23]:
                    res_msg, color = t['res_b'], "#389e0d"
                else:
                    res_msg, color = t['res_g'], "#096dd9"

                # عرض النتيجة
                st.markdown(f"""<div class="report-card" style="border-color: {color}; color: {color};">
                    <h2>{res_msg}</h2><p style='font-size: 1.1em;'>{t['advice']}</p></div>""", unsafe_allow_html=True)

                # المشاركة
                share_text = urllib.parse.quote(f"{res_msg} - {t['advice']}")
                st.markdown(f"""<div class="share-container">
                    <a href="https://wa.me/?text={share_text}" target="_blank" style="background:#25D366; color:white; padding:10px 20px; border-radius:10px; text-decoration:none; font-weight:bold;">WhatsApp</a>
                    <a href="mailto:?subject=Report&body={share_text}" style="background:#EA4335; color:white; padding:10px 20px; border-radius:10px; text-decoration:none; font-weight:bold;">Email</a>
                </div>""", unsafe_allow_html=True)

st.write("---")

# --- 6. الدليل الطبي الموسع (16 نوعاً بروابط عالمية) ---
with st.expander(f"📖 {t['guide']}"):
    tab_m, tab_b = st.tabs([t['malig'], t['benign']])
    
    with tab_m: # الأورام الخبيثة
        m_diseases = [
            ("Basal Cell Carcinoma (BCC)", "https://www.mayoclinic.org/diseases-conditions/basal-cell-carcinoma/symptoms-causes/syc-20354487"),
            ("Squamous Cell Carcinoma (SCC)", "https://www.skincancer.org/skin-cancer-information/squamous-cell-carcinoma/"),
            ("Melanoma / الورم الميلانيني", "https://www.mayoclinic.org/diseases-conditions/melanoma/symptoms-causes/syc-20374884"),
            ("Merkel Cell Carcinoma", "https://www.mayoclinic.org/diseases-conditions/merkel-cell-carcinoma/symptoms-causes/syc-20351030"),
            ("Kaposi Sarcoma", "https://www.cancer.org/cancer/types/kaposi-sarcoma.html"),
            ("Sebaceous Gland Carcinoma", "https://www.mayoclinic.org/diseases-conditions/sebaceous-carcinoma/symptoms-causes/syc-20352957"),
            ("Dermatofibrosarcoma Protuberans", "https://www.cancer.gov/types/soft-tissue-sarcoma/patient/dfsp-treatment-pdq"),
            ("Cutaneous Lymphoma", "https://www.clfoundation.org/cutaneous-lymphoma")
        ]
        for name, link in m_diseases:
            st.markdown(f'<div class="disease-item"><strong>{name}</strong><br><a href="{link}" target="_blank" class="link-btn">{t["more"]}</a></div>', unsafe_allow_html=True)

    with tab_b: # الأورام الحميدة
        b_diseases = [
            ("Nevi / Moles (الشامات)", "https://www.mayoclinic.org/diseases-conditions/moles/symptoms-causes/syc-20375200"),
            ("Seborrheic Keratosis", "https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878"),
            ("Lipomas (الأورام الشحمية)", "https://www.mayoclinic.org/diseases-conditions/lipoma/symptoms-causes/syc-20374470"),
            ("Hemangiomas (الأورام الوعائية)", "https://www.mayoclinic.org/diseases-conditions/infantile-hemangioma/symptoms-causes/syc-20353177"),
            ("Dermatofibromas", "https://my.clevelandclinic.org/health/diseases/22643-dermatofibroma"),
            ("Skin Cysts (الأكياس الجلدية)", "https://www.healthline.com/health/skin-cyst"),
            ("Skin Tags (الزوائد الجلدية)", "https://www.medicalnewstoday.com/articles/67317"),
            ("Actinic Keratosis (ما قبل السرطان)", "https://www.skincancer.org/skin-cancer-information/actinic-keratosis/")
        ]
        for name, link in b_diseases:
            st.markdown(f'<div class="disease-item" style="border-right-color:#389e0d;"><strong>{name}</strong><br><a href="{link}" target="_blank" class="link-btn">{t["more"]}</a></div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey; font-size: 0.8em;'>Global Skin Guard AI © 2026</p>", unsafe_allow_html=True)
