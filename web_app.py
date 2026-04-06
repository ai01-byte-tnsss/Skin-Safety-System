import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.parse

# --- 1. إعدادات الصفحة والقاموس اللغوي (25 لغة) ---
st.set_page_config(page_title="Global Skin Guard AI", layout="centered")

LANG_DATA = {
    "English": {"dir": "ltr", "title": "🛡️ Skin Safety AI", "upload": "📥 Upload Image", "camera": "📸 Camera", "analyze": "🚀 Analyze", "guide": "📚 Medical Guide", "malig": "Malignant", "benign": "Benign", "more": "Details", "res_m": "🚨 Malignant Suspect", "res_b": "🔍 Benign", "res_g": "🩺 General", "advice": "Consult a doctor.", "share": "Share Result"},
    "Français": {"dir": "ltr", "title": "🛡️ IA de Sécurité Cutanée", "upload": "📥 Charger l'image", "camera": "📸 Caméra", "analyze": "🚀 Analyser", "guide": "📚 Guide Médical", "malig": "Malin", "benign": "Bénin", "more": "Détails", "res_m": "🚨 Suspect Malin", "res_b": "🔍 Bénin", "res_g": "🩺 Général", "advice": "Consultez un médecin.", "share": "Partager"},
    "العربية": {"dir": "rtl", "title": "🛡️ نظام الكشف عن سلامة الجلد", "upload": "📥 ارفع صورة الفحص", "camera": "📸 صورة فورية", "analyze": "🚀 بدء التحليل", "guide": "📚 الدليل الطبي الشامل", "malig": "الأورام الخبيثة", "benign": "الأورام الحميدة", "more": "تفاصيل وصور", "res_m": "🚨 اشتباه ورم خبيث", "res_b": "🔍 ورم حميد", "res_g": "🩺 حالة عامة", "advice": "يرجى مراجعة المختص.", "share": "مشاركة النتيجة"},
    "Türkçe": {"dir": "ltr", "title": "🛡️ Cilt Güvenliği AI", "upload": "📥 Resim Yükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz Et", "guide": "📚 Tıbbi Rehber", "malig": "Kötü Huylu", "benign": "İyi Huylu", "more": "Detaylar", "res_m": "🚨 Kötü Huylu Şübhesi", "res_b": "🔍 İyi Huylu", "res_g": "🩺 Genel Durum", "advice": "Doktora danışın.", "share": "Paylaş"},
    # ... يمكن إضافة باقي اللغات بنفس النمط
}

selected_lang = st.sidebar.selectbox("🌐 Choose Language / اختر اللغة", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

# --- 2. تنسيق الواجهة (CSS) ---
st.markdown(f"""
<style>
    div[dir='{t['dir']}'] {{ text-align: {'right' if t['dir']=='rtl' else 'left'}; }}
    .report-card {{ padding: 30px; border-radius: 20px; text-align: center; border: 4px solid; margin-top: 20px; box-shadow: 0px 10px 25px rgba(0,0,0,0.1); }}
    .disease-item {{ border-right: 5px solid #0d47a1; border-left: 1px solid #eee; padding: 12px; background: #fff; margin-bottom: 10px; border-radius: 8px; font-size: 14px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
    .share-btn {{ display: inline-block; padding: 10px 20px; border-radius: 10px; text-decoration: none; color: white !important; font-weight: bold; margin: 5px; }}
    .link-style {{ color: #1a73e8; text-decoration: none; font-weight: bold; }}
</style>
""", unsafe_allow_html=True)

# --- 3. تحميل النموذج وتصحيح الخطأ التقني ---
@st.cache_resource
def load_expert_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

interpreter = load_expert_model()

# --- 4. واجهة الفحص والتحليل ---
st.markdown(f"<div dir='{t['dir']}'>", unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align: center; color: #0d47a1;'>{t['title']}</h1>", unsafe_allow_html=True)

input_choice = st.radio("", (t['upload'], t['camera']))
uploaded_file = st.file_uploader(t['upload'], type=["jpg", "png", "jpeg"]) if input_choice == t['upload'] else st.camera_input(t['camera'])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, use_container_width=True)
    
    if st.button(t['analyze']):
        if interpreter:
            with st.spinner("Analyzing..."):
                # --- تصحيح أبعاد المدخلات (The Fix) ---
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # قراءة الأبعاد المطلوبة من النموذج مباشرة [1, height, width, 3]
                h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
                dtype = input_details[0]['dtype']

                # معالجة الصورة لتطابق النموذج
                img_resized = img.convert("RGB").resize((w, h))
                arr = np.array(img_resized)

                if dtype == np.float32:
                    arr = arr.astype(np.float32) / 255.0
                
                arr = np.expand_dims(arr, axis=0) # إضافة بعد الـ Batch ليصبح [1, h, w, 3]

                try:
                    # تمرير البيانات المنقحة للنموذج
                    interpreter.set_tensor(input_details[0]['index'], arr)
                    interpreter.invoke()
                    
                    # استخراج النتيجة
                    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                    idx = np.argmax(output_data)

                    # تصنيف النتيجة (خبيث: 1,4,17 | حميد: 2,5,23)
                    if idx in [1, 4, 17]:
                        res, color = t['res_m'], "#cf1322"
                    elif idx in [2, 5, 23]:
                        res, color = t['res_b'], "#389e0d"
                    else:
                        res, color = t['res_g'], "#096dd9"

                    st.markdown(f'<div class="report-card" style="border-color:{color}; color:{color};"><h2>{res}</h2><p>{t["advice"]}</p></div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Analysis Error: {e}")

st.write("---")

# --- 5. الدليل الطبي (16 نوعاً بروابط عالمية موثوقة) ---
with st.expander(f"📖 {t['guide']}"):
    tab_m, tab_b = st.tabs([t['malig'], t['benign']])
    
    with tab_m:
        m_diseases = [
            ("Basal Cell Carcinoma (BCC)", "https://www.mayoclinic.org/diseases-conditions/basal-cell-carcinoma/symptoms-causes/syc-20354487"),
            ("Squamous Cell Carcinoma (SCC)", "https://www.skincancer.org/skin-cancer-information/squamous-cell-carcinoma/"),
            ("Melanoma (الورم الميلانيني)", "https://www.mayoclinic.org/diseases-conditions/melanoma/symptoms-causes/syc-20374884"),
            ("Merkel Cell Carcinoma", "https://www.mayoclinic.org/diseases-conditions/merkel-cell-carcinoma/symptoms-causes/syc-20351030"),
            ("Kaposi Sarcoma", "https://www.cancer.org/cancer/types/kaposi-sarcoma.html"),
            ("Sebaceous Gland Carcinoma", "https://www.mayoclinic.org/diseases-conditions/sebaceous-carcinoma/symptoms-causes/syc-20352957"),
            ("Dermatofibrosarcoma Protuberans", "https://www.cancer.gov/types/soft-tissue-sarcoma/patient/dfsp-treatment-pdq"),
            ("Cutaneous Lymphoma", "https://www.clfoundation.org/cutaneous-lymphoma")
        ]
        for name, link in m_diseases:
            st.markdown(f'<div class="disease-item"><strong>{name}</strong><br><a href="{link}" target="_blank" class="link-style">{t["more"]}</a></div>', unsafe_allow_html=True)

    with tab_b:
        b_diseases = [
            ("Nevi / Moles (الشامات)", "https://www.mayoclinic.org/diseases-conditions/moles/symptoms-causes/syc-20375200"),
            ("Seborrheic Keratosis", "https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878"),
            ("Lipomas (الأورام الشحمية)", "https://www.mayoclinic.org/diseases-conditions/lipoma/symptoms-causes/syc-20374470"),
            ("Hemangiomas (الأورام الوعائية)", "https://www.mayoclinic.org/diseases-conditions/infantile-hemangioma/symptoms-causes/syc-20353177"),
            ("Dermatofibromas", "https://my.clevelandclinic.org/health/diseases/22643-dermatofibroma"),
            ("Skin Cysts (الأكياس الجلدية)", "https://www.healthline.com/health/skin-cyst"),
            ("Skin Tags (الزوائد الجلدية)", "https://www.medicalnewstoday.com/articles/67317"),
            ("Actinic Keratosis", "https://www.skincancer.org/skin-cancer-information/actinic-keratosis/")
        ]
        for name, link in b_diseases:
            st.markdown(f'<div class="disease-item" style="border-right-color:#389e0d;"><strong>{name}</strong><br><a href="{link}" target="_blank" class="link-style">{t["more"]}</a></div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown(f"<br><p style='text-align: center; color: grey; font-size: 0.8em;'>Global Skin AI System © 2026</p>", unsafe_allow_html=True)
