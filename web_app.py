import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.parse

# --- 1. القاموس اللغوي الشامل (Multi-Language Dictionary) ---
LANG_DATA = {
    "العربية": {
        "dir": "rtl", "title": "🛡️ نظام الكشف عن سلامة الجلد",
        "upload": "📥 ارفع صورة الفحص", "camera": "📸 التقاط صورة فورية",
        "analyze": "🚀 بدء التحليل الرقمي", "guide": "📚 الدليل الطبي الشامل",
        "malig": "الأورام الخبيثة", "benign": "الأورام الحميدة",
        "more": "🔗 تفاصيل وصور", "result_m": "🚨 اشتباه ورم خبيث",
        "result_b": "🔍 ورم جلدي حميد", "result_g": "🩺 حالة جلدية عامة",
        "advice": "يرجى مراجعة المختص لضمان السلامة.", "share": "مشاركة النتيجة"
    },
    "English": {
        "dir": "ltr", "title": "🛡️ Skin Safety AI System",
        "upload": "📥 Upload Scan Image", "camera": "📸 Take Instant Photo",
        "analyze": "🚀 Start Digital Analysis", "guide": "📚 Medical Guide",
        "malig": "Malignant Tumors", "benign": "Benign Tumors",
        "more": "🔗 Details & Images", "result_m": "🚨 Suspected Malignancy",
        "result_b": "🔍 Benign Skin Tumor", "result_g": "🩺 General Condition",
        "advice": "Please consult a doctor for safety.", "share": "Share Result"
    },
    "کوردی": {
        "dir": "rtl", "title": "🛡️ سیستەمی پشکنینی پێست",
        "upload": "📥 وێنەی پشکنین دابنێ", "camera": "📸 وێنەی ڕاستەوخۆ بگرە",
        "analyze": "🚀 دەستپێکردنی شیکاری", "guide": "📚 ڕێبەری پزیشکی",
        "malig": "گرێ شێرپەنجەییەکان", "benign": "گرێ بێ زیانەکان",
        "more": "🔗 زانیاری و وێنە", "result_m": "🚨 گومانی گرێی خراپ",
        "result_b": "🔍 گرێی پێستی بێ زیان", "result_g": "🩺 باری گشتی پێست",
        "advice": "تکایە سەردانی پزیشکی پسپۆڕ بکە.", "share": "ناردنی ئەنجام"
    },
    "Türkmençe": {
        "dir": "ltr", "title": "🛡️ Deri Saglygy AI Sistemasy",
        "upload": "📥 Suraty ýükle", "camera": "📸 Kamera bilen çek",
        "analyze": "🚀 Analizi başlat", "guide": "📚 Lukmançylyk Gollanmasy",
        "malig": "Howply Çişler", "benign": "Howpsuz Çişler",
        "more": "🔗 Maglumat we Suratlar", "result_m": "🚨 Howply Çiş Şübhessi",
        "result_b": "🔍 Howpsuz Deri Çişi", "result_g": "🩺 Umumy Ýagdaý",
        "advice": "Hünärmen lukmana ýüz tutuň.", "share": "Paýlaş"
    },
    "ܣܘܪܝܝܐ (Syriac)": {
        "dir": "rtl", "title": "🛡️ ܛܟܣܐ ܕܒܘܚܢܐ ܕܡܫܟܐ",
        "upload": "📥 ܐܣܩ ܨܘܪܬܐ", "camera": "📸 ܨܘܪܬܐ ܡܚܕܝܢܝܬܐ",
        "analyze": "🚀 ܫܪܝ ܒܘܚܢܐ", "guide": "📚 ܢܦܩܐ ܐܣܝܝܐ",
        "malig": "ܫܘܚܢܐ ܒܝܫܐ", "benign": "ܫܘܚܢܐ ܛܒܐ",
        "more": "🔗 ܝܬܝܪ ܝܕܥܬܐ", "result_m": "🚨 ܫܘܚܢܐ ܒܝܫܐ",
        "result_b": "🔍 ܫܘܚܢܐ ܛܒܐ", "result_g": "🩺 ܐܝܟܢܝܘܬܐ ܓܘܢܝܬܐ",
        "advice": "ܒܥܝ ܡܠܟܐ ܡܢ ܐܣܝܐ ܡܫܠܛܐ.", "share": "ܦܪܣܐ"
    }
}

# --- 2. إعدادات الصفحة والتنسيق ---
st.set_page_config(page_title="Global Skin Guard AI", layout="centered")

selected_lang = st.sidebar.selectbox("🌐 اختر اللغة / Select Language", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

st.markdown(f"""
<style>
    div[dir='{t['dir']}'] {{ text-align: {'right' if t['dir']=='rtl' else 'left'}; }}
    .report-card {{ padding: 30px; border-radius: 20px; text-align: center; border: 4px solid; margin-top: 20px; box-shadow: 0px 10px 25px rgba(0,0,0,0.1); }}
    .disease-item {{ border-right: 5px solid #0d47a1; border-left: 1px solid #eee; padding: 12px; background: #fff; margin-bottom: 10px; border-radius: 8px; }}
    .share-btn {{ display: inline-block; padding: 10px 20px; border-radius: 10px; text-decoration: none; color: white !important; font-weight: bold; margin: 5px; cursor: pointer; }}
</style>
""", unsafe_allow_html=True)

# --- 3. تحميل النموذج المخصص ---
@st.cache_resource
def load_model():
    try:
        # تأكد من وجود ملف النموذج في نفس المسار
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except: return None

interpreter = load_model()

# --- 4. واجهة المستخدم الرئيسية ---
st.markdown(f"<div dir='{t['dir']}'>", unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align: center; color: #0d47a1;'>{t['title']}</h1>", unsafe_allow_html=True)

source = st.radio("", (t['upload'], t['camera']))
input_file = st.file_uploader(t['upload'], type=["jpg", "png", "jpeg"]) if source == t['upload'] else st.camera_input(t['camera'])

if input_file:
    image = Image.open(input_file)
    st.image(image, caption=t['title'], use_container_width=True)
    
    if st.button(t['analyze']):
        if interpreter:
            with st.spinner("Analyzing..."):
                # المعالجة المسبقة للصورة
                input_details = interpreter.get_input_details()
                target_size = input_details[0]['shape'][1:3]
                img_resized = image.convert("RGB").resize(target_size)
                img_array = np.array(img_resized).astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # تنفيذ الفحص عبر النموذج
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
                idx = np.argmax(output)

                # تحديد النتيجة بناءً على الفئات (خبيث: 1,4,17 | حميد: 2,5,23)
                if idx in [1, 4, 17]:
                    res, color = t['result_m'], "#cf1322"
                elif idx in [2, 5, 23]:
                    res, color = t['result_b'], "#389e0d"
                else:
                    res, color = t['result_g'], "#096dd9"

                # عرض بطاقة النتيجة
                st.markdown(f"""
                    <div class="report-card" style="border-color: {color}; color: {color};">
                        <h2>{res}</h2>
                        <p style="font-size: 18px;">{t['advice']}</p>
                    </div>
                """, unsafe_allow_html=True)

                # نظام المشاركة الفورية
                encoded_msg = urllib.parse.quote(f"{res} - {t['advice']}")
                st.markdown(f"""
                    <div style="text-align: center; margin-top: 20px;">
                        <a class="share-btn" style="background-color: #25D366;" href="https://wa.me/?text={encoded_msg}" target="_blank">WhatsApp 💬</a>
                        <a class="share-btn" style="background-color: #EA4335;" href="mailto:?subject=Skin Report&body={encoded_msg}">Email 📧</a>
                    </div>
                """, unsafe_allow_html=True)

st.write("---")

# --- 5. الدليل الطبي في الأسفل (Expander) ---
with st.expander(f"📖 {t['guide']}"):
    tab_m, tab_b = st.tabs([t['malig'], t['benign']])
    
    with tab_m:
        m_diseases = [
            ("Melanoma / الميلانوما", "https://www.mayoclinic.org/diseases-conditions/melanoma/symptoms-causes/syc-20374884"),
            ("BCC / الخلايا القاعدية", "https://www.mayoclinic.org/diseases-conditions/basal-cell-carcinoma/symptoms-causes/syc-20354487"),
            ("SCC / الخلايا الحرشفية", "https://www.mayoclinic.org/diseases-conditions/squamous-cell-carcinoma/symptoms-causes/syc-20352480")
        ]
        for name, link in m_diseases:
            st.markdown(f'<div class="disease-item"><strong>{name}</strong><br><a href="{link}" target="_blank" style="color: #1a73e8; font-size: 13px;">{t["more"]}</a></div>', unsafe_allow_html=True)

    with tab_b:
        b_diseases = [
            ("Nevi / الشامات", "https://www.mayoclinic.org/diseases-conditions/moles/symptoms-causes/syc-20375200"),
            ("Lipomas / أورام شحمية", "https://www.mayoclinic.org/diseases-conditions/lipoma/symptoms-causes/syc-20374470"),
            ("Seborrheic Keratosis", "https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878")
        ]
        for name, link in b_diseases:
            st.markdown(f'<div class="disease-item" style="border-right-color:#389e0d;"><strong>{name}</strong><br><a href="{link}" target="_blank" style="color: #1a73e8; font-size: 13px;">{t["more"]}</a></div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown(f"<br><p style='text-align: center; color: grey; font-size: 0.8em;'>Skin Safety Detection System © 2026</p>", unsafe_allow_html=True)
