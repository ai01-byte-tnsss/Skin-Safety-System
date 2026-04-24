import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import gdown

# --- 1. إعدادات الصفحة والتصميم ---
st.set_page_config(page_title="Skin AI System", page_icon="🔍", layout="centered")

# إضافة CSS لتحسين مظهر القوائم الملونة والدليل الطبي
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: center; }
    .stSelectbox, .stExpander { border-radius: 10px; background-color: #f8f9fa; }
    .guide-benign { background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-right: 5px solid #2e7d32; margin-bottom: 10px; text-align: right; }
    .guide-malignant { background-color: #ffebee; padding: 15px; border-radius: 10px; border-right: 5px solid #c62828; margin-bottom: 10px; text-align: right; }
    .report-card { padding: 25px; border-radius: 15px; background-color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. نظام اللغات (10 لغات عالمية) ---
languages = {
    "العربية": {"title": "نظام الفحص الذكي للجلد", "upload": "ارفع صورة الفحص أو استخدم الكاميرا", "btn": "تحليل العينة", "lang_label": "اختر اللغة / Select Language"},
    "English": {"title": "Skin AI Diagnostic System", "upload": "Upload Image or Use Camera", "btn": "Analyze Sample", "lang_label": "Select Language"},
    "Français": {"title": "Système d'IA Cutanée", "upload": "Télécharger une image", "btn": "Analyser", "lang_label": "Choisir la langue"},
    "Español": {"title": "Sistema IA de Piel", "upload": "Subir imagen", "btn": "Analizar", "lang_label": "Seleccionar idioma"},
    "Deutsch": {"title": "Haut-KI-System", "upload": "Bild hochladen", "btn": "Analysieren", "lang_label": "Sprache wählen"},
    "Türkçe": {"title": "Cilt Yapay Zeka Sistemi", "upload": "Fotoğraf Yükle", "btn": "Analiz Et", "lang_label": "Dil Seçin"},
    "中文": {"title": "皮肤人工智能系统", "upload": "上传图片", "btn": "开始分析", "lang_label": "选择语言"},
    "Русский": {"title": "Система ИИ Кожи", "upload": "Загрузить фото", "btn": "Анализировать", "lang_label": "Выберите язык"},
    "Português": {"title": "Sistema de IA da Pele", "upload": "Enviar imagem", "btn": "Analisar", "lang_label": "Selecionar idioma"},
    "हिन्दी": {"title": "त्वचा एआई प्रणाली", "upload": "छवि अपलोड करें", "btn": "विश्लेषण करें", "lang_label": "भाषा चुनें"}
}

selected_lang = st.sidebar.selectbox("🌐 Language / اللغة", list(languages.keys()))
content = languages[selected_lang]

# --- 3. المصفوفة الطبية المرتبة (مطابقة لمجلداتك السبعة) ---
CLASS_MAP = {
    0: {"ar": "التقرن الضوئي (AK)", "en": "Actinic Keratosis", "type": "خبيث جزئياً / متابعة"},
    1: {"ar": "سرطان الخلايا القاعدية (BCC)", "en": "Basal Cell Carcinoma", "type": "خبيث - يتطلب تدخل"},
    2: {"ar": "آفات التقرن الحميدة (BKL)", "en": "Benign Keratosis", "type": "حميد - غير مقلق"},
    3: {"ar": "الأورام الليفية الجلدية (DF)", "en": "Dermatofibroma", "type": "حميد"},
    4: {"ar": "الميلانوما (MEL)", "en": "Melanoma", "type": "خبيث جداً"},
    5: {"ar": "الشامات والوحمات (NV)", "en": "Melanocytic Nevi", "type": "حميد - طبيعي"},
    6: {"ar": "الآفات الوعائية (VASC)", "en": "Vascular Lesions", "type": "حميد"}
}

# --- 4. تحميل المحرك وتحضير الصورة ---
@st.cache_resource
def load_model():
    f_id = '1lMGCojHeGupFunhxX5GnLOiUgxWbbRC5'
    path = "final_expert_model.h5"
    if not os.path.exists(path):
        gdown.download(f'https://drive.google.com/uc?id={f_id}', "model.zip", quiet=False)
        with zipfile.ZipFile("model.zip", 'r') as z:
            for f in z.namelist():
                if f.endswith('.h5'):
                    with open(path, "wb") as out: out.write(z.read(f))
                    break
    model = tf.keras.models.load_model(path, compile=False)
    return model

def prep_image(img):
    img = np.array(img.convert('RGB'))
    img = cv2.resize(img, (224, 224))
    return (img.astype(np.float32) / 255.0)[np.newaxis, ...]

model = load_model()

# --- 5. الواجهة الرسومية ---
st.title(content["title"])
st.warning("⚠️ تنبيه طبي: هذا النظام أداة برمجية استرشادية تعتمد على الذكاء الاصطناعي، ولا يغني عن زيارة الطبيب المختص.")

# خيارات الرفع (مثل الصورة المطلوبة)
source = st.radio("", ["Upload Image", "Use Camera"], horizontal=True)
file = st.file_uploader(content["upload"], type=["jpg", "png", "jpeg"]) if source == "Upload Image" else st.camera_input("Capture")

if file:
    img = Image.open(file)
    st.image(img, width=300)
    if st.button(content["btn"]):
        processed = prep_image(img)
        preds = model.predict(processed)[0]
        idx = np.argmax(preds)
        res = CLASS_MAP[idx]
        
        st.markdown(f"""
        <div class="report-card">
            <h3>اسم المرض المتوقع:</h3>
            <h2 style="color:#1E3A8A;">{res['ar']} <br> <small>({res['en']})</small></h2>
            <hr>
            <h4>تصنيف الحالة:</h4>
            <p style="font-size:1.2em; font-weight:bold;">{res['type']}</p>
            <p>دقة المطابقة: {preds[idx]:.2%}</p>
        </div>
        """, unsafe_allow_html=True)

# --- 6. الدليل الطبي المرجعي (منسدل وملون) ---
st.markdown("---")
with st.expander("📖 الدليل الطبي المرجعي لآفات الجلد"):
    st.markdown("""
        <div class="guide-benign">
            <h4 style="color:#2e7d32;">🟢 الآفات الحميدة (Benign)</h4>
            <p>تشمل الشامات العادية (NV)، والتقرن الدهني (BKL)، والأورام الليفية. عادة لا تشكل خطراً ولكن يفضل مراقبة أي تغير في الحجم أو اللون.</p>
        </div>
        <div class="guide-malignant">
            <h4 style="color:#c62828;">🔴 الآفات الخبيثة (Malignant)</h4>
            <p>تشمل الميلانوما (MEL) وسرطان الخلايا القاعدية (BCC). تتطلب هذه الحالات استشارة طبية فورية وعمل خزعة للتأكد، حيث أن التدخل المبكر يرفع نسب الشفاء.</p>
        </div>
    """, unsafe_allow_html=True)

st.caption("Skin Diagnostic AI System v2.0 | Powered by CNN")
