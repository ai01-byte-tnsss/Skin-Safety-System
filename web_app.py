import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import gdown

# --- 1. إعدادات الهوية البصرية والواجهة ---
st.set_page_config(page_title="Skin AI Expert System", page_icon="🧬", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: center; direction: rtl; }
    .stButton>button { width: 100%; border-radius: 12px; height: 3.5em; background-color: #1E3A8A; color: white; font-weight: bold; font-size: 1.1em; }
    .guide-benign { background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-right: 5px solid #2e7d32; margin-bottom: 10px; text-align: right; }
    .guide-malignant { background-color: #ffebee; padding: 15px; border-radius: 10px; border-right: 5px solid #c62828; margin-bottom: 10px; text-align: right; }
    .result-card { padding: 25px; border-radius: 20px; background-color: #ffffff; box-shadow: 0 10px 25px rgba(0,0,0,0.1); border: 1px solid #eee; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. نظام اللغات العشر المتكامل ---
languages = {
    "العربية": {"title": "نظام الفحص الذكي للجلد", "sub": "تحليل الأنسجة بالذكاء الاصطناعي", "upload": "ارفع صورة أو التقطها بالكاميرا", "btn": "بدء تحليل العينة", "guide": "الدليل الطبي المرجعي"},
    "English": {"title": "Skin AI Expert System", "sub": "AI Tissue Analysis", "upload": "Upload or Capture Image", "btn": "Start Analysis", "guide": "Medical Reference Guide"},
    "Français": {"title": "Système IA de la Peau", "sub": "Analyse tissulaire par IA", "upload": "Télécharger ou Capturer", "btn": "Analyser l'échantillon", "guide": "Guide Médical"},
    "Español": {"title": "Sistema de IA de Piel", "sub": "Análisis de tejidos", "upload": "Subir o Capturar imagen", "btn": "Analizar muestra", "guide": "Guía Médica"},
    "Deutsch": {"title": "Haut-KI-System", "sub": "Gewebe-KI-Analyse", "upload": "Bild hochladen oder aufnehmen", "btn": "Probe analysieren", "guide": "Medizinischer Leitfaden"},
    "Türkçe": {"title": "Cilt YZ Sistemi", "sub": "Doku Analizi", "upload": "Resim Yükle veya Çek", "btn": "Numuneyi Analiz Et", "guide": "Tıbbi Rehber"},
    "Русский": {"title": "Система ИИ Кожи", "sub": "Анализ тканей", "upload": "Загрузить или Снять", "btn": "Анализ образца", "guide": "Мед Справочник"},
    "中文": {"title": "皮肤人工智能系统", "sub": "组织分析", "upload": "上传或拍摄图片", "btn": "分析样本", "guide": "医学指南"},
    "Português": {"title": "Sistema IA de Pele", "sub": "Análise de tecidos", "upload": "Enviar ou Capturar imagem", "btn": "Analisar amostra", "guide": "Guia Médico"},
    "हिन्दी": {"title": "त्वचा एआई विशेषज्ञ", "sub": "एआई विश्लेषण", "upload": "छवि अपलोड या कैप्चر करें", "btn": "नमूना विश्लेषण", "guide": "चिकित्सा गाइड"}
}

sel_lang = st.sidebar.selectbox("🌐 اختر اللغة / Language", list(languages.keys()))
text = languages[sel_lang]

# --- 3. المصفوفة الطبية (مطابقة لترتيب مجلدات balanced_skin_dataset) ---
# الترتيب: akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6
CLASS_MAP = {
    0: {"name": "التقرن الضوئي (AK)", "type": "خبيث جزئياً (Pre-cancerous)", "color": "#c62828"},
    1: {"name": "سرطان الخلايا القاعدية (BCC)", "type": "خبيث (Malignant)", "color": "#c62828"},
    2: {"name": "آفات التقرن الحميدة (BKL)", "type": "حميد (Benign)", "color": "#2e7d32"},
    3: {"name": "الأورام الليفية الجلدية (DF)", "type": "حميد (Benign)", "color": "#2e7d32"},
    4: {"name": "الميلانوما (MEL)", "type": "خبيث جداً (Highly Malignant)", "color": "#c62828"},
    5: {"name": "الشامات والوحمات (NV)", "type": "حميد (Benign)", "color": "#2e7d32"},
    6: {"name": "الآفات الوعائية (VASC)", "type": "حميد (Benign)", "color": "#2e7d32"}
}

# --- 4. تحميل النموذج من Google Drive ومعالجة الصور ---
@st.cache_resource
def load_ai_engine():
    # الرابط الخاص بملف الأوزان الخاص بك
    file_id = '1lMGCojHeGupFunhxX5GnLOiUgxWbbRC5'
    local_h5 = "expert_model_v7.h5"
    
    if not os.path.exists(local_h5):
        try:
            # تحميل الملف المضغوط من الدرايف
            gdown.download(f'https://drive.google.com/uc?id={file_id}', "model.zip", quiet=False)
            with zipfile.ZipFile("model.zip", 'r') as z:
                for f in z.namelist():
                    if f.endswith('.h5'):
                        with open(local_h5, "wb") as out: out.write(z.read(f))
                        break
        except Exception as e:
            st.error(f"خطأ في التحميل: {e}")
            return None

    # بناء الهيكل المتوافق مع الدوال الرياضية (التلافيف، ReLU، سوفت ماكس)
    try:
        base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        output = tf.keras.layers.Dense(7, activation='softmax')(x) # 7 فئات
        model = tf.keras.Model(inputs=base.input, outputs=output)
        model.load_weights(local_h5, by_name=True, skip_mismatch=True)
        return model
    except:
        return tf.keras.models.load_model(local_h5, compile=False)

def preprocess_image(image):
    # دالة تغيير الحجم والتطبيع (Normalization)
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0  # تحويل القيم لـ [0, 1]
    return np.expand_dims(img, axis=0)

model = load_ai_engine()

# --- 5. واجهة التشخيص ---
st.title(text["title"])
st.write(f"#### {text['sub']}")

src = st.radio("", ["Upload Image", "Camera Input"], horizontal=True)
file = st.file_uploader(text["upload"], type=["jpg","png","jpeg"]) if "Upload" in src else st.camera_input("Scan")

if file:
    img = Image.open(file)
    st.image(img, width=280)
    
    if st.button(text["btn"]):
        if model:
            with st.spinner("⏳ Analyzing Tissues..."):
                ready_img = preprocess_image(img)
                preds = model.predict(ready_img)[0]
                idx = np.argmax(preds) # دالة Argmax
                res = CLASS_MAP[idx]
                
                st.markdown(f"""
                <div class="result-card">
                    <h2 style="color:{res['color']};">{res['name']}</h2>
                    <p style="font-size:1.2em;"><b>التصنيف الطبي:</b> {res['type']}</p>
                    <hr>
                    <p><b>دقة دالة Softmax:</b> {preds[idx]:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("فشل تحميل النموذج، يرجى التحقق من اتصال الإنترنت.")

# --- 6. الدليل المرجعي الملون ---
st.markdown("---")
with st.expander(f"📖 {text['guide']}"):
    st.markdown("""
        <div class="guide-benign">
            <h4>🟢 الآفات الحميدة (Benign)</h4>
            <p>تشمل (NV, BKL, DF, VASC). هي تغيرات جلدية غير سرطانية، يفضل مراقبتها دورياً لضمان عدم تغير شكلها.</p>
        </div>
        <div class="guide-malignant">
            <h4>🔴 الآفات الخبيثة (Malignant)</h4>
            <p>تشمل (MEL, BCC, AK). تتطلب استشارة فورية من طبيب الجلدية لعمل الفحوصات السريرية اللازمة.</p>
        </div>
    """, unsafe_allow_html=True)

st.caption("Developed for Medical Graduation Research © 2026")
