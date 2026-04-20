import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. إعدادات الواجهة واللغات ---
st.set_page_config(page_title="Skin AI Expert System", layout="wide")

LANG_DATA = {
    "العربية": {
        "dir": "rtl", "title": "نظام خبير الذكاء الاصطناعي لتشخيص الجلد",
        "upload": "📥 ارفع صورة الفحص", "btn": "🔍 بدء التحليل العميق",
        "advice": "⚠️ تنبيه: نظام أكاديمي استرشادي لجامعة الموصل."
    },
    "English": {
        "dir": "ltr", "title": "Skin AI Expert System",
        "upload": "📥 Upload Image", "btn": "🔍 Start Analysis",
        "advice": "⚠️ Note: Academic guidance tool."
    }
}

# --- 2. الدليل الطبي المرجعي (10 أنواع) مع الأوزان التصحيحية ---
MEDICAL_DB = {
    0: {"n": "Melanoma (ميلانوما)", "c": "#FF3B30", "s": "🚨 خبيث جداً", "w": 1.4, "d": "ورم صبغي عدواني."},
    1: {"n": "Melanocytic Nevi (وحمة صبغية)", "c": "#34C759", "s": "✅ حميد", "w": 0.6, "d": "شامة طبيعية آمنة."},
    2: {"n": "Basal Cell Carcinoma (BCC)", "c": "#FF9500", "s": "🚨 خبيث", "w": 0.5, "d": "سرطان قاعدي ينمو ببطء."},
    3: {"n": "Actinic Keratosis (AK)", "c": "#AF52DE", "s": "⚠️ ما قبل سرطاني", "w": 1.1, "d": "تلف شمسي قد يتطور."},
    4: {"n": "Benign Keratosis (BKL)", "c": "#5856D6", "s": "✅ حميد", "w": 0.9, "d": "زوائد غير سرطانية."},
    5: {"n": "Dermatofibroma (DF)", "c": "#007AFF", "s": "✅ حميد", "w": 1.2, "d": "كتلة صلبة صغيرة."},
    6: {"n": "Vascular Lesions (VASC)", "c": "#5AC8FA", "s": "✅ حميد", "w": 1.2, "d": "آفات وعائية تجمع شعيرات."},
    7: {"n": "Squamous Cell Carcinoma", "c": "#FF2D55", "s": "🚨 خبيث", "w": 1.3, "d": "سرطان الخلايا الحرشفية."},
    8: {"n": "Psoriasis (الصدفية)", "c": "#4CD964", "s": "🔍 حالة جلدية", "w": 1.0, "d": "التهاب مزمن وقشور فضية."},
    9: {"n": "Eczema (الأكزيما)", "c": "#FFCC00", "s": "🔍 حالة جلدية", "w": 1.1, "d": "التهاب جلدي وحكة وجفاف."}
}

# --- 3. بناء المحرك (حل مشكلة ValueError و الأوزان) ---
@st.cache_resource
def load_expert_engine():
    # استخدام Input layer صريحة لحل مشاكل Keras Functional API
    inp = Input(shape=(224, 224, 3))
    
    b1 = EfficientNetB0(weights=None, include_top=False)(inp)
    b2 = MobileNetV2(weights=None, include_top=False)(inp)
    
    g1 = GlobalAveragePooling2D()(b1)
    g2 = GlobalAveragePooling2D()(b2)
    
    merged = Concatenate()([g1, g2])
    out = Dense(10, activation='softmax')(Dropout(0.4)(Dense(512, activation='relu')(merged)))
    
    model = Model(inputs=inp, outputs=out)
    
    h5_path = "skin_expert_master.h5"
    ready = False
    if os.path.exists(h5_path):
        model.load_weights(h5_path)
        ready = True
    return model, ready

diag_model, is_ready = load_expert_engine()

# --- 4. واجهة المستخدم ---
lang_key = st.selectbox("🌐 Language", list(LANG_DATA.keys()))
ui = LANG_DATA[lang_key]
st.title(ui['title'])

if not is_ready:
    st.error("❌ ملف الأوزان 'skin_expert_master.h5' غير موجود!")

up_file = st.file_uploader(ui['upload'], type=["jpg", "png", "jpeg"])

if up_file and is_ready:
    img = Image.open(up_file).convert('RGB')
    st.image(img, width=400)
    
    if st.button(ui['btn']):
        with st.spinner("⏳ Analyzing..."):
            img_cv = cv2.resize(np.array(img), (224, 224))
            
            # حل مشكلة AttributeError: استخدام White Balance يدوي بدلاً من xphoto
            avg = np.mean(img_cv)
            proc = img_cv.astype(np.float32)
            for i in range(3):
                proc[:, :, i] = np.clip(img_cv[:, :, i] * (avg / np.mean(img_cv[:, :, i])), 0, 255)
            
            # تقليل حد CLAHE لمنع الانحياز لـ BCC
            lab = cv2.cvtColor(proc.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8)).apply(l)
            final = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

            # التنبوء والمعايرة
            inp_tensor = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(final, axis=0))
            preds = diag_model.predict(inp_tensor)[0]
            
            # تطبيق مصفوفة الأوزان (السر في فصل الأنواع)
            cal_w = np.array([v['w'] for v in MEDICAL_DB.values()])
            final_idx = np.argmax(preds * cal_w)
            
            res = MEDICAL_DB[final_idx]
            st.markdown(f"""
                <div style="border: 8px solid {res['c']}; padding: 20px; text-align: center; background: white;">
                    <h1 style="color: {res['c']};">{res['n']}</h1>
                    <h3>{res['s']}</h3><p>{res['d']}</p>
                </div>
            """, unsafe_allow_html=True)
