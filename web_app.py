import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os
import requests

# --- 1. إعدادات الصفحة والملفات ---
st.set_page_config(page_title="Skin AI Expert System", layout="wide")

DRIVE_FILE_ID = '135lZpgsipHNk2IZBo6H4lZZ9WzVizLqb'
MODEL_PATH = "skin_expert_hybrid_24ch.h5"

# --- 2. نظام اللغات (مترجم بالكامل) ---
LANG_DATA = {
    "العربية": {
        "dir": "rtl",
        "title": "نظام التشخيص الذكي المتطور لأمراض الجلد",
        "upload_label": "📥 ارفع صورة الفحص (Dataset Test)",
        "btn_analyze": "🔍 بدء التحليل والفحص البرمجي",
        "advice": "⚠️ تنبيه طبي: هذا النظام أداة برمجية استرشادية تعتمد على عتبات ثقة محددة برمجياً.",
        "result_text": "نتيجة التشخيص المعتمدة:",
        "confidence": "نسبة اليقين:",
        "inconclusive": "تحليل غير حاسم: لم تتجاوز العتبة المطلوبة أو الصورة غير واضحة.",
        "malig_title": "آفة جلدية خبيثة (Malignant)",
        "benign_title": "حالة جلدية حميدة (Benign)",
        "others_title": "عدوى أو التهاب جلدي (Infections/Others)",
        "malig_note": "⚠️ خطورة عالية: يرجى مراجعة المختص فوراً للفحص السريري.",
        "benign_note": "✅ حالة مستقرة: المؤشرات تدل على طبيعة حميدة.",
        "others_note": "🔎 حالة عدوى: يرجى استشارة الطبيب للعلاج."
    },
    "English": {
        "dir": "ltr",
        "title": "Advanced Skin AI Diagnostic System",
        "upload_label": "📥 Upload Scan Image (Dataset Test)",
        "btn_analyze": "🔍 Start Analysis & Threshold Check",
        "advice": "⚠️ Medical Note: This tool uses programmed confidence thresholds for safety.",
        "result_text": "Validated Diagnosis:",
        "confidence": "Confidence Level:",
        "inconclusive": "Inconclusive: Confidence below threshold or image is unclear.",
        "malig_title": "Malignant Skin Lesion",
        "benign_title": "Benign Skin Condition",
        "others_title": "Skin Infection / Others",
        "malig_note": "⚠️ High Risk: Please consult a specialist immediately.",
        "benign_note": "✅ Stable: Indicators suggest a benign nature.",
        "others_note": "🔎 Infection: Please consult a doctor for treatment."
    }
}

# --- 3. تحميل الموديل الهجين ---
@st.cache_resource
def load_hybrid_model():
    input_layer = Input(shape=(224, 224, 3))
    b1 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False)(input_layer)
    b2 = tf.keras.applications.MobileNetV2(weights=None, include_top=False)(input_layer)
    merged = Concatenate()([GlobalAveragePooling2D()(b1), GlobalAveragePooling2D()(b2)])
    d = Dense(512, activation='relu')(merged)
    out = Dense(24, activation='softmax')(Dropout(0.4)(d))
    model = Model(inputs=input_layer, outputs=out)
    
    if not os.path.exists(MODEL_PATH):
        with st.spinner("جاري جلب أوزان الموديل..."):
            URL = "https://docs.google.com/uc?export=download"
            session = requests.Session()
            r = session.get(URL, params={'id': DRIVE_FILE_ID}, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(32768):
                    if chunk: f.write(chunk)
    model.load_weights(MODEL_PATH)
    return model

diag_model = load_hybrid_model()

# --- 4. واجهة المستخدم ---
selected_lang = st.sidebar.selectbox("Select Language / اختر اللغة", list(LANG_DATA.keys()))
T = LANG_DATA[selected_lang]

st.markdown(f"<div dir='{T['dir']}' style='text-align:center;'><h1 style='color:#1E3A8A;'>{T['title']}</h1></div>", unsafe_allow_html=True)

uploaded = st.file_uploader(T['upload_label'], type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, use_container_width=True)
    with col2:
        if st.button(T['btn_analyze']):
            # معالجة الصورة
            img_res = cv2.resize(np.array(img), (224, 224))
            d_in = np.expand_dims(img_res, axis=0) / 255.0
            preds = diag_model.predict(d_in)[0]
            idx = np.argmax(preds)
            score = np.max(preds)
            
            # --- منطق الاستبعاد والعتبات الجديدة ---
            # 1. تعريف المجموعات (بناءً على مجلدات Dataset الخاصة بك)
            MALIGNANT_IDS = [1, 11] # bcc, mel
            BENIGN_IDS = [0, 2, 5, 9, 13, 14, 16, 20] # nv, akiec, bkl, df...
            
            # 2. العتبات التي طلبتها
            THRESH_MALIG = 0.40
            THRESH_BENIGN = 0.45
            
            f_label, f_color, f_note = T['inconclusive'], "#8E8E93", ""
            
            # 3. التحقق البرمجي (الفلتر)
            # تم استثناء "الأرشيف 23" هنا لإلغاء ظهوره المتكرر
            if idx in MALIGNANT_IDS:
                if score >= THRESH_MALIG:
                    f_label, f_color, f_note = T['malig_title'], "#FF3B30", T['malig_note']
            elif idx in BENIGN_IDS:
                if score >= THRESH_BENIGN:
                    f_label, f_color, f_note = T['benign_title'], "#34C759", T['benign_note']
            
            # عرض النتيجة المصفاة
            st.markdown(f"""
            <div dir='{T['dir']}' style="padding:30px; border-radius:20px; border:10px solid {f_color}; text-align:center; background:white;">
                <h2 style="color:{f_color};">{T['result_text']}</h2>
                <h1 style="color:{f_color};">{f_label}</h1>
                <div style="background:{f_color}10; padding:10px; border-radius:10px; margin-top:10px;">
                    <strong>{T['confidence']} {score*100:.2f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"<div style='text-align:center; color:#888;'><small>Hybrid AI v8.6 | Mosul 2026</small></div>", unsafe_allow_html=True)
