import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os
import requests

# --- 1. إعدادات الصفحة العامة ---
st.set_page_config(page_title="Skin AI Expert System", layout="wide")

DRIVE_FILE_ID = '135lZpgsipHNk2IZBo6H4lZZ9WzVizLqb'
MODEL_PATH = "skin_expert_hybrid_24ch.h5"

# --- 2. نظام الترجمة الشامل (عربي / English) ---
LANG_DATA = {
    "العربية": {
        "dir": "rtl",
        "title": "نظام التشخيص الذكي المتطور لأمراض الجلد",
        "upload_label": "📥 ارفع صورة الفحص (Dataset Test)",
        "btn_analyze": "🔍 بدء التحليل والفحص البرمجي",
        "advice": "⚠️ تنبيه طبي: هذا النظام أداة برمجية استرشادية تعتمد على عتبات ثقة محددة برمجياً.",
        "result_text": "نتيجة التشخيص المعتمدة:",
        "confidence": "نسبة اليقين:",
        "inconclusive": "تحليل غير حاسم: النسبة لم تتجاوز عتبة الأمان المطلوبة (0.40/0.45).",
        "malig_title": "آفة جلدية خبيثة (Malignant)",
        "benign_title": "حالة جلدية حميدة (Benign)",
        "others_title": "عدوى أو التهاب جلدي (Infections/Others)",
        "malig_note": "⚠️ خطورة عالية: يرجى مراجعة المختص فوراً للفحص السريري.",
        "benign_note": "✅ حالة مستقرة: المؤشرات تدل على طبيعة حميدة أو نمو غير سرطانية.",
        "others_note": "🔎 حالة عدوى: يرجى استشارة الطبيب للعلاج الموضعي المناسب.",
        "guide_title": "📖 الدليل المرجعي والمصادر العالمية",
        "link_cancer": "🔗 مصدر عالمي: سرطان الجلد (Mayo Clinic)",
        "link_others": "🔗 مصدر عالمي: الأمراض الجلدية (NHS)"
    },
    "English": {
        "dir": "ltr",
        "title": "Advanced Skin AI Diagnostic System",
        "upload_label": "📥 Upload Scan Image (Dataset Test)",
        "btn_analyze": "🔍 Start Analysis & Threshold Check",
        "advice": "⚠️ Medical Note: This tool uses programmed confidence thresholds for safety.",
        "result_text": "Validated Diagnosis:",
        "confidence": "Confidence Level:",
        "inconclusive": "Inconclusive: Confidence below programmed threshold (0.40/0.45).",
        "malig_title": "Malignant Skin Lesion",
        "benign_title": "Benign Skin Condition",
        "others_title": "Skin Infection / Others",
        "malig_note": "⚠️ High Risk: Please consult a specialist immediately for clinical exam.",
        "benign_note": "✅ Stable: Indicators suggest a benign nature or non-cancerous growth.",
        "others_note": "🔎 Infection: Please consult a doctor for appropriate treatment.",
        "guide_title": "📖 Medical Reference & Global Sources",
        "link_cancer": "🔗 Global Source: Skin Cancer (Mayo Clinic)",
        "link_others": "🔗 Global Source: Skin Conditions (NHS)"
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
        with st.spinner("جاري استيراد أوزان الذكاء الاصطناعي..."):
            URL = "https://docs.google.com/uc?export=download"
            session = requests.Session()
            r = session.get(URL, params={'id': DRIVE_FILE_ID}, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(32768):
                    if chunk: f.write(chunk)
    model.load_weights(MODEL_PATH)
    return model

diag_model = load_hybrid_model()

# --- 4. واجهة المستخدم والتفاعل ---
selected_lang = st.sidebar.selectbox("Select Language / اختر اللغة", list(LANG_DATA.keys()))
T = LANG_DATA[selected_lang]

st.markdown(f"<div dir='{T['dir']}' style='text-align:center;'><h1 style='color:#1E3A8A;'>{T['title']}</h1></div>", unsafe_allow_html=True)
st.warning(T['advice'])

uploaded = st.file_uploader(T['upload_label'], type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, use_container_width=True)
    with col2:
        if st.button(T['btn_analyze']):
            # --- معالجة الصورة وتحويلها لمصفوفة ---
            img_res = cv2.resize(np.array(img), (224, 224))
            d_in = np.expand_dims(img_res, axis=0) / 255.0
            preds = diag_model.predict(d_in)[0]
            idx = np.argmax(preds)
            score = np.max(preds)
            
            # --- تطبيق العتبات المطلوبة كشرط برمجتي (Threshold Logic) ---
            MALIGNANT_IDS = [1, 11, 23] # الأصناف الخبيثة
            BENIGN_IDS = [0, 2, 5, 9, 13, 14, 16, 20] # الأصناف الحميدة
            
            # العتبات الجديدة التي طلبتها
            THRESHOLD_MALIG = 0.40
            THRESHOLD_BENIGN = 0.45
            
            final_label, final_color, final_note = T['inconclusive'], "#8E8E93", ""
            
            # الفحص البرمجي بناءً على العتبات
            if idx in MALIGNANT_IDS:
                if score >= THRESHOLD_MALIG: # شرط المرور للخبيث
                    final_label, final_color, final_note = T['malig_title'], "#FF3B30", T['malig_note']
            elif idx in BENIGN_IDS:
                if score >= THRESHOLD_BENIGN: # شرط المرور للحميد
                    final_label, final_color, final_note = T['benign_title'], "#34C759", T['benign_note']
            else:
                if score >= 0.35: # شرط تلقائي للحالات الأخرى
                    final_label, final_color, final_note = T['others_title'], "#FF9500", T['others_note']

            # عرض النتيجة النهائية بناءً على تحقق الشروط
            st.markdown(f"""
            <div dir='{T['dir']}' style="padding:30px; border-radius:20px; border:10px solid {final_color}; text-align:center; background:white; box-shadow: 0px 4px 15px rgba(0,0,0,0.1);">
                <h2 style="color:{final_color};">{T['result_text']}</h2>
                <h1 style="color:{final_color}; font-size:2.5em;">{final_label}</h1>
                <p style='font-size:1.1em; color:#333;'>{final_note}</p>
                <div style="background:{final_color}15; padding:10px; border-radius:10px; margin-top:10px;">
                    <strong>{T['confidence']} {score*100:.2f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- 5. الدليل المرجعي والمصادر (مترجم) ---
st.markdown("---")
with st.expander(T['guide_title']):
    st.markdown(f"<div dir='{T['dir']}'>", unsafe_allow_html=True)
    st.write(f"### {T['malig_title']}")
    st.markdown(f"[{T['link_cancer']}](https://www.mayoclinic.org/diseases-conditions/skin-cancer/symptoms-causes/syc-20377605)")
    st.write("---")
    st.write(f"### {T['benign_title']}")
    st.write(f"### {T['others_title']}")
    st.markdown(f"[{T['link_others']}](https://www.nhs.uk/conditions/skin-conditions/)")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(f"<div style='text-align:center; color:#888; margin-top:50px;'><small>Skin AI v8.5 | Final Build | Mosul-Nineveh 2026</small></div>", unsafe_allow_html=True)
