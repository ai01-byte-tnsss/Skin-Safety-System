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
st.set_page_config(page_title="Skin AI System - نظام فحص الجلد", layout="wide")

# إعدادات الملف السحابي (الهجين فقط)
DRIVE_FILE_ID = '135lZpgsipHNk2IZBo6H4lZZ9WzVizLqb'
MODEL_PATH = "skin_expert_hybrid_24ch.h5"

# --- 2. دالة تحميل الموديل من Google Drive ---
def download_model_from_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    if token:
        response = session.get(URL, params={'id': id, 'confirm': token}, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk: f.write(chunk)

# --- 3. إعدادات اللغات العشر والواجهة ---
LANGS_CONFIG = {
    "العربية": "rtl", "English": "ltr", "Français": "ltr", "Deutsch": "ltr",
    "Español": "ltr", "Türkçe": "ltr", "Русский": "ltr", "中文": "ltr",
    "हिन्दी": "ltr", "Kurdî": "rtl"
}

UI_LABELS = {
    "العربية": {
        "title": "نظام التشخيص الذكي لأمراض الجلد",
        "upload": "📥 ارفع صورة الفحص أو استخدم الكاميرا",
        "btn": "🔍 بدء التحليل الفوري",
        "advice": "⚠️ تنبيه طبي: هذا النظام أداة برمجية استرشادية تعتمد على الذكاء الاصطناعي، ولا يغني عن زيارة الطبيب المختص.",
        "guide_title": "📖 الدليل المرجعي لتصنيفات النظام",
        "invalid": "❌ الصورة المرفوعة لا تبدو كفحص جلدي واضح."
    },
    "English": {
        "title": "Skin AI Diagnostic System",
        "upload": "📥 Upload scan or use camera",
        "btn": "🔍 Start Analysis",
        "advice": "⚠️ Medical Note: This AI tool is for guidance only.",
        "guide_title": "📖 Medical Reference Guide",
        "invalid": "❌ Invalid image format for skin scan."
    }
}

# --- 4. منطق التصنيف الثلاثي وعتبات الثقة ---
def analyze_condition(idx, score):
    percent = score * 100
    
    # مصفوفة التقسيم (بناءً على الـ 24 صنفاً)
    MALIGNANT_IDS = [1, 11, 23] 
    BENIGN_IDS = [0, 2, 5, 9, 13, 14, 16, 20]
    # البقية تعتبر عدوى أو حالات أخرى

    # دالة جلب الاسم الطبي الدقيق (للعرض الداخلي)
    DIAG_NAMES = {
        1: "Basal Cell Carcinoma (BCC)", 11: "Melanoma", 23: "Malignant Tissue Analysis",
        0: "Acne/Rosacea", 2: "Atopic Dermatitis", 5: "Eczema", 14: "Psoriasis",
        18: "Tinea/Fungal", 12: "Nail Fungus", 19: "Urticaria"
    }
    spec_name = DIAG_NAMES.get(idx, "Skin Condition")

    # تطبيق العتبات (30% للخبيث، 35% للحميد، 45% للأخرى)
    if idx in MALIGNANT_IDS:
        if percent >= 30:
            return f"آفة خبيثة - {spec_name}", "#FF3B30", "⚠️ حالة عالية الخطورة: يجب استشارة طبيب جراح أو أخصائي جلدية فوراً.", percent
    elif idx in BENIGN_IDS:
        if percent >= 35:
            return f"حالة حميدة - {spec_name}", "#34C759", "✅ حالة مستقرة: المؤشرات تدل على طبيعة حميدة، تابع الحالة بانتظام.", percent
    else:
        if percent >= 45:
            return f"عدوى/حساسية - {spec_name}", "#FF9500", "🔎 حالة عدوى: يرجى مراجعة الصيدلي أو الطبيب لوصف العلاج الموضعي المناسب.", percent
    
    return "تحليل غير حاسم", "#8E8E93", "المعطيات غير كافية للتشخيص الدقيق، يرجى تحسين الإضاءة وإعادة التصوير.", percent

# --- 5. تحميل الموديل الهجين ---
@st.cache_resource
def load_final_model():
    input_layer = Input(shape=(224, 224, 3))
    b1 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False)(input_layer)
    b2 = tf.keras.applications.MobileNetV2(weights=None, include_top=False)(input_layer)
    merged = Concatenate()([GlobalAveragePooling2D()(b1), GlobalAveragePooling2D()(b2)])
    d = Dense(512, activation='relu')(merged)
    out = Dense(24, activation='softmax')(Dropout(0.4)(d))
    model = Model(inputs=input_layer, outputs=out)
    
    if not os.path.exists(MODEL_PATH):
        with st.spinner("جاري استرداد نظام الهجين السحابي..."):
            download_model_from_drive(DRIVE_FILE_ID, MODEL_PATH)
    model.load_weights(MODEL_PATH)
    return model

diag_model = load_final_model()

# --- 6. بناء واجهة المستخدم ---
selected_lang = st.sidebar.selectbox("Language / اللغة", list(LANGS_CONFIG.keys()))
dir_ui = LANGS_CONFIG[selected_lang]
labels = UI_LABELS.get(selected_lang, UI_LABELS["العربية"])

st.markdown(f"<div dir='{dir_ui}' style='text-align:center;'><h1 style='color:#1E3A8A;'>{labels['title']}</h1></div>", unsafe_allow_html=True)
st.warning(labels['advice'])

col_up, col_res = st.columns([1, 1])

with col_up:
    st.markdown(f"<div dir='{dir_ui}'><strong>{labels['upload']}</strong></div>", unsafe_allow_html=True)
    src_opt = st.radio("", ["Upload", "Camera"], label_visibility="collapsed")
    uploaded = st.file_uploader("", type=["jpg", "png", "jpeg"]) if src_opt == "Upload" else st.camera_input("")

if uploaded:
    img_in = Image.open(uploaded).convert('RGB')
    with col_res:
        st.image(img_in, use_container_width=True)
        if st.button(labels['btn']):
            with st.spinner("⏳ فحص عتبات الثقة..."):
                img_res = cv2.resize(np.array(img_in), (224, 224))
                d_in = np.expand_dims(img_res, axis=0) / 255.0
                prediction = diag_model.predict(d_in)[0]
                idx = np.argmax(prediction)
                
                # استدعاء منطق التشخيص المعتمد على النسب
                res_label, res_color, res_note, res_acc = analyze_condition(idx, np.max(prediction))
                
                st.markdown(f"""
                <div style="padding:30px; border-radius:20px; border:10px solid {res_color}; text-align:center; background:white; box-shadow: 0px 4px 15px rgba(0,0,0,0.1);">
                    <h2 style="color:{res_color}; margin:0;">{res_label}</h2>
                    <p style="color:#333; font-size:1.1em; margin:15px 0;">{res_note}</p>
                    <div style="background:{res_color}15; padding:10px; border-radius:10px;">
                        <strong>نسبة اليقين: {res_acc:.2f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# --- 7. الدليل الطبي الملون (منسدل) ---
st.markdown("---")
with st.expander(f" {labels['guide_title']}"):
    st.markdown(f"<div dir='{dir_ui}'>", unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    g1.markdown("<div style='padding:10px; border-right:5px solid #FF3B30; background:#FFF5F5;'><strong>خبيث (Malignant)</strong><br><small>عتبة القرار: 30%</small></div>", unsafe_allow_html=True)
    g2.markdown("<div style='padding:10px; border-right:5px solid #34C759; background:#F5FFF5;'><strong>حميد (Benign)</strong><br><small>عتبة القرار: 35%</small></div>", unsafe_allow_html=True)
    g3.markdown("<div style='padding:10px; border-right:5px solid #FF9500; background:#FFF9F5;'><strong>عدوى/أخرى</strong><br><small>عتبة القرار: 45%</small></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center; color:#888; margin-top:50px;'><small>Skin Hybrid AI v6.0 | Graduation Project</small></div>", unsafe_allow_html=True)
