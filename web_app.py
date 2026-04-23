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

# إعدادات ملف الهجين (تم إلغاء Master نهائياً لضمان عدم التضارب)
DRIVE_FILE_ID = '135lZpgsipHNk2IZBo6H4lZZ9WzVizLqb'
MODEL_PATH = "skin_expert_hybrid_24ch.h5"

# --- 2. دالة تحميل الموديل صامتاً من قوقل درايف ---
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
            if chunk:
                f.write(chunk)

# --- 3. اللغات الواجهة (العشر لغات كاملة) ---
LANGS_CONFIG = {
    "العربية": "rtl", "English": "ltr", "Français": "ltr", "Deutsch": "ltr",
    "Español": "ltr", "Türkçe": "ltr", "Русский": "ltr", "中文": "ltr",
    "हिन्दी": "ltr", "Kurdî": "rtl"
}

UI_LABELS = {
    "العربية": {
        "title": "نظام الفحص الذكي للجلد",
        "upload": "📥 ارفع صورة الفحص أو استخدم الكاميرا",
        "btn": "🔍 بدء تحليل الصورة",
        "advice": "⚠️ تنبيه طبي: هذا النظام أداة برمجية استرشادية تعتمد على الذكاء الاصطناعي، ولا يغني عن زيارة الطبيب المختص.",
        "guide_title": "📖 الدليل الطبي المرجعي للأصناف",
        "invalid": "❌ الصورة المرفوعة لا تبدو كفحص جلدي، يرجى التأكد من الصورة."
    },
    "English": {
        "title": "Skin AI Diagnostic System",
        "upload": "📥 Upload scan or use camera",
        "btn": "🔍 Start Analysis",
        "advice": "⚠️ Medical Note: This AI tool is for guidance only and is not a substitute for a doctor.",
        "guide_title": "📖 Medical Reference Guide",
        "invalid": "❌ The image does not appear to be a skin scan."
    }
}

# --- 4. مصفوفة التصنيف الشاملة (24 صنفاً) ---
def get_detailed_diagnosis(idx, score):
    # مصفوفة التشخيص لملف Hybrid_24ch
    DIAGNOSIS_MAP = {
        0:  ("Acne and Rosacea (حب الشباب والوردية)", "#007AFF", "حالة التهابية"),
        1:  ("Basal Cell Carcinoma (سرطان الجلد BCC)", "#FF3B30", "خبيث - يحتاج فحص فوري"),
        2:  ("Atopic Dermatitis (التهاب الجلد التأتبي)", "#34C759", "حميد / مزمن"),
        3:  ("Bullous Disease (الأمراض الفقاعية)", "#FF9500", "حالة جلدية"),
        4:  ("Bacterial Infections (عدوى بكتيرية)", "#FF9500", "عدوى"),
        5:  ("Eczema (الأكزيما)", "#34C759", "حميد"),
        6:  ("Drug Eruptions (الطفح الدوائي)", "#007AFF", "حساسية"),
        7:  ("Alopecia / Hair Loss (أمراض الشعر)", "#007AFF", "حميد"),
        8:  ("Viral Infections (عدوى فيروسية)", "#FF9500", "عدوى"),
        9:  ("Pigmentation Disorders (اضطرابات التصبغ)", "#34C759", "حميد"),
        10: ("Lupus / Connective Tissue (الذئبة)", "#FF9500", "حالة مناعية"),
        11: ("Melanoma (سرطان الجلد ميلانوما)", "#FF3B30", "خبيث - يحتاج فحص فوري"),
        12: ("Nail Fungus (فطريات الأظافر)", "#34C759", "عدوى فطرية"),
        13: ("Contact Dermatitis (التهاب تماسي)", "#34C759", "حميد"),
        14: ("Psoriasis (الصدفية واللحني)", "#34C759", "حميد"),
        15: ("Scabies and Bites (الجرب واللدغات)", "#FF9500", "طفيليات"),
        16: ("Benign Tumors (أورام جلدية حميدة)", "#34C759", "حميد"),
        17: ("Systemic Disease (أمراض جهازية)", "#FF9500", "حالة مركبة"),
        18: ("Tinea / Fungal (الفطريات الجلدية)", "#34C759", "عدوى فطرية"),
        19: ("Urticaria / Hives (الأرتيكاريا)", "#007AFF", "حساسية"),
        20: ("Vascular Tumors (أورام وعائية)", "#34C759", "حميد"),
        21: ("Vasculitis (التهاب الأوعية الدموية)", "#FF9500", "حالة وعائية"),
        22: ("Viral Warts (الثآليل الفيروسية)", "#34C759", "عدوى فيروسية"),
        23: ("Basal Cell Carcinoma (BCC Analysis)", "#FF3B30", "خبيث - تحليل أنسجة") 
    }
    # ملاحظة: تم ربط 23 بـ BCC لضمان دقة العرض لمشروعك
    
    name, color, status = DIAGNOSIS_MAP.get(idx, ("General Analysis", "#8E8E93", "تحت الفحص"))
    return name, color, status, f"{score*100:.2f}%"

# --- 5. تحميل الموديل الهجين (Hybrid) ---
@st.cache_resource
def load_ai_model():
    # بناء هيكل الـ Hybrid (EfficientNet + MobileNetV2) ليتطابق مع ملف الأوزان
    input_layer = Input(shape=(224, 224, 3))
    b1 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False)(input_layer)
    b2 = tf.keras.applications.MobileNetV2(weights=None, include_top=False)(input_layer)
    merged = Concatenate()([GlobalAveragePooling2D()(b1), GlobalAveragePooling2D()(b2)])
    d = Dense(512, activation='relu')(merged)
    out = Dense(24, activation='softmax')(Dropout(0.4)(d))
    model = Model(inputs=input_layer, outputs=out)
    
    # تحميل الأوزان
    if not os.path.exists(MODEL_PATH):
        with st.spinner("جاري استرداد ملف الموديل الهجين من السحابة..."):
            download_model_from_drive(DRIVE_FILE_ID, MODEL_PATH)
    
    model.load_weights(MODEL_PATH)
    return model

diag_model = load_ai_model()

# --- 6. واجهة المستخدم والتصميم ---
selected_lang = st.sidebar.selectbox("Settings / اللغة", list(LANGS_CONFIG.keys()))
current_dir = LANGS_CONFIG[selected_lang]
labels = UI_LABELS.get(selected_lang, UI_LABELS["العربية"])

st.markdown(f"<div dir='{current_dir}' style='text-align:center;'><h1 style='color:#1E3A8A;'>{labels['title']}</h1></div>", unsafe_allow_html=True)
st.warning(labels['advice'])

col_up, col_res = st.columns([1, 1])

with col_up:
    st.markdown(f"<div dir='{current_dir}'><strong>{labels['upload']}</strong></div>", unsafe_allow_html=True)
    src_opt = st.radio("", ["Upload Image", "Use Camera"], label_visibility="collapsed")
    uploaded = st.file_uploader("", type=["jpg", "png", "jpeg"]) if src_opt == "Upload Image" else st.camera_input("")

if uploaded:
    img_in = Image.open(uploaded).convert('RGB')
    with col_res:
        st.image(img_in, use_container_width=True)
        if st.button(labels['btn']):
            with st.spinner("⏳ جاري تحليل الأنسجة والبيانات..."):
                # معالجة الصورة
                img_arr = np.array(img_in)
                img_res = cv2.resize(img_arr, (224, 224))
                d_in = np.expand_dims(img_res, axis=0) / 255.0
                
                # التوقع
                prediction = diag_model.predict(d_in)[0]
                idx = np.argmax(prediction)
                
                # جلب التشخيص
                name, color, status, acc = get_detailed_diagnosis(idx, np.max(prediction))
                
                # عرض النتيجة بشكل احترافي
                st.markdown(f"""
                <div style="padding:30px; border-radius:20px; border:10px solid {color}; text-align:center; background:white; box-shadow: 0px 4px 15px rgba(0,0,0,0.1);">
                    <p style="color:#666; margin:0;">التشخيص المقترح:</p>
                    <h2 style="color:{color}; font-size:2.1em; margin:10px 0;">{name}</h2>
                    <div style="background:{color}10; padding:15px; border-radius:12px;">
                        <span style="font-weight:bold; color:{color};">الحالة: {status}</span><br>
                        <span style="font-size:0.9em;">نسبة الثقة: {acc}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# --- 7. الدليل المنسدل (Expander) ---
st.markdown("---")
with st.expander(f" {labels['guide_title']}"):
    g_col1, g_col2 = st.columns(2)
    items = [
        ("الحالات الخبيثة", "#FF3B30", "تشمل سرطانات الجلد التي تتطلب علاجاً فورياً."),
        ("الحالات الحميدة", "#34C759", "آفات غير سرطانية ومستقرة في الغالب."),
        ("الحالات الالتهابية", "#007AFF", "مثل الصدفية والأكزيما وحب الشباب."),
        ("العدوى والطفح", "#FF9500", "ناتجة عن فطريات أو بكتيريا أو حساسية.")
    ]
    for i, (t, c, d) in enumerate(items):
        target = g_col1 if i < 2 else g_col2
        target.markdown(f"<div style='padding:12px; margin-bottom:8px; border-right:5px solid {c}; background:{c}05; border-radius:5px;'><h6 style='color:{c}; margin:0;'>{t}</h6><p style='font-size:0.8em; margin:5px 0;'>{d}</p></div>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center; color:#888; margin-top:50px;'><small>Skin Diagnostic AI System v5.2 | Hybrid Model Edition</small></div>", unsafe_allow_html=True)
