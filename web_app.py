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

# إعدادات الملف والربط مع Google Drive
# معرف الملف المستخرج من رابطك: 135lZpgsipHNk2IZBo6H4lZZ9WzVizLqb
DRIVE_FILE_ID = '135lZpgsipHNk2IZBo6H4lZZ9WzVizLqb'
MODEL_PATH = "skin_expert_hybrid_24ch.h5"

# --- 2. دالة تحميل الموديل صامتاً من السحابة ---
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

# --- 3. اللغات العشر الأساسية (بدون أي نقص) ---
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
        "guide_title": "📖 الدليل الطبي المرجعي للآفات والأنواع",
        "invalid": "❌ الصورة المرفوعة لا تبدو كفحص جلدي، يرجى التأكد من الصورة."
    }
}

# --- 4. مصفوفة الـ 24 صنفاً وتفاصيل التشخيص (الدقة القصوى) ---
CLASSES_24 = [
    'Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 
    'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Cellulitis Impetigo and other Bacterial Infections', 
    'Eczema Photos', 'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases', 
    'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation', 
    'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles', 
    'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis', 
    'Psoriasis pictures Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites', 
    'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease', 
    'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 
    'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections', 'archive'
]

def get_detailed_diagnosis(idx, score):
    DIAGNOSIS_MAP = {
        0: ("حب الشباب والوردية (Acne/Rosacea)", "#007AFF", "حالة جلدية"),
        1: ("سرطان الخلايا القاعدية / آفات خبيثة", "#FF3B30", "خبيث - يحتاج فحص"),
        2: ("التهاب الجلد التأتبي (Atopic Dermatitis)", "#34C759", "حميد"),
        3: ("أمراض فقاعية (Bullous Disease)", "#FF9500", "تحتاج فحص"),
        4: ("عدوى بكتيرية (Bacterial Infections)", "#FF9500", "عدوى بكتيرية"),
        5: ("أكزيما (Eczema)", "#34C759", "حميد"),
        6: ("الطفح الدوائي (Exanthems)", "#007AFF", "حساسية"),
        7: ("أمراض الشعر والفروة (Hair Diseases)", "#007AFF", "حميد"),
        8: ("أمراض فيروسية (Herpes/HPV)", "#FF9500", "عدوى فيروسية"),
        9: ("اضطرابات الصبغة (Pigmentation)", "#34C759", "حميد"),
        10: ("أمراض النسيج الضام (Lupus)", "#FF9500", "حالة مناعية"),
        11: ("ميلانوما / سرطان الجلد وشامات", "#FF3B30", "خبيث - يحتاج فحص"),
        12: ("فطريات الأظافر (Nail Fungus)", "#34C759", "حميد"),
        13: ("التهاب الجلد التماسي (Contact)", "#34C759", "حميد"),
        14: ("الصدفية واللحني (Psoriasis)", "#34C759", "حميد"),
        15: ("الجرب ولدغات الحشرات (Scabies)", "#FF9500", "طفيليات"),
        16: ("أورام جلدية حميدة (Benign Tumors)", "#34C759", "حميد"),
        17: ("أمراض جلدية جهازية (Systemic Disease)", "#FF9500", "حالة مركبة"),
        18: ("القوباء الحلقية والفطريات (Tinea)", "#34C759", "حميد"),
        19: ("الأرتيكاريا والشرى (Urticaria)", "#007AFF", "حساسية"),
        20: ("الأورام الوعائية (Vascular Tumors)", "#34C759", "حميد"),
        21: ("التهاب الأوعية (Vasculitis)", "#FF9500", "حالة وعائية"),
        22: ("الثآليل والعدوى الفيروسية (Warts)", "#34C759", "حميد"),
        23: ("محتوى أرشيفي (Archive)", "#8E8E93", "غير محدد")
    }
    name, color, status = DIAGNOSIS_MAP.get(idx, ("غير معروف", "#8E8E93", "فحص طبي"))
    return name, color, status, f"{score*100:.2f}%"

# --- 5. تحميل النماذج (تنزيل صامت من الدرايف) ---
@st.cache_resource
def load_ai_system():
    # موديل الفلترة
    filter_net = tf.keras.applications.MobileNetV2(weights="imagenet")
    
    # بناء الهيكل الهجين
    input_layer = Input(shape=(224, 224, 3))
    b1 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False)(input_layer)
    b2 = tf.keras.applications.MobileNetV2(weights=None, include_top=False)(input_layer)
    merged = Concatenate()([GlobalAveragePooling2D()(b1), GlobalAveragePooling2D()(b2)])
    d = Dense(512, activation='relu')(merged)
    out = Dense(24, activation='softmax')(Dropout(0.4)(d))
    model = Model(inputs=input_layer, outputs=out)
    
    # تحميل الأوزان (من السحابة إذا لم تكن موجودة)
    if not os.path.exists(MODEL_PATH):
        with st.spinner("جاري تهيئة النظام السحابي واسترداد البيانات (مرة واحدة فقط)..."):
            download_model_from_drive(DRIVE_FILE_ID, MODEL_PATH)
    
    model.load_weights(MODEL_PATH)
    return filter_net, model

filter_model, diag_model = load_ai_system()

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
            with st.spinner("⏳ جاري تحليل البيانات..."):
                img_arr = np.array(img_in)
                img_res = cv2.resize(img_arr, (224, 224))
                
                # الفلترة الذكية
                f_in = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_res.copy(), axis=0))
                preds = filter_model.predict(f_in)
                decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]
                
                if any(k in decoded[0][1].lower() for k in ['car', 'dog', 'furniture', 'building']):
                    st.error(labels['invalid'])
                else:
                    # التشخيص العميق
                    d_in = np.expand_dims(img_res, axis=0) / 255.0
                    prediction = diag_model.predict(d_in)[0]
                    idx = np.argmax(prediction)
                    name, color, status, acc = get_detailed_diagnosis(idx, np.max(prediction))
                    
                    st.markdown(f"""
                    <div style="padding:30px; border-radius:20px; border:10px solid {color}; text-align:center; background:white; box-shadow: 0px 4px 15px rgba(0,0,0,0.1);">
                        <p style="color:#666; margin:0;">النتيجة التقديرية:</p>
                        <h2 style="color:{color}; font-size:2.2em; margin:10px 0;">{name}</h2>
                        <div style="background:{color}10; padding:15px; border-radius:12px;">
                            <span style="font-weight:bold; color:{color};">التصنيف الطبي: {status}</span><br>
                            <span style="font-size:0.9em;">نسبة الثقة: {acc}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# --- 7. الدليل المنسدل (Expander) ---
st.markdown("---")
with st.expander(f" {labels['guide_title']}"):
    g_col1, g_col2 = st.columns(2)
    guide_items = [
        ("الحالات الخبيثة", "#FF3B30", "سرطانات الجلد التي تتطلب تدخلاً طبياً."),
        ("الحالات الحميدة", "#34C759", "تجمعات غير سرطانية ومستقرة غالباً."),
        ("الحالات الالتهابية", "#007AFF", "مثل الصدفية والأكزيما وحب الشباب."),
        ("العدوى والطفح", "#FF9500", "ناتجة عن فطريات أو بكتيريا أو حساسية.")
    ]
    for i, (t, c, d) in enumerate(guide_items):
        target = g_col1 if i < 2 else g_col2
        target.markdown(f"<div style='padding:12px; margin-bottom:8px; border-right:5px solid {c}; background:{c}05; border-radius:5px;'><h6 style='color:{c}; margin:0;'>{t}</h6><p style='font-size:0.8em; margin:5px 0;'>{d}</p></div>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center; color:#888; margin-top:50px;'><small>Skin Diagnostic AI System v4.0 | Final Project</small></div>", unsafe_allow_html=True)
