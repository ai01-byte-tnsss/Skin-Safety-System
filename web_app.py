import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os
import requests

# --- 1. إعدادات الصفحة العامة ---
st.set_page_config(page_title="Skin AI System - نظام فحص الجلد", layout="wide")

# --- 2. إعدادات التحميل من Google Drive ---
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    # المحاولة الأولى للتحميل
    response = session.get(URL, params={'id': id}, stream=True)
    
    # التعامل مع تحذير الفيروسات للملفات الكبيرة من قوقل درايف
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# معرف الملف من الرابط الذي أرسلته
DRIVE_FILE_ID = '135lZpgsipHNk2IZBo6H4lZZ9WzVizLqb'
MODEL_PATH = 'skin_expert_hybrid_24ch.h5'

# --- 3. تعريف اللغات والواجهة (بدون تغيير) ---
LANGS_CONFIG = {"العربية": "rtl", "English": "ltr"}
UI_LABELS = {
    "العربية": {
        "title": "نظام الفحص الذكي للجلد",
        "upload": "📥 ارفع صورة الفحص أو استخدم الكاميرا",
        "btn": "🔍 بدء تحليل الصورة",
        "advice": "⚠️ تنبيه طبي: هذا النظام أداة برمجية استرشادية تعتمد على الذكاء الاصطناعي، ولا يغني عن زيارة الطبيب المختص.",
        "guide_title": "📖 الدليل الطبي المرجعي لآفات الجلد",
        "invalid": "❌ الصورة المرفوعة لا تبدو كفحص جلدي، يرجى التأكد من الصورة."
    }
}

# --- 4. قائمة الأصناف الـ 24 المحدثة ---
# ملاحظة: تم توزيع الحالات بين "حميد" و "خبيث" تقديراً بناءً على المسميات الطبية لظهور النتيجة
CLASS_NAMES_24 = [
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

# خريطة الألوان للنتائج (تلقائياً: الحالات السرطانية باللون الأحمر والبقية بالأخضر/البرتقالي)
def get_diagnosis_style(class_name):
    malignant_keywords = ['Malignant', 'Carcinoma', 'Melanoma', 'Cancer']
    if any(word in class_name for word in malignant_keywords):
        return {"status": "خبيث / يحتاج فحص", "color": "#FF3B30"}
    return {"status": "حميد / حالة جلدية", "color": "#34C759"}

# --- 5. تحميل النماذج ---
@st.cache_resource
def load_full_system():
    # موديل الفلترة (MobileNetV2 الأصلي)
    filter_net = tf.keras.applications.MobileNetV2(weights="imagenet")
    
    # بناء هيكل الموديل الهجين (ليتطابق مع تدريبك)
    input_layer = Input(shape=(224, 224, 3))
    branch1 = EfficientNetB0(weights=None, include_top=False)(input_layer)
    branch2 = MobileNetV2(weights=None, include_top=False)(input_layer)
    merged = Concatenate()([GlobalAveragePooling2D()(branch1), GlobalAveragePooling2D()(branch2)])
    dense1 = Dense(512, activation='relu')(merged)
    dropout = Dropout(0.4)(dense1)
    output_layer = Dense(24, activation='softmax')(dropout) # تم التعديل لـ 24 صنف
    
    diagnostic_model = Model(inputs=input_layer, outputs=output_layer)
    
    # تحميل الملف من درايف إذا لم يكن موجوداً
    if not os.path.exists(MODEL_PATH):
        with st.spinner('جاري جلب نموذج الذكاء الاصطناعي من السحابة (87MB)...'):
            download_file_from_google_drive(DRIVE_FILE_ID, MODEL_PATH)
            
    diagnostic_model.load_weights(MODEL_PATH)
    return filter_net, diagnostic_model

filter_model, diag_model = load_full_system()

# --- 6. بناء الواجهة ---
st.sidebar.markdown("### 🌐 ضبط اللغة / Settings")
selected_language = st.sidebar.selectbox("اختر اللغة المعتمدة للواجهة:", list(LANGS_CONFIG.keys()))
labels = UI_LABELS.get(selected_language, UI_LABELS["العربية"])

st.markdown(f"<div dir='rtl' style='text-align:center;'><h1 style='color:#1E3A8A;'>{labels['title']}</h1></div>", unsafe_allow_html=True)
st.warning(labels['advice'])

col_upload, col_result = st.columns([1, 1])

with col_upload:
    st.markdown(f"<div dir='rtl'><strong>{labels['upload']}</strong></div>", unsafe_allow_html=True)
    source_option = st.radio("", ["Upload Image", "Use Camera"], label_visibility="collapsed")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"]) if source_option == "Upload Image" else st.camera_input("")

if uploaded_file:
    input_img = Image.open(uploaded_file).convert('RGB')
    with col_result:
        st.image(input_img, caption="الصورة المرفوعة", use_container_width=True)
        if st.button(labels['btn']):
            with st.spinner("⏳ جاري تحليل الأنسجة..."):
                img_array = np.array(input_img)
                img_resized = cv2.resize(img_array, (224, 224))
                
                # التحقق من أن الصورة جلدية
                check_input = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_resized.copy(), axis=0))
                filter_preds = filter_model.predict(check_input)
                decoded = tf.keras.applications.mobilenet_v2.decode_predictions(filter_preds, top=3)[0]
                
                # فحص الكلمات الممنوعة (لضمان دقة البحث)
                if any(key in decoded[0][1].lower() for key in ['dog', 'car', 'cat', 'building']):
                    st.error(labels['invalid'])
                else:
                    # التشخيص الفعلي
                    diag_input = np.expand_dims(img_resized, axis=0) / 255.0
                    prediction = diag_model.predict(diag_input)[0]
                    result_idx = np.argmax(prediction)
                    class_name = CLASS_NAMES_24[result_idx]
                    style = get_diagnosis_style(class_name)
                    
                    st.markdown(f"""
                    <div style="padding:30px; border-radius:20px; border:10px solid {style['color']}; text-align:center; background:white; margin-top:20px;">
                        <h4 style="color:#666;">التشخيص المتوقع:</h4>
                        <h2 style="color:{style['color']}; font-size:2.5em;">{style['status']}</h2>
                        <hr>
                        <p style="font-size:1.2em;">الحالة: <strong>{class_name}</strong></p>
                        <p>دقة التنبؤ: {np.max(prediction)*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

# --- 7. الدليل الطبي المرجعي (بدون تغيير) ---
st.markdown("---")
with st.expander(f" {labels['guide_title']}"):
    st.info("يحتوي هذا القسم على معلومات إرشادية حول أشهر آفات الجلد للثقافة العامة.")
    # (بقية كود الدليل الطبي الملون كما في ملفك الأصلي)

st.markdown("<div style='text-align:center; color:#888; margin-top:50px;'><small>Skin Diagnostic AI System v2.1 | Graduation Project</small></div>", unsafe_allow_html=True)
