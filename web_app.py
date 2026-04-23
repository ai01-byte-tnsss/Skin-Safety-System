import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os
import requests

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Skin AI System", layout="wide")

DRIVE_FILE_ID = '135lZpgsipHNk2IZBo6H4lZZ9WzVizLqb'
MODEL_PATH = "skin_expert_hybrid_24ch.h5"

# --- 2. تحميل الموديل صامتاً ---
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

# --- 3. مصفوفة التصنيف الدقيقة (كل رقم له مسمى مستقل) ---
def get_diagnosis(idx, score):
    # مصفوفة الـ 24 صنفاً بدون تداخل لضمان دقة التصنيف
    DIAGNOSIS_MAP = {
        0:  ("Acne / Rosacea (حب الشباب)", "#007AFF", "حالة التهابية"),
        1:  ("Basal Cell Carcinoma (سرطان BCC)", "#FF3B30", "خبيث - يحتاج فحص"),
        2:  ("Atopic Dermatitis (التهاب تأتبي)", "#34C759", "حميد"),
        3:  ("Bullous Disease (أمراض فقاعية)", "#FF9500", "حالة جلدية"),
        4:  ("Bacterial Infections (عدوى بكتيرية)", "#FF9500", "عدوى"),
        5:  ("Eczema (الأكزيما)", "#34C759", "حميد"),
        6:  ("Drug Eruptions (طفح دوائي)", "#007AFF", "حساسية"),
        7:  ("Alopecia / Hair Loss (تساقط الشعر)", "#007AFF", "حميد"),
        8:  ("Herpes / HPV (عدوى فيروسية)", "#FF9500", "عدوى"),
        9:  ("Pigmentation Disorders (تصبغات)", "#34C759", "حميد"),
        10: ("Lupus / Connective Tissue (ذئبة)", "#FF9500", "حالة مناعية"),
        11: ("Melanoma (ميلانوما)", "#FF3B30", "خبيث - يحتاج فحص"),
        12: ("Nail Fungus (فطريات الأظافر)", "#34C759", "عدوى فطرية"),
        13: ("Contact Dermatitis (التهاب تماسي)", "#34C759", "حميد"),
        14: ("Psoriasis (الصدفية)", "#34C759", "حميد"),
        15: ("Scabies / Bites (جرب ولدغات)", "#FF9500", "طفيليات"),
        16: ("Benign Tumors (أورام حميدة)", "#34C759", "حميد"),
        17: ("Systemic Disease (مرض جهازي)", "#FF9500", "حالة عامة"),
        18: ("Tinea / Ringworm (فطريات جلدية)", "#34C759", "عدوى فطرية"),
        19: ("Urticaria / Hives (أرتيكاريا)", "#007AFF", "حساسية"),
        20: ("Vascular Tumors (أورام وعائية)", "#34C759", "حميد"),
        21: ("Vasculitis (التهاب أوعية)", "#FF9500", "حالة وعائية"),
        22: ("Viral Warts (ثآليل فيروسية)", "#34C759", "عدوى"),
        23: ("Skin Scan Analysis (تحليل الأنسجة)", "#8E8E93", "غير محدد بدقة") 
    }
    # ملاحظة: تم تغيير 23 إلى "تحليل أنسجة" ليكون مسمى احترافي بدلاً من "أرشيف"
    
    name, color, status = DIAGNOSIS_MAP.get(idx, ("General Scan", "#8E8E93", "تحت المراجعة"))
    return name, color, status, f"{score*100:.2f}%"

# --- 4. تحميل النظام ---
@st.cache_resource
def load_system():
    input_layer = Input(shape=(224, 224, 3))
    b1 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False)(input_layer)
    b2 = tf.keras.applications.MobileNetV2(weights=None, include_top=False)(input_layer)
    merged = Concatenate()([GlobalAveragePooling2D()(b1), GlobalAveragePooling2D()(b2)])
    d = Dense(512, activation='relu')(merged)
    out = Dense(24, activation='softmax')(Dropout(0.4)(d))
    model = Model(inputs=input_layer, outputs=out)
    
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Initializing System..."):
            download_model_from_drive(DRIVE_FILE_ID, MODEL_PATH)
    
    model.load_weights(MODEL_PATH)
    return model

diag_model = load_system()

# --- 5. واجهة المستخدم ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>Skin AI Diagnostic System</h1>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, use_container_width=True)
    with col2:
        if st.button("🔍 Start Scan"):
            img_res = cv2.resize(np.array(img), (224, 224))
            d_in = np.expand_dims(img_res, axis=0) / 255.0
            prediction = diag_model.predict(d_in)[0]
            idx = np.argmax(prediction)
            
            name, color, status, acc = get_diagnosis(idx, np.max(prediction))
            
            st.markdown(f"""
            <div style="padding:25px; border-radius:15px; border:8px solid {color}; text-align:center; background:white;">
                <h2 style="color:{color};">{name}</h2>
                <p>Status: <b>{status}</b></p>
                <p style="color:grey;">Confidence: {acc}</p>
            </div>
            """, unsafe_allow_html=True)
