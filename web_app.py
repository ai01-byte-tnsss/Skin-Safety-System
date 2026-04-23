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

DRIVE_FILE_ID = '135lZpgsipHNk2IZBo6H4lZZ9WzVizLqb'
MODEL_PATH = "skin_expert_hybrid_24ch.h5"

# --- 2. دالة تنزيل الموديل صامتاً ---
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

# --- 3. خريطة التشخيص الشاملة (لجميع الأصناف الـ 24) ---
def get_comprehensive_diagnosis(idx, score):
    # تم ترتيب وتعديل المسميات لتكون دقيقة طبياً ومناسبة للعرض
    DIAGNOSIS_MAP = {
        0:  ("حب الشباب والوردية (Acne/Rosacea)", "#007AFF", "حالة التهابية جلدية"),
        1:  ("سرطان الخلايا القاعدية (BCC) / خبيث", "#FF3B30", "خبيث - يحتاج فحص فوري"),
        2:  ("التهاب الجلد التأتبي (Atopic Dermatitis)", "#34C759", "حميد / مزمن"),
        3:  ("أمراض فقاعية (Bullous Disease)", "#FF9500", "حالة جلدية نادرة"),
        4:  ("عدوى بكتيرية (Cellulitis/Impetigo)", "#FF9500", "عدوى بكتيرية"),
        5:  ("الأكزيما (Eczema)", "#34C759", "حميد / حساسية"),
        6:  ("طفح جلدي دوائي (Exanthems)", "#007AFF", "حساسية دوائية"),
        7:  ("أمراض تساقط الشعر (Alopecia)", "#007AFF", "حميد"),
        8:  ("أمراض فيروسية (Herpes/HPV)", "#FF9500", "عدوى فيروسية"),
        9:  ("اضطرابات التصبغ (Pigmentation)", "#34C759", "حميد"),
        10: ("أمراض النسيج الضام (Lupus)", "#FF9500", "حالة مناعية"),
        11: ("الميلانوما (Melanoma) / خبيث", "#FF3B30", "خبيث - يحتاج فحص فوري"),
        12: ("فطريات الأظافر (Nail Fungus)", "#34C759", "عدوى فطرية"),
        13: ("التهاب الجلد التماسي (Contact Dermatitis)", "#34C759", "حميد"),
        14: ("الصدفية واللحني (Psoriasis)", "#34C759", "حميد / مزمن"),
        15: ("الجرب ولدغات الحشرات (Scabies)", "#FF9500", "طفيليات جلدية"),
        16: ("أورام جلدية حميدة (Benign Tumors)", "#34C759", "حميد"),
        17: ("أمراض جلدية جهازية (Systemic Disease)", "#FF9500", "حالة مرتبطة بالجسم"),
        18: ("القوباء الحلقية والفطريات (Tinea)", "#34C759", "عدوى فطرية"),
        19: ("الأرتيكاريا والشرى (Hives)", "#007AFF", "حساسية"),
        20: ("أورام وعائية (Vascular Tumors)", "#34C759", "حميد"),
        21: ("التهاب الأوعية الدموية (Vasculitis)", "#FF9500", "حالة وعائية"),
        22: ("الثآليل والعدوى الفيروسية (Warts)", "#34C759", "عدوى فيروسية"),
        23: ("سرطان الخلايا القاعدية (BCC) / خبيث", "#FF3B30", "خبيث - يحتاج فحص فوري") # دمج الأرشيف مع الخبيث
    }
    
    name, color, status = DIAGNOSIS_MAP.get(idx, ("فحص عام", "#8E8E93", "حالة غير محددة"))
    return name, color, status, f"{score*100:.2f}%"

# --- 4. تحميل النماذج ---
@st.cache_resource
def load_ai_model():
    filter_net = tf.keras.applications.MobileNetV2(weights="imagenet")
    input_layer = Input(shape=(224, 224, 3))
    b1 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False)(input_layer)
    b2 = tf.keras.applications.MobileNetV2(weights=None, include_top=False)(input_layer)
    merged = Concatenate()([GlobalAveragePooling2D()(b1), GlobalAveragePooling2D()(b2)])
    d = Dense(512, activation='relu')(merged)
    out = Dense(24, activation='softmax')(Dropout(0.4)(d))
    model = Model(inputs=input_layer, outputs=out)
    
    if not os.path.exists(MODEL_PATH):
        with st.spinner("جاري تهيئة النظام السحابي..."):
            download_model_from_drive(DRIVE_FILE_ID, MODEL_PATH)
    
    model.load_weights(MODEL_PATH)
    return filter_net, model

filter_model, diag_model = load_ai_model()

# --- 5. واجهة المستخدم (متعددة اللغات) ---
LANGS = {"العربية": "rtl", "English": "ltr"}
selected_lang = st.sidebar.selectbox("Language / اللغة", list(LANGS.keys()))
current_dir = LANGS[selected_lang]

st.markdown(f"<div dir='{current_dir}' style='text-align:center;'><h1 style='color:#1E3A8A;'>نظام الفحص الذكي للجلد</h1></div>", unsafe_allow_html=True)
st.info("⚠️ هذا النظام أداة استرشادية تعتمد على الذكاء الاصطناعي لفحص 24 نوعاً من الآفات الجلدية.")

uploaded = st.file_uploader("ارفع صورة الفحص أو استخدم الكاميرا", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, use_container_width=True, caption="الصورة المرفوعة")
        
    with col2:
        if st.button("🔍 بدء التحليل"):
            with st.spinner("⏳ جاري تحليل الأنسجة والبيانات..."):
                img_res = cv2.resize(np.array(img), (224, 224))
                
                # الفلترة (اختياري لسرعة العرض)
                d_in = np.expand_dims(img_res, axis=0) / 255.0
                prediction = diag_model.predict(d_in)[0]
                idx = np.argmax(prediction)
                
                name, color, status, acc = get_comprehensive_diagnosis(idx, np.max(prediction))
                
                st.markdown(f"""
                <div style="padding:30px; border-radius:20px; border:10px solid {color}; text-align:center; background:white; box-shadow: 0px 4px 15px rgba(0,0,0,0.1);">
                    <p style="color:#666; margin:0;">التشخيص المقترح:</p>
                    <h2 style="color:{color}; font-size:2.2em; margin:10px 0;">{name}</h2>
                    <div style="background:{color}10; padding:15px; border-radius:12px;">
                        <span style="font-weight:bold; color:{color};">التصنيف: {status}</span><br>
                        <span style="font-size:0.9em;">نسبة الثقة: {acc}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# --- 6. الدليل المرجعي المنسدل ---
st.markdown("---")
with st.expander("📖 الدليل الطبي المرجعي لجميع الحالات"):
    st.write("يغطي النظام 24 صنفاً طبياً تم دمجها برمجياً لضمان دقة النتائج:")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<span style='color:#FF3B30;'>●</span> **الحالات الخبيثة:** تشمل BCC والميلانوما.", unsafe_allow_html=True)
        st.markdown("<span style='color:#34C759;'>●</span> **الحالات الحميدة:** تشمل الأكزيما، الصدفية، والتصبغات.", unsafe_allow_html=True)
    with c2:
        st.markdown("<span style='color:#007AFF;'>●</span> **الحالات الالتهابية:** تشمل حب الشباب والشرى.", unsafe_allow_html=True)
        st.markdown("<span style='color:#FF9500;'>●</span> **العدوى:** تشمل الفطريات، البكتيريا، والعدوى الفيروسية.", unsafe_allow_html=True)
