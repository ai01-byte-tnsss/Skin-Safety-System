import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. إعدادات الصفحة العامة ---
st.set_page_config(page_title="Skin AI System - نظام فحص الجلد", layout="wide")

# --- 2. تعريف اللغات العشر الأساسية ---
LANGS_CONFIG = {
    "العربية": "rtl",
    "English": "ltr",
    "Français": "ltr",
    "Deutsch": "ltr",
    "Español": "ltr",
    "Türkçe": "ltr",
    "Русский": "ltr",
    "中文": "ltr",
    "हिन्दी": "ltr",
    "Kurdî": "rtl"
}

# نصوص واجهة المستخدم حسب اللغة المختارة
UI_LABELS = {
    "العربية": {
        "title": "نظام الفحص الذكي للجلد",
        "upload": "📥 ارفع صورة الفحص أو استخدم الكاميرا",
        "btn": "🔍 بدء تحليل الصورة",
        "advice": "⚠️ تنبيه طبي: هذا النظام أداة برمجية استرشادية تعتمد على الذكاء الاصطناعي، ولا يغني عن زيارة الطبيب المختص.",
        "guide_title": "📖 الدليل الطبي المرجعي لآفات الجلد",
        "invalid": "❌ الصورة المرفوعة لا تبدو كفحص جلدي، يرجى التأكد من الصورة."
    },
    "English": {
        "title": "Skin AI Diagnostic System",
        "upload": "📥 Upload scan or use camera",
        "btn": "🔍 Start Analysis",
        "advice": "⚠️ Medical Note: This AI tool is for guidance only and is not a substitute for a doctor.",
        "guide_title": "📖 Medical Reference Guide for Skin Lesions",
        "invalid": "❌ The image does not appear to be a skin scan."
    }
}

# --- 3. بيانات التصنيف (لربط مخرجات الموديل بكلمة حميد/خبيث) ---
DIAGNOSIS_MAP = {
    0: {"status": "خبيث", "color": "#FF3B30"}, # Melanoma
    1: {"status": "حميد", "color": "#34C759"}, # Melanocytic Nevi
    2: {"status": "خبيث", "color": "#FF3B30"}, # Basal Cell Carcinoma
    3: {"status": "خبيث", "color": "#FF9500"}, # Actinic Keratosis
    4: {"status": "حميد", "color": "#34C759"}, # Benign Keratosis
    5: {"status": "حميد", "color": "#34C759"}, # Dermatofibroma
    6: {"status": "حميد", "color": "#34C759"}, # Vascular Lesions
    7: {"status": "خبيث", "color": "#FF3B30"}, # Squamous Cell Carcinoma
    8: {"status": "حميد", "color": "#34C759"}, # Psoriasis
    9: {"status": "حميد", "color": "#34C759"}  # Eczema
}

# --- 4. الدليل الطبي الملون (10 أنواع منفصلة للعرض فقط) ---
GUIDE_CONTENT = {
    "Melanoma (ميلانوما)": {"color": "#FF3B30", "desc": "أخطر أنواع سرطان الجلد، يظهر عادةً كشامة غير منتظمة الشكل أو اللون وتنمو بسرعة."},
    "Basal Cell (خلايا قاعدية)": {"color": "#FF9500", "desc": "أكثر أنواع سرطان الجلد شيوعاً، ينمو ببطء ونادراً ما ينتشر، لكنه يتطلب علاجاً."},
    "Squamous Cell (خلايا حرشفية)": {"color": "#FF2D55", "desc": "ثاني أكثر الأنواع شيوعاً، يظهر كبقع حمراء قشرية أو قروح مفتوحة."},
    "Actinic Keratosis (تقران شمسي)": {"color": "#AF52DE", "desc": "بقع خشنة ناتجة عن التعرض المفرط للشمس، وتعتبر مرحلة ما قبل سرطانية."},
    "Nevi (الشامات)": {"color": "#34C759", "desc": "تجمعات صبغية طبيعية تظهر على الجلد، وتكون حميدة في الغالبية العظمى من الحالات."},
    "Benign Keratosis (تقران حميد)": {"color": "#5856D6", "desc": "زوائد جلدية غير سرطانية تظهر عادةً مع التقدم في السن وتكون ذات ملمس شمعي."},
    "Dermatofibroma (ليفي جلدي)": {"color": "#007AFF", "desc": "نمو حميد صغير وصلب يظهر غالباً بعد إصابة بسيطة أو لدغة حشرة."},
    "Vascular Lesions (آفات وعائية)": {"color": "#5AC8FA", "desc": "تشمل الوحمات الدموية والنقاط الحمراء الناتجة عن تجمع الشعيرات الدموية."},
    "Psoriasis (الصدفية)": {"color": "#4CD964", "desc": "مرض جلدي مزمن يسبب بقعاً حمراء مغطاة بقشور فضية نتيجة تسارع نمو الخلايا."},
    "Eczema (الأكزيما)": {"color": "#FFCC00", "desc": "التهاب جلدي يسبب حكة شديدة واحمراراً، وغالباً ما يرتبط بالحساسية أو العوامل الوراثية."}
}

# --- 5. تحميل النماذج الذكية ---
@st.cache_resource
def load_ai_models():
    # موديل التحقق من نوع الصورة (فلترة)
    filter_net = tf.keras.applications.MobileNetV2(weights="imagenet")
    
    # بناء الموديل الهجين للتشخيص
    input_layer = Input(shape=(224, 224, 3))
    branch1 = EfficientNetB0(weights=None, include_top=False)(input_layer)
    branch2 = MobileNetV2(weights=None, include_top=False)(input_layer)
    
    merged = Concatenate()([GlobalAveragePooling2D()(branch1), GlobalAveragePooling2D()(branch2)])
    dense1 = Dense(512, activation='relu')(merged)
    dropout = Dropout(0.4)(dense1)
    output_layer = Dense(10, activation='softmax')(dropout)
    
    diagnostic_model = Model(inputs=input_layer, outputs=output_layer)
    
    # تحميل الأوزان إذا كان الملف موجوداً
    weights_path = "skin_expert_master.h5"
    if os.path.exists(weights_path):
        try:
            diagnostic_model.load_weights(weights_path)
        except:
            pass
            
    return filter_net, diagnostic_model

filter_model, diag_model = load_ai_models()

# --- 6. بناء واجهة المستخدم ---
st.sidebar.markdown("### 🌐 ضبط اللغة / Settings")
selected_language = st.sidebar.selectbox("اختر اللغة المعتمدة للواجهة:", list(LANGS_CONFIG.keys()))
current_dir = LANGS_CONFIG[selected_language]

# جلب النصوص بناءً على اللغة (الافتراضي إنجليزي إذا لم تتوفر ترجمة للعربية)
labels = UI_LABELS.get(selected_language, UI_LABELS["English"])

st.markdown(f"<div dir='{current_dir}' style='text-align:center;'><h1 style='color:#1E3A8A;'>{labels['title']}</h1></div>", unsafe_allow_html=True)
st.warning(labels['advice'])

# مساحة العمل الأساسية
col_upload, col_result = st.columns([1, 1])

with col_upload:
    st.markdown(f"<div dir='{current_dir}'><strong>{labels['upload']}</strong></div>", unsafe_allow_html=True)
    source_option = st.radio("", ["Upload Image", "Use Camera"], label_visibility="collapsed")
    
    if source_option == "Upload Image":
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    else:
        uploaded_file = st.camera_input("")

if uploaded_file:
    input_img = Image.open(uploaded_file).convert('RGB')
    
    with col_result:
        st.image(input_img, caption="الصورة المرفوعة", use_container_width=True)
        
        if st.button(labels['btn']):
            with st.spinner("⏳ جاري معالجة البيانات وتحليل الأنسجة..."):
                # تحضير الصورة
                img_array = np.array(input_img)
                img_resized = cv2.resize(img_array, (224, 224))
                
                # المرحلة 1: التحقق من أن الصورة هي للجلد فعلاً
                check_input = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_resized.copy(), axis=0))
                filter_preds = filter_model.predict(check_input)
                decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(filter_preds, top=3)[0]
                
                is_skin_image = True
                invalid_keywords = ['car', 'dog', 'cat', 'flower', 'building', 'laptop', 'furniture']
                for _, label, score in decoded_preds:
                    if any(key in label.lower() for key in invalid_keywords) and score > 0.45:
                        is_skin_image = False
                
                if not is_skin_image:
                    st.error(labels['invalid'])
                else:
                    # المرحلة 2: التحليل والتشخيص
                    diag_input = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_resized, axis=0))
                    prediction = diag_model.predict(diag_input)[0]
                    
                    # استخراج النتيجة (حميد أو خبيث)
                    result_idx = np.argmax(prediction)
                    result_data = DIAGNOSIS_MAP[result_idx]
                    
                    # عرض النتيجة النهائية بشكل بارز جداً
                    st.markdown(f"""
                    <div style="padding:50px; border-radius:25px; border:12px solid {result_data['color']}; text-align:center; background:white; margin-top:20px; box-shadow: 0px 4px 15px rgba(0,0,0,0.1);">
                        <p style="font-size:1.2em; color:#666; margin-bottom:5px;">نتيجة الفحص التقديرية:</p>
                        <h1 style="color:{result_data['color']}; font-size:6em; margin:0; font-weight:900;">{result_data['status']}</h1>
                    </div>
                    """, unsafe_allow_html=True)

# --- 7. الدليل الطبي المرجعي (منسدل ومرتب) ---
st.markdown("---")
with st.expander(f" {labels['guide_title']}"):
    st.markdown("<br>", unsafe_allow_html=True)
    g_col1, g_col2 = st.columns(2)
    
    guide_items = list(GUIDE_CONTENT.items())
    
    for i in range(len(guide_items)):
        title, content = guide_items[i]
        # تقسيم الدليل إلى عمودين
        target_column = g_col1 if i < 5 else g_col2
        
        target_column.markdown(f"""
        <div style="padding:15px; margin-bottom:15px; border-right:8px solid {content['color']}; background-color:{content['color']}10; border-radius:10px;">
            <h4 style="color:{content['color']}; margin:0;">{title}</h4>
            <p style="color:#444; font-size:0.95em; margin-top:8px; line-height:1.4;">{content['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

# تذييل الصفحة
st.markdown(f"<div style='text-align:center; color:#888; margin-top:50px;'><small>Skin Diagnostic AI System v2.0</small></div>", unsafe_allow_html=True)
