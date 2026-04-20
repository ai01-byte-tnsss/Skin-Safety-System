import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Skin AI System", layout="wide")

# --- 2. اللغات العشر ---
LANGS_CONFIG = {
    "العربية": "rtl", "English": "ltr", "Français": "ltr", "Deutsch": "ltr",
    "Español": "ltr", "Türkçe": "ltr", "Русский": "ltr", "中文": "ltr",
    "हिन्दी": "ltr", "Kurdî": "rtl"
}

UI_LABELS = {
    "العربية": {
        "title": "نظام فحص الجلد الذكي",
        "upload": "📥 ارفع صورة الفحص",
        "btn": "🔍 بدء التحليل",
        "advice": "⚠️ تنبيه: هذا النظام للاسترشاد فقط وليس بديلاً عن الطبيب.",
        "guide_title": "📖 الدليل الطبي المرجعي",
        "invalid": "❌ الصورة لا تبدو كفحص جلدي."
    },
    "English": {
        "title": "Skin AI Diagnostic System",
        "upload": "📥 Upload Scan",
        "btn": "🔍 Start Analysis",
        "advice": "⚠️ Note: This tool is for guidance only.",
        "guide_title": "📖 Medical Reference Guide",
        "invalid": "❌ Invalid skin image."
    }
}

# --- 3. خريطة التصنيف (حميد/خبيث) ---
DIAGNOSIS_MAP = {
    0: {"status": "خبيث", "color": "#FF3B30"},
    1: {"status": "حميد", "color": "#34C759"},
    2: {"status": "خبيث", "color": "#FF3B30"},
    3: {"status": "خبيث", "color": "#FF9500"},
    4: {"status": "حميد", "color": "#34C759"},
    5: {"status": "حميد", "color": "#34C759"},
    6: {"status": "حميد", "color": "#34C759"},
    7: {"status": "خبيث", "color": "#FF3B30"},
    8: {"status": "حميد", "color": "#34C759"},
    9: {"status": "حميد", "color": "#34C759"}
}

# --- 4. محرك الذكاء الاصطناعي (موديل واحد موحد) ---
@st.cache_resource
def load_ai_system():
    # استخدام MobileNetV2 كموديل أساسي وحيد
    input_tensor = Input(shape=(224, 224, 3))
    base_model = MobileNetV2(weights=None, include_top=False, input_tensor=input_tensor)
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    # محاولة تحميل الأوزان مع إظهار الحالة في القائمة الجانبية
    h5_path = "skin_expert_master.h5"
    if os.path.exists(h5_path):
        try:
            model.load_weights(h5_path)
            st.sidebar.success("✅ تم تحميل الأوزان بنجاح")
        except Exception as e:
            st.sidebar.error(f"❌ خطأ في مطابقة الأوزان: {e}")
    else:
        st.sidebar.warning("⚠️ ملف الأوزان غير موجود")
            
    return model

diag_model = load_ai_system()

# --- 5. واجهة المستخدم ---
st.sidebar.markdown("### 🌐 Settings / الإعدادات")
selected_lang = st.sidebar.selectbox("Language", list(LANGS_CONFIG.keys()))
labels = UI_LABELS.get(selected_lang, UI_LABELS["English"])
dir_ui = LANGS_CONFIG[selected_lang]

st.markdown(f"<div dir='{dir_ui}' style='text-align:center;'><h1 style='color:#1E3A8A;'>{labels['title']}</h1></div>", unsafe_allow_html=True)
st.warning(labels['advice'])

col1, col2 = st.columns(2)
with col1:
    file = st.file_uploader(labels['upload'], type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, use_container_width=True)
        if st.button(labels['btn']):
            with st.spinner("⏳ Analyzing..."):
                # معالجة الصورة
                img_resized = cv2.resize(np.array(img), (224, 224))
                img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_resized, axis=0))
                
                # التنبؤ
                prediction = diag_model.predict(img_preprocessed)[0]
                res_idx = np.argmax(prediction)
                result = DIAGNOSIS_MAP[res_idx]
                
                # عرض النتيجة (كلمة واحدة)
                st.markdown(f"""
                <div style="padding:40px; border-radius:20px; border:10px solid {result['color']}; text-align:center; background:white; margin-top:10px;">
                    <h1 style="color:{result['color']}; font-size:5em; margin:0;">{result['status']}</h1>
                </div>
                """, unsafe_allow_html=True)

# --- 6. الدليل الطبي المنسدل (10 أنواع) ---
st.write("---")
GUIDE_DATA = {
    "Melanoma": "#FF3B30", "Basal Cell": "#FF9500", "Squamous Cell": "#FF2D55",
    "Nevi": "#34C759", "Psoriasis": "#4CD964", "Eczema": "#FFCC00",
    "Dermatofibroma": "#007AFF", "Vascular Lesions": "#5AC8FA",
    "Actinic Keratosis": "#AF52DE", "Benign Keratosis": "#5856D6"
}

with st.expander(labels['guide_title']):
    g_cols = st.columns(2)
    items = list(GUIDE_DATA.items())
    for i in range(10):
        c = g_cols[0] if i < 5 else g_cols[1]
        name, color = items[i]
        c.markdown(f"<div style='border-right:5px solid {color}; padding:5px; margin:5px; background:{color}10;'>{name}</div>", unsafe_allow_html=True)
