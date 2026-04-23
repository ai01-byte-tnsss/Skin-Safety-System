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

# --- 2. تحميل الموديل الهجين ---
@st.cache_resource
def load_skin_model():
    input_layer = Input(shape=(224, 224, 3))
    b1 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False)(input_layer)
    b2 = tf.keras.applications.MobileNetV2(weights=None, include_top=False)(input_layer)
    merged = Concatenate()([GlobalAveragePooling2D()(b1), GlobalAveragePooling2D()(b2)])
    d = Dense(512, activation='relu')(merged)
    out = Dense(24, activation='softmax')(Dropout(0.4)(d))
    model = Model(inputs=input_layer, outputs=out)
    
    if not os.path.exists(MODEL_PATH):
        with st.spinner("جاري تهيئة النظام السحابي..."):
            URL = "https://docs.google.com/uc?export=download"
            session = requests.Session()
            r = session.get(URL, params={'id': DRIVE_FILE_ID}, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(32768):
                    if chunk: f.write(chunk)
    model.load_weights(MODEL_PATH)
    return model

diag_model = load_skin_model()

# --- 3. منطق التشخيص مع العتبات المطلوبة ---
def get_diagnosis_logic(idx, score):
    # المجموعات البرمجية للأصناف الـ 24
    MALIGNANT_IDS = [1, 11, 23] # BCC, Melanoma, High Risk Analysis
    BENIGN_IDS = [0, 2, 5, 9, 13, 14, 16, 20] # Acne, Eczema, Psoriasis, etc.
    
    # العتبات المطلوبة
    THRESHOLD_MALIGNANT = 0.65
    THRESHOLD_BENIGN = 0.80
    THRESHOLD_OTHERS = 0.50 # عتبة افتراضية للأنواع الأخرى لضمان الدقة

    if idx in MALIGNANT_IDS:
        if score >= THRESHOLD_MALIGNANT:
            return "آفة خبيثة (Malignant)", "#FF3B30", "تحذير: مؤشرات مرتفعة للإصابة بسرطان الجلد. يرجى مراجعة المختص فوراً."
    elif idx in BENIGN_IDS:
        if score >= THRESHOLD_BENIGN:
            return "آفة حميدة (Benign)", "#34C759", "الحالة مستقرة: النتائج تشير إلى نمو حميد أو حالة جلدية غير سرطانية."
    else:
        if score >= THRESHOLD_OTHERS:
            return "عدوى أو التهاب جلدي", "#FF9500", "تشخيص ثانوي: الحالة تظهر مؤشرات لعدوى فطرية أو بكتيرية أو حساسية."

    return "تحليل غير حاسم", "#8E8E93", "تنبيه: نسبة اليقين لم تتجاوز العتبة المطلوبة. يرجى إعادة التصوير بوضوح أعلى."

# --- 4. واجهة المستخدم (اللغات والتحليل) ---
selected_lang = st.sidebar.selectbox("Language / اللغة", ["العربية", "English"])
dir_ui = "rtl" if selected_lang == "العربية" else "ltr"

st.markdown(f"<div dir='{dir_ui}' style='text-align:center;'><h1 style='color:#1E3A8A;'>نظام الفحص الذكي المتطور للجلد</h1></div>", unsafe_allow_html=True)

uploaded = st.file_uploader("ارفع صورة الفحص", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, use_container_width=True)
    with col2:
        if st.button("🔍 بدء التحليل"):
            img_res = cv2.resize(np.array(img), (224, 224))
            d_in = np.expand_dims(img_res, axis=0) / 255.0
            preds = diag_model.predict(d_in)[0]
            best_idx = np.argmax(preds)
            best_score = np.max(preds)
            
            label, color, note = get_diagnosis_logic(best_idx, best_score)
            
            st.markdown(f"""
            <div style="padding:30px; border-radius:20px; border:10px solid {color}; text-align:center; background:white;">
                <h2 style="color:{color};">{label}</h2>
                <p style='font-size:1.1em;'>{note}</p>
                <div style="background:{color}10; padding:10px; border-radius:10px;">
                    <strong>نسبة الثقة المتحققة: {best_score*100:.2f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- 5. الدليل الطبي المطور مع الروابط العالمية ---
st.markdown("---")
with st.expander("📖 الدليل الطبي المرجعي للأنواع"):
    st.markdown(f"<div dir='{dir_ui}'>", unsafe_allow_html=True)
    
    st.subheader("1. الآفات الخبيثة (Malignant Lesions)")
    st.write("""
    تشمل سرطانات الجلد التي تنشأ نتيجة نمو غير طبيعي لخلايا الجلد، وأبرزها:
    * **BCC:** سرطان الخلايا القاعدية، وهو الأكثر شيوعاً.
    * **Melanoma:** الميلانوما، وهو النوع الأخطر الذي قد ينتقل لأعضاء أخرى.
    """)
    st.markdown("[🔗 لمزيد من المعلومات حول سرطان الجلد (Mayo Clinic)](https://www.mayoclinic.org/diseases-conditions/skin-cancer/symptoms-causes/syc-20377605)")
    st.markdown(f"**عتبة القرار البرمجية (Threshold): {0.65}**")

    st.write("---")
    
    st.subheader("2. الآفات الحميدة (Benign Lesions)")
    st.write("""
    هي نموات غير سرطانية ولا تشكل خطراً على الحياة، ومنها:
    * **الأكزيما والصدفية:** حالات التهابية مزمنة.
    * **الشامات الحميدة:** تجمعات صبغية طبيعية.
    * **الأورام الوعائية:** نموات حميدة في الأوعية الدموية.
    """)
    st.markdown(f"**عتبة القرار البرمجية (Threshold): {0.80}**")

    st.write("---")

    st.subheader("3. العدوى والالتهابات (Infections & Others)")
    st.write("""
    تشمل الأمراض الجلدية الناتجة عن مسببات خارجية أو حساسية:
    * **الفطريات:** مثل فطريات الأظافر والقوباء الحلقية.
    * **العدوى البكتيرية:** مثل الالتهاب الخلوي.
    * **الحساسية:** مثل الأرتيكاريا والطفح الدوائي.
    """)
    st.markdown("[🔗 دليل الأمراض الجلدية والعدوى (NHS)](https://www.nhs.uk/conditions/skin-conditions/)")
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center; color:#888; margin-top:50px;'><small>Graduation Project | Skin AI Diagnostics v7.0</small></div>", unsafe_allow_html=True)
