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
st.set_page_config(page_title="Skin AI System v7.8", layout="wide")

DRIVE_FILE_ID = '135lZpgsipHNk2IZBo6H4lZZ9WzVizLqb'
MODEL_PATH = "skin_expert_hybrid_24ch.h5"

# --- 2. قاموس اللغات الكامل (ترجمة شاملة) ---
LANG_DATA = {
    "العربية": {
        "dir": "rtl",
        "title": "نظام التشخيص الذكي المتطور للجلد",
        "upload_label": "📥 ارفع صورة الفحص أو استخدم الكاميرا",
        "btn_analyze": "🔍 بدء التحليل الفوري",
        "advice": "⚠️ تنبيه طبي: هذا النظام أداة برمجية استرشادية، ولا يغني عن زيارة الطبيب المختص.",
        "guide_title": "📖 الدليل الطبي المرجعي للأنواع",
        "malig_title": "1. الآفات الخبيثة (Malignant)",
        "malig_desc": "تشمل سرطانات الجلد مثل BCC والميلانوما. تتطلب تدخلاً طبياً عاجلاً.",
        "benign_title": "2. الآفات الحميدة (Benign)",
        "benign_desc": "نموات غير سرطانية مثل الأكزيما والصدفية والشامات الحميدة.",
        "others_title": "3. العدوى والالتهابات",
        "others_desc": "أمراض ناتجة عن فطريات أو بكتيريا أو حساسية جلدية.",
        "link_cancer": "🔗 مصدر عالمي: سرطان الجلد (Mayo Clinic)",
        "link_others": "🔗 مصدر عالمي: الأمراض الجلدية (NHS)",
        "result_text": "التشخيص المقترح:",
        "confidence": "نسبة اليقين:",
        "inconclusive": "تحليل غير حاسم: لم تصل النسبة للحد الأدنى (يرجى إعادة التصوير).",
        "malig_note": "⚠️ خطورة عالية: يرجى مراجعة المختص فوراً.",
        "benign_note": "✅ حالة مستقرة: المؤشرات تدل على طبيعة حميدة.",
        "others_note": "🔎 حالة عدوى: يرجى استشارة الطبيب للعلاج الموضعي."
    },
    "English": {
        "dir": "ltr",
        "title": "Advanced Skin AI Diagnostic System",
        "upload_label": "📥 Upload Scan Image or Use Camera",
        "btn_analyze": "🔍 Start Instant Analysis",
        "advice": "⚠️ Medical Note: This AI tool is for guidance only.",
        "guide_title": "📖 Medical Reference Guide",
        "malig_title": "1. Malignant Lesions",
        "malig_desc": "Skin cancers like BCC and Melanoma. Requires urgent medical intervention.",
        "benign_title": "2. Benign Lesions",
        "benign_desc": "Non-cancerous growths like Eczema and Psoriasis.",
        "others_title": "3. Infections & Others",
        "others_desc": "Conditions caused by fungi, bacteria, or allergies.",
        "link_cancer": "🔗 Global Source: Skin Cancer (Mayo Clinic)",
        "link_others": "🔗 Global Source: Skin Conditions (NHS)",
        "result_text": "Suggested Diagnosis:",
        "confidence": "Confidence Level:",
        "inconclusive": "Inconclusive: Accuracy below required threshold.",
        "malig_note": "⚠️ High Risk: Please consult a specialist immediately.",
        "benign_note": "✅ Stable: Indicators suggest a benign nature.",
        "others_note": "🔎 Infection: Please consult a doctor for treatment."
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
        with st.spinner("Downloading AI Weights..."):
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
selected_lang = st.sidebar.selectbox("Language Selection / اختيار اللغة", list(LANG_DATA.keys()))
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
            img_res = cv2.resize(np.array(img), (224, 224))
            d_in = np.expand_dims(img_res, axis=0) / 255.0
            preds = diag_model.predict(d_in)[0]
            idx = np.argmax(preds)
            score = np.max(preds)
            
            # --- تعديل الـ Thresholds (الحل المقترح للتنوع) ---
            MALIGNANT_IDS = [1, 11, 23]
            BENIGN_IDS = [0, 2, 5, 9, 13, 14, 16, 20]
            
            # خفضنا العتبات للسماح بظهور الأنواع الأخرى
            THRESH_MALIG = 0.40 # بدلاً من 0.65
            THRESH_BENIGN = 0.45 # بدلاً من 0.80
            
            final_label, final_color, final_note = T['inconclusive'], "#8E8E93", ""
            
            if idx in MALIGNANT_IDS:
                if score >= THRESH_MALIG:
                    final_label, final_color, final_note = T['malig_title'], "#FF3B30", T['malig_note']
            elif idx in BENIGN_IDS:
                if score >= THRESH_BENIGN:
                    final_label, final_color, final_note = T['benign_title'], "#34C759", T['benign_note']
            else:
                if score >= 0.35:
                    final_label, final_color, final_note = T['others_title'], "#FF9500", T['others_note']

            st.markdown(f"""
            <div dir='{T['dir']}' style="padding:30px; border-radius:20px; border:10px solid {final_color}; text-align:center; background:white;">
                <h2 style="color:{final_color};">{T['result_text']} {final_label}</h2>
                <p style='font-size:1.1em;'>{final_note}</p>
                <div style="background:{final_color}10; padding:10px; border-radius:10px;">
                    <strong>{T['confidence']} {score*100:.2f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- 5. الدليل المرجعي المحدث ---
st.markdown("---")
with st.expander(T['guide_title']):
    st.markdown(f"<div dir='{T['dir']}'>", unsafe_allow_html=True)
    
    st.subheader(T['malig_title'])
    st.write(T['malig_desc'])
    st.markdown(f"**Threshold: 0.40**")
    st.markdown(f"[Mayo Clinic - Skin Cancer](https://www.mayoclinic.org/diseases-conditions/skin-cancer/symptoms-causes/syc-20377605)")

    st.write("---")
    
    st.subheader(T['benign_title'])
    st.write(T['benign_desc'])
    st.markdown(f"**Threshold: 0.45**")

    st.write("---")

    st.subheader(T['others_title'])
    st.write(T['others_desc'])
    st.markdown(f"[NHS - Skin Conditions](https://www.nhs.uk/conditions/skin-conditions/)")
    
    st.markdown("</div>", unsafe_allow_html=True)
