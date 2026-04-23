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
st.set_page_config(page_title="Skin AI Expert System", layout="wide")

DRIVE_FILE_ID = '135lZpgsipHNk2IZBo6H4lZZ9WzVizLqb'
MODEL_PATH = "skin_expert_hybrid_24ch.h5"

# --- 2. قاموس اللغات والترجمة الكاملة للواجهة ---
LANG_DATA = {
    "العربية": {
        "dir": "rtl",
        "title": "نظام التشخيص الذكي المتطور لأمراض الجلد",
        "upload_label": "📥 ارفع صورة من مجلد Dataset للاختبار",
        "btn_analyze": "🔍 بدء التحليل الفوري",
        "advice": "⚠️ تنبيه طبي: هذا النظام أداة برمجية استرشادية، ولا يغني عن زيارة الطبيب المختص.",
        "guide_title": "📖 الدليل الطبي المرجعي والنسب المعتمدة",
        "malig_title": "1. الآفات الخبيثة (Malignant)",
        "malig_desc": "تشمل الحالات الموجودة في مجلد bcc و mel. تتطلب تدخل طبي عاجل.",
        "benign_title": "2. الآفات الحميدة (Benign)",
        "benign_desc": "تشمل الحالات في مجلدات nv, bkl, akiec. نموات غير سرطانية غالباً.",
        "others_title": "3. حالات أخرى (Others)",
        "others_desc": "تشمل الحالات في مجلد df والالتهابات الجلدية المختلفة.",
        "result_text": "التشخيص المقترح:",
        "confidence": "نسبة اليقين:",
        "inconclusive": "تحليل غير حاسم: لم تتجاوز النسبة حد الثقة المطلوب برمجياً.",
        "malig_note": "⚠️ خطورة عالية: يرجى مراجعة المختص فوراً.",
        "benign_note": "✅ حالة مستقرة: المؤشرات تدل على طبيعة حميدة.",
        "others_note": "🔎 حالة عدوى/التهاب: يرجى استشارة الطبيب للعلاج."
    },
    "English": {
        "dir": "ltr",
        "title": "Advanced Skin AI Diagnostic System",
        "upload_label": "📥 Upload an image from Dataset for testing",
        "btn_analyze": "🔍 Start Instant Analysis",
        "advice": "⚠️ Medical Note: This AI tool is for guidance only.",
        "guide_title": "📖 Medical Reference Guide & Thresholds",
        "malig_title": "1. Malignant Lesions",
        "malig_desc": "Includes cases from bcc and mel folders. Requires urgent medical care.",
        "benign_title": "2. Benign Lesions",
        "benign_desc": "Includes cases from nv, bkl, and akiec. Mostly non-cancerous.",
        "others_title": "3. Other Conditions",
        "others_desc": "Includes cases from df and various skin inflammations.",
        "result_text": "Suggested Diagnosis:",
        "confidence": "Confidence Level:",
        "inconclusive": "Inconclusive: Confidence below required threshold.",
        "malig_note": "⚠️ High Risk: Please consult a specialist immediately.",
        "benign_note": "✅ Stable: Indicators suggest a benign nature.",
        "others_note": "🔎 Infection/Others: Please consult a doctor for treatment."
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
        with st.spinner("جاري جلب أوزان الموديل الهجين..."):
            URL = "https://docs.google.com/uc?export=download"
            session = requests.Session()
            r = session.get(URL, params={'id': DRIVE_FILE_ID}, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(32768):
                    if chunk: f.write(chunk)
    model.load_weights(MODEL_PATH)
    return model

diag_model = load_hybrid_model()

# --- 4. واجهة المستخدم والتفاعل مع اختيار اللغة ---
selected_lang = st.sidebar.selectbox("Language / اللغة", list(LANG_DATA.keys()))
T = LANG_DATA[selected_lang]

st.markdown(f"<div dir='{T['dir']}' style='text-align:center;'><h1 style='color:#1E3A8A;'>{T['title']}</h1></div>", unsafe_allow_html=True)
st.warning(T['advice'])

uploaded = st.file_uploader(T['upload_label'], type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, use_container_width=True, caption="Image from Balanced_Skin_Dataset")
    with col2:
        if st.button(T['btn_analyze']):
            # --- معالجة الصورة وتحليل النتائج ---
            img_res = cv2.resize(np.array(img), (224, 224))
            d_in = np.expand_dims(img_res, axis=0) / 255.0
            preds = diag_model.predict(d_in)[0]
            idx = np.argmax(preds)
            score = np.max(preds)
            
            # --- ضبط الـ Thresholds برمجياً (مطابق للدليل) ---
            # تم ضبطها لتسمح بالتنوع في النتائج بناءً على مجلداتك
            THRESH_MALIG = 0.40
            THRESH_BENIGN = 0.45
            THRESH_OTHERS = 0.35

            # ربط الأصناف بمجلدات الـ Dataset (bcc=1, mel=11, nv=9, akiec=0, df=4, bkl=16)
            MALIGNANT_IDS = [1, 11, 23] 
            BENIGN_IDS = [0, 2, 5, 9, 13, 14, 16, 20]
            
            f_label, f_color, f_note = T['inconclusive'], "#8E8E93", ""
            
            if idx in MALIGNANT_IDS:
                if score >= THRESH_MALIG:
                    f_label, f_color, f_note = T['malig_title'], "#FF3B30", T['malig_note']
            elif idx in BENIGN_IDS:
                if score >= THRESH_BENIGN:
                    f_label, f_color, f_note = T['benign_title'], "#34C759", T['benign_note']
            else:
                if score >= THRESH_OTHERS:
                    f_label, f_color, f_note = T['others_title'], "#FF9500", T['others_note']

            # عرض النتيجة المعتمدة على المعالجة والنسبة
            st.markdown(f"""
            <div dir='{T['dir']}' style="padding:30px; border-radius:20px; border:10px solid {f_color}; text-align:center; background:white; box-shadow: 0px 4px 15px rgba(0,0,0,0.1);">
                <h2 style="color:{f_color};">{T['result_text']} {f_label}</h2>
                <p style='font-size:1.1em;'>{f_note}</p>
                <div style="background:{f_color}10; padding:10px; border-radius:10px;">
                    <strong>{T['confidence']} {score*100:.2f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- 5. الدليل المرجعي (مدمج مع نسب المعالجة) ---
st.markdown("---")
with st.expander(T['guide_title']):
    st.markdown(f"<div dir='{T['dir']}'>", unsafe_allow_html=True)
    
    # خبيث
    st.subheader(T['malig_title'])
    st.write(T['malig_desc'])
    st.info(f"Threshold (نسبة المعالجة برمجياً): 0.40")
    st.markdown("[🔗 Mayo Clinic - Skin Cancer](https://www.mayoclinic.org/diseases-conditions/skin-cancer/symptoms-causes/syc-20377605)")

    st.write("---")
    
    # حميد
    st.subheader(T['benign_title'])
    st.write(T['benign_desc'])
    st.success(f"Threshold (نسبة المعالجة برمجياً): 0.45")

    st.write("---")

    # أخرى
    st.subheader(T['others_title'])
    st.write(T['others_desc'])
    st.warning(f"Threshold (نسبة المعالجة برمجياً): 0.35")
    st.markdown("[🔗 NHS - Skin Conditions](https://www.nhs.uk/conditions/skin-conditions/)")
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(f"<div style='text-align:center; color:#888; margin-top:50px;'><small>Hybrid Skin AI v7.9 | {selected_lang} Edition</small></div>", unsafe_allow_html=True)
