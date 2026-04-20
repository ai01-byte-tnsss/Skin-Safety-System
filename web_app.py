import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. الإعدادات واللغات الثابتة ---
st.set_page_config(page_title="Skin AI System", layout="wide")

LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "نظام الفحص الذكي للجلد", "upload": "📥 ارفع صورة", "cam": "📸 كاميرا", "btn": "🔍 بدء الفحص", "invalid": "❌ عذراً، الصورة ليست فحصاً جلدياً.", "advice": "⚠️ تنبيه: هذا النظام أداة برمجية استرشادية فقط وليس بديلاً عن الطبيب المختص."},
    "English": {"dir": "ltr", "title": "Skin AI Diagnostic System", "upload": "📥 Upload", "cam": "📸 Camera", "btn": "🔍 Start Analysis", "invalid": "❌ Invalid skin image.", "advice": "⚠️ Note: This system is a guidance tool and not a substitute for a professional doctor."},
    "Kurdî": {"dir": "rtl", "title": "ژیری پێست", "upload": "📥 وێنە", "cam": "📸 کامێرا", "btn": "🔍 شیکاری", "invalid": "❌ هەڵە.", "advice": "⚠️ ئاگاداري: جێگرەوەی پزیشک نییە."}
}

# --- 2. الدليل الطبي المرجعي (10 أنواع) ---
MEDICAL_INFO = {
    0: {"n": "Melanoma (ميلانوما)", "c": "#FF3B30", "s": "🚨 خبيث جداً", "d": "أخطر أنواع سرطان الجلد، يتطلب فحصاً طبياً فورياً."},
    1: {"n": "Melanocytic Nevi (وحمة صبغية)", "c": "#34C759", "s": "✅ حميد", "d": "شامة طبيعية، آمنة ومستقرة في أغلب الحالات."},
    2: {"n": "Basal Cell Carcinoma (BCC)", "c": "#FF9500", "s": "🚨 خبيث", "d": "سرطان الخلايا القاعدية، ينمو ببطء ويجب استئصاله."},
    3: {"n": "Actinic Keratosis (AK)", "c": "#AF52DE", "s": "⚠️ ما قبل سرطاني", "d": "بقع ناتجة عن الشمس قد تتطور لسرطان مستقبلاً."},
    4: {"n": "Benign Keratosis (BKL)", "c": "#5856D6", "s": "✅ حميد", "d": "زوائد جلدية غير سرطانية تظهر مع تقدم العمر."},
    5: {"n": "Dermatofibroma (DF)", "c": "#007AFF", "s": "✅ حميد", "d": "كتلة صلبة صغيرة، غير ضارة تماماً."},
    6: {"n": "Vascular Lesions (VASC)", "c": "#5AC8FA", "s": "✅ حميد", "d": "آفات وعائية ناتجة عن تجمع الشعيرات الدموية."},
    7: {"n": "Squamous Cell Carcinoma", "c": "#FF2D55", "s": "🚨 خبيث", "d": "سرطان الخلايا الحرشفية، يتطلب تدخلاً طبياً مختصاً."},
    8: {"n": "Psoriasis (الصدفية)", "c": "#4CD964", "s": "🔍 حالة جلدية", "d": "مرض مناعي يسبب قشور فضية وبقع حمراء."},
    9: {"n": "Eczema (الأكزيما)", "c": "#FFCC00", "s": "🔍 حالة جلدية", "d": "التهاب جلدي يسبب حكة واحمرار وجفاف."}
}

# --- 3. تحميل المحركات والذكاء الاصطناعي (تم الإصلاح الجذري هنا) ---
@st.cache_resource
def load_full_system():
    # موديل الفلترة (MobileNetV2 الأصلي)
    f_mod = tf.keras.applications.MobileNetV2(weights="imagenet")
    
    # بناء الموديل الهجين بمدخل واحد لتجنب خطأ ValueError
    shared_input = Input(shape=(224, 224, 3))
    
    b1 = EfficientNetB0(weights=None, include_top=False)(shared_input)
    b2 = MobileNetV2(weights=None, include_top=False)(shared_input)
    
    comb = Concatenate()([GlobalAveragePooling2D()(b1), GlobalAveragePooling2D()(b2)])
    # تأكد أن المخرج يطابق عدد الأصناف في ملف skin_expert_master.h5 (سواء كان 7 أو 10)
    out = Dense(10, activation='softmax')(Dropout(0.4)(Dense(512, activation='relu')(comb)))
    
    d_mod = Model(inputs=shared_input, outputs=out)
    
    h5_path = "skin_expert_master.h5"
    if os.path.exists(h5_path):
        try:
            d_mod.load_weights(h5_path)
        except:
            st.error("⚠️ تنبيه: ملف الأوزان لا يتوافق مع عدد الأصناف الحالي (10).")
            
    return f_mod, d_mod

filter_m, diag_m = load_full_system()

# --- 4. واجهة المستخدم ---
selected_lang = st.selectbox("🌐 Choose Language / اختر اللغة", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

st.markdown(f"<h1 style='text-align:center; color:#1E3A8A;'>{t['title']}</h1>", unsafe_allow_html=True)
st.warning(t['advice'])

c1, c2 = st.columns(2)
with c1:
    m = st.radio("", [t['upload'], t['cam']], horizontal=True)
    file = st.file_uploader("", type=["jpg", "png", "jpeg"]) if "ارفع" in m or "Upload" in m else st.camera_input("")

if file:
    img = Image.open(file).convert('RGB')
    with c2: st.image(img, use_container_width=True)
    
    if st.button(t['btn']):
        with st.spinner("⏳ Analyzing..."):
            img_np = np.array(img)
            img_res = cv2.resize(img_np, (224, 224))
            
            # المرحلة 1: فلترة الكائنات
            xf = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_res.copy(), axis=0))
            f_preds = filter_m.predict(xf)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(f_preds, top=3)[0]
            
            is_valid = True
            for _, label, score in decoded:
                if any(x in label.lower() for x in ['car', 'wheel', 'dog', 'cat', 'flower', 'laptop', 'building']) and score > 0.4:
                    is_valid = False
            
            if not is_valid:
                st.error(t['invalid'])
            else:
                # المرحلة 2: تحسين الصورة (CLAHE)
                lab = cv2.cvtColor(img_res, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                img_proc = cv2.merge((clahe.apply(l), a, b))
                img_proc = cv2.cvtColor(img_proc, cv2.COLOR_LAB2RGB)
                
                # المرحلة 3: التشخيص (بمدخل واحد الآن)
                inp = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_proc, axis=0))
                res_preds = diag_m.predict(inp)[0] # تم تعديله ليرسل مدخلاً واحداً فقط
                
                idx = np.argmax(res_preds)
                res = MEDICAL_INFO[idx]
                
                st.markdown(f"""
                <div style="padding:30px; border-radius:15px; border:10px solid {res['c']}; text-align:center; background:white; margin-top:20px;">
                    <h1 style="color:{res['c']}; font-size:2.4em;">{res['n']}</h1>
                    <h2 style="color:#444;">التصنيف: {res['s']}</h2>
                    <hr style="border:1px solid {res['c']}; width:40%; margin:auto;">
                    <p style="font-size:1.3em; color:#333; margin-top:15px; font-weight:bold;">{res['d']}</p>
                    <p>نسبة التأكد: {res_preds[idx]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

# --- 5. الدليل المرجعي ---
st.write("---")
st.subheader("📖 الدليل المرجعي")
selected_info = st.selectbox("اختر فئة لعرض التفاصيل:", [v['n'] for v in MEDICAL_INFO.values()])
for k, v in MEDICAL_INFO.items():
    if v['n'] == selected_info:
        st.markdown(f"<div style='background-color:{v['c']}10; padding:20px; border-right:10px solid {v['c']};'><h3>{v['n']}</h3><p>{v['d']}</p></div>", unsafe_allow_html=True)
