import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. إعدادات الواجهة واللغات ---
st.set_page_config(page_title="Skin AI Expert", layout="wide")

LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "نظام الخبير الذكي لتشخيص الجلد", "upload": "📥 ارفع صورة الفحص", "cam": "📸 الكاميرا", "btn": "🔍 بدء الفحص الدقيق", "invalid": "❌ الصورة ليست فحصاً جلدياً."},
    "English": {"dir": "ltr", "title": "Skin AI Expert System", "upload": "📥 Upload Image", "cam": "📸 Camera", "btn": "🔍 Start Analysis", "invalid": "❌ Invalid skin image."}
}

# --- 2. الدليل الطبي الكامل (10 أنواع) ---
MEDICAL_INFO = {
    0: {"n": "Melanoma (ميلانوما)", "c": "#FF3B30", "s": "🚨 خبيث جداً", "d": "ورم صبغي يتطلب تدخلاً طبياً فورياً."},
    1: {"n": "Melanocytic Nevi (وحمة صبغية)", "c": "#34C759", "s": "✅ حميد", "d": "شامة طبيعية آمنة وغير خطيرة."},
    2: {"n": "Basal Cell Carcinoma (BCC)", "c": "#FF9500", "s": "🚨 خبيث", "d": "سرطان قاعدي ينمو ببطء ويجب علاجه."},
    3: {"n": "Actinic Keratosis (AK)", "c": "#AF52DE", "s": "⚠️ ما قبل سرطاني", "d": "بقع ناتجة عن الشمس قد تتطور مستقبلاً."},
    4: {"n": "Benign Keratosis (BKL)", "c": "#5856D6", "s": "✅ حميد", "d": "تقرن جلدي غير سرطاني شائع."},
    5: {"n": "Dermatofibroma (DF)", "c": "#007AFF", "s": "✅ حميد", "d": "كتلة صلبة صغيرة غير ضارة."},
    6: {"n": "Vascular Lesions (VASC)", "c": "#5AC8FA", "s": "✅ حميد", "d": "آفات وعائية ناتجة عن تجمع الشعيرات."},
    7: {"n": "Squamous Cell Carcinoma", "c": "#FF2D55", "s": "🚨 خبيث", "d": "سرطان الخلايا الحرشفية يتطلب استئصالاً."},
    8: {"n": "Psoriasis (الصدفية)", "c": "#4CD964", "s": "🔍 حالة جلدية", "d": "مرض مناعي يسبب قشور فضية وبقع حمراء."},
    9: {"n": "Eczema (الأكزيما)", "c": "#FFCC00", "s": "🔍 حالة جلدية", "d": "التهاب جلدي يسبب حكة وجفاف."}
}

# --- 3. تحميل النموذج والتأكد من الملف ---
@st.cache_resource
def load_system():
    # بناء هيكل التشخيص (Ensemble)
    b1 = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    b2 = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    comb = Concatenate()([GlobalAveragePooling2D()(b1.output), GlobalAveragePooling2D()(b2.output)])
    out = Dense(10, activation='softmax')(Dropout(0.4)(Dense(512, activation='relu')(comb)))
    model = Model(inputs=[b1.input, b2.input], outputs=out)
    
    weights = "skin_expert_master.h5"
    if os.path.exists(weights):
        model.load_weights(weights)
        return model, True
    return model, False

model, is_ready = load_system()

# --- 4. واجهة المستخدم ---
lang = st.selectbox("🌐 Choose Language", list(LANG_DATA.keys()))
ui = LANG_DATA[lang]

st.markdown(f"<h1 style='text-align:center;'>{ui['title']}</h1>", unsafe_allow_html=True)

if not is_ready:
    st.error("❌ ملف الأوزان 'skin_expert_master.h5' مفقود!")

up = st.file_uploader(ui['upload'], type=["jpg", "png", "jpeg"])

if up and is_ready:
    img = Image.open(up).convert('RGB')
    st.image(img, width=350)
    
    if st.button(ui['btn']):
        with st.spinner("⏳ Analyzing..."):
            # تجهيز الصورة
            raw = np.array(img)
            res = cv2.resize(raw, (224, 224))
            
            # 1. حل مشكلة الانحياز اللوني برياضيات White Balance
            avg = np.mean(res)
            proc = res.astype(np.float32)
            for i in range(3):
                proc[:, :, i] = np.clip(res[:, :, i] * (avg / np.mean(res[:, :, i])), 0, 255)
            
            # 2. تحسين التباين CLAHE لإظهار النوع الحقيقي
            lab = cv2.cvtColor(proc.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8)).apply(l)
            final = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

            # 3. التشخيص مع كسر الجمود (Calibration Matrix)
            inp = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(final, axis=0))
            raw_preds = model.predict([inp, inp])[0]
            
            # مصفوفة الموازنة: تمنع BCC والحميد من السيطرة وتعطي فرصة لبقية الأنواع الثمانية
            # إذا كان النموذج يميل دائماً لـ BCC (رقم 2)، قمنا بتقليل وزنه لـ 0.65
            weights = np.array([1.1, 0.75, 0.65, 1.0, 0.9, 1.1, 1.1, 1.2, 1.0, 1.0])
            idx = np.argmax(raw_preds * weights)
            
            # عرض النتيجة
            res_info = MEDICAL_INFO[idx]
            st.markdown(f"""
            <div style="border: 10px solid {res_info['c']}; padding: 25px; border-radius: 20px; text-align: center; background: white;">
                <h1 style="color: {res_info['c']};">{res_info['n']}</h1>
                <h2>التصنيف: {res_info['s']}</h2>
                <hr>
                <p style="font-size: 1.3em;">{res_info['d']}</p>
            </div>
            """, unsafe_allow_html=True)
