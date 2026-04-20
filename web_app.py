import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. إعدادات الصفحة واللغات ---
st.set_page_config(page_title="Global Skin AI Expert", layout="wide")

LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "نظام التشخيص العالمي الذكي للجلد", "upload": "📥 ارفع صورة", "cam": "📸 كاميرا", "btn": "🔍 تحليل الأنسجة", "invalid": "❌ الصورة لا تبدو فحصاً جلدياً.", "advice": "⚠️ تنبيه: استشر الطبيب فوراً."},
    "English": {"dir": "ltr", "title": "Global AI Skin Diagnostic", "upload": "📥 Upload", "cam": "📸 Camera", "btn": "🔍 Analyze Tissue", "invalid": "❌ Invalid Image.", "advice": "⚠️ Note: Consult a doctor."},
    "Kurdî": {"dir": "rtl", "title": "ژیری پێست", "upload": "📥 وێنە", "cam": "📸 کامێرا", "btn": "🔍 شیکاری", "invalid": "❌ هەڵە.", "advice": "⚠️ پزیشک ببینە."}
    # ... بقية اللغات تعمل بنفس النمط
}

# --- 2. الدليل الطبي الملون (7 أصناف للتشخيص + أصناف للشرح) ---
MEDICAL_INFO = {
    0: {"n": "Melanoma (ميلانوما)", "c": "#FF0000", "s": "🚨 خبيث جداً", "d": "أخطر أنواع سرطان الجلد."},
    1: {"n": "Melanocytic Nevi (وحمة)", "c": "#27AE60", "s": "✅ حميد", "d": "شامات طبيعية آمنة."},
    2: {"n": "Basal Cell Carcinoma (BCC)", "c": "#C0392B", "s": "🚨 خبيث", "d": "سرطان الخلايا القاعدية، ينمو ببطء."},
    3: {"n": "Actinic Keratosis (AK)", "c": "#E67E22", "s": "⚠️ ما قبل سرطاني", "d": "بقع ناتجة عن الشمس قد تتحول لسرطان."},
    4: {"n": "Benign Keratosis (BKL)", "c": "#2ECC71", "s": "✅ حميد", "d": "زوائد جلدية غير سرطانية."},
    5: {"n": "Dermatofibroma (DF)", "c": "#16A085", "s": "✅ حميد", "d": "كتلة صلبة صغيرة تظهر في الساقين."},
    6: {"n": "Vascular Lesions (VASC)", "c": "#8E44AD", "s": "✅ حميد", "d": "آفات وعائية دموية."}
}

# --- 3. محركات الذكاء الاصطناعي (الإصلاح التقني الجذري) ---
@st.cache_resource
def load_engines():
    # موديل الفلترة
    f_mod = tf.keras.applications.MobileNetV2(weights="imagenet")
    
    # بناء الموديل الهجين بمدخل واحد (حل خطأ ValueError)
    master_in = Input(shape=(224, 224, 3))
    
    b1 = EfficientNetB0(weights=None, include_top=False)(master_in)
    b2 = MobileNetV2(weights=None, include_top=False)(master_in)
    
    comb = Concatenate()([GlobalAveragePooling2D()(b1), GlobalAveragePooling2D()(b2)])
    # تأكد أن المخرج 7 ليطابق ملف الأوزان
    out = Dense(7, activation='softmax')(Dropout(0.5)(Dense(512, activation='relu')(comb)))
    
    d_mod = Model(inputs=master_in, outputs=out)
    
    # تحميل الأوزان مع التحقق من المسار
    h5_file = "skin_expert_master.h5"
    ready = False
    if os.path.exists(h5_file):
        try:
            d_mod.load_weights(h5_file)
            ready = True
        except: st.error("❌ ملف الأوزان لا يطابق الهيكلية!")
    
    return f_mod, d_mod, ready

filter_m, diag_m, is_ready = load_engines()

# --- 4. واجهة المستخدم ---
selected_lang = st.selectbox("🌐 Choose Language / اختر اللغة", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

st.markdown(f"<h1 style='text-align:center;'>{t['title']}</h1>", unsafe_allow_html=True)

if not is_ready:
    st.error(f"⚠️ ملف الأوزان مفقود في: {os.getcwd()}")

file = st.file_uploader(t['upload'], type=["jpg", "png", "jpeg"])

if file and is_ready:
    img = Image.open(file).convert('RGB')
    st.image(img, width=300)
    
    if st.button(t['btn']):
        with st.spinner("Processing..."):
            img_np = np.array(img)
            img_res = cv2.resize(img_np, (224, 224))
            
            # فلترة الصور (التأكد أنها جلد)
            xf = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_res, axis=0))
            f_preds = filter_m.predict(xf)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(f_preds, top=3)[0]
            
            is_skin = any(x in str(decoded).lower() for x in ['skin', 'face', 'hand', 'arm', 'leg', 'neck'])
            
            if not is_skin:
                st.error(t['invalid'])
            else:
                # تحسين الصورة
                lab = cv2.cvtColor(img_res, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_proc = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)
                
                # التشخيص
                inp = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_proc, axis=0))
                res_preds = diag_m.predict(inp)[0] # مدخل واحد الآن
                idx = np.argmax(res_preds)
                
                info = MEDICAL_INFO[idx]
                st.markdown(f"""
                <div style="padding:20px; border:5px solid {info['c']}; border-radius:15px; text-align:center; background:white;">
                    <h2 style="color:{info['c']};">{info['n']}</h2>
                    <h3>{info['s']}</h3>
                    <hr>
                    <h4>الدقة التقنية: {res_preds[idx]*100:.1f}%</h4>
                </div>
                """, unsafe_allow_html=True)
