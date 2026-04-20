import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. إعدادات الواجهة الرسومية ---
st.set_page_config(page_title="Skin AI Expert System", layout="wide")

# القاموس الكامل لـ 20 لغة (كما في النسخ السابقة)
LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "النظام الخبير لتشخيص أمراض الجلد", "upload": "📥 ارفع صورة", "cam": "📸 الكاميرا", "btn": "🔍 تحليل الحالة", "invalid": "❌ الصورة ليست فحصاً جلدياً.", "advice": "⚠️ تنبيه: استشر الطبيب فوراً."},
    "English": {"dir": "ltr", "title": "Skin AI Expert Diagnostic System", "upload": "📥 Upload", "cam": "📸 Camera", "btn": "🔍 Analyze Now", "invalid": "❌ Invalid skin image.", "advice": "⚠️ Note: Consult a doctor."},
    # ... يمكن إضافة بقية اللغات هنا بنفس النمط
}

# الدليل الطبي مع مصفوفة الأوزان (Weight Matrix) لكسر الانحياز
MEDICAL_INFO = {
    0: {"n": "Melanoma", "c": "#D32F2F", "s": "🚨 خبيث جداً", "w": 1.45, "d": "أخطر أنواع سرطان الجلد."},
    1: {"n": "Melanocytic Nevi", "c": "#388E3C", "s": "✅ حميد", "w": 0.55, "d": "شامة طبيعية آمنة."},
    2: {"n": "Basal Cell Carcinoma (BCC)", "c": "#F57C00", "s": "🚨 خبيث", "w": 0.50, "d": "سرطان قاعدي ينمو ببطء."},
    3: {"n": "Actinic Keratosis", "c": "#7B1FA2", "s": "⚠️ ما قبل سرطاني", "w": 1.15, "d": "تلف شمسي قد يتطور."},
    4: {"n": "Benign Keratosis", "c": "#1976D2", "s": "✅ حميد", "w": 0.85, "d": "زوائد غير سرطانية."},
    5: {"n": "Dermatofibroma", "c": "#00796B", "s": "✅ حميد", "w": 1.20, "d": "كتلة صلبة صغيرة."},
    6: {"n": "Vascular Lesions", "c": "#C2185B", "s": "✅ حميد", "w": 1.25, "d": "آفات وعائية دموية."},
    7: {"n": "Squamous Cell Carcinoma", "c": "#E64A19", "s": "🚨 خبيث", "w": 1.35, "d": "سرطان الخلايا الحرشفية."},
    8: {"n": "Psoriasis", "c": "#512DA8", "s": "🔍 حالة جلدية", "w": 1.05, "d": "صدفية: قشور فضية."},
    9: {"n": "Eczema", "c": "#FFA000", "s": "🔍 حالة جلدية", "w": 1.15, "d": "أكزيما: جفاف وحكة."}
}

# --- 2. بناء الموديل (الحل النهائي لخطأ ValueError في صورك) ---
@st.cache_resource
def load_stable_engine():
    # استخدام مدخل واحد موحد يغذي المسارين
    master_input = Input(shape=(224, 224, 3), name="master_input")
    
    # تعريف المسارات الهجينة (Ensemble)
    base_eff = EfficientNetB0(weights=None, include_top=False)(master_input)
    base_mob = MobileNetV2(weights=None, include_top=False)(master_input)
    
    # دمج المخرجات
    gap_eff = GlobalAveragePooling2D()(base_eff)
    gap_mob = GlobalAveragePooling2D()(base_mob)
    merged = Concatenate()([gap_eff, gap_mob])
    
    # الطبقات الكثيفة النهائية (10 أصناف)
    x = Dense(512, activation='relu')(merged)
    x = Dropout(0.4)(x)
    output = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=master_input, outputs=output)
    
    # محاولة تحميل الأوزان مع التحقق من المسار
    h5_path = "skin_expert_master.h5"
    ready = False
    if os.path.exists(h5_path):
        try:
            model.load_weights(h5_path)
            ready = True
        except Exception as e:
            st.error(f"خطأ في توافق الأوزان: {e}")
    
    return model, ready

main_model, is_ready = load_stable_engine()

# --- 3. واجهة المستخدم والمنطق البرمجي ---
sel_lang = st.selectbox("🌐 اللغة / Language", list(LANG_DATA.keys()))
ui = LANG_DATA[sel_lang]

st.markdown(f"<h1 style='text-align:center;'>{ui['title']}</h1>", unsafe_allow_html=True)

if not is_ready:
    st.error(f"❌ لم يتم العثور على ملف 'skin_expert_master.h5' في المسار: {os.getcwd()}")

uploaded_file = st.file_uploader(ui['upload'], type=["jpg", "png", "jpeg"])

if uploaded_file and is_ready:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, width=350, caption="الصورة الحالية")
    
    if st.button(ui['btn'], use_container_width=True):
        with st.spinner("⏳ جاري تحليل النسيج..."):
            img_np = np.array(img)
            img_res = cv2.resize(img_np, (224, 224))
            
            # موازنة الألوان (Gray World) يدوياً لحل مشكلة AttributeError
            avg_color = np.mean(img_res)
            proc_img = img_res.astype(np.float32)
            for i in range(3):
                proc_img[:,:,i] = np.clip(img_res[:,:,i] * (avg_color / (np.mean(img_res[:,:,i]) + 1e-6)), 0, 255)
            
            # تحسين التباين (CLAHE)
            lab = cv2.cvtColor(proc_img.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
            final_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

            # إجراء التنبوء وتطبيق الأوزان لكسر الانحياز
            input_tensor = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(final_img, axis=0))
            raw_scores = main_model.predict(input_tensor)[0]
            
            # ضرب النتائج في الأوزان التصحيحية (W) لضمان دقة التصنيف
            cal_weights = np.array([v['w'] for v in MEDICAL_INFO.values()])
            final_idx = np.argmax(raw_scores * cal_weights)
            
            # عرض النتيجة المنسقة
            res = MEDICAL_INFO[final_idx]
            st.markdown(f"""
                <div style="border: 8px solid {res['c']}; padding: 25px; border-radius: 15px; text-align: center; background: white;">
                    <h1 style="color: {res['c']};">{res['n']}</h1>
                    <h3 style="background: #f8f9fa;">{res['s']}</h3>
                    <p style="font-size: 1.2em;">{res['d']}</p>
                    <strong>نسبة المطابقة النسيجية: {raw_scores[final_idx]*100:.2f}%</strong>
                </div>
            """, unsafe_allow_html=True)

# --- 4. الدليل الطبي المرجعي (مرتب) ---
st.write("---")
st.subheader("📚 الدليل الطبي المرجعي")
cols = st.columns(2)
for i, (k, v) in enumerate(MEDICAL_INFO.items()):
    target_col = cols[i % 2]
    target_col.markdown(f"<span style='color:{v['c']};'>●</span> **{v['n']}**: {v['s']}", unsafe_allow_html=True)
