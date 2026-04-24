import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import time

# --- 1. إعدادات الصفحة والتنسيق ---
st.set_page_config(page_title="Skin AI Expert System", page_icon="🧬", layout="wide")

# تنسيق الواجهة ودعم اللغة العربية
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
    .main { background-color: #f9fbfb; }
    .stButton>button { width: 100%; border-radius: 12px; height: 3.5em; background-color: #1E3A8A; color: white; font-weight: bold; }
    .report-card { padding: 20px; border-radius: 15px; background-color: white; border-right: 10px solid #1E3A8A; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. الدليل الطبي المنسدل (24 فئة) ---
MEDICAL_INFO = {
    0: {"n": "Acne and Rosacea", "s": "✅ حالة شائعة", "c": "#34C759", "d": "حب الشباب والوردية؛ حالات تتعلق بالتهاب الغدد الدهنية."},
    1: {"n": "Actinic Keratosis / BCC", "s": "🚨 ما قبل سرطاني", "c": "#FF3B30", "d": "تقرن ضوئي أو سرطان الخلايا القاعدية؛ يتطلب فحصاً."},
    2: {"n": "Atopic Dermatitis", "s": "🔍 حالة جلدية", "c": "#5856D6", "d": "التهاب الجلد التأتبي؛ نوع من الحساسية الجلدية."},
    # ... يمكن إضافة بقية الـ 24 فئة هنا بنفس التنسيق
}

# --- 3. محرك تحميل النموذج (التعامل مع ZIP) ---
@st.cache_resource
def load_ai_engine():
    zip_path = "skin_expert_hybrid_24ch.zip"
    extract_path = "temp_model_dir"
    try:
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            time.sleep(2) # لضمان اكتمال فك الضغط
            
            h5_path = None
            for root, dirs, files in os.walk(extract_path):
                for f in files:
                    if f.endswith(".h5"):
                        h5_path = os.path.join(root, f)
                        break
            
            if h5_path:
                # بناء الهيكل الهجين
                b1 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
                b2 = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
                merge = Concatenate()([GlobalAveragePooling2D()(b1.output), GlobalAveragePooling2D()(b2.output)])
                out = Dense(24, activation='softmax')(Dropout(0.4)(Dense(512, activation='relu')(merge)))
                
                m = Model(inputs=[b1.input, b2.input], outputs=out)
                m.load_weights(h5_path)
                return m
        return None
    except Exception as e:
        return f"Error: {e}"

# محاولة تحميل النموذج
model_engine = load_ai_engine()

# --- 4. واجهة المستخدم الرسومية ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>الذكاء الاصطناعي لفحص سلامة الجلد 🧬</h1>", unsafe_allow_html=True)

# أ) شريط الأدوات العلوية (اللغات والدليل)
col_a, col_b = st.columns(2)
with col_a:
    lang = st.selectbox("🌐 اختر اللغة (Language)", ["العربية", "English", "Kurdish"])

with col_b:
    with st.expander("📖 الدليل الإرشادي للأمراض (24 فئة)"):
        for key, val in MEDICAL_INFO.items():
            st.write(f"**{val['n']}**: {val['d']}")

st.write("---")

# ب) خيارات إدخال الصورة (تحميل أو كاميرا)
st.subheader("📸 خطوة 1: تزويد النظام بالصورة")
tab_upload, tab_camera = st.tabs(["📤 رفع ملف من الجهاز", "📷 التقاط صورة بالكاميرا"])

with tab_upload:
    file_input = st.file_uploader("اختر صورة واضحة (JPG, PNG)", type=["jpg", "png", "jpeg"])

with tab_camera:
    cam_input = st.camera_input("وجه الكاميرا نحو المنطقة المصابة")

active_image = file_input if file_input else cam_input

# ج) منطقة العرض وزر الفحص
if active_image:
    st.write("---")
    col_view, col_action = st.columns([1, 1])
    
    with col_view:
        img_display = Image.open(active_image).convert('RGB')
        st.image(img_display, caption="الصورة المراد فحصها", use_container_width=True)

    with col_action:
        st.subheader("🔍 خطوة 2: التحليل")
        if st.button("بدء فحص الصورة الآن"):
            if model_engine and not isinstance(model_engine, str):
                with st.spinner("⏳ جاري تحليل الأنماط الحيوية..."):
                    # المعالجة المسبقة
                    img_cv = cv2.resize(np.array(img_display), (224, 224))
                    tensor = (img_cv.astype(np.float32) / 255.0)[np.newaxis, ...]
                    
                    # التنبؤ
                    preds = model_engine.predict([tensor, tensor])[0]
                    idx = np.argmax(preds)
                    conf = preds[idx]
                    
                    # عرض النتيجة
                    res_data = MEDICAL_INFO.get(idx, {"n": "غير محدد", "s": "🔍", "c": "#8E8E93", "d": "لم يتم التعرف بدقة."})
                    
                    st.markdown(f"""
                    <div class="report-card" style="border-right-color: {res_data['c']};">
                        <h2 style="color:{res_data['c']};">{res_data['n']}</h2>
                        <p><b>نسبة الثقة:</b> {conf:.2%}</p>
                        <p>{res_data['d']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("⚠️ المحرك غير جاهز حالياً. تأكد من رفع ملف skin_expert_hybrid_24ch.zip بشكل صحيح.")

st.write("---")
st.caption("نظام Skin Safety System V2.1 | تم التطوير لصالح مشروع التخرج 2026")
