import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import gdown

# --- 1. إعدادات الواجهة الاحترافية ---
st.set_page_config(page_title="Skin Safety AI Expert", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
    .report-box { padding: 30px; border-radius: 20px; background-color: white; border-right: 15px solid #1E3A8A; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
    .status-danger { color: #D32F2F; background-color: #FFEBEE; padding: 10px; border-radius: 10px; font-weight: bold; }
    .status-safe { color: #388E3C; background-color: #E8F5E9; padding: 10px; border-radius: 10px; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 12px; height: 3.5em; background-color: #1E3A8A; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. مصفوفة التصنيف المرتبة (مطابقة لمجلداتك السبعة 100%) ---
# الترتيب الأبجدي للمجلدات كما ظهر في صورتك: akiec, bcc, bkl, df, mel, nv, vasc
CLASS_MAPPING = {
    0: {"name": "التقرن الضوئي (Actinic Keratosis)", "status": "خبيث جزئياً / متابعة"},
    1: {"name": "سرطان الخلايا القاعدية (BCC)", "status": "خبيث - يتطلب تدخل"},
    2: {"name": "آفات التقرن الحميدة (BKL)", "status": "حميد - غير مقلق"},
    3: {"name": "الأورام الليفية الجلدية (DF)", "status": "حميد - زوائد ليفية"},
    4: {"name": "الميلانوما - سرطان الجلد (Melanoma)", "status": "خبيث جداً - مراجعة فورية"},
    5: {"name": "الشامات والوحمات (Nevus)", "status": "حميد - شامة طبيعية"},
    6: {"name": "الآفات الوعائية (Vascular Lesions)", "status": "حميد - أوعية دموية"}
}

# --- 3. الدوال الرياضية ومعالجة المصفوفات (Normalization & Resizing) ---
def apply_skin_math(image):
    # أ. دالة تغيير الحجم لمصفوفة ثابتة (224x224)
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    
    # ب. فلاتر تحسين التباين لإبراز حدود الإصابة
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    # ج. دالة التطبيع (Normalization) تحويل النطاق إلى [0, 1]
    return (img.astype(np.float32) / 255.0)[np.newaxis, ...]

# --- 4. تحميل المحرك (التلافيف وسوفت ماكس) ---
@st.cache_resource
def load_expert_engine():
    f_id = '1lMGCojHeGupFunhxX5GnLOiUgxWbbRC5'
    path = "skin_model_final.h5"
    if not os.path.exists(path):
        gdown.download(f'https://drive.google.com/uc?id={f_id}', "model.zip", quiet=False)
        with zipfile.ZipFile("model.zip", 'r') as z:
            for f in z.namelist():
                if f.endswith('.h5'):
                    with open(path, "wb") as out: out.write(z.read(f))
                    break
    try:
        # بناء هيكل CNN ليتناسب مع الـ 7 فئات فقط
        base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dense(512, activation='relu')(x) # ReLU Function
        # دالة سوفت ماكس لـ 7 فئات
        output = tf.keras.layers.Dense(7, activation='softmax')(x) 
        
        model = tf.keras.Model(inputs=base.input, outputs=output)
        model.load_weights(path, by_name=True, skip_mismatch=True)
        return model
    except:
        return tf.keras.models.load_model(path, compile=False)

model = load_expert_engine()

# --- 5. الواجهة والتشخيص الفوري ---
st.markdown("<h1 style='text-align:center;'>🧬 خبير سلامة الجلد - نظام التصنيف السباعي</h1>", unsafe_allow_html=True)

up = st.file_uploader("ارفع صورة الفحص من المجلدات الموضحة في صورتك", type=["jpg", "png", "jpeg"])

if up:
    col1, col2 = st.columns([1, 1.2], gap="large")
    img_raw = Image.open(up)
    with col1:
        st.image(img_raw, use_container_width=True, caption="العينة المرفوعة")
    
    with col2:
        if st.button("🚀 بدء تحليل الأنسجة (Softmax)"):
            if model:
                with st.spinner("⏳ جاري معالجة المصفوفات واستخراج الميزات..."):
                    # المعالجة والتطبيع
                    processed_img = apply_skin_math(img_raw)
                    # التنبؤ (دالة التلافيف وسوفت ماكس)
                    preds = model.predict(processed_img)[0]
                    # دالة Argmax لاختيار الفئة الصحيحة
                    idx = np.argmax(preds)
                    
                    res = CLASS_MAPPING[idx]
                    css = "status-danger" if "خبيث" in res['status'] else "status-safe"

                    st.markdown(f"""
                    <div class="report-box">
                        <h3 style="color:#1E3A8A;">التشخيص المكتشف:</h3>
                        <p style="font-size:1.7em; font-weight:bold;">{res['name']}</p>
                        <h3 style="color:#1E3A8A;">تصنيف الحالة:</h3>
                        <div class="{css}">{res['status']}</div>
                        <hr>
                        <p><b>دقة دالة Softmax:</b> {preds[idx]:.2%}</p>
                        <p style="font-size:0.85em; color:gray;">يتم التصنيف بناءً على ترتيب المجلدات السبعة المستخرجة.</p>
                    </div>
                    """, unsafe_allow_html=True)
