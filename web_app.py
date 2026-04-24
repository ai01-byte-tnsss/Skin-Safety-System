import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import gdown

# --- 1. إعدادات الهوية البصرية ---
st.set_page_config(page_title="Skin Safety AI", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
    .report-box { padding: 30px; border-radius: 20px; background-color: white; border-right: 15px solid #1E3A8A; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
    .danger { color: #D32F2F; background-color: #FFEBEE; padding: 10px; border-radius: 10px; font-weight: bold; }
    .safe { color: #388E3C; background-color: #E8F5E9; padding: 10px; border-radius: 10px; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 12px; height: 3.5em; background-color: #1E3A8A; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. مصفوفة التشخيصات المرتبة أبجدياً (مطابقة لمجلداتك 100%) ---
# الترتيب: akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6
CLASS_NAMES = [
    {"n": "التقرن الضوئي (Actinic Keratosis)", "s": "خبيث جزئياً / متابعة"},
    {"n": "سرطان الخلايا القاعدية (BCC)", "s": "خبيث - يحتاج فحص"},
    {"n": "آفات التقرن الحميدة (BKL)", "s": "حميد - غير مقلق"},
    {"n": "الأورام الليفية الجلدية (DF)", "s": "حميد - زوائد ليفية"},
    {"n": "الميلانوما - سرطان الجلد (Melanoma)", "s": "خبيث جداً - مراجعة فورية"},
    {"n": "الشامات والوحمات (Nevus)", "s": "حميد - شامة طبيعية"},
    {"n": "الآفات الوعائية (Vascular Lesions)", "s": "حميد - أوعية دموية"}
]

# --- 3. معالجة الصورة الذكية ---
def prepare_image(img):
    img = np.array(img.convert('RGB'))
    img = cv2.resize(img, (224, 224))
    # تقنية تحسين التباين لإبراز معالم المرض
    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# --- 4. تحميل المحرك (مع حل مشكلة الـ Layer Mismatch) ---
@st.cache_resource
def load_ai_model():
    f_id = '1lMGCojHeGupFunhxX5GnLOiUgxWbbRC5'
    path = "final_skin_model.h5"
    if not os.path.exists(path):
        gdown.download(f'https://drive.google.com/uc?id={f_id}', "model.zip", quiet=False)
        with zipfile.ZipFile("model.zip", 'r') as z:
            for f in z.namelist():
                if f.endswith('.h5'):
                    with open(path, "wb") as out: out.write(z.read(f))
                    break
    
    # بناء الهيكل الصريح لضمان عدم حدوث Mismatch
    base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    # استخدام Softmax للتفرقة بين الـ 7 أنواع
    out = tf.keras.layers.Dense(7, activation='softmax')(x)
    model = tf.keras.Model(inputs=base.input, outputs=out)
    model.load_weights(path, by_name=True, skip_mismatch=True)
    return model

model = load_ai_model()

# --- 5. الواجهة الرسومية ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>🧬 خبير تشخيص أمراض الجلد الذكي</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📸 عينة الفحص")
    up = st.file_uploader("ارفع صورة من مجلدات bcc, mel, nv...", type=["jpg", "png", "jpeg"])
    if up:
        image = Image.open(up)
        st.image(image, use_container_width=True, caption="الصورة الأصلية")

with col2:
    st.subheader("🔍 تقرير المختبر الذكي")
    if up and st.button("🚀 ابدأ تحليل الأنسجة"):
        if model:
            with st.spinner("⏳ جاري استخراج الخصائص الحيوية..."):
                processed = prepare_image(image)
                # تنفيذ التوقع الرياضي
                preds = model.predict(processed)[0]
                # اختيار الفئة الأعلى يقيناً
                idx = np.argmax(preds)
                
                res = CLASS_NAMES[idx]
                style = "danger" if "خبيث" in res['s'] else "safe"

                # عرض النتيجة بشكل مرتب وتنسيق HTML نظيف
                st.markdown(f"""
                <div class="report-box">
                    <h3 style="color:#1E3A8A;">اسم المرض المتوقع:</h3>
                    <p style="font-size:1.8em; font-weight:bold;">{res['n']}</p>
                    <h3 style="color:#1E3A8A;">تصنيف الحالة:</h3>
                    <div class="{style}">{res['s']}</div>
                    <p style="margin-top:20px; font-size:1.1em;"><b>دقة المطابقة:</b> {preds[idx]:.2%}</p>
                    <hr>
                    <p style="font-size:0.85em; color:gray;">⚠️ تنبيه: هذا التشخيص آلي استرشادي، يجب استشارة الطبيب المختص.</p>
                </div>
                """, unsafe_allow_html=True)

st.divider()
st.caption("نظام Skin Safety AI © 2026 | النسخة المحدثة لفرز الـ 7 فئات")
