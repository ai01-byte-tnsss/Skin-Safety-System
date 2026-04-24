import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import gdown

# --- 1. إعدادات الواجهة ---
st.set_page_config(page_title="Skin Expert AI", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
    .stButton>button { width: 100%; border-radius: 12px; background-color: #1E3A8A; color: white; font-weight: bold; height: 3.5em; }
    .report-box { padding: 25px; border-radius: 15px; background-color: white; border-right: 10px solid #1E3A8A; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    .danger { color: #D32F2F; background-color: #FFEBEE; padding: 10px; border-radius: 8px; font-weight: bold; }
    .safe { color: #388E3C; background-color: #E8F5E9; padding: 10px; border-radius: 8px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. الترتيب الأبجدي الدقيق للمجلدات (هذا هو سر التفرقة بين الأنواع) ---
# الموديل يقرأ المجلدات بترتيب أبجدي: akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6
DISEASES_DB = {
    0: {"name": "التقرن الضوئي (akiec)", "status": "خبيث جزئياً - Pre-cancerous"},
    1: {"name": "سرطان الخلايا القاعدية (bcc)", "status": "خبيث - Basal Cell Carcinoma"},
    2: {"name": "آفات التقرن الحميدة (bkl)", "status": "حميد - Benign Keratosis"},
    3: {"name": "الأورام الليفية الجلدية (df)", "status": "حميد - Dermatofibroma"},
    4: {"name": "الميلانوما - سرطان الجلد (mel)", "status": "خبيث جداً - Melanoma"},
    5: {"name": "الشامات والوحمات (nv)", "status": "حميد - Melanocytic Nevi"},
    6: {"name": "الآفات الوعائية (vasc)", "status": "حميد - Vascular Lesions"}
}

# --- 3. معالجة الصورة المتقدمة ---
def preprocess_input(img):
    img = np.array(img.convert('RGB'))
    img = cv2.resize(img, (224, 224))
    # التحسين لزيادة الفوارق بين الخلايا
    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    return (img.astype(np.float32) / 255.0)[np.newaxis, ...]

# --- 4. المحرك (بناء نظيف وشامل) ---
@st.cache_resource
def load_model_fixed():
    f_id = '1lMGCojHeGupFunhxX5GnLOiUgxWbbRC5'
    model_h5 = "final_fix.h5"
    
    if not os.path.exists(model_h5):
        try:
            gdown.download(f'https://drive.google.com/uc?id={f_id}', "data.zip", quiet=False)
            with zipfile.ZipFile("data.zip", 'r') as z:
                for f in z.namelist():
                    if f.endswith('.h5'):
                        with open(model_h5, "wb") as out: out.write(z.read(f))
                        break
        except: return None

    try:
        # بناء الهيكل المتوافق مع عدد مجلداتك السبعة
        base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(7, activation='softmax')(x) # 7 فئات فقط
        model = tf.keras.Model(inputs=base.input, outputs=out)
        
        # تحميل الأوزان بدقة
        model.load_weights(model_h5, by_name=True, skip_mismatch=True)
        return model
    except:
        return tf.keras.models.load_model(model_h5, compile=False)

model = load_model_fixed()

# --- 5. الواجهة والتشخيص ---
st.markdown("<h1 style='text-align:center;'>🧬 نظام التفرقة بين أمراض الجلد الذكي</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📸 رفع العينة")
    file = st.file_uploader("اختر صورة من مجلدات bcc أو mel أو nv...", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file)
        st.image(img, use_container_width=True)

with col2:
    st.subheader("🔍 نتيجة التقرير")
    if file and st.button("🚀 ابدأ التحليل"):
        if model:
            with st.spinner("⏳ جاري التفرقة بين الأنواع الـ 7..."):
                processed = preprocess_input(img)
                preds = model.predict(processed)[0]
                idx = np.argmax(preds)
                
                # جلب البيانات بناءً على الترتيب الصحيح
                info = DISEASES_DB.get(idx, {"name": "غير معروف", "status": "فحص نسيجي مطلوب"})
                style = "danger" if "خبيث" in info['status'] else "safe"

                st.markdown(f"""
                <div class="report-box">
                    <h3>اسم المرض المكتشف:</h3>
                    <p style="font-size:1.6em; font-weight:bold; color:#1E3A8A;">{info['name']}</p>
                    <h3>التصنيف الطبي:</h3>
                    <div class="{style}">{info['status']}</div>
                    <p style="margin-top:15px;"><b>دقة التشخيص:</b> {preds[idx]:.2%}</p>
                    <hr>
                    <p style="font-size:0.8em; color:gray;">ملاحظة: هذا النظام يفرق الآن بين الأنواع السبعة بناءً على ترتيب المجلدات الأبجدي.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("المحرك لا يعمل، يرجى إعادة التشغيل.")
