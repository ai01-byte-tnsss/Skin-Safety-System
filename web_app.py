import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import gdown

# --- 1. إعدادات الصفحة والتصميم ---
st.set_page_config(page_title="Skin Safety Expert", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.5em; 
        background-color: #1E3A8A; color: white; font-weight: bold; 
        transition: 0.3s; border: none;
    }
    .stButton>button:hover { background-color: #2563EB; transform: translateY(-2px); }
    .report-box { 
        padding: 30px; border-radius: 20px; background-color: white; 
        border-right: 15px solid #1E3A8A; box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .danger { color: #D32F2F; background-color: #FFEBEE; padding: 10px; border-radius: 8px; font-weight: bold; }
    .safe { color: #388E3C; background-color: #E8F5E9; padding: 10px; border-radius: 8px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. مطابقة الفئات مع مجلدات Dataset الخاصة بك ---
# الترتيب الأبجدي للمجلدات هو المعيار هنا لضمان مطابقة الصورة مع النتيجة
CLASS_MAP = {
    0: {"name": "التقرن الضوئي (akiec)", "status": "خبيث جزئياً - خطر محتمل"},
    1: {"name": "سرطان الخلايا القاعدية (bcc)", "status": "خبيث - يحتاج تدخل طبي"},
    2: {"name": "آفات التقرن الحميدة (bkl)", "status": "حميد - غير مقلق"},
    3: {"name": "الأورام الليفية الجلدية (df)", "status": "حميد - زوائد ليفية"},
    4: {"name": "الميلانوما (mel)", "status": "خبيث جداً - مراجعة فورية!"},
    5: {"name": "الشامات والوحمات (nv)", "status": "حميد - شامة طبيعية"},
    6: {"name": "الآفات الوعائية (vasc)", "status": "حميد - أوعية دموية"}
}

# --- 3. وحدة معالجة الصور ---
def process_image(img):
    img = np.array(img.convert('RGB'))
    img = cv2.resize(img, (224, 224))
    # تحسين التباين وإزالة الضوضاء
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return (img.astype(np.float32) / 255.0)[np.newaxis, ...]

# --- 4. تحميل المحرك (الحل الجذري) ---
@st.cache_resource
def load_expert_model():
    file_id = '1lMGCojHeGupFunhxX5GnLOiUgxWbbRC5'
    local_h5 = "skin_expert_model.h5"
    
    if not os.path.exists(local_h5):
        try:
            with st.spinner("⏳ جاري تهيئة محرك الذكاء الاصطناعي..."):
                gdown.download(f'https://drive.google.com/uc?id={file_id}', "model.zip", quiet=False)
                with zipfile.ZipFile("model.zip", 'r') as z:
                    for f in z.namelist():
                        if f.endswith('.h5'):
                            with open(local_h5, "wb") as out: out.write(z.read(f))
                            break
        except: return None

    try:
        # بناء هيكل Model (MobileNetV2) مطابق للتدريب
        base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        out = tf.keras.layers.Dense(len(CLASS_MAP), activation='softmax')(x)
        model = tf.keras.Model(inputs=base.input, outputs=out)
        
        # تحميل الأوزان بمرونة
        model.load_weights(local_h5, by_name=True, skip_mismatch=True)
        return model
    except:
        return tf.keras.models.load_model(local_h5, compile=False)

model = load_expert_model()

# --- 5. ترتيب واجهة المستخدم ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>🧬 نظام خبير سلامة الجلد الذكي</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.2em;'>تحليل متطور للآفات الجلدية باستخدام التعلم العميق</p>", unsafe_allow_html=True)

st.divider()

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.subheader("📸 منطقة الفحص")
    src = st.toggle("🎥 تفعيل الكاميرا المباشرة")
    file = st.camera_input("التقط صورة للإصابة") if src else st.file_uploader("📤 ارفع صورة من مجلدات Dataset", type=["jpg", "png", "jpeg"])

with col_right:
    st.subheader("🔍 التقرير التحليلي")
    if file:
        raw_img = Image.open(file)
        st.image(raw_img, caption="الصورة التي سيتم تحليلها", use_container_width=True)
        
        if st.button("🚀 بدء التحليل والمطابقة"):
            if model:
                with st.spinner("⏳ جاري استخراج الأنماط الحيوية..."):
                    processed_img = process_image(raw_img)
                    preds = model.predict(processed_img)[0]
                    idx = np.argmax(preds)
                    
                    info = CLASS_MAP.get(idx, {"name": "غير محدد", "status": "غير معروف"})
                    is_danger = "خبيث" in info['status']
                    style_class = "danger" if is_danger else "safe"

                    st.markdown(f"""
                    <div class="report-box">
                        <h3 style="color:#1E3A8A;">التشخيص المقترح:</h3>
                        <p style="font-size:1.5em; font-weight:bold;">{info['name']}</p>
                        
                        <h3 style="color:#1E3A8A;">تصنيف الأمان:</h3>
                        <div class="{style_class}">{info['status']}</div>
                        
                        <div style="margin-top:20px; padding:10px; border-top:1px solid #eee;">
                            <b>نسبة الثقة في التشخيص:</b> 
                            <span style="font-size:1.2em; color:#1E3A8A;">{preds[idx]:.2%}</span>
                        </div>
                        <p style="font-size:0.8em; color:gray; margin-top:15px;">
                        ⚠️ ملاحظة: هذا التشخيص آلي لأغراض البحث. يرجى مراجعة طبيب الجلدية المختص.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("❌ المحرك غير جاهز. تأكد من اتصال الإنترنت ورابط الأوزان.")

st.divider()
st.caption("نظام الحماية الجلدية الذكي © 2026 - مشروع تطوير الأبحاث الطبية")
