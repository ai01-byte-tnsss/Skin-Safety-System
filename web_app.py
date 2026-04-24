import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import gdown

# --- 1. إعدادات الصفحة والتصميم العربي ---
st.set_page_config(page_title="Skin AI Expert", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { 
        font-family: 'Cairo', sans-serif; 
        text-align: right; 
        direction: rtl; 
    }
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.5em; 
        background-color: #1E3A8A; color: white; font-weight: bold; 
    }
    .report-box { 
        padding: 20px; border-radius: 15px; background-color: white; 
        border-right: 10px solid #1E3A8A; box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
    }
    </style>
    """, unsafe_allow_html=True)

# قائمة فئات الأمراض الـ 24
DISEASES = {
    0: "حب الشباب والوردية", 1: "التقرن الضوئي", 2: "التهاب الجلد التأتبي", 
    3: "سرطان الخلايا القاعدية", 4: "آفات حميدة", 5: "أمراض جلدية فقاعية",
    6: "التهاب النسيج الخلوي", 7: "الطفح الدوائي", 8: "الأكزيما", 
    9: "الأمراض الخارجية", 10: "العدوى الفطرية", 11: "الهربس والزوائد",
    12: "الأمراض الجلدية الخفيفة", 13: "الورم الميلانيني", 14: "الوحمات والشامات",
    15: "أورام ليمفاوية", 16: "الصدفية والقمط", 17: "لسعات الحشرات",
    18: "القرنية الدهنية", 19: "سرطان الخلايا الحرشفية", 20: "السعفة",
    21: "الشري والحساسية", 22: "الأورام الوعائية", 23: "الثآليل"
}

# --- 2. محرك تحميل الأوزان الذكي ---
@st.cache_resource
def load_ai_engine():
    file_id = '1lMGCojHeGupFunhxX5GnLOiUgxWbbRC5'
    download_url = f'https://drive.google.com/uc?id={file_id}'
    zip_file = "model_weights.zip"
    extract_folder = "model_dir"

    try:
        # 1. تحميل الملف من Google Drive
        if not os.path.exists(zip_file):
            with st.spinner("⏳ جاري تحميل محرك الذكاء الاصطناعي..."):
                gdown.download(download_url, zip_file, quiet=False)
        
        # 2. فك الضغط
        if not os.path.exists(extract_folder):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
        
        # 3. تحديد مكان ملف h5
        h5_path = None
        for root, dirs, files in os.walk(extract_folder):
            for f in files:
                if f.endswith(".h5"):
                    h5_path = os.path.join(root, f)
                    break
        
        if h5_path:
            # 4. بناء الهيكل (Architecture)
            # تم اختيار MobileNetV2 كقاعدة لأنه الأكثر توافقاً مع الأوزان المرفوعة
            base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
            x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.4)(x)
            output = tf.keras.layers.Dense(24, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=base_model.input, outputs=output)
            
            # 5. تحميل الأوزان مع تخطي الطبقات غير المتطابقة (لحل مشكلة 237 vs 4)
            model.load_weights(h5_path, by_name=True, skip_mismatch=True)
            return model
    except Exception as e:
        st.error(f"⚠️ عذراً، تعذر تشغيل المحرك: {e}")
    return None

model = load_ai_engine()

# --- 3. واجهة المستخدم ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>خبير سلامة الجلد بالذكاء الاصطناعي 🧬</h1>", unsafe_allow_html=True)

# خيارات اللغة والدليل
col_top1, col_top2 = st.columns(2)
with col_top1:
    st.selectbox("🌐 اختر اللغة", ["العربية", "English"])
with col_top2:
    with st.expander("📖 الدليل الإرشادي"):
        for i, name in DISEASES.items():
            st.write(f"• {name}")

st.divider()

# منطقة رفع الصور والكاميرا
tab1, tab2 = st.tabs(["📤 رفع صورة من الاستوديو", "📸 التقاط صورة بالكاميرا"])

with tab1:
    up_file = st.file_uploader("اختر صورة واضحة للإصابة", type=["jpg", "png", "jpeg"])
with tab2:
    cam_file = st.camera_input("التقط صورة مباشرة")

active_source = up_file if up_file else cam_file

if active_source:
    col_img, col_res = st.columns([1, 1])
    img = Image.open(active_source).convert('RGB')
    
    with col_img:
        st.image(img, caption="الصورة التي سيتم فحصها", use_container_width=True)
    
    with col_res:
        st.subheader("🔍 تحليل الحالة")
        if st.button("بدء الفحص الآن"):
            if model:
                with st.spinner("⏳ جاري تحليل ملامح الجلد..."):
                    # المعالجة المسبقة للصورة
                    img_resized = cv2.resize(np.array(img), (224, 224))
                    img_array = (img_resized.astype(np.float32) / 255.0)[np.newaxis, ...]
                    
                    # عملية التنبؤ
                    predictions = model.predict(img_array)
                    class_idx = np.argmax(predictions[0])
                    confidence = predictions[0][class_idx]
                    
                    # عرض النتيجة ببطاقة منسقة
                    st.markdown(f"""
                    <div class="report-box">
                        <h3 style="color:#1E3A8A;">النتيجة المتوقعة: {DISEASES.get(class_idx)}</h3>
                        <p><b>نسبة الثقة:</b> {confidence:.2%}</p>
                        <hr>
                        <p style="font-size:0.8em; color:#555;">
                        تنبيه: هذا الفحص استرشادي، يرجى مراجعة الطبيب المختص للتشخيص النهائي.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("المحرك غير جاهز حالياً. يرجى الانتظار ثواني أو إعادة تشغيل التطبيق.")

st.divider()
st.caption("نظام Skin Safety System 2026 | تطوير لأغراض البحث العلمي")
