import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import gdown

# --- 1. إعدادات الصفحة والتنسيق ---
st.set_page_config(page_title="Skin AI Expert System", page_icon="🧬", layout="wide")

# تنسيق الواجهة ودعم اللغة العربية
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
    .main { background-color: #f0f2f6; }
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.5em; 
        background-color: #1E3A8A; color: white; font-weight: bold; 
        font-size: 1.1em; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #2563EB; border: none; }
    .report-card { 
        padding: 25px; border-radius: 15px; background-color: white; 
        border-right: 10px solid #1E3A8A; box-shadow: 0 10px 20px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# قائمة الأمراض الـ 24
DISEASES = {
    0: "Acne and Rosacea", 1: "Actinic Keratosis", 2: "Atopic Dermatitis", 
    3: "Basal Cell Carcinoma", 4: "Benign Keratosis", 5: "Bullous Disease",
    6: "Cellulitis", 7: "Drug Eruptions", 8: "Eczema", 9: "Exanthems",
    10: "Fungal Infections", 11: "Herpes HPV", 12: "Light Diseases",
    13: "Melanoma", 14: "Nevi and Moles", 15: "Lymphoma",
    16: "Psoriasis Lichen Planus", 17: "Scabies and Bites",
    18: "Seborrheic Keratosis", 19: "Squamous Cell Carcinoma",
    20: "Tinea Ringworm", 21: "Urticaria Hives", 22: "Vascular Tumors", 23: "Warts"
}

# --- 2. محرك تحميل النموذج الذكي ---
@st.cache_resource
def load_ai_engine():
    file_id = '1lMGCojHeGupFunhxX5GnLOiUgxWbbRC5'
    download_url = f'https://drive.google.com/uc?id={file_id}'
    zip_file = "model_weights.zip"
    extract_folder = "model_dir"

    try:
        # تحميل من Google Drive
        if not os.path.exists(zip_file):
            with st.spinner("⏳ جاري تحميل الأوزان من سحابة Google..."):
                gdown.download(download_url, zip_file, quiet=False)
        
        # فك الضغط
        if not os.path.exists(extract_folder):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
        
        h5_path = None
        for root, dirs, files in os.walk(extract_folder):
            for f in files:
                if f.endswith(".h5"):
                    h5_path = os.path.join(root, f)
                    break
        
        if h5_path:
            # بناء الهيكل الهجين (Hybrid) ليتناسب مع أوزانك المكونة من +200 طبقة
            base1 = tf.keras.applications.EfficientNetB0(input_shape=(224,224,3), include_top=False, weights=None)
            base2 = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
            
            x1 = tf.keras.layers.GlobalAveragePooling2D()(base1.output)
            x2 = tf.keras.layers.GlobalAveragePooling2D()(base2.output)
            merged = tf.keras.layers.Concatenate()([x1, x2])
            
            top = tf.keras.layers.Dense(512, activation='relu')(merged)
            top = tf.keras.layers.Dropout(0.4)(top)
            output = tf.keras.layers.Dense(24, activation='softmax')(top)
            
            model = tf.keras.Model(inputs=[base1.input, base2.input], outputs=output)
            
            # تحميل الأوزان
            model.load_weights(h5_path)
            return model
    except Exception as e:
        st.error(f"⚠️ خطأ في المحرك: {e}")
    return None

model = load_ai_engine()

# --- 3. تصميم واجهة المستخدم ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>الذكاء الاصطناعي لفحص الجلد 🧬</h1>", unsafe_allow_html=True)

# الأدوات العلوية
col_lang, col_guide = st.columns(2)
with col_lang:
    st.selectbox("🌐 اختر لغة التقارير", ["العربية (Arabic)", "English"])
with col_guide:
    with st.expander("📖 الدليل الطبي للفئات"):
        for i, name in DISEASES.items():
            st.write(f"• {name}")

st.divider()

# منطقة الإدخال
col_in, col_out = st.columns([1, 1])

with col_in:
    st.subheader("📸 رفع أو التقاط صورة")
    source_type = st.radio("وسيلة الإدخال:", ["المعرض 📤", "الكاميرا 📷"])
    
    if source_type == "المعرض 📤":
        file = st.file_uploader("اختر صورة الجلد", type=["jpg", "png", "jpeg"])
    else:
        file = st.camera_input("وجه الكاميرا نحو الإصابة")

# منطقة النتائج
with col_out:
    st.subheader("🔍 نتيجة الفحص")
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, use_container_width=True)
        
        if st.button("بدء تحليل الصورة الآن"):
            if model:
                with st.spinner("⏳ جاري تحليل ملامح الأنسجة..."):
                    # المعالجة المسبقة
                    img_ready = cv2.resize(np.array(img), (224, 224))
                    img_ready = (img_ready.astype(np.float32) / 255.0)[np.newaxis, ...]
                    
                    # التنبؤ (إرسال مدخلين للموديل الهجين)
                    prediction = model.predict([img_ready, img_ready])
                    class_id = np.argmax(prediction[0])
                    score = prediction[0][class_id]
                    
                    # عرض النتيجة
                    st.markdown(f"""
                    <div class="report-card">
                        <h2 style="color:#1E3A8A;">{DISEASES.get(class_id, "غير محدد")}</h2>
                        <p style="font-size:1.2em;"><b>الدقة:</b> {score:.2%}</p>
                        <hr>
                        <p style="color:#666; font-size:0.9em;">
                        ⚠️ ملاحظة: هذا التقرير آلي. يجب مراجعة الطبيب للتشخيص السريري.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("المحرك لم يكتمل تحميله بعد، يرجى الانتظار قليلاً.")

st.divider()
st.caption("Skin Safety System V3.0 - 졸업 프로젝트 2026")
