import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import gdown

# --- 1. الهوية البصرية (تنسيق الواجهة) ---
st.set_page_config(page_title="Skin AI Expert System", page_icon="🧬", layout="wide")

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

# --- 2. مصفوفة التصنيف النهائي (تعتمد على مخرجات Softmax) ---
# الترتيب الأبجدي للمجلدات كما في الـ Dataset الخاصة بك
CLASS_LABELS = {
    0: {"name": "التقرن الضوئي (akiec)", "status": "خبيث جزئياً / ما قبل سرطان"},
    1: {"name": "سرطان الخلايا القاعدية (bcc)", "status": "خبيث - يتطلب تدخل"},
    2: {"name": "آفات التقرن الحميدة (bkl)", "status": "حميد - غير مقلق"},
    3: {"name": "الأورام الليفية الجلدية (df)", "status": "حميد - زوائد ليفية"},
    4: {"name": "الميلانوما - سرطان الجلد (mel)", "status": "خبيث جداً - مراجعة فورية"},
    5: {"name": "الشامات والوحمات (nv)", "status": "حميد - شامة طبيعية"},
    6: {"name": "الآفات الوعائية (vasc)", "status": "حميد - شعيرات دموية"}
}

# --- 3. دوال المعالجة والمصفوفات (Preprocessing & Normalization) ---
def apply_expert_preprocessing(pil_image):
    # أ. دالة تغيير الحجم (Resizing) باستخدام OpenCV لمصفوفة ثابتة (224x224)
    img = np.array(pil_image.convert('RGB'))
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    
    # ب. فلاتر تحسين التباين (Contrast Enhancement) لإبراز حدود الإصابة
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    # ج. دالة التطبيع (Normalization) تحويل النطاق من [0, 255] إلى [0, 1]
    img_normalized = img.astype(np.float32) / 255.0
    
    # د. مصفوفة الصورة المدخلة (Input Matrix Expansion)
    return np.expand_dims(img_normalized, axis=0)

# --- 4. محرك الـ CNN (التلافيف، التجميع، وسوفت ماكس) ---
@st.cache_resource
def load_cnn_engine():
    f_id = '1lMGCojHeGupFunhxX5GnLOiUgxWbbRC5'
    local_path = "cnn_expert_model.h5"
    if not os.path.exists(local_path):
        gdown.download(f'https://drive.google.com/uc?id={f_id}', "model.zip", quiet=False)
        with zipfile.ZipFile("model.zip", 'r') as z:
            for f in z.namelist():
                if f.endswith('.h5'):
                    with open(local_path, "wb") as out: out.write(z.read(f))
                    break

    try:
        # بناء هيكل CNN (تطبيق دالة التلافيف، ReLU، وMax Pooling)
        base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output) # Pooling Filter
        x = tf.keras.layers.Dense(512, activation='relu')(x)      # ReLU Function
        # دالة سوفت ماكس (Softmax) لتحويل المخرجات لقيم احتمالية
        output = tf.keras.layers.Dense(7, activation='softmax')(x) 
        
        model = tf.keras.Model(inputs=base.input, outputs=output)
        model.load_weights(local_path, by_name=True, skip_mismatch=True)
        return model
    except:
        return tf.keras.models.load_model(local_path, compile=False)

model = load_cnn_engine()

# --- 5. الواجهة الرسومية والتشخيص ---
st.markdown("<h1 style='text-align:center;'>🧬 الخبير الذكي لتشخيص آفات الجلد</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📸 مصفوفة البيانات المدخلة")
    uploaded_file = st.file_uploader("ارفع صورة الفحص (JPG, PNG)", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        raw_img = Image.open(uploaded_file)
        st.image(raw_img, use_container_width=True, caption="الصورة الأصلية قبل المعالجة")

with col2:
    st.subheader("🔍 نتائج تحليل خوارزمية CNN")
    if uploaded_file and st.button("🚀 تنفيذ عمليات المعالجة والتصنيف"):
        if model:
            with st.spinner("⏳ جاري تطبيق الدوال الرياضية والفلاتر..."):
                # تنفيذ المعالجة والتطبيع
                input_matrix = apply_expert_preprocessing(raw_img)
                
                # تنفيذ التنبؤ (حساب دالة التلافيف وسوفت ماكس)
                predictions = model.predict(input_matrix)[0]
                # استخدام Argmax لاختيار الفئة الأعلى من مصفوفة الاحتمالات
                result_idx = np.argmax(predictions)
                
                res_data = CLASS_LABELS[result_idx]
                style_class = "status-danger" if "خبيث" in res_data['status'] else "status-safe"

                st.markdown(f"""
                <div class="report-box">
                    <h3 style="color:#1E3A8A;">التشخيص المكتشف:</h3>
                    <p style="font-size:1.7em; font-weight:bold;">{res_data['name']}</p>
                    <h3 style="color:#1E3A8A;">تصنيف الحالة الحيوية:</h3>
                    <div class="{style_class}">{res_data['status']}</div>
                    <hr>
                    <p style="font-size:1.1em;"><b>دقة دالة Softmax:</b> {predictions[result_idx]:.2%}</p>
                    <p style="font-size:0.85em; color:gray; margin-top:15px;">
                    تم استخراج الميزات باستخدام فلاتر التلافيف (Convolutional Filters) المدمجة في نموذج MobileNetV2.
                    </p>
                </div>
                """, unsafe_allow_html=True)

st.divider()
st.caption("نظام الحماية الذكي Skin Safety AI 2026 | تطبيق مفاهيم شبكات CNN")
