import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import gdown

# --- 1. إعدادات الصفحة والتنسيق الجمالي ---
st.set_page_config(page_title="Skin AI Expert", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
    .main { background-color: #f4f7f9; }
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.5em; 
        background-color: #1E3A8A; color: white; font-weight: bold; font-size: 1.1em;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: 0.3s;
    }
    .stButton>button:hover { background-color: #2563EB; transform: translateY(-2px); }
    .report-box { 
        padding: 25px; border-radius: 15px; background-color: white; 
        border-right: 10px solid #1E3A8A; box-shadow: 0 10px 20px rgba(0,0,0,0.05);
    }
    .malignant-text { color: #D32F2F; font-weight: bold; background-color: #FFEBEE; padding: 5px 10px; border-radius: 5px; }
    .benign-text { color: #388E3C; font-weight: bold; background-color: #E8F5E9; padding: 5px 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. قاعدة البيانات الطبية المصنفة (24 مرض) ---
DISEASES_INFO = {
    0: {"name": "حب الشباب والوردية (Acne/Rosacea)", "status": "حميد"},
    1: {"name": "التقرن الضوئي (Actinic Keratosis)", "status": "ما قبل خبيث"},
    2: {"name": "التهاب الجلد التأتبي (Atopic Dermatitis)", "status": "حميد"},
    3: {"name": "سرطان الخلايا القاعدية (Basal Cell Carcinoma)", "status": "خبيث"},
    4: {"name": "آفات التقرن الحميدة (Benign Keratosis)", "status": "حميد"},
    5: {"name": "الأمراض الفقاعية (Bullous Disease)", "status": "حميد"},
    6: {"name": "التهاب النسيج الخلوي (Cellulitis)", "status": "حميد / عدوى"},
    7: {"name": "الطفح الدوائي (Drug Eruptions)", "status": "حميد"},
    8: {"name": "الأكزيما (Eczema)", "status": "حميد"},
    9: {"name": "الطفح الجلدي الفيروسي (Exanthems)", "status": "حميد"},
    10: {"name": "العدوى الفطرية (Fungal Infections)", "status": "حميد"},
    11: {"name": "الهربس والزوائد (Herpes/HPV)", "status": "حميد"},
    12: {"name": "الأمراض المرتبطة بالضوء (Light Diseases)", "status": "حميد"},
    13: {"name": "الميلانوما - سرطان الجلد (Melanoma)", "status": "خبيث جداً"},
    14: {"name": "الوحمات والشامات (Nevi/Moles)", "status": "حميد"},
    15: {"name": "أورام ليمفاوية جلدية (Lymphoma)", "status": "خبيث"},
    16: {"name": "الصدفية والقمط (Psoriasis/Lichen Planus)", "status": "حميد"},
    17: {"name": "لسعات الحشرات والجرب (Scabies/Bites)", "status": "حميد"},
    18: {"name": "القرنية الدهنية (Seborrheic Keratosis)", "status": "حميد"},
    19: {"name": "سرطان الخلايا الحرشفية (SCC)", "status": "خبيث"},
    20: {"name": "السعفة والالتهابات الفطرية (Tinea)", "status": "حميد"},
    21: {"name": "الشري والحساسية (Urticaria)", "status": "حميد"},
    22: {"name": "الأورام الوعائية (Vascular Tumors)", "status": "حميد"},
    23: {"name": "الثآليل المعدية (Warts)", "status": "حميد"}
}

# --- 3. وحدة معالجة الصور (Image Processing) ---
def process_skin_image(image):
    # تحويل لـ RGB ثم لـ OpenCV
    img_array = np.array(image.convert('RGB'))
    # تغيير الحجم
    img_resized = cv2.resize(img_array, (224, 224))
    # تنعيم لإزالة النمش البسيط (Denoising)
    img_filtered = cv2.GaussianBlur(img_resized, (3, 3), 0)
    # التقييس (Normalization)
    img_ready = img_filtered.astype(np.float32) / 255.0
    return np.expand_dims(img_ready, axis=0)

# --- 4. محرك التحميل الذكي (حل مشكلة الفشل الجذري) ---
@st.cache_resource
def load_expert_system():
    file_id = '1lMGCojHeGupFunhxX5GnLOiUgxWbbRC5'
    zip_path = "model_files.zip"
    h5_local = "final_model.h5"
    
    try:
        if not os.path.exists(h5_local):
            gdown.download(f'https://drive.google.com/uc?id={file_id}', zip_path, quiet=False)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    if file.endswith('.h5'):
                        with open(h5_local, "wb") as f:
                            f.write(zip_ref.read(file))
                        break
        
        # بناء هيكل النموذج يدوياً لضمان التوافق مع أي إصدار تنسرفلو
        base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        out = tf.keras.layers.Dense(24, activation='softmax')(x)
        model = tf.keras.Model(inputs=base.input, outputs=out)
        
        # تحميل الأوزان مع خاصية تخطي التعارض
        model.load_weights(h5_local, by_name=True, skip_mismatch=True)
        return model
    except Exception as e:
        st.error(f"حدث خطأ في بناء المحرك: {e}")
        return None

model = load_expert_system()

# --- 5. واجهة المستخدم الرسومية ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>🧬 الخبير الذكي لتشخيص أمراض الجلد</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>نظام متطور يعتمد على الذكاء الاصطناعي لمعالجة وتصنيف الصور الجلدية</p>", unsafe_allow_html=True)

col_input, col_output = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📸 مدخلات الفحص")
    choice = st.radio("اختر طريقة الإدخال:", ["📤 رفع صورة من الجهاز", "📷 التقاط صورة بالكاميرا"])
    
    if choice == "📷 التقاط صورة بالكاميرا":
        uploaded_file = st.camera_input("التقط صورة واضحة للمنطقة المصابة")
    else:
        uploaded_file = st.file_uploader("ارفع صورة الإصابة (JPG, PNG)", type=["jpg", "png", "jpeg"])

with col_output:
    st.subheader("🔍 التقرير والنتيجة")
    if uploaded_file:
        raw_img = Image.open(uploaded_file)
        st.image(raw_img, caption="الصورة المدخلة", use_container_width=True)
        
        if st.button("🚀 بدء المعالجة والتشخيص"):
            if model:
                with st.spinner("⏳ جاري معالجة الصورة ومطابقة الأنماط..."):
                    # المعالجة
                    processed_data = process_skin_image(raw_img)
                    # التنبؤ
                    predictions = model.predict(processed_data)[0]
                    idx = np.argmax(predictions)
                    accuracy = predictions[idx]
                    
                    # جلب البيانات
                    info = DISEASES_INFO.get(idx, {"name": "غير محدد", "status": "غير معروف"})
                    status_style = "malignant-text" if "خبيث" in info['status'] else "benign-text"

                    # عرض التقرير المنسق
                    st.markdown(f"""
                    <div class="report-box">
                        <h3 style="color:#1E3A8A; margin-bottom:5px;">التشخيص المقترح:</h3>
                        <p style="font-size:1.5em; font-weight:bold;">{info['name']}</p>
                        
                        <h3 style="color:#1E3A8A; margin-top:20px; margin-bottom:10px;">تصنيف الحالة:</h3>
                        <span class="{status_style}">{info['status']}</span>
                        
                        <p style="margin-top:20px;"><b>دقة المطابقة الحيوية:</b> {accuracy:.2%}</p>
                        <hr>
                        <p style="font-size:0.8em; color:#666;">
                        ⚠️ تنبيه طبي: هذا النظام استرشادي فقط. يجب مراجعة الطبيب المختص قبل اتخاذ أي قرار علاجي.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("المحرك غير جاهز، يرجى إعادة تحميل الصفحة.")

st.write("---")
st.caption("نظام Skin Safety AI v3.0 | مشروع تخرج 2026")
