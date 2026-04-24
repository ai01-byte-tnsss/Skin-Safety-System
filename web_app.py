import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import gdown

# --- 1. إعدادات الصفحة والتصميم ---
st.set_page_config(page_title="Skin AI Expert System", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
    .stButton>button { width: 100%; border-radius: 12px; height: 3.5em; background-color: #1E3A8A; color: white; font-weight: bold; }
    .status-badge { padding: 8px 15px; border-radius: 20px; font-weight: bold; display: inline-block; margin-top: 5px; }
    .malignant-badge { background-color: #FFEBEE; color: #D32F2F; border: 1px solid #D32F2F; }
    .benign-badge { background-color: #E8F5E9; color: #388E3C; border: 1px solid #388E3C; }
    .report-box { padding: 25px; border-radius: 15px; background-color: white; border-right: 10px solid #1E3A8A; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. قاعدة البيانات الطبية الدقيقة (24 فئة) ---
DISEASES_DB = {
    0: {"name": "حب الشباب والوردية (Acne/Rosacea)", "status": "حميد"},
    1: {"name": "التقرن الضوئي (Actinic Keratosis)", "status": "خبيث / ما قبل خبيث"},
    2: {"name": "التهاب الجلد التأتبي (Atopic Dermatitis)", "status": "حميد"},
    3: {"name": "سرطان الخلايا القاعدية (Basal Cell Carcinoma)", "status": "خبيث"},
    4: {"name": "آفات التقرن الحميدة (Benign Keratosis)", "status": "حميد"},
    5: {"name": "الأمراض الفقاعية (Bullous Disease)", "status": "حميد"},
    6: {"name": "التهاب النسيج الخلوي (Cellulitis)", "status": "عدوى / حميد"},
    7: {"name": "الطفح الدوائي (Drug Eruptions)", "status": "حميد"},
    8: {"name": "الأكزيما (Eczema)", "status": "حميد"},
    9: {"name": "الطفح الجلدي الفيروسي (Exanthems)", "status": "فيروسي / حميد"},
    10: {"name": "العدوى الفطرية (Fungal Infections)", "status": "حميد"},
    11: {"name": "الهربس والزوائد (Herpes/HPV)", "status": "فيروسي / حميد"},
    12: {"name": "الأمراض المرتبطة بالضوء (Light Diseases)", "status": "حميد"},
    13: {"name": "الميلانوما - سرطان الجلد (Melanoma)", "status": "خبيث جدًا"},
    14: {"name": "الشامات والوحمات (Nevi/Moles)", "status": "حميد"},
    15: {"name": "أورام ليمفاوية جلدية (Lymphoma)", "status": "خبيث"},
    16: {"name": "الصدفية والقمط (Psoriasis/Lichen Planus)", "status": "حميد"},
    17: {"name": "لسعات الحشرات والجرب (Scabies/Bites)", "status": "حميد"},
    18: {"name": "القرنية الدهنية (Seborrheic Keratosis)", "status": "حميد"},
    19: {"name": "سرطان الخلايا الحرشفية (SCC)", "status": "خبيث"},
    20: {"name": "السعفة والالتهابات الفطرية (Tinea)", "status": "فطري / حميد"},
    21: {"name": "الشري والحساسية (Urticaria)", "status": "حميد"},
    22: {"name": "الأورام الوعائية (Vascular Tumors)", "status": "حميد"},
    23: {"name": "الثآليل المعدية (Warts)", "status": "حميد"}
}

# --- 3. وحدة معالجة الصورة (Preprocessing) ---
def preprocess_image(pil_image):
    # تحويل وتغيير الحجم
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_cv, (224, 224), interpolation=cv2.INTER_AREA)
    # تنعيم وتحسين التباين
    img_blurred = cv2.GaussianBlur(img_resized, (3, 3), 0)
    img_yuv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_final = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    # التقييس (Normalization)
    img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    return img_normalized[np.newaxis, ...]

# --- 4. محرك بناء النموذج وتحميل الأوزان (الحل لمشكلة الـ 4%) ---
@st.cache_resource
def load_ai_engine():
    file_id = '1lMGCojHeGupFunhxX5GnLOiUgxWbbRC5'
    url = f'https://drive.google.com/uc?id={file_id}'
    zip_f, dir_f = "weights.zip", "weights_dir"
    try:
        if not os.path.exists(zip_f): gdown.download(url, zip_f, quiet=False)
        if not os.path.exists(dir_f):
            with zipfile.ZipFile(zip_f, 'r') as z: z.extractall(dir_f)
        h5 = next((os.path.join(r, f) for r, d, fs in os.walk(dir_f) for f in fs if f.endswith(".h5")), None)
        
        if h5:
            # إعادة بناء هيكل الـ 237 طبقة بالكامل (MobileNetV2)
            base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
            x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.4)(x)
            out = tf.keras.layers.Dense(24, activation='softmax')(x)
            model = tf.keras.Model(inputs=base.input, outputs=out)
            
            # تحميل الأوزان في الهيكل المكتمل
            model.load_weights(h5)
            return model
    except Exception as e:
        st.error(f"⚠️ فشل في بناء المحرك: {e}")
    return None

model = load_ai_engine()

# --- 5. الواجهة والتشخيص ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>🧬 نظام التشخيص الذكي لخبير الجلد</h1>", unsafe_allow_html=True)

col_up, col_res = st.columns([1, 1.3])

with col_up:
    st.subheader("📸 إدخال عينة الفحص")
    input_file = st.camera_input("التقط صورة") if st.toggle("استخدم الكاميرا") else st.file_uploader("ارفع الصورة", type=["jpg", "png", "jpeg"])

with col_res:
    st.subheader("🔍 المعالجة والمطابقة الطبية")
    if input_file:
        img = Image.open(input_file).convert('RGB')
        st.image(img, caption="الصورة الأصلية", width=300)
        
        if st.button("🚀 بدء تحليل الأنسجة والمطابقة"):
            if model:
                # 1. المعالجة المسبقة
                processed_img = preprocess_image(img)
                
                with st.spinner("⏳ جاري تحليل الخصائص المورفولوجية..."):
                    # 2. التنبؤ
                    preds = model.predict(processed_img)[0]
                    idx = np.argmax(preds)
                    
                    # 3. جلب معلومات المرض والتصنيف
                    disease = DISEASES_DB.get(idx)
                    status_class = "malignant-badge" if "خبيث" in disease["status"] else "benign-badge"

                    # 4. التقرير النهائي
                    st.markdown(f"""
                    <div class="report-box">
                        <h3 style="color:#1E3A8A;">التشخيص المكتشف:</h3>
                        <p style="font-size:1.6em; font-weight:bold;">{disease['name']}</p>
                        <h3 style="color:#1E3A8A;">تصنيف الحالة الحيوية:</h3>
                        <span class="status-badge {status_class}">{disease['status']}</span>
                        <p style="margin-top:15px; font-size:1.1em;"><b>دقة المطابقة:</b> {preds[idx]:.2%}</p>
                        <hr>
                        <p style="font-size:0.85em; color:#666;">⚠️ تنبيه: هذا الفحص يعتمد على الذكاء الاصطناعي للأغراض البحثية، استشر الطبيب دائماً.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("المحرك غير جاهز. تأكد من تحديث ملف الأوزان.")

st.divider()
st.caption("Skin Safety AI 2026 | البحث والتطوير الطبي")
