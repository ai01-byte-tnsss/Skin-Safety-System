import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import gdown

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Skin AI Expert System", page_icon="🧬", layout="wide")

# تخصيص واجهة المستخدم
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
    .stButton>button { width: 100%; border-radius: 12px; height: 3.5em; background-color: #1E3A8A; color: white; font-weight: bold; }
    .status-badge { padding: 8px 15px; border-radius: 20px; font-weight: bold; display: inline-block; }
    .malignant-badge { background-color: #FFEBEE; color: #D32F2F; border: 1px solid #D32F2F; } /* خبيث */
    .benign-badge { background-color: #E8F5E9; color: #388E3C; border: 1px solid #388E3C; }    /* حميد */
    .report-box { padding: 25px; border-radius: 15px; background-color: white; border-right: 10px solid #1E3A8A; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. وحدة معالجة الصورة المتقدمة (Preprocessing Module) ---
def preprocess_image(pil_image):
    """
    تقوم هذه الوظيفة بمعالجة الصورة الخام لتكون جاهزة للمطابقة الطبية.
    """
    with st.spinner("⏳ جاري تنظيف ومعالجة الصورة (Preprocessing)..."):
        # 1. تحويل صورة PIL إلى مصفوفة OpenCV (RGB)
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 2. تغيير الحجم إلى الأبعاد القياسية التي تدرب عليها النموذج (224x224)
        # Standard input for most skin lesion models
        img_resized = cv2.resize(img_cv, (224, 224), interpolation=cv2.INTER_AREA)
        
        # 3. إزالة الضوضاء وتنعيم الصورة (Gaussian Blur)
        # Helps remove small skin imperfections
        img_blurred = cv2.GaussianBlur(img_resized, (3, 3), 0)
        
        # 4. تحسين التباين (Histogram Equalization) في مسار الـ YUV
        # Essential for analyzing lesion borders, similar to ABCD features
        img_yuv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_final_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        # 5. التقييس (Normalization) - تحويل قيم البكسل إلى (0-1)
        img_rgb = cv2.cvtColor(img_final_bgr, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # 6. إضافة بُعد الحزمة (Batch Dimension)
        img_batch = img_normalized[np.newaxis, ...]
        
        return img_batch

# --- 3. قاعدة البيانات الطبية المطابقة (24 مرض) ---
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
    16: {"name": "الصدفية (Psoriasis)", "status": "حميد"},
    17: {"name": " لسعات الحشرات (Scabies/Bites)", "status": "حميد"},
    18: {"name": "القرنية الدهنية (Seborrheic Keratosis)", "status": "حميد"},
    19: {"name": "سرطان الخلايا الحرشفية (SCC)", "status": "خبيث"},
    20: {"name": "السعفة والالتهابات (Tinea)", "status": "فطري / حميد"},
    21: {"name": "الشري والحساسية (Urticaria)", "status": "حميد"},
    22: {"name": "الأورام الوعائية (Vascular Tumors)", "status": "حميد"},
    23: {"name": "الثآليل المعدية (Warts)", "status": "حميد"}
}

# --- 4. تحميل النموذج ---
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
            # بناء النموذج بما يتوافق مع الأوزان (استخدام MobileNetV2 كجسم مرن)
            base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
            x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            out = tf.keras.layers.Dense(24, activation='softmax')(x)
            model = tf.keras.Model(inputs=base.input, outputs=out)
            model.load_weights(h5, by_name=True, skip_mismatch=True)
            return model
    except Exception as e: st.error(f"Error: {e}"); return None

model = load_ai_engine()

# --- 5. الواجهة والتشخيص ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>🧬 نظام التشخيص الجلدي بالذكاء الاصطناعي</h1>", unsafe_allow_html=True)

col_up, col_res = st.columns([1, 1.3])

with col_up:
    st.subheader("📸 الخطوة الأولى: إدخال الصورة")
    input_file = st.camera_input("التقط صورة") if st.toggle("استخدم الكاميرا") else st.file_uploader("ارفع الصورة من المعرض", type=["jpg", "png", "jpeg"])

with col_res:
    st.subheader("🔍 الخطوة الثانية: المعالجة والمطابقة")
    if input_file:
        img = Image.open(input_file).convert('RGB')
        st.image(img, caption="الصورة الأصلية", width=300)
        
        if st.button("بدء الفحص السريري الآن"):
            if model:
                # 1. تنفيذ معالجة الصورة أولاً
                processed_img = preprocess_image(img)
                
                # 2. عرض الصورة المعالجة لزيادة شفافية النظام (اختياري)
                # st.image(processed_img[0], caption="الصورة بعد المعالجة", width=300)

                with st.spinner("⏳ جاري المطابقة مع أنماط الأمراض الـ 24..."):
                    # 3. المطابقة والتنبؤ
                    preds = model.predict(processed_img)[0]
                    idx = np.argmax(preds)
                    
                    disease_info = DISEASES_DB.get(idx)
                    
                    # تحديد تصنيف الحالة للتنسيق الجمالي
                    status_class = "malignant-badge" if "خبيث" in disease_info["status"] else "benign-badge"

                    # 4. عرض التقرير النهائي
                    st.markdown(f"""
                    <div class="report-box">
                        <h3 style="color:#1E3A8A;">اسم المرض المتوقع:</h3>
                        <p style="font-size:1.6em; font-weight:bold;">{disease_info['name']}</p>
                        <h3 style="color:#1E3A8A;">تصنيف الحالة:</h3>
                        <span class="status-badge {status_class}">{disease_info['status']}</span>
                        <p style="margin-top:15px;"><b>نسبة المطابقة:</b> {preds[idx]:.2%}</p>
                        <hr>
                        <p style="font-size:0.8em; color:#666;">⚠️ ملاحظة: هذا التشخيص استرشادي. يجب استشارة الطبيب المختص.</p>
                    </div>
                    """, unsafe_allow_html=True)
