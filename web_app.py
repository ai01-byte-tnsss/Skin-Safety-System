import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import gdown

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Skin Health AI Expert", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
    .stButton>button { width: 100%; border-radius: 12px; height: 3.5em; background-color: #1E3A8A; color: white; font-weight: bold; }
    .malignant { color: #D32F2F; background-color: #FFEBEE; padding: 10px; border-radius: 8px; font-weight: bold; } /* تنبيه أحمر */
    .benign { color: #388E3C; background-color: #E8F5E9; padding: 10px; border-radius: 8px; font-weight: bold; }    /* تنبيه أخضر */
    .report-box { padding: 25px; border-radius: 15px; background-color: white; border-right: 10px solid #1E3A8A; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. قاعدة البيانات الطبية المصنفة (24 فئة) ---
DISEASES_DB = {
    0: {"name": "حب الشباب والوردية (Acne/Rosacea)", "status": "حميد / التهابي"},
    1: {"name": "التقرن الضوئي (Actinic Keratosis)", "status": "ما قبل خبيث - يحتاج متابعة"},
    2: {"name": "التهاب الجلد التأتبي (Atopic Dermatitis)", "status": "حميد / التهابي"},
    3: {"name": "سرطان الخلايا القاعدية (Basal Cell Carcinoma)", "status": "خبيث - راجع الطبيب"},
    4: {"name": "التقرن الحميد (Benign Keratosis)", "status": "حميد"},
    5: {"name": "الأمراض الفقاعية (Bullous Disease)", "status": "مناعي / يحتاج فحص"},
    6: {"name": "التهاب النسيج الخلوي (Cellulitis)", "status": "عدوى بكتيرية - تحتاج علاج"},
    7: {"name": "الطفح الدوائي (Drug Eruptions)", "status": "تحسسي"},
    8: {"name": "الأكزيما (Eczema)", "status": "حميد / التهابي"},
    9: {"name": "الطفح الفيروسي (Exanthems)", "status": "عدوى فيروسية"},
    10: {"name": "العدوى الفطرية (Fungal Infections)", "status": "عدوى فطرية"},
    11: {"name": "الهربس / زوائد فيروسية (Herpes/HPV)", "status": "فيروسي معدي"},
    12: {"name": "أمراض الحساسية الضوئية (Light Diseases)", "status": "تحسس ضوئي"},
    13: {"name": "الميلانوما (Melanoma)", "status": "خبيث جداً - مراجعة فورية"},
    14: {"name": "الشامات والوحمات (Nevi/Moles)", "status": "حميد"},
    15: {"name": "أورام ليمفاوية جلدية (Lymphoma)", "status": "خبيث / فحص نسيجي"},
    16: {"name": "الصدفية (Psoriasis)", "status": "حميد / مزمن"},
    17: {"name": "الجرب ولسعات الحشرات (Scabies/Bites)", "status": "طفيلي معدي"},
    18: {"name": "التقرن الدهني (Seborrheic Keratosis)", "status": "حميد"},
    19: {"name": "سرطان الخلايا الحرشفية (SCC)", "status": "خبيث - تدخل جراحي"},
    20: {"name": "السعفة (Tinea)", "status": "فطري معدي"},
    21: {"name": "الشري / الحساسية (Urticaria)", "status": "تحسس مؤقت"},
    22: {"name": "الأورام الوعائية (Vascular Tumors)", "status": "حميد وعائي"},
    23: {"name": "الثآليل (Warts)", "status": "حميد فيروسي"}
}

# --- 3. محرك التحميل ---
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
            base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
            x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            out = tf.keras.layers.Dense(24, activation='softmax')(x)
            model = tf.keras.Model(inputs=base.input, outputs=out)
            model.load_weights(h5, by_name=True, skip_mismatch=True)
            return model
    except Exception as e: st.error(f"Error: {e}"); return None

model = load_ai_engine()

# --- 4. الواجهة والنتائج ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>🧬 نظام تشخيص أمراض الجلد الذكي</h1>", unsafe_allow_html=True)

col_in, col_res = st.columns([1, 1.2])

with col_in:
    st.subheader("📸 إدخال الصورة")
    input_file = st.camera_input("التقط صورة للإصابة") if st.toggle("استخدم الكاميرا") else st.file_uploader("ارفع صورة من الجهاز", type=["jpg", "png", "jpeg"])

with col_res:
    st.subheader("🔍 التقرير الفني")
    if input_file:
        img = Image.open(input_file).convert('RGB')
        st.image(img, use_container_width=True)
        
        if st.button("بدء الفحص السريع"):
            if model:
                with st.spinner("⏳ جاري المطابقة مع قاعدة البيانات الطبية..."):
                    img_arr = cv2.resize(np.array(img), (224, 224))
                    img_arr = (img_arr.astype(np.float32) / 255.0)[np.newaxis, ...]
                    preds = model.predict(img_arr)[0]
                    idx = np.argmax(preds)
                    
                    # استخراج المعلومات
                    disease = DISEASES_DB.get(idx, {"name": "غير محدد", "status": "غير معروف"})
                    status_class = "malignant" if "خبيث" in disease["status"] else "benign"

                    st.markdown(f"""
                    <div class="report-box">
                        <h3 style="color:#1E3A8A;">اسم المرض:</h3>
                        <p style="font-size:1.4em; font-weight:bold;">{disease['name']}</p>
                        <h3 style="color:#1E3A8A;">تصنيف الحالة:</h3>
                        <div class="{status_class}">{disease['status']}</div>
                        <p style="margin-top:10px;"><b>نسبة المطابقة:</b> {preds[idx]:.2%}</p>
                        <hr>
                        <p style="font-size:0.8em; color:#666;">⚠️ ملاحظة: هذا التقرير استرشادي، يرجى مراجعة الطبيب المختص.</p>
                    </div>
                    """, unsafe_allow_html=True)
