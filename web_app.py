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
    .malignant { color: #D32F2F; font-weight: bold; } /* لون أحمر للخبيث */
    .benign { color: #388E3C; font-weight: bold; }    /* لون أخضر للحميد */
    .report-box { padding: 25px; border-radius: 15px; background-color: white; border-right: 10px solid #1E3A8A; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. قاعدة البيانات الطبية الدقيقة ---
# مراجعة الأسماء لتطابق مخرجات الأوزان العالمية
DISEASES_INFO = {
    0: {"name": "حب الشباب والوردية (Acne/Rosacea)", "type": "التهابي/حميد"},
    1: "التقرن الضوئي (Actinic Keratosis)",
    2: "التهاب الجلد التأتبي (Atopic Dermatitis)",
    3: {"name": "سرطان الخلايا القاعدية (BCC)", "type": "خبيث - يتطلب استشارة فورية"},
    4: "آفات التقرن الحميدة (Benign Keratosis)",
    5: "الأمراض الفقاعية (Bullous Disease)",
    6: "التهاب النسيج الخلوي (Cellulitis)",
    7: "الطفح الدوائي (Drug Eruptions)",
    8: "الأكزيما (Eczema)",
    9: "الطفح الجلدي الفيروسي (Exanthems)",
    10: "العدوى الفطرية (Fungal Infections)",
    11: "الهربس والزوائد الفيروسية (Herpes/HPV)",
    12: "الأمراض المرتبطة بالضوء (Light Diseases)",
    13: {"name": "الميلانوما - سرطان الجلد (Melanoma)", "type": "خبيث جداً - مراجعة طبيب فوراً"},
    14: "الوحمات والشامات (Nevi/Moles)",
    15: {"name": "أورام ليمفاوية جلدية (Lymphoma)", "type": "خبيث/تحتاج فحوصات"},
    16: "الصدفية والقمط (Psoriasis/Lichen Planus)",
    17: "لسعات الحشرات والجرب (Scabies/Bites)",
    18: "القرنية الدهنية (Seborrheic Keratosis)",
    19: {"name": "سرطان الخلايا الحرشفية (SCC)", "type": "خبيث - يحتاج تدخل"},
    20: "السعفة والالتهابات الفطرية (Tinea)",
    21: "الشري والحساسية (Urticaria)",
    22: "الأورام الوعائية (Vascular Tumors)",
    23: "الثآليل المعدية (Warts)"
}

# --- 3. محرك تحميل الأوزان ---
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

# --- 4. الواجهة ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>🧬 نظام الخبير الذكي لتشخيص أمراض الجلد</h1>", unsafe_allow_html=True)

col_up, col_res = st.columns([1, 1.2])

with col_up:
    st.subheader("📸 إدخال الصورة")
    src = st.radio("المصدر:", ["المعرض 📤", "الكاميرا 📷"])
    input_file = st.file_uploader("ارفع الصورة هنا", type=["jpg", "png", "jpeg"]) if "المعرض" in src else st.camera_input("التقط صورة")

with col_res:
    st.subheader("🔍 نتيجة التقرير الطبي")
    if input_file:
        img = Image.open(input_file).convert('RGB')
        st.image(img, use_container_width=True)
        
        if st.button("بدء فحص الأنسجة الآن"):
            if model:
                with st.spinner("⏳ جاري تحليل الخصائص المجهرية..."):
                    img_arr = cv2.resize(np.array(img), (224, 224))
                    img_arr = (img_arr.astype(np.float32) / 255.0)[np.newaxis, ...]
                    preds = model.predict(img_arr)[0]
                    idx = np.argmax(preds)
                    
                    info = DISEASES_INFO.get(idx, "غير معروف")
                    name = info["name"] if isinstance(info, dict) else info
                    dtype = info["type"] if isinstance(info, dict) else "حميد / التهابي"
                    color_class = "malignant" if "خبيث" in dtype else "benign"

                    st.markdown(f"""
                    <div class="report-box">
                        <h2 style="color:#1E3A8A;">التشخيص المتوقع:</h2>
                        <h3 class="{color_class}">{name}</h3>
                        <p><b>تصنيف الحالة:</b> {dtype}</p>
                        <p><b>دقة النظام:</b> {preds[idx]:.2%}</p>
                        <hr>
                        <p style="font-size:0.8em; color:#666;">⚠️ ملاحظة: هذا التقرير تم إنشاؤه بواسطة الذكاء الاصطناعي لمساعدة الأطباء ولا يعتبر تشخيصاً نهائياً دون فحص سريري.</p>
                    </div>
                    """, unsafe_allow_html=True)

st.write("---")
st.caption("نظام الحماية الجلدية 2026 | النسخة الاحترافية")
