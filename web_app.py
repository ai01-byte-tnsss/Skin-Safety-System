import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import gdown

# --- 1. إعدادات الصفحة والهوية البصرية ---
st.set_page_config(page_title="Skin Safety System", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
    .main { background-color: #f0f2f6; }
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.5em; 
        background-color: #1E3A8A; color: white; font-weight: bold; font-size: 1.1em;
        border: none; box-shadow: 0 4px 12px rgba(0,0,0,0.1); transition: 0.3s;
    }
    .stButton>button:hover { background-color: #2563EB; transform: translateY(-2px); }
    .report-box { 
        padding: 30px; border-radius: 20px; background-color: white; 
        border-right: 12px solid #1E3A8A; box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-top: 10px;
    }
    .danger-badge { color: #D32F2F; background-color: #FFEBEE; padding: 10px 15px; border-radius: 10px; font-weight: bold; display: inline-block; }
    .safe-badge { color: #388E3C; background-color: #E8F5E9; padding: 10px 15px; border-radius: 10px; font-weight: bold; display: inline-block; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. مصفوفة التصنيف المرتبة (Mapping Matrix) ---
# مطابقة 100% مع ترتيب مجلداتك: akiec, bcc, bkl, df, mel, nv, vasc
LABELS = {
    0: {"id": "akiec", "name": "التقرن الضوئي (Actinic Keratosis)", "status": "خبيث جزئياً / متابعة"},
    1: {"id": "bcc", "name": "سرطان الخلايا القاعدية (Basal Cell Carcinoma)", "status": "خبيث - يحتاج تدخل"},
    2: {"id": "bkl", "name": "آفات التقرن الحميدة (Benign Keratosis)", "status": "حميد - غير مقلق"},
    3: {"id": "df", "name": "الأورام الليفية الجلدية (Dermatofibroma)", "status": "حميد - زوائد ليفية"},
    4: {"name": "الميلانوما (Melanoma)", "status": "خبيث جداً - مراجعة فورية!"},
    5: {"name": "الشامات والوحمات (Melanocytic Nevi)", "status": "حميد - شامة طبيعية"},
    6: {"name": "الآفات الوعائية (Vascular Lesions)", "status": "حميد - أوعية دموية"}
}

# --- 3. معالجة الصور المتقدمة ---
def preprocess_skin_image(image):
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (224, 224))
    # تنعيم الصورة لزيادة دقة التعرف على الحواف
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# --- 4. تحميل المحرك الذكي (الحل الجذري) ---
@st.cache_resource
def load_expert_engine():
    file_id = '1lMGCojHeGupFunhxX5GnLOiUgxWbbRC5'
    h5_file = "validated_model.h5"
    
    if not os.path.exists(h5_file):
        try:
            gdown.download(f'https://drive.google.com/uc?id={file_id}', "model.zip", quiet=False)
            with zipfile.ZipFile("model.zip", 'r') as z:
                for f in z.namelist():
                    if f.endswith('.h5'):
                        with open(h5_file, "wb") as out: out.write(z.read(f))
                        break
        except: return None

    try:
        # بناء الهيكل المتوافق مع عدد مجلداتك السبعة لضمان دقة Softmax
        base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        output = tf.keras.layers.Dense(7, activation='softmax')(x)
        model = tf.keras.Model(inputs=base.input, outputs=output)
        model.load_weights(h5_file, by_name=True, skip_mismatch=True)
        return model
    except:
        return tf.keras.models.load_model(h5_file, compile=False)

model = load_expert_engine()

# --- 5. بناء واجهة المستخدم ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>🧬 نظام الخبير الذكي لسلامة الجلد</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>تحليل فوري وتصنيف دقيق للآفات الجلدية باستخدام مصفوفة Softmax</p>", unsafe_allow_html=True)

st.divider()

col_up, col_res = st.columns([1, 1.3], gap="large")

with col_up:
    st.subheader("📸 إدخال عينة الفحص")
    use_cam = st.toggle("🎥 تشغيل الكاميرا المباشرة")
    file = st.camera_input("التقط صورة") if use_cam else st.file_uploader("📤 ارفع صورة من الـ Dataset", type=["jpg", "png", "jpeg"])
    
    if file:
        img_display = Image.open(file)
        st.image(img_display, caption="العينة المراد فحصها", use_container_width=True)

with col_res:
    st.subheader("🔍 التقرير التحليلي النهائي")
    if file:
        if st.button("🚀 بدء تحليل الخصائص المورفولوجية"):
            if model:
                with st.spinner("⏳ جاري مطابقة الأنماط مع الفئات السبع..."):
                    # 1. المعالجة
                    proc_img = preprocess_skin_image(img_display)
                    # 2. التنبؤ
                    predictions = model.predict(proc_img)[0]
                    # 3. اختيار النتيجة الأقوى
                    idx = np.argmax(predictions)
                    accuracy = predictions[idx]
                    
                    # 4. جلب البيانات والتنسيق
                    data = LABELS.get(idx, {"name": "غير محدد", "status": "فحص سريري مطلوب"})
                    badge_class = "danger-badge" if "خبيث" in data['status'] else "safe-badge"

                    st.markdown(f"""
                    <div class="report-box">
                        <h3 style="color:#1E3A8A; margin-bottom:10px;">اسم المرض المتوقع:</h3>
                        <p style="font-size:1.7em; font-weight:bold; margin-bottom:20px;">{data['name']}</p>
                        
                        <h3 style="color:#1E3A8A; margin-bottom:10px;">التصنيف الطبي للحالة:</h3>
                        <div class="{badge_class}">{data['status']}</div>
                        
                        <div style="margin-top:25px; border-top:1px solid #eee; padding-top:15px;">
                            <p style="font-size:1.1em;"><b>دقة المطابقة الرقمية:</b> <span style="color:#1E3A8A; font-weight:bold;">{accuracy:.2%}</span></p>
                        </div>
                        <p style="font-size:0.8em; color:gray; margin-top:20px;">
                        ⚠️ ملاحظة: هذا النظام يعتمد على الذكاء الاصطناعي لمساعدة الأطباء ولا يغني عن الخزعة الطبية.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("فشل في تحميل محرك الذكاء الاصطناعي. تأكد من رابط الأوزان.")

st.divider()
st.caption("نظام Skin AI Expert © 2026 | تطوير لأغراض التشخيص المتقدم")
