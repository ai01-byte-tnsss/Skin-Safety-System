import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import gdown

# --- 1. إعدادات الواجهة الاحترافية ---
st.set_page_config(page_title="Skin Safety AI Expert", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.8em; 
        background-color: #0A2342; color: white; font-weight: bold; font-size: 1.1em;
        border: none; box-shadow: 0 4px 15px rgba(0,0,0,0.2); transition: 0.3s;
    }
    .stButton>button:hover { background-color: #1E3A8A; transform: translateY(-2px); }
    .report-box { 
        padding: 30px; border-radius: 20px; background-color: white; 
        border-right: 12px solid #0A2342; box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .result-name { font-size: 1.8em; color: #0A2342; font-weight: bold; margin-bottom: 10px; }
    .status-badge { padding: 10px 20px; border-radius: 10px; font-weight: bold; display: inline-block; font-size: 1.1em; }
    .danger { background-color: #FFE5E5; color: #D32F2F; border: 1px solid #D32F2F; }
    .safe { background-color: #E5F9E5; color: #2E7D32; border: 1px solid #2E7D32; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. مصفوفة التصنيف المرتبة أبجدياً (مطابقة لمجلدات balanced_skin_dataset) ---
# هذا الترتيب هو ما يعتمده ImageDataGenerator في كولاب تلقائياً
CLASS_MAP = {
    0: {"id": "akiec", "name": "التقرن الضوئي (Actinic Keratosis)", "status": "خبيث جزئياً / ما قبل سرطان"},
    1: {"id": "bcc", "name": "سرطان الخلايا القاعدية (Basal Cell Carcinoma)", "status": "خبيث - يحتاج تدخل طبي"},
    2: {"id": "bkl", "name": "آفات التقرن الحميدة (Benign Keratosis)", "status": "حميد - آفات تقرنية"},
    3: {"id": "df", "name": "الأورام الليفية الجلدية (Dermatofibroma)", "status": "حميد - عقدة ليفية"},
    4: {"id": "mel", "name": "الميلانوما - سرطان الجلد (Melanoma)", "status": "خبيث جداً - مراجعة فورية!"},
    5: {"id": "nv", "name": "الشامات والوحمات (Melanocytic Nevi)", "status": "حميد - شامة طبيعية"},
    6: {"id": "vasc", "name": "الآفات الوعائية (Vascular Lesions)", "status": "حميد - عيوب وعائية"}
}

# --- 3. معالجة الصور المتقدمة (Preprocessing) ---
def process_skin_image(image):
    # تحويل الصورة إلى مصفوفة RGB
    img = np.array(image.convert('RGB'))
    # تغيير الحجم إلى (224, 224) كما تم التدريب في كولاب
    img = cv2.resize(img, (224, 224))
    # تحسين جودة التفاصيل (Denoising)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # التقييس (Normalization) ليكون بين 0 و 1
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# --- 4. محرك تحميل النموذج الذكي ---
@st.cache_resource
def load_expert_model():
    file_id = '1lMGCojHeGupFunhxX5GnLOiUgxWbbRC5'
    local_h5 = "skin_expert_v5.h5"
    
    if not os.path.exists(local_h5):
        try:
            gdown.download(f'https://drive.google.com/uc?id={file_id}', "model.zip", quiet=False)
            with zipfile.ZipFile("model.zip", 'r') as z:
                for f in z.namelist():
                    if f.endswith('.h5'):
                        with open(local_h5, "wb") as out: out.write(z.read(f))
                        break
        except: return None

    try:
        # بناء الهيكل الصارم ليتناسب مع الـ 7 فئات فقط
        base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        # دالة Softmax هنا هي المسؤولة عن توزيع الاحتمالات بين الأنواع الـ 7
        out = tf.keras.layers.Dense(7, activation='softmax')(x)
        model = tf.keras.Model(inputs=base.input, outputs=out)
        
        # تحميل الأوزان مع تخطي أي عدم تطابق قديم
        model.load_weights(local_h5, by_name=True, skip_mismatch=True)
        return model
    except:
        return tf.keras.models.load_model(local_h5, compile=False)

model = load_expert_model()

# --- 5. واجهة المستخدم والتحليل ---
st.markdown("<h1 style='text-align:center; color:#0A2342;'>🧬 نظام خبير تشخيص أمراض الجلد الذكي</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>تحليل مورفولوجي دقيق يعتمد على مصفوفة Balanced Skin Dataset</p>", unsafe_allow_html=True)

st.divider()

col_input, col_report = st.columns([1, 1.3], gap="large")

with col_input:
    st.subheader("📸 عينة الفحص")
    source = st.radio("طريقة الإدخال:", ["📤 رفع صورة", "📷 استخدام الكاميرا"])
    
    if source == "📷 استخدام الكاميرا":
        file = st.camera_input("التقط صورة واضحة")
    else:
        file = st.file_uploader("اختر صورة من مجلدات bcc, mel, nv...", type=["jpg", "png", "jpeg"])

with col_report:
    st.subheader("🔍 التقرير التحليلي النهائي")
    if file:
        img_raw = Image.open(file)
        st.image(img_raw, caption="الصورة المراد تحليلها", use_container_width=True)
        
        if st.button("🚀 بدء تحليل الأنسجة والمطابقة"):
            if model:
                with st.spinner("⏳ جاري استخراج السمات الحيوية وتطبيق Softmax..."):
                    # 1. المعالجة
                    processed = process_skin_image(img_raw)
                    # 2. التنبؤ (إخراج مصفوفة من 7 أرقام)
                    predictions = model.predict(processed)[0]
                    # 3. اختيار الفئة الأعلى (Argmax)
                    idx = np.argmax(predictions)
                    
                    # 4. جلب البيانات من المصفوفة المرتبة
                    info = CLASS_MAP.get(idx)
                    is_bad = "خبيث" in info['status']
                    badge_style = "danger" if is_bad else "safe"

                    st.markdown(f"""
                    <div class="report-box">
                        <h3 style="color:#0A2342; margin-bottom:5px;">التشخيص المكتشف:</h3>
                        <div class="result-name">{info['name']}</div>
                        
                        <h3 style="color:#0A2342; margin-top:20px; margin-bottom:10px;">التصنيف الطبي:</h3>
                        <div class="status-badge {badge_style}">{info['status']}</div>
                        
                        <div style="margin-top:25px; border-top:1px solid #eee; padding-top:15px;">
                            <b>نسبة الثقة في المطابقة:</b> 
                            <span style="font-size:1.3em; color:#0A2342;">{predictions[idx]:.2%}</span>
                        </div>
                        <p style="font-size:0.85em; color:gray; margin-top:20px;">
                        ⚠️ ملاحظة: هذا التشخيص استرشادي مبني على الذكاء الاصطناعي، يرجى استشارة الطبيب.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("المحرك غير جاهز. يرجى مراجعة اتصال الإنترنت ورابط الأوزان.")

st.divider()
st.caption("نظام Skin Safety AI v5.0 | النسخة المخصصة لـ 7 فئات")
