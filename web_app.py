import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import gdown

# --- 1. إعدادات الصفحة والتنسيق العربي ---
st.set_page_config(page_title="Skin AI Expert", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; border-radius: 10px; height: 3.5em; 
        background-color: #1E3A8A; color: white; font-weight: bold; font-size: 1.2em;
    }
    .report-box { 
        padding: 20px; border-radius: 15px; background-color: white; 
        border-right: 10px solid #1E3A8A; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. الدليل الطبي (24 فئة) ---
DISEASES = {
    0: "حب الشباب والوردية (Acne/Rosacea)", 1: "التقرن الضوئي (Actinic Keratosis)", 
    2: "التهاب الجلد التأتبي (Atopic Dermatitis)", 3: "سرطان الخلايا القاعدية (BCC)",
    4: "آفات حميدة (Benign Keratosis)", 5: "أمراض جلدية فقاعية (Bullous Disease)",
    6: "التهاب النسيج الخلوي (Cellulitis)", 7: "الطفح الدوائي (Drug Eruptions)",
    8: "الأكزيما (Eczema)", 9: "الأمراض الخارجية (Exanthems)",
    10: "العدوى الفطرية (Fungal Infections)", 11: "الهربس والزوائد (Herpes/HPV)",
    12: "الأمراض الجلدية الخفيفة (Light Diseases)", 13: "الورم الميلانيني (Melanoma)",
    14: "الوحمات والشامات (Nevi/Moles)", 15: "أورام ليمفاوية (Lymphoma)",
    16: "الصدفية والقمط (Psoriasis/Lichen Planus)", 17: "لسعات الحشرات (Scabies/Bites)",
    18: "القرنية الدهنية (Seborrheic Keratosis)", 19: "سرطان الخلايا الحرشفية (SCC)",
    20: "السعفة والالتهابات الفطرية (Tinea)", 21: "الشري والحساسية (Urticaria)",
    22: "الأورام الوعائية (Vascular Tumors)", 23: "الثآليل والعدوى الفيروسية (Warts)"
}

# --- 3. محرك تحميل النموذج من Google Drive ---
@st.cache_resource
def load_ai_engine():
    # الرابط المباشر للملف الذي رفعته أنت على Drive
    file_id = '1lMGCojHeGupFunhxX5GnLOiUgxWbbRC5'
    download_url = f'https://drive.google.com/uc?id={file_id}'
    zip_file = "model_weights.zip"
    extract_folder = "model_dir"

    try:
        # تحميل الملف من الدرايف
        if not os.path.exists(zip_file):
            with st.spinner("⏳ جاري جلب محرك الذكاء الاصطناعي من Google Drive..."):
                gdown.download(download_url, zip_file, quiet=False)
        
        # فك الضغط
        if not os.path.exists(extract_folder):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
        
        # البحث عن ملف .h5
        h5_path = None
        for root, dirs, files in os.walk(extract_folder):
            for f in files:
                if f.endswith(".h5"):
                    h5_path = os.path.join(root, f)
                    break
        
        if h5_path:
            # بناء هيكل النموذج الهجين (EfficientNet + MobileNet)
            base1 = tf.keras.applications.EfficientNetB0(input_shape=(224,224,3), include_top=False, weights=None)
            base2 = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
            
            x1 = GlobalAveragePooling2D()(base1.output)
            x2 = GlobalAveragePooling2D()(base2.output)
            merged = Concatenate()([x1, x2])
            
            top = Dense(512, activation='relu')(merged)
            top = Dropout(0.4)(top)
            output = Dense(24, activation='softmax')(top)
            
            model = Model(inputs=[base1.input, base2.input], outputs=output)
            model.load_weights(h5_path)
            return model
    except Exception as e:
        st.error(f"حدث خطأ أثناء التحميل: {e}")
    return None

model = load_ai_engine()

# --- 4. واجهة المستخدم الرئيسية ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>الخبير الذكي لفحص سلامة الجلد 🧬</h1>", unsafe_allow_html=True)

# شريط الأدوات
c1, c2 = st.columns(2)
with c1:
    st.selectbox("🌐 لغة العرض", ["العربية", "English", "Kurdish"])
with c2:
    with st.expander("📖 دليل الأمراض الـ 24"):
        for i, name in DISEASES.items():
            st.write(f"• {name}")

st.write("---")

# منطقة الإدخال
col_input, col_result = st.columns([1, 1])

with col_input:
    st.subheader("📸 الخطوة 1: تزويد الصورة")
    mode = st.radio("اختر طريقة الإدخال:", ["📤 رفع ملف من الجهاز", "📷 التقاط صورة مباشرة"])
    
    source = None
    if mode == "📤 رفع ملف من الجهاز":
        source = st.file_uploader("اختر صورة JPG أو PNG", type=["jpg", "png", "jpeg"])
    else:
        source = st.camera_input("التقط صورة واضحة للمنطقة المصابة")

# منطقة التحليل
if source:
    img = Image.open(source).convert('RGB')
    
    with col_result:
        st.subheader("🔍 الخطوة 2: الفحص والنتيجة")
        st.image(img, caption="الصورة الحالية", use_container_width=True)
        
        if st.button("إجراء الفحص الآن"):
            if model:
                with st.spinner("⏳ جاري تحليل الأنماط الحيوية..."):
                    # المعالجة المسبقة
                    img_resized = cv2.resize(np.array(img), (224, 224))
                    img_array = (img_resized.astype(np.float32) / 255.0)[np.newaxis, ...]
                    
                    # التنبؤ
                    prediction = model.predict([img_array, img_array])[0]
                    class_idx = np.argmax(prediction)
                    confidence = prediction[class_idx]
                    
                    # عرض النتيجة بجمالية
                    st.markdown(f"""
                    <div class="report-box">
                        <h3 style="color:#1E3A8A;">النتيجة المتوقعة: {DISEASES[class_idx]}</h3>
                        <p><b>نسبة الثقة:</b> {confidence:.2%}</p>
                        <hr>
                        <p style="font-size:0.9em; color:#555;">
                        ⚠️ تنبيه: هذا الفحص يعتمد على الذكاء الاصطناعي للأغراض التعليمية فقط، يرجى استشارة الطبيب المختص للتشخيص النهائي.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("المحرك غير جاهز. تأكد من اتصال الإنترنت لتحميل الملف من Google Drive.")

st.write("---")
st.caption("نظام Skin Safety System 2026 | تطوير لأغراض البحث العلمي")
