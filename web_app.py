import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الصفحة
st.set_page_config(page_title="Skin Expert System", page_icon="🩺", layout="centered")

# تنسيق الواجهة بالعربي والانجليزي
st.title("🩺 Skin Disease Expert System")
st.subheader("نظام خبير لتشخيص الأمراض الجلدية")
st.write("---")

# 2. تحميل النموذج
# نستخدم @st.cache_resource لكي يتم تحميل النموذج مرة واحدة فقط وتسريع التطبيق
@st.cache_resource
def load_my_model():
    model_path = 'skin_expert_master.h5' # تأكد أن هذا الاسم مطابق للملف في مجلدك
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error("❌ Model file 'skin_expert_master.h5' not found!")
        return None

model = load_my_model()

# 3. قائمة الأمراض (24 صنفاً) 
# ملاحظة: يجب أن تكون مرتبة أبجدياً كما فعلنا في التدريب (sorted listdir)
labels = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 
    'Bullous Disease', 'Cellulitis Impetigo', 'Eczema', 
    'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 'Herpes HPV', 
    'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
    'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 
    'Psoriasis and Lichen Planus', 'Scabies and Bites', 'Seborrheic Keratoses', 
    'Systemic Disease', 'Tinea Ringworm', 'Urticaria Hives', 
    'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum'
]

# 4. واجهة رفع الصور
uploaded_file = st.file_uploader("Upload a clear photo of the skin condition / ارفع صورة واضحة للحالة الجلدية", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # عرض الصورة المرفوعة
    image = Image.open(uploaded_file)
    st.image(image, caption='Image uploaded successfully', use_container_width=True)
    
    with st.spinner('Analyzing... جاري التحليل'):
        # معالجة الصورة لتناسب النموذج
        img = image.resize((224, 224))
        img_array = np.array(img)
        
        # التأكد من أن الصورة 3 قنوات (RGB)
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
            
        img_array = img_array / 255.0  # التطبيع كما فعلنا في التدريب
        img_array = np.expand_dims(img_array, axis=0)
        
        # إجراء التوقع
        predictions = model.predict(img_array)
        result_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        
        # 5. عرض النتائج
        st.write("---")
        st.success(f"### Prediction: {labels[result_index]}")
        st.info(f"### Confidence Level: {confidence:.2f}%")
        
        # تنبيه طبي (ضروري لمشاريع التخرج الطبية)
        st.warning("⚠️ Disclaimer: This is an AI-assisted tool for educational purposes. Please consult a professional dermatologist.")

st.write("---")
st.caption("Developed as a Graduation Project - Skin Safety System 2026")