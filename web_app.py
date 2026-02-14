import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الصفحة
st.set_page_config(page_title="Skin Expert System", page_icon="🩺", layout="centered")

st.title("🩺 Skin Disease Expert System")
st.subheader("نظام خبير لتشخيص الأمراض الجلدية")
st.write("---")

# 2. تحميل النموذج مع معالجة الخطأ المعماري
@st.cache_resource
def load_my_model():
    model_path = 'skin_expert_master.h5'
    if os.path.exists(model_path):
        try:
            # استخدام compile=False لحل مشكلة Layer dense_1 expects 1 input
            model = tf.keras.models.load_model(model_path, compile=False)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error("❌ Model file not found in repository!")
        return None

model = load_my_model()

# 3. قائمة الأصناف (مرتبة أبجدياً)
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
uploaded_file = st.file_uploader("Upload Skin Image / ارفع صورة الجلد", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Start Diagnosis / بدء التشخيص'):
        with st.spinner('Analyzing...'):
            # معالجة الصورة
            img = image.resize((224, 224))
            img_array = np.array(img.convert('RGB')) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # التوقع
            predictions = model.predict(img_array)
            result_idx = np.argmax(predictions)
            confidence = np.max(predictions) * 100
            
            # عرض النتائج
            st.write("---")
            st.success(f"### Prediction: {labels[result_idx]}")
            st.info(f"### Confidence: {confidence:.2f}%")
            st.warning("⚠️ This is an AI tool. Consult a doctor for medical advice.")

st.write("---")
st.caption("Graduation Project - Skin Safety System 2026")
