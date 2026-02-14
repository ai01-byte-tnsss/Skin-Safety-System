import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الصفحة
st.set_page_config(page_title="Skin Safety System", page_icon="🩺", layout="centered")

st.title("🩺 Skin Disease Expert System")
st.subheader("نظام خبير لتشخيص الأمراض الجلدية")
st.write("---")

# 2. تحميل نموذج TFLite
@st.cache_resource
def load_tflite_model():
    model_path = "skin_expert_lite.tflite"
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    else:
        st.error("❌ ملف النموذج غير موجود!")
        return None

interpreter = load_tflite_model()

if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# 3. قائمة الأمراض (24 صنفاً)
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

# 4. واجهة التطبيق
uploaded_file = st.file_uploader("ارفع صورة الجلد لفحصها...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    if st.button('بدء التشخيص'):
        with st.spinner('جاري التحليل...'):
            try:
                # حل مشكلة Dimension mismatch: تغيير الحجم إلى 150
                img = image.resize((150, 150)) 
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # إدخال البيانات للنموذج
                interpreter.set_tensor(input_details[0]['index'], img_array)
                
                # تشغيل التوقع
                interpreter.invoke()
                
                # استخراج النتيجة
                output_data = interpreter.get_tensor(output_details[0]['index'])
                result_idx = np.argmax(output_data)
                confidence = np.max(output_data) * 100
                
                # 5. عرض النتائج النهائية
                st.write("---")
                st.success(f"### التشخيص المتوقع: {labels[result_idx]}")
                st.info(f"### نسبة الثقة: {confidence:.2f}%")
                st.warning("⚠️ تنبيه: هذا النظام للاستخدام التعليمي فقط.")
                
            except Exception as e:
                st.error(f"حدث خطأ أثناء المعالجة: {e}")

st.write("---")
st.caption("Graduation Project - Skin Safety System 2026")
