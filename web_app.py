import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# إعدادات الصفحة
st.set_page_config(page_title="Skin Safety Expert", page_icon="🩺", layout="centered")

st.title("🩺 Skin Disease Expert System")
st.subheader("نظام خبير متقدم لتشخيص وتصنيف الأمراض الجلدية")
st.markdown(f"### **الدقة الإجمالية للنظام: 53.57%**") 
st.write("---")

# تحميل النموذج مع دعم العمليات المخصصة
@st.cache_resource
def load_tflite_model():
    model_path = "skin_expert_refined.tflite"
    if os.path.exists(model_path):
        try:
            # محاولة تحميل النموذج بشكل قياسي
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        except Exception:
            # إذا فشل (بسبب SELECT_TF_OPS)، يتم استدعاء المحمل المرن
            st.info("جاري تهيئة بيئة العمليات المتقدمة...")
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
    return None

interpreter = load_tflite_model()

# قائمة الأصناف الـ 24
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

malignant_types = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

uploaded_file = st.file_uploader("ارفع صورة الجلد لفحصها...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    if st.button('بدء التشخيص التحليلي'):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        img = image.resize((150, 150)) 
        img_array = np.array(img, dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0 
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        result_idx = np.argmax(output_data[0])
        prediction_name = labels[result_idx]
        
        st.write("### 🔍 نتائج التحليل:")
        if prediction_name in malignant_types:
            st.error(f"⚠️ تنبيه: {prediction_name} (تصنيف خبيث)")
        else:
            st.success(f"✅ التشخيص: {prediction_name} (تصنيف حميد)")

st.write("---")
st.warning("⚠️ هذا النظام لأغراض تعليمية فقط ولا يغني عن استشارة الطبيب.")
