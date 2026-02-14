import streamlit as st
from PIL import Image
import numpy as np
import os

# إعدادات الواجهة
st.set_page_config(page_title="Skin Safety Expert", page_icon="🩺")
st.title("🩺 Skin Disease Expert System")
st.markdown("### **الدقة الإجمالية للنظام: 53.57%**")

# تحميل المفسر بطريقة لا تحتاج مكتبات ثقيلة
@st.cache_resource
def load_model():
    # سنحاول استيراد المكتبة محلياً فقط عند الحاجة لتجنب أخطاء التثبيت العامة
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except:
        # إذا فشلت، سنستخدم البديل الخفيف المدمج في النظام
        from tensorflow.lite.python.interpreter import Interpreter
        interpreter = Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter

interpreter = load_model()

labels = ['Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
          'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
          'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
          'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
          'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
          'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum']

malignant_types = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

uploaded_file = st.file_uploader("ارفع صورة الجلد للفحص...", type=["jpg", "png"])

if uploaded_file and interpreter:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
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
        
        res_idx = np.argmax(output_data[0])
        prediction_name = labels[res_idx]
        
        if prediction_name in malignant_types:
            st.error(f"⚠️ التشخيص: {prediction_name} (تصنيف خبيث)")
        else:
            st.success(f"✅ التشخيص: {prediction_name} (تصنيف حميد)")

st.warning("⚠️ ملاحظة: هذا النظام لأغراض تعليمية وبحثية فقط ولا يغني عن استشارة الطبيب.")
