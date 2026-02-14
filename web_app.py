import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# إعداد واجهة المستخدم
st.set_page_config(page_title="Skin Safety Expert", page_icon="🩺")
st.title("🩺 Skin Disease Expert System")
st.markdown("### **الدقة الإجمالية للنظام: 53.57%**")

@st.cache_resource
def load_model():
    model_path = "skin_expert_refined.tflite"
    if os.path.exists(model_path):
        try:
            # تحميل المفسر القياسي
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            st.error(f"خطأ في توافق العمليات: {e}")
            return None
    return None

interpreter = load_model()

# قائمة الأمراض الـ 24
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
        img_array = (img_array / 127.5) - 1.0 # التطبيع
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        prediction_name = labels[np.argmax(output_data[0])]
        
        st.write("### 🔍 النتائج:")
        if prediction_name in malignant_types:
            st.error(f"⚠️ التشخيص المتوقع: {prediction_name} (خبيث)")
        else:
            st.success(f"✅ التشخيص المتوقع: {prediction_name} (حميد)")

st.warning("هذا النظام لأغراض تعليمية فقط.")

