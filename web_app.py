import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="Skin Safety System", page_icon="🩺")
st.title("🩺 Skin Disease Expert System")
st.write("---")

# 2. تحميل نموذج TFLite (أكثر استقراراً على Streamlit)
@st.cache_resource
def load_tflite_model():
    # تأكد أن هذا الملف موجود في المستودع بنفس الاسم
    interpreter = tf.lite.Interpreter(model_path="skin_expert_lite.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
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
uploaded_file = st.file_uploader("ارفع صورة الجلد...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المختارة', use_container_width=True)
    
    if st.button('بدء التشخيص'):
        # معالجة الصورة لتناسب TFLite
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # تنفيذ التوقع
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        result_idx = np.argmax(output_data)
        confidence = np.max(output_data) * 100
        
        # عرض النتيجة
        st.success(f"التشخيص المتوقع: {labels[result_idx]}")
        st.info(f"نسبة التأكد: {confidence:.2f}%")
