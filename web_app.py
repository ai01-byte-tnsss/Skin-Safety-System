import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# إعدادات الواجهة
st.set_page_config(page_title="Skin Safety Expert", page_icon="🩺", layout="centered")

st.title("🩺 Skin Disease Expert System")
st.markdown(f"### **الدقة الإجمالية للنظام: 53.57%**") #
st.write("---")

@st.cache_resource
def load_model():
    model_path = "skin_expert_refined.tflite" #
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

labels = ['Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
          'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
          'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
          'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
          'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
          'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum']

malignant_types = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

uploaded_file = st.file_uploader("ارفع صورة الجلد لفحصها...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    if st.button('بدء التشخيص التحليلي'):
        input_details = interpreter.get_input_details()
        # تصحيح الأبعاد ديناميكياً لمنع التشخيص الخاطئ
        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
        
        img = image.resize((w, h))
        img_array = np.array(img, dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0 # التطبيع الصحيح
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        
        result_idx = np.argmax(output_data[0])
        prediction_name = labels[result_idx]
        
        st.write("### 🔍 النتيجة:")
        if prediction_name in malignant_types:
            st.error(f"⚠️ النوع المتوقع: {prediction_name} (خبيث)")
        else:
            st.success(f"✅ النوع المتوقع: {prediction_name} (حميد)")

# الملاحظة القانونية كما طلبت
st.write("---")
st.warning("""
**⚠️ ملاحظة هامة جداً (إخلاء مسؤولية):**
* هذا النظام يعتمد كلياً على تقنيات الذكاء الاصطناعي وتم تطويره لأغراض بحثية فقط.
* هذا البرنامج **ليس تشخيصاً طبياً حقيقياً** ولا يغني عن مراجعة الطبيب.
""")

