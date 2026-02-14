import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الواجهة
st.set_page_config(page_title="Skin Safety Expert", page_icon="🩺", layout="centered")

st.title("🩺 Skin Disease Expert System")
st.markdown(f"### **الدقة الإجمالية للنظام: 53.57%**") #
st.write("---")

# 2. تحميل النموذج
@st.cache_resource
def load_model():
    model_path = "skin_expert_refined.tflite"
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

# قائمة الأصناف الـ 24
labels = ['Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
          'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
          'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
          'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
          'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
          'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum']

malignant_types = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

# 3. واجهة رفع الصور
uploaded_file = st.file_uploader("ارفع صورة الجلد لفحصها...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    if st.button('بدء التشخيص التحليلي'):
        # --- حل مشكلة السطر 67 (المصفوفة العامة) ---
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # قراءة الأبعاد المطلوبة من النموذج نفسه
        new_height = input_details[0]['shape'][1]
        new_width = input_details[0]['shape'][2]
        
        # تغيير حجم الصورة وتحويلها لمصفوفة float32 حصراً
        img = image.resize((new_width, new_height))
        img_array = np.array(img, dtype=np.float32)
        
        # التطبيع (Normalization)
        img_array = (img_array / 127.5) - 1.0 
        
        # إضافة بعد الدفعة لتصبح المصفوفة [1, Height, Width, 3]
        img_array = np.expand_dims(img_array, axis=0)
        
        try:
            # السطر 67: إرسال المصفوفة للنموذج
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # عرض النتيجة
            result_idx = np.argmax(output_data[0])
            prediction_name = labels[result_idx]
            
            st.write(f"### 🔍 التشخيص: {prediction_name}")
            if prediction_name in malignant_types:
                st.error("التصنيف: خبيث (يستوجب فحص طبي)")
            else:
                st.success("التصنيف: حميد")
        except Exception as e:
            st.error(f"خطأ في معالجة المصفوفة: {e}")

# 4. الملاحظة الطبية
st.write("---")
st.warning("""
**⚠️ ملاحظة إخلاء مسؤولية:**
هذا النظام يعتمد على الذكاء الاصطناعي للأغراض البحثية فقط، وليس تشخيصاً طبياً حقيقياً.
""")

