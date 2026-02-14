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

labels = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
    'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
    'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
    'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
    'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
    'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum'
]

malignant_types = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

# 3. واجهة رفع الصور
uploaded_file = st.file_uploader("ارفع صورة الجلد للفحص...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    if st.button('بدء التشخيص التحليلي'):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # قراءة الأبعاد والنوع (FLOAT16) ديناميكياً
        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
        required_dtype = input_details[0]['dtype']
        
        # --- الحل الجذري لمشكلة التشخيص الخاطئ (The Secret Sauce) ---
        # 1. تغيير الحجم بدقة عالية
        img = image.resize((w, h), Image.LANCZOS)
        img_array = np.array(img).astype(np.float32)
        
        # 2. تطبيق معادلة MobileNet الرسمية: (pixel / 127.5) - 1.0
        # هذه المعادلة هي التي تجعل النموذج يفرق بين الألوان بدقة
        img_array = (img_array / 127.5) - 1.0
        
        # 3. تحويل النوع ليتوافق مع FLOAT16 وإضافة بعد الدفعة
        img_array = np.expand_dims(img_array, axis=0).astype(required_dtype)
        
        try:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # استخراج النتائج (Probabilities)
            probabilities = output_data[0]
            result_idx = np.argmax(probabilities)
            prediction_name = labels[result_idx]
            confidence = probabilities[result_idx] * 100 # نسبة الثقة
            
            st.write(f"### 🔍 التشخيص المكتشف: {prediction_name}")
            # st.write(f"**نسبة الثقة في التشخيص:** {confidence:.2f}%")
            
            if prediction_name in malignant_types:
                st.error("تصنيف الحالة: خبيث (يستوجب فحصاً طبياً فورياً)")
            else:
                st.success("تصنيف الحالة: حميد")
        except Exception as e:
            st.error(f"خطأ في معالجة البيانات: {e}")

# 4. الملاحظة الطبية
st.write("---")
st.warning("⚠️ ملاحظة إخلاء مسؤولية: هذا النظام يعتمد على الذكاء الاصطناعي للأغراض البحثية فقط وليس تشخيصاً طبياً حقيقياً.")


