import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الصفحة والواجهة
st.set_page_config(page_title="Skin Cancer Expert", page_icon="🩺")
st.title("🩺 النظام الخبير لتشخيص سرطان الجلد")
st.write("---")

@st.cache_resource
def load_model():
    model_path = "skin_expert_refined.tflite"
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

# قائمة الأصناف الـ 24 المعتمدة
labels = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
    'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
    'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
    'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
    'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
    'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum'
]

# أنواع السرطان المستهدفة في المشروع
cancer_labels = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

uploaded_file = st.file_uploader("ارفع صورة الفحص الجلدي...", type=["jpg", "png", "jpeg"])

if uploaded_file and interpreter:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="الصورة المرفوعة للفحص", use_container_width=True)
    
    if st.button('إجراء التشخيص النهائي'):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
        
        # --- الخطوة 1: معالجة الصورة وتحويلها لنوع البيانات الصحيح (INT8/FLOAT16) ---
        img = image.resize((w, h), Image.Resampling.BILINEAR)
        img_array = np.array(img).astype(np.float32)
        
        # التطبيع المعياري (Standardization)
        img_array = (img_array / 127.5) - 1.0 
        
        # التحقق من الـ Quantization الخاص بالمدخلات
        if input_details[0]['dtype'] == np.int8 or input_details[0]['dtype'] == np.uint8:
            scale, zero_point = input_details[0]['quantization']
            img_array = (img_array / scale + zero_point).astype(input_details[0]['dtype'])
        else:
            img_array = img_array.astype(input_details[0]['dtype'])
        
        img_array = np.expand_dims(img_array, axis=0)
        
        try:
            # --- الخطوة 2: تنفيذ النموذج ---
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # --- الخطوة 3: إعادة Scaling للمخرجات (De-quantization) ---
            if output_details[0]['dtype'] == np.int8 or output_details[0]['dtype'] == np.uint8:
                scale, zero_point = output_details[0]['quantization']
                probs = (output_data[0].astype(np.float32) - zero_point) * scale
            else:
                probs = output_data[0]
            
            # اختيار التشخيص الأعلى بدقة
            result_idx = np.argmax(probs)
            prediction = labels[result_idx]
            
            st.write("---")
            st.write("### 🔍 النتيجة النهائية للتشخيص:")

            # تصنيف الحالة مباشرة (خبيث/حميد) بدون نسب تشتيت
            if prediction in cancer_labels:
                st.error(f"⚠️ المرض المكتشف: {prediction}")
                st.subheader("التصنيف الطبي: [خبيث - سرطان]")
            else:
                st.success(f"✅ المرض المكتشف: {prediction}")
                st.subheader("التصنيف الطبي: [حميد - ليس سرطان]")
                
        except Exception as e:
            st.error(f"حدث خطأ تقني في معالجة البيانات: {e}")

# الملاحظة القانونية والطبية
st.write("---")
st.warning("⚠️ إخلاء مسؤولية: هذا النظام تعليمي ولا يغني عن مراجعة الطبيب المختص.")
