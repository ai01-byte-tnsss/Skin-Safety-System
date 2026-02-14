import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الواجهة
st.set_page_config(page_title="Skin Cancer Expert", page_icon="🩺")
st.title("🩺 نظام تشخيص سرطان الجلد (النسخة المصححة)")

@st.cache_resource
def load_model():
    model_path = "skin_expert_refined.tflite"
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

# الأصناف الـ 24 المعتمدة
labels = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
    'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
    'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
    'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
    'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
    'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum'
]

# أنواع السرطان (الخبيثة)
cancer_labels = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

uploaded_file = st.file_uploader("ارفع صورة الجلد...", type=["jpg", "png", "jpeg"])

if uploaded_file and interpreter:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="الصورة المرفوعة", use_container_width=True)
    
    if st.button('بدء الفحص النهائي'):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
        
        # --- حل المشكلة 1: التعامل مع النوع المطلوب (int8 أو float16) ---
        input_type = input_details[0]['dtype']
        
        img = image.resize((w, h), Image.Resampling.BILINEAR)
        img_array = np.array(img).astype(np.float32)
        
        # عمل Scaling للمدخلات (المعياري لـ MobileNet)
        img_array = (img_array / 127.5) - 1.0 
        
        # تحويل نوع البيانات ليتوافق مع الـ Quantization الخاص بالنموذج
        if input_type == np.int8:
            # إذا كان النموذج int8 نحتاج تحويل القيم من [-1, 1] إلى نطاق التكميم
            scale, zero_point = input_details[0]['quantization']
            img_array = (img_array / scale + zero_point).astype(np.int8)
        else:
            img_array = img_array.astype(input_type) # تحويل لـ float16 أو النوع المطلوب
        
        img_array = np.expand_dims(img_array, axis=0)
        
        try:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # --- حل المشكلة 2: إعادة Scaling للمخرجات (De-quantization) ---
            if output_details[0]['dtype'] == np.int8:
                scale, zero_point = output_details[0]['quantization']
                probs = (output_data[0].astype(np.float32) - zero_point) * scale
            else:
                probs = output_data[0]
            
            # تشخيص واحد فقط للسرطان
            result_idx = np.argmax(probs)
            prediction = labels[result_idx]
            
            st.write("---")
            if prediction in cancer_labels:
                st.error(f"🔴 التشخيص: {prediction}")
                st.subheader("التصنيف النهائي: [خبيث - سرطان]")
            else:
                st.success(f"🟢 التشخيص: {prediction}")
                st.subheader("التصنيف النهائي: [حميد - ليس سرطان]")
                
        except Exception as e:
            st.error(f"خطأ في معالجة النموذج: {e}")

st.warning("⚠️ ملاحظة إخلاء مسؤولية: هذا النظام تعليمي ولا يغني عن التشخيص الطبي.")
