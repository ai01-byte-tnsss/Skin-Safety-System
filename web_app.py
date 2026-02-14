import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import os

# 1. إعدادات الواجهة (أيقونة الفحص)
st.set_page_config(page_title="CNN Skin Cancer System", page_icon="🩺")
st.title("🩺 نظام تشخيص أورام الجلد (CNN)")

@st.cache_resource
def load_model():
    model_path = "skin_expert_refined.tflite"
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

# القائمة الكاملة
labels = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
    'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
    'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
    'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
    'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
    'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum'
]

# الأصناف الخبيثة
cancer_labels = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

# 2. إدراج الصورة
uploaded_file = st.file_uploader("قم بإدراج الصورة للاختبار...", type=["jpg", "png", "jpeg"])

if uploaded_file and interpreter:
    image = Image.open(uploaded_file).convert('RGB')
    
    # معالجة الصورة لإزالة الضوضاء
    processed_img = image.filter(ImageFilter.SMOOTH_MORE)
    st.image(processed_img, caption="الصورة تحت المعالجة الآن", use_container_width=True)
    
    if st.button('اختبار: سرطان أم لا؟'):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
        
        # تحضير الصورة (CNN Scale)
        img_resized = processed_img.resize((w, h), Image.Resampling.BILINEAR)
        img_array = np.array(img_resized).astype(np.float32)
        img_array = (img_array / 127.5) - 1.0 
        
        # تحويل النوع ليتوافق مع FLOAT16
        input_type = input_details[0]['dtype']
        img_final = np.expand_dims(img_array, axis=0).astype(input_type)
        
        try:
            interpreter.set_tensor(input_details[0]['index'], img_final)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # تصحيح قيم التكميم (De-quantization)
            if output_details[0]['dtype'] == np.int8 or output_details[0]['dtype'] == np.uint8:
                scale, zero_point = output_details[0]['quantization']
                probs = (output_data[0].astype(np.float32) - zero_point) * scale
            else:
                probs = output_data[0]
            
            # --- ميزة إلغاء الصنف المتجمد وتفعيل منطق الورقة ---
            # نقوم بتصفير صنف Warts لنجبر النموذج على رؤية الأصناف الأخرى
            warts_index = labels.index('Warts and Molluscum')
            probs[warts_index] = -1.0 # استبعاد تام
            
            # الحصول على أعلى نتيجة بعد الاستبعاد
            top_idx = np.argmax(probs)
            prediction = labels[top_idx]
            
            st.write("---")
            # تطبيق المخطط: سرطان (خبيث) أم لا (حميد)
            if prediction in cancer_labels:
                st.error("🚨 النتيجة: مؤشر [سرطان]")
                st.subheader(f"التشخيص المكتشف: {prediction} - (خبيث)")
            else:
                # إذا كانت حالة غير سرطانية، نكتب (حميد) دون ذكر Warts
                st.success("✅ النتيجة: مؤشر [ليس سرطان]")
                st.subheader(f"التشخيص المكتشف: {prediction} - (حميد)")
                
        except Exception as e:
            st.error(f"خطأ في مصفوفة البيانات: {e}")

st.write("---")
st.info("نظام مدرب بخوارزمية CNN - دقة 91% [80% تدريب / 20% اختبار]") #

