import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import os

# 1. إعدادات الواجهة الاحترافية (بدون نسب)
st.set_page_config(page_title="CNN Skin Cancer System", page_icon="🩺")
st.title("🩺 نظام CNN المتطور لتشخيص الأورام")
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

# قائمة الأصناف كما هي في تدريبك
labels = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
    'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
    'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
    'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
    'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
    'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum'
]

# تصنيفات السرطان (الخبيث) حسب مخططك
cancer_labels = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

# 2. منطقة إدراج الصورة (أيقونة الاختبار)
uploaded_file = st.file_uploader("قم بإدراج صورة الجلد للفحص...", type=["jpg", "png", "jpeg"])

if uploaded_file and interpreter:
    # معالجة قوية للصورة (إزالة الضوضاء وتصحيح الإضاءة)
    image = Image.open(uploaded_file).convert('RGB')
    image = ImageOps.autocontrast(image) # تحسين التباين لكشف الأورام
    image = image.filter(ImageFilter.SHARPEN) # توضيح الحواف لخوارزمية CNN
    
    st.image(image, caption="الصورة المعالجة رقمياً", use_container_width=True)
    
    if st.button('اختبار: سرطان أم لا؟'):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
        
        # تحضير المصفوفة للنموذج المكمم
        img = image.resize((w, h), Image.Resampling.LANCZOS)
        img_array = np.array(img).astype(np.float32)
        img_array = (img_array / 127.5) - 1.0 # Scaling الرسمي
        
        # تحويل النوع ليتوافق مع FLOAT16 أو INT8 تلقائياً
        input_type = input_details[0]['dtype']
        img_array = np.expand_dims(img_array, axis=0).astype(input_type)
        
        try:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # تصحيح مخرجات التكميم (De-quantization) لضمان الدقة العالية
            if output_details[0]['dtype'] == np.int8 or output_details[0]['dtype'] == np.uint8:
                scale, zero_point = output_details[0]['quantization']
                probs = (output_data[0].astype(np.float32) - zero_point) * scale
            else:
                probs = output_data[0]
            
            # --- منطق التشخيص النهائي حسب الورقة (خبيث أم حميد) ---
            # 1. فحص مؤشرات السرطان أولاً
            cancer_idx = [labels.index(c) for c in cancer_labels]
            cancer_prob_sum = sum([probs[i] for i in cancer_idx])
            
            # النتيجة العامة
            top_idx = np.argmax(probs)
            prediction = labels[top_idx]
            
            st.write("---")
            # تطبيق المخطط: سرطان أم لا؟ -> النوع -> خبيث/حميد
            if prediction in cancer_labels or cancer_prob_sum > 0.1: # حساسية عالية للسرطان
                # إذا كان أحد أنواع السرطان هو الأعلى، أو مجموع احتمالات السرطان كافٍ
                final_diag = prediction if prediction in cancer_labels else "Melanoma (مؤشر مرتفع)"
                st.error("⚠️ نتيجة الفحص: [سرطان]")
                st.subheader(f"التصنيف: {final_diag} - (خبيث)")
            else:
                st.success("✅ نتيجة الفحص: [ليس سرطان]")
                st.subheader(f"التصنيف: {prediction} - (حميد)")
                
        except Exception as e:
            st.error(f"خطأ في مصفوفة التصنيف: {e}")

st.write("---")
st.info("نظام CNN - دقة التدريب: 80% / دقة الاختبار المستهدفة: 91%") #
