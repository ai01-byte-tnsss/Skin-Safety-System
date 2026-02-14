import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الواجهة الاحترافية
st.set_page_config(page_title="Skin Safety Expert", page_icon="🩺", layout="centered")
st.title("🩺 Skin Disease Expert System")
st.markdown(f"### **نظام تحليل وتصنيف الأمراض الجلدية**")
st.write("---")

# 2. تحميل النموذج برمجياً
@st.cache_resource
def load_model():
    model_path = "skin_expert_refined.tflite"
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

# قائمة الأصناف الـ 24 (تأكد أنها مطابقة لترتيب مجلدات التدريب)
labels = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
    'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
    'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
    'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
    'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
    'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum'
]

malignant_types = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

# 3. واجهة الرفع والمعالجة
uploaded_file = st.file_uploader("ارفع صورة الجلد للفحص التحليلي...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    if st.button('بدء التشخيص التحليلي'):
        # كشف تفاصيل النموذج تلقائياً لمنع أعطال المصفوفة
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
        required_dtype = input_details[0]['dtype'] # التعامل مع FLOAT16
        
        # معالجة الصورة بأعلى جودة (LANCZOS) وتطبيع معيار MobileNet
        img = image.resize((w, h), Image.LANCZOS)
        img_array = np.array(img).astype(np.float32)
        img_array = (img_array / 127.5) - 1.0 # المعالجة الرسمية للنماذج المطورة
        img_array = np.expand_dims(img_array, axis=0).astype(required_dtype)
        
        try:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # --- ميزة كشف العطل وترتيب الأصناف ---
            probabilities = output_data[0]
            # الحصول على أعلى 3 احتمالات لضمان كشف "تداخل الأصناف"
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            
            st.write("### 📊 نتائج التحليل الاحتمالي:")
            for i in top_3_indices:
                score = probabilities[i] * 100
                st.info(f"النوع: **{labels[i]}** | نسبة الثقة: **{score:.2f}%**")
            
            # النتيجة النهائية (صاحبة أعلى احتمال)
            final_pred = labels[top_3_indices[0]]
            
            st.write("---")
            if final_pred in malignant_types:
                st.error(f"⚠️ التشخيص النهائي: {final_pred} (تصنيف خبيث)")
            else:
                st.success(f"✅ التشخيص النهائي: {final_pred} (تصنيف حميد)")

        except Exception as e:
            st.error(f"فشل في مصفوفة البيانات: {e}")

# 4. ملاحظة إخلاء المسؤولية
st.write("---")
st.warning("⚠️ ملاحظة: هذا النظام يعتمد على الذكاء الاصطناعي للأغراض البحثية والتعليمية فقط وليس تشخيصاً طبياً حقيقياً.")
