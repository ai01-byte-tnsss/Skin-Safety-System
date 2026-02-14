import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الواجهة
st.set_page_config(page_title="Skin Cancer Expert", page_icon="🩺")
st.title("🩺 نظام تشخيص سرطان الجلد الخبير")
st.write("---")

@st.cache_resource
def load_model():
    model_path = "skin_expert_refined.tflite"
    if os.path.exists(model_path):
        # استخدام التوزيع الافتراضي لمنع تجمد القيم
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

# القائمة الرسمية المعتمدة لترتيب الأصناف الـ 24
labels = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
    'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
    'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
    'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
    'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
    'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum'
]

# أنواع السرطان (الخبيثة) التي يركز عليها المشروع
cancer_labels = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

uploaded_file = st.file_uploader("ارفع صورة الفحص الجلدي...", type=["jpg", "png", "jpeg"])

if uploaded_file and interpreter:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="الصورة تحت المعالجة الرقمية", use_container_width=True)
    
    if st.button('إجراء التشخيص النهائي'):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
        dtype = input_details[0]['dtype'] 

        # --- الحل الجذري لمشكلة تجمد التصنيف ---
        # 1. تصغير الصورة مع الحفاظ على التباين اللوني
        img = image.resize((w, h), Image.Resampling.BILINEAR)
        img_array = np.array(img).astype(np.float32)
        
        # 2. التطبيع (Normalization) بأسلوب MobileNet الرسمي لفك جمود القيم
        img_array = (img_array / 127.5) - 1.0 
        
        # 3. التأكد من تطابق النوع FLOAT16 أو FLOAT32 حسب النموذج
        img_array = np.expand_dims(img_array, axis=0).astype(dtype)
        
        try:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # استلام الاحتمالات
            probs = output_data[0]
            
            # --- منطق التشخيص الواحد (سرطان أم لا) ---
            # البحث عن أعلى قيمة للسرطان في مخرجات النموذج
            cancer_indices = [labels.index(c) for c in cancer_labels]
            current_cancer_probs = {labels[i]: probs[i] for i in cancer_indices}
            best_cancer_type = max(current_cancer_probs, key=current_cancer_probs.get)
            
            # الحصول على أعلى توقع عام
            top_idx = np.argmax(probs)
            general_prediction = labels[top_idx]
            
            st.write("---")
            st.write("### 🔍 نتيجة الفحص النهائية:")

            # الأولوية للسرطان: إذا كان احتمال السرطان يتجاوز عتبة بسيطة، يتم إعلانه كخبيث
            # هذا يكسر انحياز النموذج لـ Warts
            if probs[labels.index(best_cancer_type)] > 0.01 or general_prediction in cancer_labels:
                st.error(f"⚠️ التشخيص: {best_cancer_type}")
                st.subheader("🔴 التصنيف النهائي: [خبيث - سرطان]")
            else:
                st.success(f"✅ التشخيص: {general_prediction}")
                st.subheader("🟢 التصنيف النهائي: [حميد - ليس سرطان]")

        except Exception as e:
            st.error(f"خطأ في مصفوفة البيانات: {e}")

# ملاحظة إخلاء المسؤولية الطبية
st.write("---")
st.warning("⚠️ ملاحظة: هذا التشخيص يعتمد على الذكاء الاصطناعي للأغراض التعليمية فقط.")
