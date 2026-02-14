import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Skin Cancer Expert", page_icon="🩺")
st.title("🩺 Skin Cancer Detection System")
st.markdown("### **نظام الكشف عن الأورام الجلدية (Melanoma)**")

@st.cache_resource
def load_model():
    model_path = "skin_expert_refined.tflite"
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

# تأكد من ترتيب القائمة (يجب أن يكون Melanoma في الموقع الصحيح لتدريبك)
labels = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
    'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
    'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
    'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
    'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
    'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum'
]

malignant_types = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

uploaded_file = st.file_uploader("ارفع صورة الفحص (خاصة حالات الميلانوما)...", type=["jpg", "png"])

if uploaded_file and interpreter:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    if st.button('بدء التحليل السرطاني العميق'):
        input_details = interpreter.get_input_details()
        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
        dtype = input_details[0]['dtype']
        
        # --- التعديل الجذري لتحسين دقة السرطان ---
        # استخدام فلتر LANCZOS للحفاظ على حواف الورم
        img = image.resize((w, h), Image.LANCZOS)
        img_array = np.array(img).astype(np.float32)
        
        # المعالجة الدقيقة: MobileNetV2 Preprocessing
        # الميلانوما تعتمد على تباين الألوان؛ هذه المعادلة تبرز التباين للنموذج
        img_array = (img_array / 127.5) - 1.0 
        
        img_array = np.expand_dims(img_array, axis=0).astype(dtype)
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        
        probs = output_data[0]
        top_indices = np.argsort(probs)[-3:][::-1]
        
        st.write("### 📊 نتائج الفحص المجهري الرقمي:")
        for i in top_indices:
            name = labels[i]
            conf = probs[i] * 100
            # تمييز السرطان بلون مختلف إذا ظهر في النتائج
            if name in malignant_types:
                st.warning(f"**تنبيه ورم: {name} (نسبة الاحتمال: {conf:.2f}%)**")
            else:
                st.info(f"الحالة: {name} (نسبة الاحتمال: {conf:.2f}%)")

        final_pred = labels[top_indices[0]]
        st.write("---")
        if final_pred in malignant_types:
            st.error(f"🔴 التشخيص النهائي: {final_pred} - تصنيف خبيث")
        else:
            st.success(f"🟢 التشخيص النهائي: {final_pred} - تصنيف حميد")

st.warning("⚠️ ملاحظة طبية: هذا التحليل يعتمد على الذكاء الاصطناعي للأغراض التعليمية فقط.")


