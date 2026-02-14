import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Skin Cancer Expert", page_icon="🩺")
st.title("🩺 نظام فحص وتشخيص سرطان الجلد")

@st.cache_resource
def load_model():
    model_path = "skin_expert_refined.tflite"
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

# قائمة الأصناف
labels = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
    'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
    'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
    'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
    'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
    'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum'
]

# الأصناف الخبيثة (سرطان)
cancer_labels = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

uploaded_file = st.file_uploader("ارفع صورة الفحص الجلدي...", type=["jpg", "png", "jpeg"])

if uploaded_file and interpreter:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="الصورة المرفوعة", use_container_width=True)
    
    if st.button('إجراء التشخيص النهائي'):
        input_details = interpreter.get_input_details()
        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
        dtype = input_details[0]['dtype'] 
        
        # --- التعديل الجوهري لكسر جمود التشخيص ---
        img = image.resize((w, h), Image.Resampling.LANCZOS)
        img_array = np.array(img).astype(np.float32)
        
        # تغيير معادلة التطبيع لفك الارتباط بـ Warts
        # تجربة التطبيع من 0 إلى 1 (غالبية نماذج الكولاب تعمل هكذا)
        img_array = img_array / 255.0 
        
        img_array = np.expand_dims(img_array, axis=0).astype(dtype)
        
        try:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            
            probs = output_data[0]
            
            # البحث عن أعلى نسبة لسرطان موجودة في النتائج حتى لو لم تكن الأولى
            cancer_indices = [labels.index(c) for c in cancer_labels]
            cancer_probs = {labels[i]: probs[i] for i in cancer_indices}
            highest_cancer = max(cancer_probs, key=cancer_probs.get)
            
            # الحصول على التوقع العام الأعلى
            top_idx = np.argmax(probs)
            prediction = labels[top_idx]
            
            st.write("---")
            
            # منطق الأولوية للسرطان: إذا كانت نسبة السرطان > 1% اعتبره خبيثاً للأمان
            if cancer_probs[highest_cancer] > 0.01: 
                st.error(f"⚠️ التشخيص المكتشف: {highest_cancer}")
                st.subheader("🔴 التصنيف: [خبيث - سرطان]")
            else:
                st.success(f"✅ التشخيص المكتشف: {prediction}")
                st.subheader("🟢 التصنيف: [حميد - ليس سرطان]")
                
        except Exception as e:
            st.error(f"خطأ تقني: {e}")

st.write("---")
st.warning("⚠️ ملاحظة إخلاء مسؤولية: هذا النظام تعليمي ولا يغني عن التشخيص الطبي.")
