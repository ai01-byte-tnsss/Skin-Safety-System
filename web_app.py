import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# إعدادات الواجهة
st.set_page_config(page_title="Skin Cancer Expert", page_icon="🩺")
st.title("🩺 نظام الكشف المبكر عن سرطان الجلد")
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

# قائمة الأصناف الـ 24 (تأكد من ترتيبها الصحيح حسب تدريبك)
labels = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
    'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
    'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
    'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
    'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
    'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum'
]

# أنواع السرطان المستهدفة في مشروعك
cancer_types = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

uploaded_file = st.file_uploader("ارفع صورة حالة مشتبه بها (سرطان/حميد)...", type=["jpg", "png", "jpeg"])

if uploaded_file and interpreter:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="الصورة تحت الفحص المجهري الرقمي", use_container_width=True)
    
    if st.button('بدء فحص مؤشرات الأورام'):
        input_details = interpreter.get_input_details()
        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
        dtype = input_details[0]['dtype']
        
        # معالجة الصورة لتوضيح حدود الورم
        img = image.resize((w, h), Image.Resampling.LANCZOS)
        img_array = np.array(img).astype(np.float32)
        
        # التطبيع المعياري لإبراز تباين الألوان الغامقة (السرطان)
        img_array = (img_array / 127.5) - 1.0 
        img_array = np.expand_dims(img_array, axis=0).astype(dtype)
        
        try:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            
            probs = output_data[0]
            top_idx = np.argsort(probs)[-3:][::-1] # أفضل 3 توقعات
            
            # --- منطق الفحص الذكي للسرطان أولاً ---
            found_cancer = False
            primary_prediction = labels[top_idx[0]]
            
            st.write("### 🔍 نتائج الفحص التحليلي:")
            
            # عرض النتائج مع تمييز السرطان فورا
            for i in top_idx:
                name = labels[i]
                confidence = probs[i] * 100
                if name in cancer_types:
                    st.warning(f"🚨 مؤشر خبيث: {name} (نسبة اليقين: {confidence:.2f}%)")
                    found_cancer = True if i == top_idx[0] else found_cancer
                else:
                    st.info(f"🔹 حالة حميدة: {name} (نسبة اليقين: {confidence:.2f}%)")

            st.write("---")
            # التصنيف النهائي الصارم
            if primary_prediction in cancer_types:
                st.error(f"🔴 النتيجة النهائية: {primary_prediction} - تصنيف [خبيث]")
            else:
                st.success(f"🟢 النتيجة النهائية: {primary_prediction} - تصنيف [حميد]")
                
        except Exception as e:
            st.error(f"حدث خطأ في قراءة بيانات المصفوفة: {e}")

st.write("---")
st.caption("ملاحظة: هذا النظام مخصص للكشف عن مؤشرات السرطان لأغراض بحثية.")
