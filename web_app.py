import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الواجهة
st.set_page_config(page_title="Skin Cancer Expert", page_icon="🩺")
st.title("🩺 نظام الفحص الذكي للأورام الجلدية")
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

# الأصناف الـ 24 المعتمدة
labels = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
    'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
    'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
    'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
    'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
    'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum'
]

# تحديد أنواع السرطان (الأصناف الخبيثة) لزيادة حساسيتها
cancer_labels = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

uploaded_file = st.file_uploader("ارفع صورة الفحص الجلدي...", type=["jpg", "png", "jpeg"])

if uploaded_file and interpreter:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="الصورة تحت التحليل الرقمي", use_container_width=True)
    
    if st.button('بدء فحص مؤشرات الأورام الخبيثة'):
        input_details = interpreter.get_input_details()
        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
        dtype = input_details[0]['dtype'] # للتعامل مع FLOAT16
        
        # تحسين معالجة الصورة (LANCZOS) وتطبيع MobileNet
        img = image.resize((w, h), Image.Resampling.LANCZOS)
        img_array = np.array(img).astype(np.float32)
        img_array = (img_array / 127.5) - 1.0 
        img_array = np.expand_dims(img_array, axis=0).astype(dtype)
        
        try:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            
            probs = output_data[0]
            
            # --- منطق الفحص ذو الأولوية (السرطان أولاً) ---
            st.write("### 🔍 نتائج الفحص التحليلي:")
            
            # فحص إذا كان أي نوع من السرطان موجود في أعلى 5 احتمالات
            top_5_indices = np.argsort(probs)[-5:][::-1]
            cancer_detected_in_top = [i for i in top_5_indices if labels[i] in cancer_labels]
            
            # عرض كل الاحتمالات المهمة مع التمييز اللوني
            for i in top_5_indices:
                name = labels[i]
                confidence = probs[i] * 100
                if name in cancer_labels:
                    st.warning(f"🚨 تنبيه مؤشر خبيث: {name} ({confidence:.2f}%)")
                else:
                    st.info(f"🔹 حالة حميدة: {name} ({confidence:.2f}%)")

            st.write("---")
            
            # التصنيف النهائي: إذا وجد سرطان بنسبة معقولة (حتى لو ليس الأول) يتم التحذير منه
            # هنا نكسر "جمود" التصنيف الخاطئ
            highest_cancer_idx = cancer_detected_in_top[0] if cancer_detected_in_top else None
            
            if highest_cancer_idx is not None and probs[highest_cancer_idx] > 0.05: # عتبة 5% لكشف السرطان المتربص
                st.error(f"🔴 النتيجة النهائية: تم رصد مؤشرات لمرض {labels[highest_cancer_idx]} - [خبيث]")
            else:
                final_name = labels[top_5_indices[0]]
                st.success(f"🟢 النتيجة النهائية: {final_name} - [حميد]")
                
        except Exception as e:
            st.error(f"خطأ في قراءة البيانات الرقمية: {e}")

st.write("---")
st.warning("⚠️ ملاحظة: هذا النظام بحثي للكشف عن مؤشرات السرطان ولا يغني عن زيارة الطبيب.")
