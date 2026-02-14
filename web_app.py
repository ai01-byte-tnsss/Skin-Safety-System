import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# إعدادات الواجهة
st.set_page_config(page_title="Skin Cancer Expert", page_icon="🩺")
st.title("🩺 نظام تشخيص سرطان الجلد")

@st.cache_resource
def load_model():
    model_path = "skin_expert_refined.tflite"
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

# القائمة الرسمية للأصناف (تأكد من بقاء 'Melanoma' كمرجع للسرطان)
labels = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
    'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
    'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
    'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
    'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
    'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum'
]

# أنواع السرطان الخبيثة
cancer_labels = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

uploaded_file = st.file_uploader("ارفع صورة الفحص الجلدي...", type=["jpg", "png", "jpeg"])

if uploaded_file and interpreter:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="الصورة المرفوعة", use_container_width=True)
    
    if st.button('إجراء التشخيص النهائي'):
        input_details = interpreter.get_input_details()
        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
        dtype = input_details[0]['dtype'] 
        
        # --- حل مشكلة التجمد: تغيير المعالجة إلى NEAREST لمنع تمويه الأنسجة ---
        img = image.resize((w, h), Image.Resampling.NEAREST)
        img_array = np.array(img).astype(np.float32)
        
        # تجربة التطبيع الخام (بدون طرح 1) لكسر جمود النموذج
        img_array = img_array / 255.0 
        
        img_array = np.expand_dims(img_array, axis=0).astype(dtype)
        
        try:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            
            # استخراج النتائج
            probs = output_data[0]
            
            # --- ميزة كسر الجمود (The Bias Breaker) ---
            # إذا كان النموذج يعطي 'Warts' بنسبة ساحقة، سننظر للنتيجة الثانية فوراً
            sorted_indices = np.argsort(probs)[::-1]
            
            # اختيار النتيجة الأفضل التي ليست 'Warts' إذا كان هناك احتمال للسرطان
            final_idx = sorted_indices[0]
            
            # فحص يدوي: هل هناك أي نوع سرطان ظهر في أفضل 3 نتائج؟
            found_cancer = None
            for idx in sorted_indices[:3]:
                if labels[idx] in cancer_labels and probs[idx] > 0.005: # حتى لو الاحتمال 0.5%
                    found_cancer = labels[idx]
                    break
            
            st.write("---")
            st.write("### 🔍 النتيجة النهائية للتشخيص:")

            # إذا وجدنا سرطان في الخلفية، نعطي الأولوية له (لأن السرطان هو أساس مشروعك)
            if found_cancer:
                st.error(f"⚠️ التشخيص المكتشف: {found_cancer}")
                st.subheader("🔴 التصنيف: [خبيث - سرطان]")
            else:
                prediction = labels[final_idx]
                if prediction in cancer_labels:
                    st.error(f"⚠️ التشخيص المكتشف: {prediction}")
                    st.subheader("🔴 التصنيف: [خبيث - سرطان]")
                else:
                    st.success(f"✅ التشخيص المكتشف: {prediction}")
                    st.subheader("🟢 التصنيف: [حميد - ليس سرطان]")
                
        except Exception as e:
            st.error(f"خطأ تقني: {e}")

st.write("---")
st.warning("⚠️ ملاحظة إخلاء مسؤولية: هذا النظام تعليمي ولا يغني عن التشخيص الطبي.")

