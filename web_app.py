import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الصفحة
st.set_page_config(page_title="Skin Safety Expert", page_icon="🩺", layout="centered")

st.title("🩺 Skin Disease Expert System")
st.subheader("نظام خبير متقدم لتشخيص وتصنيف الأمراض الجلدية")
st.markdown(f"### **الدقة الإجمالية للنظام: 53.57%**") # عرض الدقة الكلية للنظام
st.write("---")

# 2. تحميل نموذج TFLite
@st.cache_resource
def load_tflite_model():
    model_path = "skin_expert_lite.tflite"
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_tflite_model()

# 3. قائمة الأصناف الـ 24
labels = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 
    'Bullous Disease', 'Cellulitis Impetigo', 'Eczema', 
    'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 'Herpes HPV', 
    'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
    'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 
    'Psoriasis and Lichen Planus', 'Scabies and Bites', 'Seborrheic Keratoses', 
    'Systemic Disease', 'Tinea Ringworm', 'Urticaria Hives', 
    'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum'
]

# الأنواع التي يصنفها النظام كأورام (خبيثة أو ما قبل سرطانية)
malignant_types = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

# 4. واجهة التطبيق
uploaded_file = st.file_uploader("ارفع صورة الجلد لفحصها...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    if st.button('بدء التشخيص التحليلي'):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # --- حل مشكلة التصنيف الواحد (Preprocessing Fix) ---
        img = image.resize((150, 150)) 
        img_array = np.array(img, dtype=np.float32)
        
        # استخدام معيار التطبيع الخاص بـ MobileNet (حل مشكلة الـ 100% الثابتة)
        img_array = (img_array / 127.5) - 1.0 
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # الحصول على أعلى نتيجة
        result_idx = np.argmax(output_data)
        prediction_name = labels[result_idx]
        
        st.write("### 🔍 نتائج التحليل المخبري الرقمي:")
        
        # التفريق بين السرطان وغيره
        if prediction_name in malignant_types:
            st.error(f"⚠️ تنبيه: تم رصد مؤشرات لنوع من الأورام ({prediction_name})")
            st.subheader("التصنيف: خبيث أو يحتاج مراجعة فورية")
        else:
            st.success(f"✅ التشخيص المتوقع: {prediction_name}")
            st.subheader("التصنيف: حميد (ليس سرطان)")
            st.write(f"هذه الحالة تندرج تحت الأمراض الجلدية غير السرطانية التي تدرب عليها النظام.")

# 5. التنبيه الطبي وإخلاء المسؤولية كما طلبت
st.write("---")
st.warning("""
**⚠️ إخلاء مسؤولية طبي هام:**
* هذا النظام يعتمد بالكامل على تقنيات الذكاء الاصطناعي (AI) للأغراض البحثية والتعليمية فقط.
* هذا التطبيق **ليس تشخيصاً طبياً حقيقياً أو واقعياً** ولا يغني عن استشارة الطبيب المختص.
* التشخيص النهائي يجب أن يتم من خلال الفحص السريري والمخبري داخل العيادات الطبية المعتمدة.
""")
st.caption("Graduation Project - Skin Safety System 2026")
