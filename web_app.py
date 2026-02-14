import streamlit as st
import tflite_runtime.interpreter as tflite # استخدام المكتبة الخفيفة لضمان التشغيل
from PIL import Image
import numpy as np
import os

# 1. إعدادات واجهة المستخدم
st.set_page_config(page_title="Skin Safety Expert", page_icon="🩺", layout="centered")

st.title("🩺 Skin Disease Expert System")
st.markdown(f"### **الدقة الإجمالية للنظام: 53.57%**") # عرض الدقة كما في التقارير
st.write("---")

# 2. تحميل النموذج الجديد باستخدام tflite_runtime
@st.cache_resource
def load_model():
    model_path = "skin_expert_refined.tflite" # الملف الذي استخرجناه بنجاح
    if os.path.exists(model_path):
        # تحميل المفسر (Interpreter)
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

# 3. قائمة التشخيصات (24 صنفاً)
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

# تحديد الأصناف الخبيثة للمناقشة
malignant_types = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

# 4. رفع ومعالجة الصور
uploaded_file = st.file_uploader("ارفع صورة الجلد للفحص...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    if st.button('بدء التشخيص التحليلي'):
        # إعداد المدخلات والمخرجات
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # معالجة الصورة (150x150) كما في مرحلة التدريب
        img = image.resize((150, 150))
        img_array = np.array(img, dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0 # التطبيع القياسي
        img_array = np.expand_dims(img_array, axis=0)
        
        # تنفيذ التنبؤ
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # الحصول على النتيجة
        result_idx = np.argmax(output_data[0])
        prediction_name = labels[result_idx]
        
        st.write("### 🔍 نتائج التحليل:")
        
        if prediction_name in malignant_types:
            st.error(f"⚠️ تنبيه: {prediction_name}")
            st.subheader("التصنيف: خبيث (يستوجب مراجعة طبيب)")
        else:
            st.success(f"✅ التشخيص المتوقع: {prediction_name}")
            st.subheader("التصنيف: حميد")

# 5. إخلاء المسؤولية الطبي
st.write("---")
st.warning("""
**⚠️ ملاحظة هامة:**
هذا النظام للأغراض التعليمية والبحثية فقط، ولا يعتبر بديلاً عن الفحص الطبي المتخصص.
""")

