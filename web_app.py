import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الصفحة
st.set_page_config(page_title="Skin Safety Expert", page_icon="🩺", layout="centered")

st.title("🩺 Skin Disease Expert System")
st.markdown(f"### **الدقة الإجمالية للنظام: 53.57%**") #
st.write("---")

# 2. تحميل النموذج
@st.cache_resource
def load_model():
    model_path = "skin_expert_refined.tflite"
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

# قائمة الأصناف الـ 24 المعتمدة
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

malignant_types = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

# 3. واجهة المستخدم لرفع الصور
uploaded_file = st.file_uploader("ارفع صورة الجلد لفحصها...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المرفوعة للفحص', use_container_width=True)
    
    if st.button('بدء التشخيص التحليلي'):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # قراءة الأبعاد والأنواع المطلوبة ديناميكياً لتجنب الأخطاء التقنية
        target_height = input_details[0]['shape'][1]
        target_width = input_details[0]['shape'][2]
        required_dtype = input_details[0]['dtype'] # حل مشكلة FLOAT16
        
        # معالجة الصورة
        img = image.resize((target_width, target_height))
        img_array = np.array(img, dtype=np.float32) # نبدأ بـ float32 للحسابات
        
        # --- تعديل التطبيع لتحسين دقة التشخيص (0 إلى 1) ---
        img_array = img_array / 255.0 
        
        # تحويل المصفوفة للنوع النهائي المطلوب (FLOAT16) وإضافة البعد الرابع
        img_array = np.expand_dims(img_array, axis=0).astype(required_dtype)
        
        try:
            # تنفيذ التنبؤ
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # الحصول على النتيجة الأعلى
            result_idx = np.argmax(output_data[0])
            prediction_name = labels[result_idx]
            
            st.write(f"### 🔍 نتيجة التحليل: {prediction_name}")
            
            if prediction_name in malignant_types:
                st.error("التصنيف الطبي للمرض: خبيث (يستوجب مراجعة طبيب)")
            else:
                st.success("التصنيف الطبي للمرض: حميد")
                
        except Exception as e:
            st.error(f"حدث خطأ في معالجة البيانات: {e}")

# 4. الملاحظة القانونية والطبية
st.write("---")
st.warning("""
**⚠️ ملاحظة إخلاء مسؤولية:**
هذا النظام يعتمد على الذكاء الاصطناعي للأغراض التعليمية والبحثية فقط، ولا يعتبر تشخيصاً طبياً حقيقياً.
يجب مراجعة الطبيب المختص للحصول على تشخيص دقيق.
""")

