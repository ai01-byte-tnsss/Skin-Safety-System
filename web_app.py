import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# إعدادات الواجهة
st.set_page_config(page_title="Skin Safety Expert", page_icon="🩺", layout="centered")
st.title("🩺 Skin Disease Expert System")
st.markdown(f"### **الدقة الإجمالية للنظام: 53.57%**") #
st.write("---")

# تحميل النموذج
@st.cache_resource
def load_model():
    model_path = "skin_expert_refined.tflite"
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

labels = ['Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
          'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
          'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
          'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
          'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
          'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum']

malignant_types = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

uploaded_file = st.file_uploader("ارفع صورة الجلد لفحصها...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    if st.button('بدء التشخيص التحليلي'):
        # --- معالجة المصفوفة العامة لتجنب خطأ السطر 67 ---
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # قراءة الأبعاد المطلوبة ديناميكياً من النموذج
        target_height = input_details[0]['shape'][1]
        target_width = input_details[0]['shape'][2]
        
        # 1. تغيير حجم الصورة لتطابق مصفوفة النموذج
        img = image.resize((target_width, target_height))
        
        # 2. تحويل الصورة إلى مصفوفة Numpy مع التأكد من نوع float32 (أساسي لحل الخطأ)
        img_array = np.array(img, dtype=np.float32)
        
        # 3. التطبيع (Normalization) لمجال [-1, 1]
        img_array = (img_array / 127.5) - 1.0 
        
        # 4. إضافة بعد الدفعة لتصبح المصفوفة رباعية الأبعاد [1, H, W, 3]
        img_array = np.expand_dims(img_array, axis=0)
        
        try:
            # السطر 67: تغذية النموذج بالمصفوفة
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # استخراج النتيجة
            result_idx = np.argmax(output_data[0])
            prediction_name = labels[result_idx]
            
            st.write(f"### 🔍 نتيجة التحليل: {prediction_name}")
            if prediction_name in malignant_types:
                st.error("التصنيف الطبي: خبيث")
            else:
                st.success("التصنيف الطبي: حميد")
        except Exception as e:
            # عرض الخطأ بشكل مفصل في حال حدوثه
            st.error(f"حدث خطأ تقني في مصفوفة البيانات: {e}")

# الملاحظة الطبية كما في المشروع القديم
st.write("---")
st.warning("""
**⚠️ ملاحظة إخلاء مسؤولية:**
هذا النظام يعتمد على الذكاء الاصطناعي للأغراض التعليمية فقط، وليس تشخيصاً طبياً حقيقياً.
""")
