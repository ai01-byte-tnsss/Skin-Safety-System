import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الواجهة
st.set_page_config(page_title="Skin Safety Expert", page_icon="🩺", layout="centered")

st.title("🩺 Skin Disease Expert System")
st.markdown(f"### **الدقة الإجمالية للنظام: 53.57%**") #
st.write("---")

# 2. تحميل النموذج
@st.cache_resource
def load_model():
    model_path = "skin_expert_refined.tflite" #
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

# قائمة الأصناف الـ 24 المعتمدة
labels = ['Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
          'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
          'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
          'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
          'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
          'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum']

malignant_types = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

# 3. واجهة رفع الصور
uploaded_file = st.file_uploader("ارفع صورة الجلد لفحصها...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    if st.button('بدء التشخيص التحليلي'):
        # --- الجزء الخاص بتعديل حجم المصفوفة بشكل عام ---
        # الحصول على تفاصيل المدخلات من النموذج مباشرة
        input_details = interpreter.get_input_details()
        
        # استخراج الأبعاد المطلوبة ديناميكياً من النموذج
        # المصفوفة عادة تكون بتنسيق [Batch, Height, Width, Channels]
        target_height = input_details[0]['shape'][1]
        target_width = input_details[0]['shape'][2]
        
        # تعديل حجم الصورة المرفوعة لتطابق أبعاد مصفوفة النموذج
        img = image.resize((target_width, target_height))
        
        # تحويل الصورة إلى مصفوفة Numpy مع تحديد نوع البيانات float32
        img_array = np.array(img, dtype=np.float32)
        
        # تطبيق عملية التطبيع (Normalization) لتناسب لغة النموذج
        img_array = (img_array / 127.5) - 1.0 
        
        # إضافة بعد الدفعة (Expand Dimensions) لتصبح المصفوفة رباعية الأبعاد [1, H, W, 3]
        img_array = np.expand_dims(img_array, axis=0)
        # ------------------------------------------------
        
        # تنفيذ التنبؤ
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        
        # عرض النتيجة
        result_idx = np.argmax(output_data[0])
        prediction_name = labels[result_idx]
        
        st.write("### 🔍 نتيجة التحليل:")
        if prediction_name in malignant_types:
            st.error(f"⚠️ النوع: {prediction_name} (خبيث)")
        else:
            st.success(f"✅ النوع: {prediction_name} (حميد)")

# 4. الملاحظة القانونية
st.write("---")
st.warning("""
**⚠️ ملاحظة إخلاء مسؤولية:**
هذا النظام يعتمد على الذكاء الاصطناعي للأغراض التعليمية والبحثية فقط، ولا يعتبر تشخيصاً طبياً نهائياً.
""")
