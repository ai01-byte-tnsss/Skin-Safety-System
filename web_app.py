import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الصفحة الاحترافية
st.set_page_config(page_title="Skin Safety Expert", page_icon="🩺", layout="centered")

st.title("🩺 Skin Disease Expert System")
st.subheader("نظام خبير متقدم لتشخيص وتصنيف الأمراض الجلدية")
st.markdown(f"### **الدقة الإجمالية للنظام: 53.57%**") # عرض الدقة الكلية للنظام
st.write("---")

# 2. تحميل نموذج TFLite المطور (النسخة الجديدة)
@st.cache_resource
def load_tflite_model():
    # تأكد من رفع هذا الملف الجديد إلى مستودع GitHub الخاص بك
    model_path = "skin_expert_refined.tflite" 
    if os.path.exists(model_path):
        # استخدام التوزيع العملياتي SELECT_TF_OPS المدعوم في النسخة الجديدة
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_tflite_model()

# 3. قائمة الأصناف الـ 24 المعتمدة في التدريب
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

# تصنيف الأنواع السرطانية أو شديدة الخطورة
malignant_types = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

# 4. واجهة التطبيق والمعالجة
uploaded_file = st.file_uploader("ارفع صورة الجلد لفحصها...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if interpreter is None:
        st.error("فشل تحميل ملف النموذج 'skin_expert_refined.tflite'. تأكد من وجوده في المسار الصحيح.")
    else:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='الصورة المرفوعة للفحص', use_container_width=True)
        
        if st.button('بدء التشخيص التحليلي'):
            # الحصول على تفاصيل المدخلات والمخرجات
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # معالجة الصورة بنفس إعدادات التدريب (150x150)
            img = image.resize((150, 150)) 
            img_array = np.array(img, dtype=np.float32)
            
            # تطبيق التطبيع المتوافق مع نموذج MobileNet (من -1 إلى 1)
            img_array = (img_array / 127.5) - 1.0 
            img_array = np.expand_dims(img_array, axis=0)
            
            # تنفيذ التنبؤ
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # الحصول على الفئة الأعلى احتمالاً
            result_idx = np.argmax(output_data[0])
            prediction_name = labels[result_idx]
            
            st.write("### 🔍 نتائج التحليل المخبري الرقمي:")
            
            # منطق التصنيف (خبيث/حميد)
            if prediction_name in malignant_types:
                st.error(f"⚠️ تنبيه طبي: تم رصد مؤشرات لنوع من الأورام ({prediction_name})")
                st.subheader("التصنيف الطبي للمرض: خبيث / يستوجب مراجعة فورية")
                st.info("النموذج صنف هذه الحالة ضمن الفئات السرطانية أو ما قبل السرطانية التي تتطلب فحصاً سريرياً عاجلاً.")
            else:
                st.success(f"✅ التشخيص المبدئي المتوقع: {prediction_name}")
                st.subheader("التصنيف الطبي للمرض: حميد (ليس سرطان)")
                st.write(f"هذه الحالة تندرج تحت فئة الأمراض الجلدية غير السرطانية وفقاً لقاعدة البيانات التي تدرب عليها النظام.")

# 5. الملاحظة القانونية والطبية (أسفل الصفحة) كما طلبت
st.write("---")
st.warning("""
**⚠️ ملاحظة هامة جداً (إخلاء مسؤولية):**
* هذا النظام يعتمد كلياً على تقنيات الذكاء الاصطناعي (AI) وتم تطويره لأغراض بحثية وتعليمية فقط.
* هذا البرنامج **ليس تشخيصاً طبياً حقيقياً أو واقعياً** ولا يمكن اعتباره بديلاً عن رأي الطبيب المختص.
* النتائج المقدمة هي مجرد احتمالات رقمية، ويجب دائماً مراجعة العيادات المختصة لإجراء الفحوصات اللازمة.
""")
st.caption("مشروع تخرج - نظام خبير لسلامة الجلد 2026")
