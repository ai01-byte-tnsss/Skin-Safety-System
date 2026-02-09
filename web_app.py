import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# إعدادات الصفحة
st.set_page_config(page_title="Skin AI Expert", layout="centered")
st.title("🩺 نظام التشخيص الذكي المطور")
st.write("هذا النظام مدرب للتعرف على حب الشباب وسرطانات الجلد الشائعة.")

# تحميل نموذج TFLite المصغر
@st.cache_resource
def load_lite_model():
    # البحث عن الملف في المسار الحالي
    model_path = "skin_expert_lite.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_lite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"خطأ في تحميل النموذج: {e}")
    st.stop()

# 2. ترتيب الفئات (هذا الترتيب الأبجدي هو الافتراضي لـ ImageDataGenerator)
# تم ترتيبها لتطابق مخرجات التدريب (0 لـ Acne، و1 لـ AKIEC، وهكذا)
class_names = [
    'Acne (حب شباب)',                         # Index 0
    'AKIEC (سرطان الطبقة السطحية)',            # Index 1
    'BCC (سرطان خلايا قاعدية)',                # Index 2
    'BKL (آفات حميدة تشبه الثآليل)',           # Index 3
    'DF (ألياف جلدية حميدة)',                  # Index 4
    'Melanoma (ميلانوما - خبيث)',             # Index 5
    'Nevus (شامة طبيعية)',                    # Index 6
    'VASC (آفات وعائية)'                      # Index 7
]

uploaded_file = st.file_uploader("ارفع صورة الآفة الجلدية للفحص...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='الصورة المختارة', use_column_width=True)
    
    with st.spinner('جاري تحليل الأنماط البصرية وحساب الاحتمالات...'):
        # معالجة الصورة
        img_resized = img.resize((150, 150))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # التنبؤ باستخدام TFLite
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        # الحصول على النتائج الخام (Raw logits)
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # تحويل النتائج إلى احتمالات مئوية (Softmax)
        exp_data = np.exp(output_data - np.max(output_data))
        probabilities = exp_data / exp_data.sum()
        
        # تحديد النتيجة الأعلى
        top_index = np.argmax(probabilities)
        result_label = class_names[top_index]
        confidence_score = probabilities[top_index] * 100

        # 3. عرض النتائج النهائية
        st.divider()
        st.subheader(f"التشخيص المتوقع: {result_label}")
        
        # تلوين شريط اليقين (أحمر للحالات الخطرة، أخضر للحميدة)
        if "Melanoma" in result_label or "BCC" in result_label:
            st.warning(f"نسبة اليقين: {confidence_score:.2f}%")
            st.progress(int(confidence_score))
            st.error("⚠️ ملاحظة: النموذج يشتبه في حالة تستدعي فحص طبي عاجل.")
        else:
            st.success(f"نسبة اليقين: {confidence_score:.2f}%")
            st.progress(int(confidence_score))

        # عرض تحليل تفصيلي لجميع الفئات
        with st.expander("إظهار الدقة الكاملة لجميع الفئات"):
            for name, prob in zip(class_names, probabilities):
                st.write(f"**{name}**: {prob*100:.2f}%")

st.info("تنبيه: هذا النموذج للأغراض الأكاديمية فقط ولا يغني عن زيارة الطبيب المختص.")

