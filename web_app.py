import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# إعداد واجهة المستخدم
st.set_page_config(page_title="Skin AI Expert", layout="centered")
st.title("🩺 نظام التشخيص الذكي (النسخة الاحترافية)")

# تحميل نموذج TFLite
@st.cache_resource
def load_lite_model():
    interpreter = tf.lite.Interpreter(model_path="skin_expert_lite.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_lite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# قائمة الفئات بالترتيب الصحيح (تأكد من مطابقتها لنتائج Colab)
class_names = [
    'Acne (حب شباب)', 
    'AKIEC (سرطان الطبقة السطحية)', 
    'BCC (سرطان خلايا قاعدية)', 
    'BKL (آفات حميدة)', 
    'DF (ألياف جلدية)', 
    'Melanoma (ميلانوما - خبيث)', 
    'Nevus (شامة طبيعية)', 
    'VASC (آفات وعائية)'
]

uploaded_file = st.file_uploader("ارفع صورة الآفة الجلدية...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='الصورة المرفوعة', use_column_width=True)
    
    # معالجة الصورة (نفس مقاس التدريب 150x150)
    img_resized = img.resize((150, 150))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # تنفيذ التنبؤ
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # --- الخطوة السحرية لحل مشكلة "كل شيء حب شباب" ---
    # تحويل المخرجات إلى احتمالات حقيقية مئوية
    probabilities = tf.nn.softmax(output_data).numpy()
    
    # اختيار الفئة الأعلى يقيناً
    top_index = np.argmax(probabilities)
    result = class_names[top_index]
    confidence = probabilities[top_index] * 100

    # عرض النتائج
    st.divider()
    st.subheader(f"التشخيص المتوقع: {result}")
    st.write(f"**درجة اليقين:** {confidence:.2f}%")
    st.progress(int(confidence))

    # عرض تحليل جميع الفئات للتأكد من عدم وجود انحياز
    with st.expander("إظهار تفاصيل الدقة لكل الأمراض"):
        for i, name in enumerate(class_names):
            st.write(f"{name}: {probabilities[i]*100:.2f}%")


