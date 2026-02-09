import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# تحميل النموذج المصغر (TFLite)
interpreter = tf.lite.Interpreter(model_path="skin_expert_lite.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# قائمة التصنيفات المحدثة (تأكد من مطابقتها لترتيب التدريب لديك)
class_names = ['Acne', 'BCC', 'Melanoma', 'Nevus', 'BKL', 'DF', 'VASC', 'AKIEC']

st.title("🩺 فحص الجلد الذكي (النسخة الخفيفة)")

uploaded_file = st.file_uploader("ارفع صورة الفحص...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='الصورة المختارة', use_column_width=True)
    
    # معالجة الصورة
    img = img.resize((150, 150))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # التنبؤ باستخدام TFLite
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    result = class_names[np.argmax(predictions)]
    st.success(f"التشخيص المتوقع: {result}")

