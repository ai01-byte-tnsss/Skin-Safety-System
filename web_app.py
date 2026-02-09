import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Skin AI Expert", layout="centered")
st.title("🩺 نظام التشخيص الذكي المطور")

@st.cache_resource
def load_lite_model():
    # تأكد أن الملف مرفوع في GitHub بنفس هذا الاسم تماماً
    interpreter = tf.lite.Interpreter(model_path="skin_expert_lite.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_lite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# القائمة المحدثة بالترتيب الصحيح
# ملاحظة: إذا كان النموذج قديماً، فقد يتجاهل الفئات الإضافية تلقائياً الآن
class_names = ['Acne (حب شباب)', 'AKIEC', 'BCC', 'BKL', 'DF', 'Melanoma', 'Nevus', 'VASC']

uploaded_file = st.file_uploader("ارفع صورة الفحص...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='الصورة المختارة', use_column_width=True)
    
    # معالجة الصورة
    img_resized = img.resize((150, 150))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # التنبؤ
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # تحويل النتائج إلى احتمالات (Softmax)
    probabilities = tf.nn.softmax(output_data).numpy()
    
    # تصحيح الخطأ: التأكد من أن القائمة والنتائج متساويتان في العدد
    num_results = len(probabilities)
    final_labels = class_names[:num_results] 

    top_index = np.argmax(probabilities)
    result = final_labels[top_index]
    confidence = probabilities[top_index] * 100

    st.divider()
    st.subheader(f"التشخيص المتوقع: {result}")
    st.write(f"**درجة اليقين:** {confidence:.2f}%")
    st.progress(int(confidence))

    with st.expander("إظهار الدقة الكاملة لجميع الفئات"):
        for i in range(num_results):
            st.write(f"{final_labels[i]}: {probabilities[i]*100:.2f}%")


