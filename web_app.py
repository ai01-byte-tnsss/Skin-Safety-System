import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# إعدادات واجهة التطبيق
st.set_page_config(page_title="Skin Safety AI", layout="centered")
st.title("🩺 نظام الكشف عن سلامة الجلد")
st.write("تحليل ذكي للكشف عن آفات الجلد وتحديد ما إذا كانت حميدة أم خبيثة.")

@st.cache_resource
def load_lite_model():
    # تأكد من رفع ملف skin_expert_lite.tflite على GitHub
    interpreter = tf.lite.Interpreter(model_path="skin_expert_lite.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_lite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# قائمة الأصناف الكاملة بالترتيب الصحيح
class_names = [
    'Acne (حب شباب)', 
    'AKIEC (آفات ما قبل السرطان)', 
    'BCC (سرطان خلايا قاعدية - خبيث)', 
    'BKL (آفات حميدة)', 
    'DF (ألياف جلدية حميدة)', 
    'Melanoma (ميلانوما - خبيث جداً)', 
    'Nevus (شامة طبيعية)', 
    'VASC (آفات وعائية)'
]

uploaded_file = st.file_uploader("ارفع صورة الآفة الجلدية للفحص...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='الصورة المختارة للفحص', use_column_width=True)
    
    with st.spinner('جاري تحليل سلامة الجلد...'):
        # معالجة الصورة
        img_resized = img.resize((150, 150))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # التنبؤ
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # تحويل النتائج لنسب مئوية (Softmax)
        probabilities = tf.nn.softmax(output_data).numpy()
        top_index = np.argmax(probabilities)
        result = class_names[top_index]
        confidence = probabilities[top_index] * 100

        st.divider()

        # --- قسم الكشف عن سلامة الجلد (خبيث أم حميد) ---
        malignant_types = ['BCC', 'Melanoma', 'AKIEC']
        is_malignant = any(mtype in result for mtype in malignant_types)

        if is_malignant:
            st.error(f"⚠️ تحذير: تم اكتشاف اشتباه في حالة (خبيثة/خطيرة): {result}")
            st.info("نصيحة: يجب مراجعة طبيب جلدية فوراً لإجراء فحص سريري.")
        else:
            st.success(f"✅ الحالة المكتشفة تبدو (حميدة/غير سرطانية): {result}")
            st.info("نصيحة: إذا تغير شكل الآفة أو زاد حجمها، يرجى استشارة المختص.")

        st.write(f"**نسبة دقة التشخيص:** {confidence:.2f}%")
        st.progress(int(confidence))

        # عرض التحليل التفصيلي لجميع الأصناف
        with st.expander("عرض التقرير المفصل لجميع أصناف الجلد"):
            for i in range(len(probabilities)):
                st.write(f"{class_names[i]}: {probabilities[i]*100:.2f}%")

st.warning("إخلاء مسؤولية: هذا النظام هو مشروع ذكاء اصطناعي للأغراض التعليمية، ولا يعتبر بديلاً عن التشخيص الطبي المتخصص.")


