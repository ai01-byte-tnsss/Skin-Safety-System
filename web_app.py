import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ==========================================
# 1. إعدادات الصفحة والتصميم
# ==========================================
st.set_page_config(page_title="Skin Safety System Pro", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .report-card { padding: 25px; border-radius: 15px; background-color: white; border-left: 6px solid #1E88E5; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); margin-top: 20px; }
    .title-text { text-align: center; color: #0D47A1; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. تشغيل النظام الرئيسي
# ==========================================
st.markdown("<h1 class='title-text'>🛡️ منصة التشخيص الذكي للأمراض الجلدية</h1>", unsafe_allow_html=True)
st.divider()

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    target_dtype = input_details[0]['dtype'] # لاكتشاف FLOAT16 تلقائياً
    
    uploaded_file = st.file_uploader("📥 ارفع صورة الآفة", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة المرفوعة", use_container_width=True)
        
        if st.button("🔬 بدء التحليل"):
            with st.spinner('جاري معالجة البيانات...'):
                # 1. تجهيز الصورة
                img = image.convert('RGB').resize((224, 224))
                img_array = np.array(img).astype(np.float32) / 255.0
                img_array = img_array.astype(target_dtype) # التحويل للدقة المطلوبة FLOAT16
                img_array = np.expand_dims(img_array, axis=0)

                # 2. التنبؤ
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                
                # 3. عرض النتيجة (بدون نسبة ثقة وبدون أخطاء في العدد)
                st.markdown("<div class='report-card'>", unsafe_allow_html=True)
                st.subheader("📋 النتيجة النهائية:")
                
                max_idx = np.argmax(output_data)
                
                # قائمة الأسماء (تأكد من ترتيبها، بما أن الصورة أظهرت رقم 23، يجب أن تحتوي القائمة على 24 اسماً على الأقل)
                # سأضع أسماء عامة الآن لضمان عمل الكود
                class_names = [f"الحالة رقم {i}" for i in range(len(output_data))]
                
                # تحديث الأسماء الشائعة لأول 3 (كمثال)
                if len(class_names) > 0: class_names[0] = "سليم (Normal)"
                if len(class_names) > 1: class_names[1] = "ورم حميد (Benign)"
                if len(class_names) > 2: class_names[2] = "ورم خبيث (Malignant)"

                st.success(f"🔍 التشخيص المتوقع: **{class_names[max_idx]}**")
                
                st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"⚠️ حدث خطأ تقني: {e}")


