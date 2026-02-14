import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# إعدادات الواجهة
st.set_page_config(page_title="Skin Cancer Expert", page_icon="🩺")
st.title("🩺 نظام تشخيص سرطان الجلد الخبير")
st.write("---")

@st.cache_resource
def load_model():
    model_path = "skin_expert_refined.tflite"
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

labels = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease', 
    'Cellulitis Impetigo', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
    'Herpes HPV', 'Light Diseases', 'Lupus and Connective Tissue', 'Melanoma', 
    'Nail Fungus', 'Nevi and Moles', 'Poison Ivy', 'Psoriasis and Lichen Planus', 
    'Scabies and Bites', 'Seborrheic Keratoses', 'Systemic Disease', 'Tinea Ringworm', 
    'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts and Molluscum'
]

cancer_labels = ['Melanoma', 'Actinic Keratosis', 'Vascular Tumors']

uploaded_file = st.file_uploader("ارفع صورة الفحص الجلدي...", type=["jpg", "png", "jpeg"])

if uploaded_file and interpreter:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="الصورة تحت المعالجة الرقمية", use_container_width=True)
    
    if st.button('إجراء التشخيص النهائي'):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
        dtype = input_details[0]['dtype']

        # تصغير الصورة
        img = image.resize((w, h), Image.Resampling.BILINEAR)
        img_array = np.array(img).astype(np.float32)

        # التطبيع بأسلوب MobileNet
        img_array = (img_array / 127.5) - 1.0

        img_array = np.expand_dims(img_array, axis=0).astype(dtype)

        try:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # 🔥 الإصلاح هنا: تحويل القيم إلى احتمالات Softmax
            probs = tf.nn.softmax(output_data[0]).numpy()

            top_idx = np.argmax(probs)
            general_prediction = labels[top_idx]

            st.write("---")
            st.write("### 🔍 نتيجة الفحص النهائية:")

            if general_prediction in cancer_labels:
                st.error(f"⚠️ التشخيص: {general_prediction}")
                st.subheader("🔴 التصنيف النهائي: [خبيث - سرطان]")
            else:
                st.success(f"✅ التشخيص: {general_prediction}")
                st.subheader("🟢 التصنيف النهائي: [حميد - ليس سرطان]")

        except Exception as e:
            st.error(f"خطأ في التنفيذ: {e}")

st.write("---")
st.warning("⚠️ ملاحظة: هذا التشخيص يعتمد على الذكاء الاصطناعي للأغراض التعليمية فقط.")
