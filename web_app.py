import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. إعدادات الواجهة ---
st.set_page_config(page_title="Skin Health Expert", layout="centered")

st.markdown("""
<style>
.report-card { padding: 30px; border-radius: 20px; text-align: center; margin-top: 25px; border: 3px solid; }
.result-title { font-size: 35px; font-weight: bold; margin-bottom: 10px; }
.result-desc { font-size: 20px; font-weight: 500; }
.advice-box { background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 20px; border-right: 5px solid #6c757d; }
</style>
""", unsafe_allow_html=True)

# --- 2. تحميل النموذج ---
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except:
        st.error("⚠️ لم يتم العثور على ملف النموذج.")
        return None

interpreter = load_model()

if interpreter:

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    st.markdown("<h1 style='text-align: center; color: #1a237e;'>🛡️ الفحص الذكي للآفات الجلدية</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: center;'>نظام يعتمد على الذكاء الاصطناعي لتحليل الحالة الجلدية</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📥 قم برفع صورة الحالة الجلدية", type=["jpg", "jpeg", "png"])

    if uploaded_file:

        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        if st.button("🚀 تحليل الحالة الآن"):

            with st.spinner("جاري التحليل..."):

                # --- معالجة الصورة ---
                img = image.convert("RGB").resize((224, 224))
                img_array = np.array(img).astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # --- تشغيل النموذج ---
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]['index'])[0]

                # بعض نماذج TFLite تعطي softmax مسبقاً
                probs = output_data
                max_idx = np.argmax(probs)

                # --- تعريف العائلات ---
                malignant_ids = [1, 4, 17]
                benign_ids = [2, 5, 23]

                # --- تحديد النتيجة ---
                if max_idx in malignant_ids:

                    res_msg = "النتيجة: اشتباه ورم خبيث"
                    sub_msg = "الحالة قد تكون سرطانية وتتطلب فحصاً طبياً عاجلاً"
                    bg_color = "#ffebee"
                    txt_color = "#b71c1c"
                    advice = "ينصح بمراجعة طبيب الجلدية لإجراء فحص سريري وربما خزعة."

                elif max_idx in benign_ids:

                    res_msg = "النتيجة: ورم جلدي حميد"
                    sub_msg = "الآفة الجلدية تبدو من النوع السليم وغير الخطير"
                    bg_color = "#e8f5e9"
                    txt_color = "#1b5e20"
                    advice = "لا توجد علامات خطر واضحة، لكن يفضل مراقبة أي تغير بالحجم أو اللون."

                else:

                    res_msg = "النتيجة: غير ذلك"
                    sub_msg = "تم تصنيف الحالة كمرض جلدي غير سرطاني"
                    bg_color = "#e3f2fd"
                    txt_color = "#0d47a1"
                    advice = "قد تكون حالة جلدية شائعة مثل الالتهاب أو الحساسية."

                # --- عرض التقرير ---
                st.markdown(f"""
                <div class="report-card" style="background-color:{bg_color}; border-color:{txt_color}; color:{txt_color};">
                    <p class="result-title">{res_msg}</p>
                    <p class="result-desc">{sub_msg}</p>
                </div>

                <div class="advice-box">
                <strong>💡 نصيحة النظام:</strong> {advice}
                </div>
                """, unsafe_allow_html=True)

st.markdown("<br><hr><p style='text-align:center;color:grey;'>نظام دعم قرار طبي للأغراض التعليمية والبحثية فقط</p>", unsafe_allow_html=True)
