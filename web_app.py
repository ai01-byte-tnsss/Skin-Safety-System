import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. إعدادات الواجهة الاحترافية ---
st.set_page_config(page_title="Skin Safety System", layout="centered")

st.markdown("""
<style>
    .report-card { padding: 35px; border-radius: 25px; text-align: center; margin: 20px 0; border: 4px solid; box-shadow: 0px 10px 30px rgba(0,0,0,0.1); }
    .result-title { font-size: 32px; font-weight: bold; margin-bottom: 10px; }
    .result-desc { font-size: 19px; font-weight: 500; line-height: 1.6; }
    .advice-box { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 1px solid #eee; border-right: 10px solid #455a64; margin-top: 20px; }
    .preview-box { border: 2px dashed #0d47a1; padding: 10px; border-radius: 15px; background-color: #f0f4f8; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 2. تحميل النموذج ---
@st.cache_resource
def load_expert_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except: return None

interpreter = load_expert_model()

# تعريف مجموعات التصنيف (مباشرة دون عتبات ثابتة)
class_groups = {
    "Malignant": [1, 4, 17], 
    "Benign": [2, 5, 23]
}

if interpreter:
    input_details = interpreter.get_input_details()
    target_dtype = input_details[0]['dtype']
    input_shape = input_details[0]['shape'][1:3]

    st.markdown("<h1 style='text-align: center; color: #0d47a1;'>🛡️ الكشف عن سلامة الجلد لمرض سرطان</h1>", unsafe_allow_html=True)
    
    st.write("---")

    # اختيار مصدر الصورة
    source_option = st.radio("اختر طريقة إدخال الصورة:", ("رفع ملف من الجهاز", "التقاط صورة فورية بالكاميرا"))
    
    if source_option == "رفع ملف من الجهاز":
        uploaded_file = st.file_uploader("📥 اختر صورة واضحة", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("📸 وجه الكاميرا نحو المنطقة المصابة")

    # --- ظهور الصورة قبل الفحص ---
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        st.markdown("<p style='font-weight: bold; color: #0d47a1;'>🔍 معاينة الصورة قبل بدء الفحص:</p>", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="preview-box">', unsafe_allow_html=True)
            st.image(image, use_container_width=True, caption="الصورة الحالية جاهزة للتحليل")
            st.markdown('</div>', unsafe_allow_html=True)

        # زر بدء الفحص يظهر فقط بعد وجود الصورة
        if st.button("🚀 بدء تحليل الأنماط الجلدية الآن"):
            with st.spinner("جاري فحص خصائص الأنسجة بصرياً..."):
                try:
                    # المعالجة المسبقة
                    img = image.convert("RGB").resize(input_shape)
                    img_array = np.array(img)
                    
                    if target_dtype == np.float32:
                        img_array = img_array.astype(np.float32) / 255.0
                    else:
                        img_array = img_array.astype(target_dtype)
                    
                    img_array = np.expand_dims(img_array, axis=0)

                    # تنفيذ الفحص
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
                    
                    # اختيار أعلى فئة ثقة مباشرة (Argmax) - هذا يضمن التصنيف الصحيح
                    predicted_index = np.argmax(output_data)

                    # منطق التصنيف
                    if predicted_index in class_groups["Malignant"]:
                        res_msg, sub_msg = "🚨 النتيجة: اشتباه ورم خبيث", "تم رصد علامات نمو غير طبيعي تتطلب تقييماً طبياً فورياً."
                        bg_c, txt_c = "#fff1f0", "#cf1322"
                        advice = "يُنصح بشدة بالتوجه للطبيب المختص لإجراء فحص سريري دقيق وبحث الخطوات التالية."
                    
                    elif predicted_index in class_groups["Benign"]:
                        res_msg, sub_msg = "🔍 النتيجة: ورم جلدي حميد", "التحليل الرقمي يشير إلى أن الآفة من النوع السليم وغير المقلق."
                        bg_c, txt_c = "#f6ffed", "#389e0d"
                        advice = "الحالة لا تستدعي القلق حالياً، ولكن يُفضل مراقبتها بشكل دوري لأي تغيرات."
                    
                    else:
                        res_msg, sub_msg = "🩺 النتيجة: حالة جلدية عامة", "التحليل يرجح وجود نمط جلدي طبيعي أو غير ورمي."
                        bg_c, txt_c = "#e6f7ff", "#096dd9"
                        advice = "لا توجد مؤشرات سرطانية مقلقة؛ استشر الطبيب العام للمتابعة الروتينية."

                    # عرض تقرير النتيجة
                    st.markdown(f"""
                        <div class="report-card" style="background-color: {bg_c}; border-color: {txt_c}; color: {txt_c};">
                            <p class="result-title">{res_msg}</p>
                            <p class="result-desc">{sub_msg}</p>
                        </div>
                        <div class="advice-box">
                            <p style="font-size: 20px; font-weight: bold; color: #263238; margin-bottom: 5px;">💡 التوصية الطبية:</p>
                            <p style="font-size: 18px; color: #455a64;">{advice}</p>
                        </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error("نعتذر، حدث خطأ أثناء التحليل. يرجى محاولة رفع صورة أخرى.")

st.markdown("<br><hr><p style='text-align: center; color: #9e9e9e;'>نظام تقييم سلامة الجلد الذكي المعتمد © 2026</p>", unsafe_allow_html=True)
