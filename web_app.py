import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. إعدادات الواجهة الاحترافية ---
st.set_page_config(page_title="Skin Health Detection System", layout="centered")

st.markdown("""
<style>
    .report-card { padding: 35px; border-radius: 25px; text-align: center; margin: 20px 0; border: 4px solid; box-shadow: 0px 10px 30px rgba(0,0,0,0.1); }
    .result-title { font-size: 32px; font-weight: bold; margin-bottom: 10px; }
    .result-desc { font-size: 19px; font-weight: 500; line-height: 1.6; }
    .advice-box { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 1px solid #eee; border-right: 10px solid #455a64; margin-top: 20px; }
    .quality-alert { background-color: #fffbe6; border: 1px solid #ffe58f; padding: 15px; border-radius: 10px; color: #856404; font-size: 14px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 2. تحميل النموذج والبيانات الوصفية ---
@st.cache_resource
def load_expert_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except: return None

interpreter = load_expert_model()

# تعريف التصنيفات (تأكد من ترتيبها كما في تدريب النموذج)
# هذا الجزء يجعل الكود يقرأ النتيجة مباشرة من "هوية" الصورة
class_map = {
    "Malignant": [1, 4, 17], # فئات السرطان
    "Benign": [2, 5, 23]     # فئات الأورام الحميدة
}

if interpreter:
    input_details = interpreter.get_input_details()
    target_dtype = input_details[0]['dtype']
    input_shape = input_details[0]['shape'][1:3]

    st.markdown("<h1 style='text-align: center; color: #0d47a1;'>🛡️ الكشف عن سلامة الجلد لمرض سرطان</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="quality-alert">
            💡 <b>ملاحظة للفحص الفوري:</b> عند التقاط صورة مباشرة، يرجى التأكد من ثبات اليد واستخدام فلاش الكاميرا إذا كانت الإضاءة ضعيفة لضمان دقة تحليل الأنسجة.
        </div>
    """, unsafe_allow_html=True)

    # خيارين: رفع صورة أو التقاط صورة فورية (تعديل احترافي للمريض)
    source_option = st.radio("اختر مصدر الصورة:", ("رفع صورة محملة", "التقاط صورة فورية بالكاميرا"))
    
    if source_option == "رفع صورة محملة":
        uploaded_file = st.file_uploader("📥 اختر الصورة من الجهاز", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("📸 التقط صورة الآفة الجلدية الآن")

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        if st.button("🚀 تحليل الصورة وقرار الفحص"):
            with st.spinner("جاري فحص الأنماط البصرية مباشرة..."):
                try:
                    # معالجة الصورة مهما كان مصدرها
                    img = image.convert("RGB").resize(input_shape)
                    img_array = np.array(img)
                    if target_dtype == np.float32:
                        img_array = img_array.astype(np.float32) / 255.0
                    else:
                        img_array = img_array.astype(target_dtype)
                    img_array = np.expand_dims(img_array, axis=0)

                    # تنفيذ الفحص المباشر
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
                    
                    # اختيار أعلى فئة ثقة مباشرة (Argmax)
                    # هنا الكود يقرأ الصورة ويقرر "ما هي الفئة الأكثر شبهاً بها"
                    predicted_index = np.argmax(output_data)

                    # منطق التصنيف التلقائي
                    if predicted_index in class_map["Malignant"]:
                        res_msg, sub_msg = "🚨 النتيجة: اشتباه ورم خبيث", "تم رصد علامات نمو غير طبيعي في أنسجة الجلد."
                        bg_c, txt_c = "#fff1f0", "#cf1322"
                        advice = "يجب التوجه للطبيب المختص لإجراء فحص سريري دقيق وبحث الخطوات القادمة."
                    
                    elif predicted_index in class_map["Benign"]:
                        res_msg, sub_msg = "🔍 النتيجة: ورم جلدي حميد", "التحليل الرقمي يشير إلى أن الآفة من النوع السليم."
                        bg_c, txt_c = "#f6ffed", "#389e0d"
                        advice = "الحالة لا تستدعي القلق حالياً، ولكن يُفضل مراقبتها بشكل دوري."
                    
                    else:
                        res_msg, sub_msg = "🩺 النتيجة: حالة جلدية عامة", "التحليل يرجح وجود نمط جلدي طبيعي أو غير ورمي."
                        bg_c, txt_c = "#e6f7ff", "#096dd9"
                        advice = "لا توجد مؤشرات قلق سرطانية؛ استشر طبيبك العام للمتابعة."

                    # عرض النتيجة
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
                    st.error("نعتذر، حدث خطأ في معالجة الصورة. يرجى التأكد من وضوحها والمحاولة مرة أخرى.")

st.markdown("<br><hr><p style='text-align: center; color: #9e9e9e;'>نظام تقييم سلامة الجلد الذكي المعتمد © 2026</p>", unsafe_allow_html=True)
