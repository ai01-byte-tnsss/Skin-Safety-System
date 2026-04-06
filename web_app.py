import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. إعدادات الواجهة والجماليات ---
st.set_page_config(page_title="Skin Safety System", layout="centered")

st.markdown("""
<style>
    .report-card { padding: 30px; border-radius: 20px; text-align: center; margin-top: 25px; border: 3px solid; }
    .result-title { font-size: 30px; font-weight: bold; margin-bottom: 10px; }
    .result-desc { font-size: 18px; font-weight: 500; }
    .advice-box { background-color: #fcfcfc; padding: 20px; border-radius: 12px; margin-top: 20px; border-right: 6px solid #455a64; }
    .quality-alert { background-color: #fffbe6; border: 1px solid #ffe58f; padding: 15px; border-radius: 8px; color: #856404; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 2. تحميل المحرك ومعالجة الأخطاء التقنية ---
@st.cache_resource
def load_expert_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"⚠️ فشل تحميل النموذج: {e}")
        return None

interpreter = load_expert_model()

if interpreter:
    # الحصول على متطلبات المدخلات (هذا يحل مشكلة ValueError)
    input_details = interpreter.get_input_details()
    target_dtype = input_details[0]['dtype']
    input_shape = input_details[0]['shape'][1:3] # غالباً (224, 224)

    st.markdown("<h1 style='text-align: center; color: #0d47a1;'>🛡️ نظام الكشف عن سلامة الجلد</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="quality-alert">
            ⚠️ <b>تنبيه الجودة:</b> جودة الإضاءة ووضوح الصورة يؤثران مباشرة على دقة التشخيص.
        </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 فحص حالة جديدة"):
        st.rerun()

    uploaded_file = st.file_uploader("📥 ارفع صورة الآفة الجلدية هنا", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        if st.button("🚀 بدء التحليل الدقيق"):
            with st.spinner("جاري فحص الأنماط الجلدية..."):
                try:
                    # --- [معالجة الصورة المتقدمة لضمان الدقة] ---
                    img = image.convert("RGB").resize(input_shape)
                    img_array = np.array(img)
                    
                    # مطابقة نوع البيانات المطلوب (حل مشكلة الصورة الأخيرة)
                    if target_dtype == np.float32:
                        img_array = img_array.astype(np.float32) / 255.0
                    else:
                        img_array = img_array.astype(target_dtype)

                    img_array = np.expand_dims(img_array, axis=0)

                    # --- تشغيل النموذج ---
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
                    
                    # استخدام Softmax لضمان توزيع النسب بشكل صحيح
                    probs = tf.nn.softmax(output_data).numpy()

                    # --- [نظام التصنيف المنفصل والأولويات] ---
                    malignant_ids = [1, 4, 17] # تأكد من مطابقة هذه الأرقام لنموذجك
                    benign_ids = [2, 5, 23]
                    
                    prob_malignant = sum([probs[i] for i in malignant_ids if i < len(probs)])
                    prob_benign = sum([probs[i] for i in benign_ids if i < len(probs)])

                    # --- منطق اتخاذ القرار النهائي (بدون إظهار نسب) ---
                    if prob_malignant >= 0.25: # عتبة الأمان للخبيث
                        res_msg, sub_msg = "🚨 اشتباه ورم خبيث", "تم رصد أنماط تتطلب تقييماً طبياً فورياً."
                        bg_color, txt_color = "#fff1f0", "#cf1322"
                        advice = "التشخيص المبكر ضروري جداً؛ يرجى مراجعة طبيب الجلدية المختص."
                    
                    elif prob_benign >= 0.60: # عتبة الطمأنينة للحميد
                        res_msg, sub_msg = "🔍 ورم جلدي حميد", "الآفة المكتشفة تبدو من النوع السليم وغير المقلق."
                        bg_color, txt_color = "#f6ffed", "#389e0d"
                        advice = "لا توجد علامات خطر حالية، لكن يفضل المراقبة الدورية لأي تغير."
                    
                    else: # الحالات الجلدية العامة
                        res_msg, sub_msg = "🩺 حالة جلدية عامة", "التحليل يرجح وجود نمط جلدي غير ورمي (التهاب أو حساسية)."
                        bg_color, txt_color = "#e6f7ff", "#096dd9"
                        advice = "هذه الأعراض شائعة؛ يمكن استشارة الطبيب العام لوصف العلاج المناسب."

                    # عرض النتيجة بأسلوب التقرير الاحترافي
                    st.markdown(f"""
                        <div class="report-card" style="background-color: {bg_color}; border-color: {txt_color}; color: {txt_color};">
                            <p class="result-title">{res_msg}</p>
                            <p class="result-desc">{sub_msg}</p>
                        </div>
                        <div class="advice-box"><strong>💡 توصية النظام:</strong> {advice}</div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"حدث خطأ أثناء المعالجة: {e}")

    # قسم الأسئلة الشائعة
    st.write("---")
    with st.expander("❓ الأسئلة الشائعة حول الفحص"):
        st.write("هذا النظام هو وسيلة فحص أولية مدعومة بالذكاء الاصطناعي لمساعدتك في التقييم، ولا يغني عن الفحص السريري للطبيب المختص.")

st.markdown("<br><hr><p style='text-align: center; color: grey;'>نظام تقييم سلامة الجلد الذكي © 2026</p>", unsafe_allow_html=True)
