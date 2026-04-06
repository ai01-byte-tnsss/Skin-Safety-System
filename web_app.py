import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. إعدادات الواجهة ---
st.set_page_config(page_title="Skin Safety System 2026", layout="centered")

st.markdown("""
<style>
    .report-card { padding: 30px; border-radius: 20px; text-align: center; margin-top: 25px; border: 3px solid; }
    .result-title { font-size: 32px; font-weight: bold; margin-bottom: 12px; }
    .result-desc { font-size: 19px; font-weight: 500; }
    .advice-box { background-color: #fcfcfc; padding: 20px; border-radius: 12px; margin-top: 25px; border-right: 6px solid #455a64; }
    .quality-alert { background-color: #fffbe6; border: 1px solid #ffe58f; padding: 15px; border-radius: 8px; color: #856404; font-size: 14px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 2. تحميل النموذج ---
@st.cache_resource
def load_expert_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"⚠️ خطأ في تحميل النموذج: {e}")
        return None

interpreter = load_expert_model()

if interpreter:
    input_details = interpreter.get_input_details()
    target_dtype = input_details[0]['dtype']
    input_shape = input_details[0]['shape'][1:3]

    st.markdown("<h1 style='text-align: center; color: #0d47a1;'>🛡️ نظام الكشف عن سلامة الجلد</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="quality-alert">
            ⚠️ <b>تنبيه هام:</b> إذا كانت الصورة غير واضحة، قد يميل النظام لتصنيف الحالة كـ "عامة" حمايةً من التشخيص الخاطئ. يرجى رفع صورة مكبرة وواضحة.
        </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 فحص حالة جديدة"):
        st.rerun()

    uploaded_file = st.file_uploader("📥 ارفع صورة الآفة الجلدية"، type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        if st.button("🚀 بدء التحليل الفوري"):
            with st.spinner("جاري تدقيق الأنماط السرطانية..."):
                try:
                    # معالجة الصورة
                    img = image.convert("RGB").resize(input_shape)
                    img_array = np.array(img)
                    if target_dtype == np.float32:
                        img_array = img_array.astype(np.float32) / 255.0
                    else:
                        img_array = img_array.astype(target_dtype)
                    img_array = np.expand_dims(img_array, axis=0)

                    # تشغيل النموذج
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
                    
                    # تطبيق Softmax لضبط النسب
                    probs = tf.nn.softmax(output_data).numpy()

                    # --- [إعادة ضبط الفئات بناءً على الأهمية الطبية] ---
                    # ملاحظة: تأكد من أن كود 17 هو "خبيث" وليس "عام" في نموذجك
                    malignant_indices = [1, 4, 17] 
                    benign_indices = [2, 5, 23]
                    
                    p_malig = sum([probs[i] for i in malignant_indices if i < len(probs)])
                    p_benign = sum([probs[i] for i in benign_indices if i < len(probs)])

                    # --- [المنطق الحاسم لحل مشكلة التصنيف العام] ---
                    # تقليل عتبة الخبيث لزيادة الحساسية (Sensitivity)
                    if p_malig >= 0.15: # خفضنا النسبة من 25% إلى 15% لكشف الحالات المشتبهة فوراً
                        res_msg, sub_msg = "🚨 اشتباه ورم خبيث", "تم رصد علامات حيوية تستوجب الفحص الطبي الفوري."
                        bg_color, txt_color = "#fff1f0", "#cf1322"
                        advice = "يجب مراجعة الطبيب المختص بأسرع وقت لإجراء الفحوصات السريرية."
                    
                    elif p_benign >= 0.50: # خفضنا عتبة الحميد لضمان تصنيف أدق
                        res_msg, sub_msg = "🔍 ورم جلدي حميد", "الآفة تبدو سليمة ومن النوع غير السرطاني."
                        bg_color, txt_color = "#f6ffed", "#389e0d"
                        advice = "لا توجد علامات قلق، لكن راقب أي تغير في الحجم أو اللون."
                    
                    else: # الحالة العامة تظهر فقط إذا كانت الاحتمالات الأخرى شبه معدومة
                        res_msg, sub_msg = "🩺 حالة جلدية عامة", "النمط الجلدي المكتشف يشير إلى حالة غير ورمية (التهاب أو حساسية)."
                        bg_color, txt_color = "#e6f7ff", "#096dd9"
                        advice = "استشر طبيباً عاماً لوصف العلاج المناسب للحالة الجلدية."

                    st.markdown(f"""
                        <div class="report-card" style="background-color: {bg_color}; border-color: {txt_color}; color: {txt_color};">
                            <p class="result-title">{res_msg}</p>
                            <p class="result-desc">{sub_msg}</p>
                        </div>
                        <div class="advice-box"><strong>💡 توصية النظام:</strong> {advice}</div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"⚠️ حدث خطأ: {e}")

st.markdown("<br><hr><p style='text-align: center; color: grey;'>نظام تقييم سلامة الجلد الذكي © 2026</p>", unsafe_allow_html=True)
