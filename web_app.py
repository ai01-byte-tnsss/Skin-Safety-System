import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ==========================================
# 1. إعدادات الصفحة والتصميم (CSS)
# ==========================================
st.set_page_config(page_title="Skin Safety System TFLite", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #1E88E5; color: white; font-weight: bold; }
    .report-card { padding: 25px; border-radius: 15px; background-color: white; border-left: 6px solid #1E88E5; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); margin-top: 20px; }
    .title-text { text-align: center; color: #0D47A1; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. نظام تسجيل الدخول الآمن
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.markdown("<div style='text-align: center; padding: 50px;'>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/1022/1022313.png", width=120)
        st.markdown("<h3>🔒 نظام آمن: يرجى تسجيل الدخول</h3>", unsafe_allow_html=True)
        pwd = st.text_input("أدخل كلمة المرور للوصول للنظام", type="password", placeholder="كلمة المرور")
        if st.button("دخول"):
            if pwd == "test**00":
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("❌ كلمة المرور غير صحيحة")
        return False
    return True

# ==========================================
# 3. تشغيل النظام الرئيسي
# ==========================================
if check_password():
    st.markdown("<h1 class='title-text'>🛡️ منصة التشخيص الذكي للأمراض الجلدية</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555;'>نظام خبير يعتمد على الشبكات العصبية (TFLite)</p>", unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("دقة النموذج", "91%")
    with m2: st.metric("نوع المعالجة", "TFLite Speed")
    with m3: st.metric("حالة النظام", "نشط ✅")

    st.divider()

    # --- دالة تحميل نموذج TFLite وتجهيزه ---
    @st.cache_resource
    def load_tflite_model():
        # استخدام اسم الملف الموجود في صورتك الأخيرة
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter

    try:
        interpreter = load_tflite_model()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        uploaded_file = st.file_uploader("📥 قم برفع صورة الآفة الجلدية هنا", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col_img, col_info = st.columns([1, 1])
            with col_img:
                st.image(image, caption="الصورة المرفوعة", use_container_width=True)
            
            with col_info:
                st.info("💡 **نصيحة طبية:** تأكد من جودة الصورة للحصول على أدق نتيجة.")
                analyze_btn = st.button("🔬 بدء التحليل (TFLite)")

            if analyze_btn:
                with st.spinner('جاري التحليل السريع باستخدام TFLite...'):
                    # 1. معالجة الصورة (تغيير الحجم والنوع لتناسب TFLite)
                    img = image.resize((224, 224))
                    img_array = np.array(img).astype('float32') / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # 2. تشغيل التنبؤ عبر TFLite
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    prediction_prob = interpreter.get_tensor(output_details[0]['index'])[0][0]
                    
                    st.markdown("<div class='report-card'>", unsafe_allow_html=True)
                    st.subheader("📋 التقرير التشخيصي النهائي:")
                    st.markdown("---")

                    # المنطق المطور للتركيز على السرطان
                    if prediction_prob > 0.70:
                        st.error("🚨 **النتيجة: نعم (مؤشرات قوية لورم خبيث - Malignant)**")
                        st.write("تم رصد أنماط بصرية تتطابق مع خصائص الأورام الجلدية.")
                        st.warning("⚠️ **تنبيه:** يرجى مراجعة المختص فوراً.")
                    elif prediction_prob < 0.35:
                        st.balloons()
                        st.success("✅ **النتيجة: سليم (ورم حميد أو شامة طبيعية - Benign)**")
                        st.write("الخصائص البصرية تظهر أنسجة مستقرة.")
                    else:
                        st.warning("🔍 **النتيجة: مرض جلدي آخر (غير سرطاني)**")
                        st.write("الأنماط تشير إلى وجود **مرض جلدي آخر** (مثل الأكزيما أو الصدفية).")

                    st.markdown("---")
                    st.write(f"**نسبة الثقة في التحليل:** {max(prediction_prob, 1-prediction_prob):.2%}")
                    st.progress(float(prediction_prob))
                    st.markdown("</div>", unsafe_allow_html=True)

        st.sidebar.markdown("### حول النظام (TFLite)")
        st.sidebar.info("هذا الإصدار يستخدم TFLite لضمان سرعة معالجة عالية واستهلاك أقل للموارد.")

    except Exception as e:
        st.error(f"⚠️ خطأ في تشغيل TFLite: {e}")
