import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="Skin Safety System", layout="centered")


# 2. دالة التحقق من كلمة المرور المحدثة
def check_password():
    if "password_correct" not in st.session_state:
        st.markdown("<h3 style='text-align: center;'>🔒 نظام آمن: يرجى تسجيل الدخول</h3>", unsafe_allow_html=True)
        # وضعنا كلمة المرور الجديدة هنا: test**00
        pwd = st.text_input("أدخل كلمة المرور للوصول إلى نظام فحص الجلد", type="password")
        if st.button("دخول"):
            if pwd == "test**00":
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("❌ كلمة المرور غير صحيحة")
        return False
    return True


# 3. تشغيل النظام في حال كانت كلمة المرور صحيحة
if check_password():
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>🛡️ النظام الذكي للكشف عن سلامة الجلد</h1>",
                unsafe_allow_html=True)

    # عرض الإحصائيات (كما في صورتك السابقة)
    col1, col2, col3 = st.columns(3)
    col1.metric("دقة التدريب", "80%")
    col2.metric("نسبة الاختبار", "20%")
    col3.metric("الدقة الإجمالية", "93%")

    st.write("---")


    # تحميل النموذج
    @st.cache_resource
    def load_my_model():
        return tf.keras.models.load_model('skin_cancer_model.h5')


    try:
        model = load_my_model()

        uploaded_file = st.file_uploader("📸 إدراج صورة الفحص (JPG, PNG)", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="الصورة القيد التحليل الرقمي", use_column_width=True)

            if st.button("🔍 اختبار سرطان أم لا"):
                # معالجة الصورة بنفس أبعاد التدريب
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)

                st.subheader("📋 نتيجة التقرير النهائي:")

                if prediction[0][0] > 0.5:
                    st.success("النتيجة: نعم (سرطان)")
                    st.warning("الحالة: حميد ⚪")
                    st.error("تم رصد خصائص بصرية تستوجب المتابعة الطبية الفورية.")
                else:
                    st.balloons()
                    st.success("النتيجة: سليم (لا يوجد سرطان) ✅")
                    st.info("الحالة: طبيعية")

                st.info("⚠️ تنبيه: هذا التقرير هو تحليل أولي رقمي؛ يرجى مراجعة دكتور مختص لتأكيد التشخيص نسيجياً.")

    except Exception as e:
        st.error(f"حدث خطأ: تأكد من وجود ملف النموذج skin_cancer_model.h5 في GitHub")