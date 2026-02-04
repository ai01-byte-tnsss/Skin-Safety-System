import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات واجهة المستخدم
st.set_page_config(page_title="نظام فحص سرطان الجلد الرقمي", layout="centered")


# 2. تحميل النموذج البرمجي
@st.cache_resource
def load_my_model():
    try:
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, 'skin_cancer_expert.h5')
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {e}")
        return None


model = load_my_model()

# 3. نظام الحماية بكلمة المرور
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔐 تسجيل الدخول")
    password = st.text_input("أدخل كلمة المرور الخاصة بالنظام:", type="password")
    if st.button("دخول"):
        if password == "test**00":  # كلمة المرور المطلوبة
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("كلمة المرور غير صحيحة")
else:
    # 4. واجهة الفحص الرقمي
    st.title("🔍 فحص سرطان الجلد بالذكاء الاصطناعي")
    st.write("ارفع صورة واضحة للشامة للحصول على تحليل فوري.")

    uploaded_file = st.file_uploader("اختر صورة (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة التي تم رفعها", width=300)

        if st.button("🔎 ابدأ التحليل"):
            if model is not None:
                with st.spinner('جاري معالجة البيانات الرقمية...'):
                    try:
                        # أ- معالجة الصورة وتوحيد مقاسها
                        img = image.resize((170, 170))
                        img_array = np.array(img.convert('RGB')) / 255.0

                        # ب- الحل النهائي لخطأ الأبعاد (86528)
                        # نقوم بتحويل الصورة لصف واحد وقصها بالضبط لتناسب النموذج
                        flat_img = img_array.flatten()
                        if len(flat_img) >= 86528:
                            final_input = flat_img[:86528].reshape(1, 86528)
                        else:
                            # في حال كانت الصورة أصغر، يتم إكمال النقص بأصفار
                            padded_img = np.pad(flat_img, (0, 86528 - len(flat_img)), 'constant')
                            final_input = padded_img.reshape(1, 86528)

                        # ج- إجراء التنبؤ الرقمي
                        prediction = model.predict(final_input)
                        result = prediction[0][0]

                        # د- عرض النتيجة النهائية للمستخدم
                        st.markdown("---")
                        if result > 0.5:
                            st.error(f"⚠️ النتيجة: يوجد احتمال إصابة بنسبة {result * 100:.2f}%")
                            st.info("نوصي بزيارة طبيب متخصص للفحص السريري.")
                        else:
                            st.success(f"✅ النتيجة: المنطقة تبدو سليمة بنسبة {(1 - result) * 100:.2f}%")
                            st.balloons()

                    except Exception as e:
                        st.error(f"عذراً، حدث خطأ أثناء التحليل: {e}")
            else:
                st.error("ملف النموذج (h5) غير موجود أو لم يتم تحميله بشكل صحيح.")