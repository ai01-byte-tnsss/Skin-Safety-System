import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# عنوان التطبيق
st.title("نظام خبير لتشخيص الأمراض الجلدية")

# دالة تحميل النموذج المصححة
@st.cache_resource
def load_my_model():
    model_path = "skin_expert_master.h5"
    # إضافة compile=False هو الحل السحري لتجاوز خطأ ValueError الظاهر في الصورة
    model = tf.keras.models.load_model(model_path, compile=False)
    # إعادة بناء المعايير داخلياً لضمان عمل النموذج
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

try:
    model = load_my_model()
    st.success("✅ تم تحميل النموذج بنجاح!")
except Exception as e:
    st.error(f"❌ حدث خطأ أثناء تحميل النموذج: {e}")

# واجهة رفع الصور
uploaded_file = st.file_uploader("اختر صورة الجلد للفحص...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة', use_column_width=True)
    
    # معالجة الصورة لتناسب النموذج
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("تشخيص"):
        with st.spinner('جاري التحليل...'):
            prediction = model.predict(img_array)
            # هنا يجب وضع قائمة الأصناف الـ 24 الخاصة بك
            # classes = ['مرض 1', 'مرض 2', ...] 
            result = np.argmax(prediction)
            st.write(f"النتيجة المتوقعة: الصنف رقم {result}")
            st.write(f"نسبة التأكد: {np.max(prediction)*100:.2f}%")
