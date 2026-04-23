import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="نظام الكشف عن سلامة الجلد",
    page_icon="🛡️",
    layout="wide"
)

# --- قائمة اللغات في الشريط الجانبي ---
language = st.sidebar.selectbox(
    "🌐 اختر اللغة / Select Language",
    ["العربية", "English"]
)

# نصوص الواجهة بناءً على اللغة المختارة
if language == "العربية":
    title = "نظام التشخيص الذكي لسلامة الجلد باستخدام الذكاء الاصطناعي"
    upload_label = "قم برفع صورة الشامة أو المنطقة المصابة:"
    analyze_btn = "تحليل الصورة"
    result_title = "النتيجة المتوقعة:"
    confidence_title = "نسبة الثقة:"
    guide_title = "دليل أنواع الأورام الجلدية"
    benign_desc = "الورم الحميد: هو نمو غير سرطاني، لا ينتشر إلى أجزاء أخرى من الجسم، وغالباً ما يكون غير ضار."
    malignant_desc = "الورم الخبيث: هو ورم سرطاني يمكن أن ينمو بسرعة وينتشر إلى الأنسجة المجاورة وأجزاء أخرى من الجسم."
    other_desc = "أنواع أخرى: تشمل الحالات التي قد تكون التهابات أو آفات جلدية غير نمطية تتطلب فحصاً مخبرياً."
    warning_text = "⚠️ تنبيه: هذا التشخيص آلي لأغراض بحثية فقط. يرجى استشارة الطبيب."
else:
    title = "AI Smart Skin Safety Detection System"
    upload_label = "Upload an image of the mole or affected area:"
    analyze_btn = "Analyze Image"
    result_title = "Predicted Result:"
    confidence_title = "Confidence Level:"
    guide_title = "Skin Tumor Types Guide"
    benign_desc = "Benign: Non-cancerous growth, does not spread to other body parts, and is usually harmless."
    malignant_desc = "Malignant: Cancerous tumor that can grow rapidly and spread to nearby tissues."
    other_desc = "Other types: Includes conditions that may be infections or atypical lesions requiring lab tests."
    warning_text = "⚠️ Disclaimer: This is an AI diagnosis for research purposes only. Please consult a doctor."

# --- واجهة المستخدم الرئيسية ---
st.title(title)

# --- قائمة منسدلة لشرح الأنواع (ملونة) ---
with st.expander(guide_title):
    st.markdown(f'<p style="color:green; font-weight:bold;">🟢 {benign_desc}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:red; font-weight:bold;">🔴 {malignant_desc}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:orange; font-weight:bold;">🟠 {other_desc}</p>', unsafe_allow_html=True)

st.write("---")

# --- تحميل النموذج ---
@st.cache_resource
def load_my_model():
    # سيتم البحث عن ملف master.h5 في المجلد الرئيسي للمشروع
    model = tf.keras.models.load_model('master.h5')
    return model

try:
    model = load_my_model()
except Exception as e:
    st.error("خطأ: لم يتم العثور على ملف النموذج 'master.h5'. تأكد من رفعه في المجلد الرئيسي.")

# --- منطقة رفع الصور والمعالجة ---
uploaded_file = st.file_uploader(upload_label, type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    with col2:
        st.write("🔄 **تحليل البيانات...**")
        
        # معالجة الصورة (Preprocessing)
        size = (224, 224) 
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image)
        # تطبيع الصورة (Normalization)
        img_reshape = img_array[np.newaxis, ...] / 255.0

        # التنبؤ
        prediction = model.predict(img_reshape)
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # الأصناف الثلاثة
        class_names = ["حميد (Benign)", "خبيث (Malignant)", "غير ذلك (Other)"]
        
        # تحديد لون النتيجة
        res_color = "green" if result_index == 0 else "red" if result_index == 1 else "orange"

        st.markdown(f"### {result_title} <span style='color:{res_color}'>{class_names[result_index]}</span>", unsafe_allow_html=True)
        st.metric(label=confidence_title, value=f"{confidence:.2f}%")
        st.write("---")
st.warning(warning_text)

st.write("---")
st.warning(warning_text)
