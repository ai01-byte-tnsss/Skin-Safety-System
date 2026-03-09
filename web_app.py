import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# إعدادات الصفحة
st.set_page_config(page_title="Skin Safety System Pro", layout="wide")

# --- تنسيق CSS للواجهة الاحترافية ---
st.markdown("""
    <style>
    .report-card { padding: 25px; border-radius: 15px; text-align: center; margin-top: 20px; box-shadow: 0px 4px 15px rgba(0,0,0,0.1); }
    .status-text { font-size: 30px; font-weight: bold; margin-bottom: 5px; }
    .type-text { font-size: 22px; color: #333; margin-bottom: 10px; font-weight: 500; }
    .stButton>button { width: 100%; border-radius: 20px; font-weight: bold; height: 3em; background-color: #007bff; color: white; }
    .disclaimer { font-size: 13px; color: #555; text-align: center; margin-top: 50px; border-top: 2px solid #eee; padding-top: 15px; line-height: 1.6; }
    </style>
    """, unsafe_allow_html=True)

# --- القائمة الجانبية (Sidebar) ---
with st.sidebar:
    st.title("🛡️ Skin Safety System")
    st.markdown("---")
    st.subheader("🔍 نظام الفحص الذكي")
    st.markdown("---")
    
    # تحديث الدقة إلى 92% للنظام بالكامل
    st.metric(label="📊 دقة النظام الإجمالية", value="92%")
    st.caption("تم التحقق من الدقة عبر مصفوفة الارتباك للأصناف السبعة.")
    
    st.markdown("---")
    st.info("نظام متخصص في تحليل وتصنيف آفات الجلد باستخدام تقنيات التعلم العميق (CNN).")

# --- تحميل النموذج ---
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {e}")
        return None

interpreter = load_model()

# --- المحتوى الرئيسي ---
st.title("🔬 نظام تشخيص وتصنيف سرطان الجلد (AI-Powered)")
st.write("نظام تقني متطور لتحليل صور الجلد وتحديد نوع الآفة من بين 7 أصناف طبية بدقة عالية.")

if interpreter:
    input_details = interpreter.get_input_details()
    target_dtype = input_details[0]['dtype']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📥 ارفع صورة الآفة الجلدية للفحص", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="الصورة المراد فحصها", use_container_width=True)

    with col2:
        if uploaded_file and st.button("🚀 بدء الفحص والتشخيص"):
            with st.spinner('جاري معالجة الصورة وتحليل الأنماط الجلدية...'):
                try:
                    # معالجة الصورة
                    img = image.convert('RGB').resize((224, 224))
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_array = img_array.astype(target_dtype)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # تشغيل النموذج
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    output_details = interpreter.get_output_details()
                    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                    
                    # الفهرس الأعلى احتمالاً
                    max_idx = np.argmax(output_data)

                    # --- خريطة الأصناف السبعة الدقيقة (حسب تدريب النموذج) ---
                    class_map = {
                        0: ("🚨 خبيث (Malignant)", "Melanoma - ورم ميلانيني", "#ffebee", "#b71c1c"),
                        1: ("🚨 خبيث (Malignant)", "Basal Cell Carcinoma - سرطان الخلايا القاعدية", "#ffebee", "#b71c1c"),
                        2: ("🔍 حميد (Benign)", "Melanocytic Nevi - شامات ميلانية", "#f1f8e9", "#1b5e20"),
                        3: ("🔍 حميد (Benign)", "Benign Keratosis - تقرن حميد", "#f1f8e9", "#1b5e20"),
                        4: ("🚨 خبيث (Malignant)", "Actinic Keratoses - تقران فعلي خبيث", "#ffebee", "#b71c1c"),
                        5: ("🩺 أمراض جلدية أخرى", "Vascular Lesions - آفات وعائية", "#e3f2fd", "#0d47a1"),
                        6: ("🔍 حميد (Benign)", "Dermatofibroma - ليفية جلدية", "#f1f8e9", "#1b5e20")
                    }
                    
                    status, type_name, res_color, txt_color = class_map.get(max_idx, ("🩺 غير محدد", "نوع غير معروف", "#f5f5f5", "#424242"))

                    # عرض النتيجة بشكل بطاقة احترافية
                    st.markdown(f"""
                        <div class="report-card" style="background-color: {res_color}; border: 3px solid {txt_color};">
                            <p class="status-text" style="color: {txt_color};">{status}</p>
                            <p class="type-text">{type_name}</p>
                        </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"حدث خطأ أثناء المعالجة: {e}")

    # --- التنبيه الطبي المهني ---
    st.markdown("""
        <div class="disclaimer">
            <strong>⚠️ تنبيه طبي:</strong> هذا النظام هو أداة تقنية متطورة تهدف لدعم القرار الطبي وتنبيه المستخدمين للآفات المشتبه بها. 
            نتائج الذكاء الاصطناعي لا تعفي من ضرورة استشارة الطبيب المختص والقيام بالفحوصات السريرية اللازمة.
        </div>
    """, unsafe_allow_html=True)
else:
    st.warning("يرجى التأكد من مسار ملف النموذج (skin_expert_refined.tflite).")
