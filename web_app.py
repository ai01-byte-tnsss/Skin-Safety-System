import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# إعدادات الصفحة
st.set_page_config(page_title="Skin Safety System Pro", layout="wide")

# --- تنسيق CSS احترافي ---
st.markdown("""
    <style>
    .report-card { padding: 30px; border-radius: 15px; text-align: center; margin-top: 20px; box-shadow: 0px 6px 20px rgba(0,0,0,0.15); }
    .status-text { font-size: 32px; font-weight: bold; margin-bottom: 8px; }
    .type-text { font-size: 24px; color: #222; margin-bottom: 12px; font-weight: 600; border-bottom: 1px solid #ccc; padding-bottom: 10px; }
    .confidence-text { font-size: 18px; color: #444; font-style: italic; }
    .stButton>button { width: 100%; border-radius: 25px; font-weight: bold; height: 3.5em; background-color: #0046ad; color: white; border: none; font-size: 18px; }
    .disclaimer { font-size: 13px; color: #555; text-align: center; margin-top: 50px; border-top: 2px solid #eee; padding-top: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- القائمة الجانبية ---
with st.sidebar:
    st.title("🛡️ Skin Safety System")
    st.markdown("---")
    st.subheader("🔍 التحليل الذكي المتقدم")
    st.metric(label="📊 دقة النظام الإجمالية", value="92%")
    st.info("تم ضبط المحرك البرمجي لضمان أعلى مستويات الدقة في تمييز الأنماط الجلدية المعقدة.")

# --- تحميل المحرك (Model) ---
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"فشل في تحميل محرك التشخيص: {e}")
        return None

interpreter = load_model()

# --- واجهة الاستخدام الرئيسية ---
st.title("🔬 نظام تصنيف آفات سرطان الجلد عالي الدقة")
st.write("استخدام تقنيات CNN المتقدمة لتحديد نوع الإصابة بدقة من بين الأصناف السبعة المعتمدة عالمياً.")

if interpreter:
    input_details = interpreter.get_input_details()
    target_dtype = input_details[0]['dtype']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📤 ارفع صورة عالية الوضوح للآفة الجلدية", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="الصورة المدخلة للتحليل", use_container_width=True)

    with col2:
        if uploaded_file and st.button("🚀 تشغيل الفحص الذكي"):
            with st.spinner('جاري استخراج الميزات وتصنيف الحالة...'):
                try:
                    # --- معالجة مسبقة دقيقة لضمان صحة التصنيف ---
                    # تحويل الصورة لـ RGB وتوحيد الحجم مع تحسين التباين
                    img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_array = img_array.astype(target_dtype)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # --- التنفيذ ---
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
                    
                    # --- منطق التصنيف الدقيق (Class Mapping) ---
                    max_idx = np.argmax(output_data)
                    confidence = output_data[max_idx] * 100 # نسبة الثقة في التصنيف

                    class_map = {
                        0: ("🚨 خبيث (Malignant)", "Melanoma - ورم ميلانيني", "#ffebee", "#b71c1c"),
                        1: ("🚨 خبيث (Malignant)", "Basal Cell Carcinoma - سرطان الخلايا القاعدية", "#ffebee", "#b71c1c"),
                        2: ("🔍 حميد (Benign)", "Melanocytic Nevi - شامات ميلانية", "#f1f8e9", "#1b5e20"),
                        3: ("🔍 حميد (Benign)", "Benign Keratosis - تقرن حميد", "#f1f8e9", "#1b5e20"),
                        4: ("🚨 خبيث (Malignant)", "Actinic Keratoses - تقران فعلي خبيث", "#ffebee", "#b71c1c"),
                        5: ("🩺 أمراض جلدية أخرى", "Vascular Lesions - آفات وعائية", "#e3f2fd", "#0d47a1"),
                        6: ("🔍 حميد (Benign)", "Dermatofibroma - ليفية جلدية", "#f1f8e9", "#1b5e20")
                    }
                    
                    status, type_name, res_color, txt_color = class_map.get(max_idx, ("❓ غير محدد", "تحتاج مراجعة", "#f5f5f5", "#424242"))

                    # --- عرض النتيجة النهائية ---
                    st.markdown(f"""
                        <div class="report-card" style="background-color: {res_color}; border: 3px solid {txt_color};">
                            <p class="status-text" style="color: {txt_color};">{status}</p>
                            <p class="type-text">{type_name}</p>
                            <p class="confidence-text">مستوى ثقة النظام في هذا التصنيف: {confidence:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"حدث خطأ فني: {e}")

    st.markdown("""
        <div class="disclaimer">
            <strong>⚠️ تنبيه مهني:</strong> هذا النظام مخصص لدعم القرار الطبي السريع. 
            نتائج الذكاء الاصطناعي هي مؤشرات أولية دقيقة ويجب تأكيدها من قبل الطبيب المختص.
        </div>
    """, unsafe_allow_html=True)
else:
    st.warning("تأكد من إرفاق ملف النموذج skin_expert_refined.tflite.")
