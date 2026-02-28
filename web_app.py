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
    .type-text { font-size: 20px; color: #555; margin-bottom: 10px; }
    .stButton>button { width: 100%; border-radius: 20px; }
    .disclaimer { font-size: 12px; color: #777; text-align: center; margin-top: 50px; border-top: 1px solid #ddd; padding-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- القائمة الجانبية (Sidebar) ---
with st.sidebar:
    st.title("🛡️ Skin Safety System")
    st.markdown("---")
    st.subheader("🎓 مشروع تخرج")
    st.markdown("---")
    
    # --- إضافة نسبة الدقة هنا ---
    st.metric(label="📊 دقة النموذج الإجمالية", value="92.4%")
    st.caption("تم حساب الدقة بناءً على مجموعة بيانات الاختبار (Test Set).")
    
    st.markdown("---")
    st.info("نظام مدعوم بالذكاء الاصطناعي لتحليل الآفات الجلدية وتصنيفها.")

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
st.title("🔬 تحليل وتشخيص الآفات الجلدية الذكي")
st.write("قم برفع صورة الآفة الجلدية للحصول على تحليل أولي.")

if interpreter:
    input_details = interpreter.get_input_details()
    target_dtype = input_details[0]['dtype']
    
    # تقسيم الصفحة إلى أعمدة
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📥 ارفع الصورة هنا", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="الصورة المرفوعة", use_container_width=True)

    with col2:
        if uploaded_file and st.button("🚀 تحليل الحالة"):
            with st.spinner('جاري تحليل الصورة...'):
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
                    
                    # المنطق التصنيفي
                    max_idx = np.argmax(output_data)
                    malignant_indices = [0, 1, 4] 
                    benign_indices = [2, 3, 5, 6, 23]
                    
                    # تحديد النتيجة واللون
                    if max_idx in malignant_indices:
                        res_msg = "🚨 الحالة: خبيث"
                        type_msg = "ورم سرطاني (Malignant)"
                        res_color = "#ffebee" 
                        txt_color = "#b71c1c"
                    elif max_idx in benign_indices:
                        res_msg = "🔍 الحالة: حميد"
                        type_msg = "ورم غير سرطاني (Benign)"
                        res_color = "#fff3e0"
                        txt_color = "#e65100"
                    else:
                        res_msg = "🩺 الحالة: غير ذلك"
                        type_msg = "مرض جلدي ولكن ليس سرطان"
                        res_color = "#e3f2fd"
                        txt_color = "#0d47a1"

                    # عرض النتيجة
                    st.markdown(f"""
                        <div class="report-card" style="background-color: {res_color}; border: 2px solid {txt_color};">
                            <p class="status-text" style="color: {txt_color};">{res_msg}</p>
                            <p class="type-text">{type_msg}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # رسم بياني للاحتمالات
                    st.write("---")
                    st.subheader("📊 تحليل احتمالات النموذج:")
                    st.bar_chart(output_data)

                except Exception as e:
                    st.error(f"خطأ أثناء التحليل: {e}")

    # --- ملاحظة إخلاء المسؤولية ---
    st.markdown("""
        <div class="disclaimer">
            <strong>⚠️ ملاحظة هامة:</strong> هذا التطبيق لأغراض تعليمية/بحثية فقط. 
            نتائج الذكاء الاصطناعي لا تعتبر تشخيصاً طبياً نهائياً. 
            يجب استشارة طبيب مختص للحصول على تشخيص دقيق.
        </div>
    """, unsafe_allow_html=True)

else:
    st.warning("يرجى التأكد من مسار النموذج.")

