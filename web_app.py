import streamlit as st 
import tensorflow as tf
from PIL import Image
import numpy as np

# إعدادات الواجهة
st.set_page_config(page_title="Skin Safety System Pro", layout="centered")

st.markdown("""
    <style>
    .report-card { padding: 25px; border-radius: 15px; text-align: center; margin-top: 20px; box-shadow: 0px 4px 15px rgba(0,0,0,0.1); }
    .status-text { font-size: 28px; font-weight: bold; margin-bottom: 5px; }
    .type-text { font-size: 18px; color: #555; margin-bottom: 10px; }
    .debug-text { font-size: 14px; color: #d32f2f; background-color: #f9f9f9; padding: 5px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"خطأ: تأكد من وجود ملف النموذج. {e}")
        return None

interpreter = load_model()

if interpreter:
    input_details = interpreter.get_input_details()
    target_dtype = input_details[0]['dtype']
    
    st.markdown("<h2 style='text-align: center;'>🛡️ فحص الآفات الجلدية الذكي</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📥 قم برفع صورة الآفة الجلدية هنا", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        if st.button("🚀 تحليل الحالة"):
            with st.spinner('جاري الفحص...'):
                try:
                    img = image.convert('RGB').resize((224, 224))
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_array = img_array.astype(target_dtype)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    output_details = interpreter.get_output_details()
                    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                    
                    max_idx = np.argmax(output_data)
                    confidence = output_data[max_idx] * 100

                    # --- قسم حل مشكلة التصنيف الخبيث ---
                    # 1. قائمة الأرقام التي تمثل "خبيث" (تأكد من هذه الأرقام من ملف تدريب النموذج)
                    malignant_indices = [1, 4] 
                    # 2. قائمة الأرقام التي تمثل "حميد"
                    benign_indices = [0, 2, 3, 5, 6] 

                    if max_idx in malignant_indices:
                        res_msg, type_msg, res_color, txt_color = "🚨 الحالة: خبيث", "ورم سرطاني (Malignant)", "#ffebee", "#b71c1c"
                    elif max_idx in benign_indices:
                        res_msg, type_msg, res_color, txt_color = "🔍 الحالة: حميد", "ورم غير سرطاني (Benign)", "#fff3e0", "#e65100"
                    else:
                        res_msg, type_msg, res_color, txt_color = f"🩺 رقم الفئة: {max_idx}", "حالة غير مصنفة حالياً", "#f5f5f5", "#616161"

                    st.markdown(f"""
                        <div class="report-card" style="background-color: {res_color}; border: 2px solid {txt_color};">
                            <p class="status-text" style="color: {txt_color};">{res_msg}</p>
                            <p class="type-text">{type_msg}</p>
                            <p style="color: gray;">نسبة الثقة: {confidence:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # ملاحظة للمطور لمساعدتك في ضبط الأرقام
                    st.info(f"نصيحة للمطور: إذا كانت الصورة خبيثة وظهرت كحميدة، لاحظ الرقم {max_idx} وقم بنقله إلى قائمة malignant_indices.")

                except Exception as e:
                    st.error(f"خطأ أثناء التحليل: {e}")
