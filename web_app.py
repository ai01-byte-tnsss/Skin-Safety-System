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
    .debug-text { font-size: 12px; color: #888; text-align: center; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- تحميل النموذج ---
@st.cache_resource
def load_model():
    try:
        # تأكد من اسم ملف النموذج الصحيح
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"خطأ: تأكد من وجود ملف النموذج 'skin_expert_refined.tflite' في المجلد. {e}")
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
            with st.spinner('جاري الفحص باستخدام الذكاء الاصطناعي...'):
                try:
                    # 1. معالجة الصورة وتحويل الدقة لتوافق النموذج
                    img = image.convert('RGB').resize((224, 224))
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_array = img_array.astype(target_dtype)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # 2. تشغيل النموذج
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    output_details = interpreter.get_output_details()
                    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                    
                    # 3. المنطق التصنيفي
                    max_idx = np.argmax(output_data)
                    
                    # ---------------------------------------------------------
                    # --- [هام]: قم بتحديث هذه الأرقام بناءً على الرقم الذي سيظهر لك ---
                    # ---------------------------------------------------------
                    # أضف هنا جميع الأرقام التي يخرجها النموذج وتعتبرها خبيثة
                    malignant_indices = [1, 4] 
                    # أضف هنا جميع الأرقام التي يخرجها النموذج وتعتبرها حميدة
                    benign_indices = [2, 5, 23] # أضفت 23 بناءً على صورتك
                    # ---------------------------------------------------------
                    
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
                        # رسالة تفصيلية للمساعدة في تحديد الأرقام
                        res_msg = f"🩺 الحالة: غير ذلك (الرقم: {max_idx})"
                        type_msg = "مرض جلدي ولكن ليس سرطان"
                        res_color = "#e3f2fd"
                        txt_color = "#0d47a1"
# 4. عرض النتيجة
                    st.markdown(f"""
                        <div class="report-card" style="background-color: {res_color}; border: 2px solid {txt_color};">
                            <p class="status-text" style="color: {txt_color};">{res_msg}</p>
                            <p class="type-text">{type_msg}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"خطأ أثناء التحليل: {e}")
else:
    st.warning("جاري تهيئة النظام...")

