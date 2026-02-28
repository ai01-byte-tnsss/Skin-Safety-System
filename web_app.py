import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# إعدادات الواجهة
st.set_page_config(page_title="Skin Check Pro", layout="centered")

st.markdown("""
    <style>
    .report-card { padding: 25px; border-radius: 15px; text-align: center; margin-top: 20px; box-shadow: 0px 4px 15px rgba(0,0,0,0.1); }
    .status-text { font-size: 24px; font-weight: bold; margin-bottom: 10px; }
    .status-subtext { font-size: 16px; color: #555; }
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
        st.error(f"خطأ: تأكد من وجود ملف النموذج في المجلد الرئيسي. {e}")
        return None

interpreter = load_model()

if interpreter:
    input_details = interpreter.get_input_details()
    target_dtype = input_details[0]['dtype']
    
    st.markdown("<h2 style='text-align: center;'>🛡️ فحص الآفات الجلدية الذكي</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("قم برفع الصورة للتحليل", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        if st.button("🚀 تحليل الحالة"):
            with st.spinner('جاري الفحص...'):
                # 1. معالجة الصورة وتحويل الدقة
                img = image.convert('RGB').resize((224, 224))
                img_array = np.array(img).astype(np.float32) / 255.0
                img_array = img_array.astype(target_dtype)
                img_array = np.expand_dims(img_array, axis=0)
                
                # 2. تشغيل النموذج
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                output_details = interpreter.get_output_details()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                
                # 3. المنطق التصنيفي المصحح
                max_idx = np.argmax(output_data)
                
                # --- تحديث المنطق: اجعل المجموعات فارغة حالياً لتجنب التصنيف الخاطئ ---
                # قم بملء هذه المجموعات بناءً على ترتيب المجلدات في ملف الـ Dataset الخاص بك
                malignant_set = [] # أضف أرقام المجلدات الخبيثة هنا
                benign_set = []    # أضف أرقام المجلدات الحميدة هنا
                
                # تحديد النتيجة النهائية
                if max_idx in malignant_set:
                    res_msg = "🚨 الحالة: سرطان خبيث (Malignant)"
                    sub_msg = "يجب استشارة طبيب أورام فوراً."
                    res_color = "#ffebee" 
                    txt_color = "#b71c1c"
                elif max_idx in benign_set:
                    res_msg = "🔍 الحالة: ورم حميد (Benign)"
                    sub_msg = "مرض جلدي، ولكنه ليس سرطان خبيث."
                    res_color = "#fff3e0"
                    txt_color = "#e65100"
                else:
                    # أي رقم غير معرف يتم تصنيفه كـ "مراجعة"
                    res_msg = "⚠️ الحالة: بحاجة إلى مراجعة طبية"
                    sub_msg = "النموذج لا يستطيع تصنيف هذه الآفة بدقة حالياً. يرجى مراجعة طبيب."
                    res_color = "#eceff1"
                    txt_color = "#37474f"

                # 4. عرض النتيجة
                st.markdown(f"""
                    <div class="report-card" style="background-color: {res_color}; border: 2px solid {txt_color};">
                        <p class="status-text" style="color: {txt_color};">{res_msg}</p>
                        <p class="status-subtext">{sub_msg}</p>
                    </div>
                """, unsafe_allow_html=True)
else:
    st.warning("جاري تهيئة النظام...")
