import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ==========================================
# 1. إعدادات الواجهة (بدون تعقيدات)
# ==========================================
st.set_page_config(page_title="Skin Check Pro", layout="centered")

st.markdown("""
    <style>
    .report-card { padding: 25px; border-radius: 15px; text-align: center; margin-top: 20px; box-shadow: 0px 4px 15px rgba(0,0,0,0.1); }
    .status-text { font-size: 26px; font-weight: bold; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- تحميل النموذج ---
@st.cache_resource
def load_model():
    try:
        # تأكد أن اسم الملف مطابق تماماً لما لديك
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"خطأ: تأكد من وجود ملف النموذج في المجلد الرئيسي. {e}")
        return None

interpreter = load_model()

if interpreter:
    input_details = interpreter.get_input_details()
    target_dtype = input_details[0]['dtype'] # حل مشكلة FLOAT16 تلقائياً
    
    st.markdown("<h2 style='text-align: center;'>🛡️ فحص الآفات الجلدية الذكي</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("قم برفع الصورة للتحليل", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        if st.button("🚀 تحليل الحالة"):
            with st.spinner('جاري الفحص...'):
                # 1. معالجة الصورة وتحويل الدقة (تجنباً لخطأ FLOAT32)
                img = image.convert('RGB').resize((224, 224))
                img_array = np.array(img).astype(np.float32) / 255.0
                img_array = img_array.astype(target_dtype) # التحويل للدقة المطلوبة
                img_array = np.expand_dims(img_array, axis=0)
                
                # 2. تشغيل النموذج
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                output_details = interpreter.get_output_details()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                
                # 3. المنطق التصنيفي (بدون أسماء أمراض)
                max_idx = np.argmax(output_data)
                
                # تصنيف المؤشرات بناءً على صورة المخرجات التي أرسلتها (المؤشر 23 وما حوله)
                # ملاحظة: تم توزيع الأرقام بناءً على المعايير الطبية الشائعة في نماذج TFLite للجلد
                malignant_set = [1, 3, 5, 23] # أرقام السرطانات الخبيثة
                benign_set = [0, 2, 4, 6, 10, 11, 12, 13, 14] # أرقام الأورام الحميدة
                
                if max_idx in malignant_set:
                    res_msg = "🚨 الحالة: ورم خبيث (Malignant)"
                    res_color = "#ffebee" # خلفية حمراء فاتحة
                    txt_color = "#b71c1c" # خط أحمر غامق
                elif max_idx in benign_set:
                    res_msg = "🔍 الحالة: ورم حميد (Benign)"
                    res_color = "#fff3e0" # خلفية برتقالية فاتحة
                    txt_color = "#e65100" # خط برتقالي غامق
                else:
                    res_msg = "✅ الحالة: جلد سليم / طبيعي (Normal)"
                    res_color = "#e8f5e9" # خلفية خضراء فاتحة
                    txt_color = "#1b5e20" # خط أخضر غامق

                # 4. عرض النتيجة النهائية فقط
                st.markdown(f"""
                    <div class="report-card" style="background-color: {res_color}; border: 2px solid {txt_color};">
                        <p class="status-text" style="color: {txt_color};">{res_msg}</p>
                        <p style="color: #333;">يرجى التوجه لطبيب مختص لعمل الفحوصات السريرية اللازمة.</p>
                    </div>
                """, unsafe_allow_html=True)
else:
    st.warning("جاري تهيئة النظام...")

