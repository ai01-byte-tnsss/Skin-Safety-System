import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ==========================================
# 1. إعدادات الصفحة والتصميم (CSS)
# ==========================================
st.set_page_config(page_title="Skin Safety System Pro", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #1E88E5; color: white; font-weight: bold; }
    .report-card { padding: 25px; border-radius: 15px; background-color: white; border-left: 6px solid #1E88E5; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); margin-top: 20px; }
    .title-text { text-align: center; color: #0D47A1; }
    .status-text { font-size: 24px; font-weight: bold; margin-bottom: 10px; }
    .status-subtext { font-size: 16px; color: #555; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. تشغيل النظام الرئيسي
# ==========================================
st.markdown("<h1 class='title-text'>🛡️ منصة التشخيص الذكي للأمراض الجلدية</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>نظام خبير يعتمد على الشبكات العصبية (TFLite)</p>", unsafe_allow_html=True)

st.divider()

# --- دالة تحميل نموذج TFLite وتجهيزه ---
@st.cache_resource
def load_tflite_model():
    try:
        # تأكد أن الملف موجود في نفس مجلد التشغيل
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"خطأ في تحميل نموذج TFLite: {e}")
        return None

interpreter = load_tflite_model()

if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # --- كشف نوع البيانات المطلوب من النموذج تلقائياً ---
    target_dtype = input_details[0]['dtype']
    
    uploaded_file = st.file_uploader("📥 قم برفع صورة الآفة الجلدية هنا", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(image, caption="الصورة المرفوعة", use_container_width=True)
        
        with col_info:
            st.info("💡 **نصيحة طبية:** تأكد من جودة الصورة للحصول على أدق نتيجة.")
            analyze_btn = st.button("🔬 بدء التحليل")

        if analyze_btn:
            with st.spinner('جاري التحليل السريع باستخدام TFLite...'):
                try:
                    # 1. معالجة الصورة
                    img = image.convert('RGB')
                    img = img.resize((224, 224))
                    
                    # --- الحل الشامل لمشكلة الدقة (FLOAT) ---
                    # تحويل الصورة إلى مصفوفة بايثون (float32 افتراضياً)
                    img_array = np.array(img).astype(np.float32) / 255.0
                    
                    # تحويل المصفوفة إلى نوع البيانات الذي يتوقعه النموذج تحديداً (FLOAT16 أو FLOAT32)
                    img_array = img_array.astype(target_dtype)
                    
                    img_array = np.expand_dims(img_array, axis=0)

                    # 2. تشغيل التنبؤ عبر TFLite
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    
                    # --- استقبال النتائج ---
                    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                    
                    # -----------------------------------------------------
                    # 3. المنطق المصحح للتعامل مع نتائج النموذج (بدون ذكر أرقام)
                    # -----------------------------------------------------
                    
                    max_idx = np.argmax(output_data)
                    
                    # تحديث المجموعات بناءً على تصنيفات ISIC الدقيقة (أرقام افتراضية بناءً على طلبك)
                    # يجب تحديث هذه الأرقام بناءً على ملف التدريب الخاص بك
                    malignant_set = [1, 3, 5, 23, 4] 
                    benign_set = [0, 2, 6, 10, 11, 12, 13, 14] 
                    other_conditions_set = [7, 8, 9, 15, 16] 

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
                    elif max_idx in other_conditions_set:
                        res_msg = "🩺 الحالة: مرض جلدي (غير سرطاني)"
                        sub_msg = "آفة جلدية، ولكنها ليست من أنواع السرطان."
                        res_color = "#e3f2fd"
                        txt_color = "#0d47a1"
                    else:
                        res_msg = "⚠️ الحالة: غير معرفة - يرجى مراجعة طبيب"
                        sub_msg = "يرجى الفحص السريري للتأكد."
                        res_color = "#eceff1"
                        txt_color = "#37474f"

                    # 4. عرض النتيجة
                    st.markdown(f"""
                        <div class="report-card" style="background-color: {res_color}; border: 2px solid {txt_color};">
                            <p class="status-text" style="color: {txt_color};">{res_msg}</p>
                            <p class="status-subtext">{sub_msg}</p>
                        </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"⚠️ خطأ أثناء تحليل الصورة: {e}")

else:
    st.warning("⚠️ لم يتم العثور على ملف النموذج 'skin_expert_refined.tflite'.")
