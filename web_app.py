import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. إعدادات واجهة المستخدم ---
st.set_page_config(page_title="Skin Safety System Pro", layout="centered")

# تصميم مخصص للنتائج (CSS)
st.markdown("""
    <style>
    .report-card { padding: 25px; border-radius: 15px; text-align: center; margin-top: 20px; box-shadow: 0px 4px 15px rgba(0,0,0,0.1); }
    .status-text { font-size: 28px; font-weight: bold; margin-bottom: 5px; }
    .type-text { font-size: 18px; color: #555; margin-bottom: 10px; }
    .probability-bar { background-color: #eee; border-radius: 10px; margin: 10px 0; height: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. دالة تحميل النموذج (TFLite) ---
@st.cache_resource
def load_skin_model():
    try:
        # تأكد من أن اسم الملف مطابق تماماً لما لديك
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"⚠️ خطأ: لم يتم العثور على ملف النموذج 'skin_expert_refined.tflite'. تأكد من وجوده في المجلد الصحيح. \n{e}")
        return None

interpreter = load_skin_model()

# --- 3. بناء الواجهة والتفاعل ---
if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    target_dtype = input_details[0]['dtype']
    
    st.markdown("<h2 style='text-align: center; color: #2E4053;'>🛡️ نظام فحص صحة الجلد الذكي</h2>", unsafe_allow_html=True)
    st.write("---")
    
    uploaded_file = st.file_uploader("📥 قم برفع صورة واضحة للآفة الجلدية", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # عرض الصورة المرفوعة بشكل أنيق
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة التي سيتم تحليلها", use_container_width=True)
        
        if st.button("🚀 بدء التحليل الآن"):
            with st.spinner('جاري فحص الأنماط باستخدام الذكاء الاصطناعي...'):
                try:
                    # أ- المعالجة المسبقة للصورة (Pre-processing)
                    img = image.convert('RGB').resize((224, 224))
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_array = img_array.astype(target_dtype)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # ب- تشغيل التوقع (Inference)
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                    
                    # ج- تحسين منطق التصنيف (Softmax Logic)
                    # تحويل المخرجات لنسب مئوية لفهم أدق
                    probabilities = tf.nn.softmax(output_data).numpy()
                    max_idx = np.argmax(probabilities)
                    confidence = probabilities[max_idx] * 100

                    # --- إعدادات الفئات (قم بتعديل الأرقام حسب نموذجك) ---
                    malignant_indices = [1, 4]  # أرقام الفئات الخبيثة
                    benign_indices = [2, 5, 23] # أرقام الفئات الحميدة

                    # د- تحديد النتيجة بناءً على الحساسية الطبية
                    # إذا كانت نسبة الشك في "أي" فئة خبيثة تتجاوز 35%، نرفع التنبيه فوراً
                    malignant_total_prob = sum([probabilities[i] for i in malignant_indices if i < len(probabilities)])

                    if malignant_total_prob > 0.35:
                        res_msg = "🚨 الحالة: تستوجب فحص طبي (احتمال خبيث)"
                        type_msg = f"نسبة الشك في وجود خلايا غير طبيعية: {malignant_total_prob*100:.1f}%"
                        res_color = "#ffebee" 
                        txt_color = "#b71c1c"
                    elif max_idx in benign_indices:
                        res_msg = "🔍 الحالة: حميد (آمن مبدئياً)"
                        type_msg = f"النمط المكتشف: ورم حميد - الثقة: {confidence:.1f}%"
                        res_color = "#e8f5e9"
                        txt_color = "#1b5e20"
                    else:
                        res_msg = f"🩺 الحالة: أخرى (كود {max_idx})"
                        type_msg = f"تم التعرف على نمط جلدي غير سرطاني - الثقة: {confidence:.1f}%"
                        res_color = "#e3f2fd"
                        txt_color = "#0d47a1"

                    # هـ- عرض بطاقة النتيجة النهائية
                    st.markdown(f"""
                        <div class="report-card" style="background-color: {res_color}; border: 2px solid {txt_color};">
                            <p class="status-text" style="color: {txt_color};">{res_msg}</p>
                            <p class="type-text">{type_msg}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("💡 نصيحة: نتائج الذكاء الاصطناعي هي وسيلة مساعدة للفرز الأولي، القرار النهائي دائماً للطبيب المختص.")

                except Exception as e:
                    st.error(f"حدث خطأ أثناء معالجة الصورة: {e}")
else:
    st.info("بانتظار تهيئة النظام وتحميل ملف النموذج...")

# تذييل الصفحة
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>نظام دعم القرار الطبي - مشروع التخرج 2026</p>", unsafe_allow_html=True)
