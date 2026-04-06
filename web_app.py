import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. إعدادات الواجهة الاحترافية ---
st.set_page_config(page_title="Skin Health Analysis", layout="centered")

st.markdown("""
<style>
    .report-card { padding: 30px; border-radius: 20px; text-align: center; margin-top: 25px; border: 3px solid; }
    .result-title { font-size: 32px; font-weight: bold; margin-bottom: 12px; }
    .result-desc { font-size: 19px; font-weight: 500; line-height: 1.6; }
    .advice-box { background-color: #fcfcfc; padding: 20px; border-radius: 12px; margin-top: 25px; border-right: 6px solid #455a64; }
    .quality-alert { background-color: #fffbe6; border: 1px solid #ffe58f; padding: 15px; border-radius: 8px; color: #856404; font-size: 14px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 2. محرك تحميل النموذج ---
@st.cache_resource
def load_expert_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"⚠️ خطأ critical: تعذر تحميل النموذج الأساسي. تأكد من وجود ملف 'skin_expert_refined.tflite' في المجلد المخصص. \n[تفاصيل تقنية]: {e}")
        return None

interpreter = load_expert_model()

# التحقق من أن محرك التحليل جاهز
if interpreter:
    # جلب تفاصيل المدخلات لفهم نوع البيانات المطلوب
    input_details = interpreter.get_input_details()
    target_dtype = input_details[0]['dtype'] # هذا هو السطر الحاسم

    st.markdown("<h1 style='text-align: center; color: #0d47a1;'>🛡️ نظام الكشف عن سلامة الجلد</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="quality-alert">
            ⚠️ <b>تنبيه الجودة:</b> يرجى التأكد من توفر إضاءة قوية ووضوح الصورة لضمان دقة التحليل.
        </div>
    """, unsafe_allow_html=True)

    # ميزة إعادة المحاولة
    if st.button("🔄 فحص حالة جديدة"):
        st.rerun()

    uploaded_file = st.file_uploader("📥 قم برفع صورة الحالة الجلدية", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        if st.button("🚀 بدء التحليل"):
            with st.spinner("جاري الفحص..."):
                try:
                    # --- [المعالج المسبق المصحح] ---
                    
                    # 1. تحويل الصورة وتغيير حجمها
                    img = image.convert("RGB").resize((224, 224))
                    img_array = np.array(img) # نحتفظ بالأرقام من 0-255 كـ uint8
                    
                    # 2. المطابقة التلقائية لنوع البيانات المطلوب من النموذج
                    # إذا كان النموذج يريد float32، نقوم بالتطبيع
                    if target_dtype == np.float32:
                        img_array = img_array.astype(np.float32) / 255.0
                    # إذا كان يريد uint8 (الأكثر شيوعاً في نماذج TFLite المُحسّنة)، نتركها كما هي
                    elif target_dtype == np.uint8:
                        img_array = img_array.astype(np.uint8)
                    else:
                        # لحالات أخرى نادرة، نحاول المطابقة المباشرة
                        img_array = img_array.astype(target_dtype)

                    img_array = np.expand_dims(img_array, axis=0) # إضافة بُعد الدفعة (batch dimension)

                    # --- تشغيل التحليل ---
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
                    probs = tf.nn.softmax(output_data).numpy()

                    # --- تصنيف النتائج (المنطق الطبي) ---
                    malignant_ids = [1, 4, 17]
                    benign_ids = [2, 5, 23]
                    prob_malignant = sum([probs[i] for i in malignant_ids if i < len(probs)])
                    prob_benign = sum([probs[i] for i in benign_ids if i < len(probs)])

                    # تحديد النتيجة والرسائل
                    if prob_malignant >= 0.25:
                        res_msg, sub_msg, bg_color, txt_color = "🚨 اشتباه ورم خبيث", "تتطلب الحالة تقييماً طبياً فورياً.", "#fff1f0", "#cf1322"
                        advice_text = "التشخيص المبكر هو مفتاح الشفاء؛ يرجى زيارة المختص."
                    elif prob_benign >= 0.60:
                        res_msg, sub_msg, bg_color, txt_color = "🔍 ورم جلدي حميد", "الآفة تبدو سليمة وغير مقلقة حالياً.", "#f6ffed", "#389e0d"
                        advice_text = "راقب الحالة باستمرار واستشر الطبيب عند حدوث تغيرات."
                    else:
                        res_msg, sub_msg, bg_color, txt_color = "🩺 حالة جلدية عامة", "التحليل يرجح وجود حالة غير ورمية.", "#e6f7ff", "#096dd9"
                        advice_text = "قد تكون حساسية أو التهاباً بسيطاً."

                    # عرض التقرير
                    st.markdown(f"""
                        <div class="report-card" style="background-color: {bg_color}; border-color: {txt_color}; color: {txt_color};">
                            <p class="result-title">{res_msg}</p>
                            <p class="result-desc">{sub_msg}</p>
                        </div>
                        <div class="advice-box"><strong>💡 توصية:</strong> {advice_text}</div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"⚠️ خطأ فني أثناء التحليل. يرجى محاولة رفع صورة أخرى أو التحقق من جودة الإضاءة. \n[تفاصيل تقنية]: {e}")

    # --- 3. قسم الأسئلة الشائعة ---
    st.write("---")
    with st.expander("❓ الأسئلة الشائعة"):
        st.markdown("""
        **1. هل هذا التطبيق يعطي تشخيصاً نهائياً؟**
        لا، هذا النظام هو وسيلة فحص أولية تعتمد على الذكاء الاصطناعي، ويجب دائماً مراجعة الطبيب المختص للتشخيص النهائي.
        
        **2. كيف أحصل على أدق نتيجة؟**
        تأكد من تصوير المنطقة المصابة في ضوء النهار وان تكون الصورة واضحة.
        """)

else:
    st.error("❌ تعذر تحميل محرك التحليل. يرجى التحقق من إعدادات المجلد الأساسي.")

st.markdown("<br><hr><p style='text-align: center; color: #9e9e9e;'>نظام تقييم سلامة الجلد الذكي © 2026</p>", unsafe_allow_html=True)
