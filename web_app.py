import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. الإعدادات المتقدمة للواجهة ---
st.set_page_config(page_title="Advanced Skin Analysis System", layout="centered")

st.markdown("""
<style>
    .report-card { padding: 35px; border-radius: 25px; text-align: center; margin-top: 20px; border: 4px solid; box-shadow: 0px 10px 30px rgba(0,0,0,0.1); }
    .result-title { font-size: 34px; font-weight: bold; margin-bottom: 15px; }
    .result-desc { font-size: 21px; font-weight: 500; line-height: 1.7; }
    .advice-box { background-color: #ffffff; padding: 25px; border-radius: 15px; margin-top: 25px; border-right: 10px solid #455a64; border-left: 1px solid #eee; border-top: 1px solid #eee; border-bottom: 1px solid #eee; }
    .quality-alert { background-color: #fffbe6; border: 1px solid #ffe58f; padding: 15px; border-radius: 12px; color: #856404; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# --- 2. محرك تحميل النموذج الذكي ---
@st.cache_resource
def load_expert_engine():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ عطل في المحرك الأساسي: {e}")
        return None

interpreter = load_expert_engine()

if interpreter:
    # استخراج بصمة النموذج التقنية
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    target_dtype = input_details[0]['dtype']
    input_shape = input_details[0]['shape'][1:3]

    st.markdown("<h1 style='text-align: center; color: #0d47a1;'>🛡️ الكشف عن سلامة الجلد لمرض سرطان</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="quality-alert">
            💡 <b>بروتوكول الفحص:</b> تأكد من رفع صورة قريبة (Macro) للآفة الجلدية، مع إضاءة موزعة جيداً لضمان دقة التصنيف بين الحالات الحميدة والخبيثة.
        </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 فحص حالة جديدة"):
        st.rerun()

    uploaded_file = st.file_uploader("📥 ارفع الصورة المجهرية للآفة", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="الصورة قيد التحليل")

        if st.button("🚀 تحليل الأنماط الحيوية"):
            with st.spinner("جاري إجراء المسح المجهري والموازنة الإحصائية..."):
                try:
                    # --- المرحلة 1: المعالجة فائقة الدقة ---
                    img = image.convert("RGB").resize(input_shape)
                    img_array = np.array(img)
                    
                    # مطابقة DataType (حل نهائي لـ ValueError)
                    if target_dtype == np.float32:
                        img_array = img_array.astype(np.float32) / 255.0
                    else:
                        img_array = img_array.astype(target_dtype)
                    
                    img_array = np.expand_dims(img_array, axis=0)

                    # --- المرحلة 2: استخلاص النتائج وتطبيق Softmax ---
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    raw_output = interpreter.get_tensor(output_details[0]['index'])[0]
                    probs = tf.nn.softmax(raw_output).numpy()

                    # --- المرحلة 3: نظام الفئات الذكي (Smart Grouping) ---
                    # الأكواد المحددة بناءً على متطلباتك
                    malig_idx = [1, 4, 17]
                    benign_idx = [2, 5, 23]
                    
                    p_malig = sum([probs[i] for i in malig_idx if i < len(probs)])
                    p_benign = sum([probs[i] for i in benign_idx if i < len(probs)])
                    p_general = 1.0 - (p_malig + p_benign)

                    # --- المرحلة 4: منطق اتخاذ القرار "الاحترافي" (The Expert Logic) ---
                    # يعتمد هذا المنطق على "الأولوية الأمنية" + "قوة الدليل البصري"
                    
                    # أولاً: التحقق من وجود "خطر حقيقي" (عتبة أمان موزونة)
                    if p_malig >= 0.35 or (p_malig > p_benign and p_malig > 0.20):
                        res_msg, sub_msg = "🚨 اشتباه ورم خبيث", "تم رصد مؤشرات حيوية غير منتظمة تتطلب تدخلاً طبياً عاجلاً."
                        bg_c, txt_c = "#fff1f0", "#cf1322"
                        advice = "يُنصح بشدة بمراجعة طبيب اختصاص الجلدية فوراً. التشخيص المبكر هو مفتاح السلامة."
                    
                    # ثانياً: التحقق من "الحالة الحميدة" (يجب أن تكون الأقوى بوضوح)
                    elif p_benign > p_malig and p_benign > p_general:
                        res_msg, sub_msg = "🔍 ورم جلدي حميد", "تشير التحليلات الرقمية إلى أن الآفة من النوع السليم حالياً."
                        bg_c, txt_c = "#f6ffed", "#389e0d"
                        advice = "الحالة تبدو مستقرة. يفضل مراقبة أي تغير مفاجئ في الحواف أو اللون."
                    
                    # ثالثاً: الحالة العامة (الالتهابات، الحساسية، إلخ)
                    else:
                        res_msg, sub_msg = "🩺 حالة جلدية عامة", "التحليل يرجح وجود نمط جلدي غير ورمي (مثل الالتهاب أو الحساسية)."
                        bg_c, txt_c = "#e6f7ff", "#096dd9"
                        advice = "هذه الأعراض شائعة في العديد من الحالات الجلدية البسيطة. استشر الطبيب لوصف العلاج الموضعي."

                    # --- المرحلة 5: عرض التقرير النهائي (خالي من النسب) ---
                    st.markdown(f"""
                        <div class="report-card" style="background-color: {bg_c}; border-color: {txt_c}; color: {txt_c};">
                            <p class="result-title">{res_msg}</p>
                            <p class="result-desc">{sub_msg}</p>
                        </div>
                        <div class="advice-box">
                            <p style="font-size: 20px; font-weight: bold; color: #263238; margin-bottom: 8px;">💡 التوصية الطبية:</p>
                            <p class="result-desc" style="color: #455a64;">{advice}</p>
                        </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"⚠️ فشل في إتمام الفحص: {e}")

    # تذييل الصفحة الاحترافي
    st.write("---")
    st.markdown("<p style='text-align: center; color: #9e9e9e;'>نظام تقييم سلامة الجلد الذكي - كافة الحقوق محفوظة © 2026</p>", unsafe_allow_html=True)

else:
    st.error("❌ تعذر تحميل نظام الفحص.")
