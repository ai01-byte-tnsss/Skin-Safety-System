import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. إعدادات الواجهة الاحترافية ---
st.set_page_config(page_title="Skin Safety System 2026", layout="centered")

st.markdown("""
<style>
    /* تصميم بطاقة النتائج */
    .report-card { padding: 30px; border-radius: 20px; text-align: center; margin-top: 25px; border: 3px solid; }
    .result-title { font-size: 32px; font-weight: bold; margin-bottom: 12px; }
    .result-desc { font-size: 19px; font-weight: 500; line-height: 1.6; }
    
    /* تصميم صندوق النصيحة */
    .advice-box { background-color: #fcfcfc; padding: 20px; border-radius: 12px; margin-top: 25px; border-right: 6px solid #455a64; box-shadow: 0px 2px 10px rgba(0,0,0,0.05); }
    .advice-title { color: #263238; font-size: 20px; font-weight: bold; margin-bottom: 8px; }
    
    /* تنبيه الجودة */
    .quality-alert { background-color: #fffbe6; border: 1px solid #ffe58f; padding: 15px; border-radius: 8px; color: #856404; font-size: 14px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 2. محرك تحميل وتجهيز النموذج ---
@st.cache_resource
def load_expert_model():
    try:
        # تأكد من أن اسم الملف يطابق الملف المرفوع في مشروعك
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"⚠️ فشل في تحميل المحرك التقني: {e}")
        return None

interpreter = load_expert_model()

if interpreter:
    # جلب متطلبات المدخلات من النموذج مباشرة لمنع أخطاء DataType
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    target_dtype = input_details[0]['dtype']
    input_shape = input_details[0]['shape'][1:3] # الحجم المطلوب (مثل 224x224)

    # الواجهة الرئيسية
    st.markdown("<h1 style='text-align: center; color: #0d47a1;'>🛡️ نظام الكشف عن سلامة الجلد</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: center; font-size: 1.1em;'>نظام ذكي متطور لتقييم الحالات الجلدية والاشتباه السرطاني</p>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="quality-alert">
            ⚠️ <b>تنبيه الجودة:</b> دقة التحليل تعتمد بشكل كبير على جودة الصورة. يرجى التأكد من توفر <b>إضاءة جيدة</b>، ووضوح <b>التركيز</b>، وعدم وجود ظلال.
        </div>
    """, unsafe_allow_html=True)

    # زر إعادة الضبط
    if st.button("🔄 فحص حالة جديدة"):
        st.rerun()

    uploaded_file = st.file_uploader("📥 قم برفع صورة واضحة للمنطقة المراد فحصها", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        if st.button("🚀 بدء التحليل الدقيق"):
            with st.spinner("جاري تحليل الخصائص الحيوية للصورة..."):
                try:
                    # --- [المعالجة المسبقة الاحترافية] ---
                    # 1. تغيير الحجم والمطابقة
                    img = image.convert("RGB").resize(input_shape)
                    img_array = np.array(img)
                    
                    # 2. حل مشكلة ValueError بمطابقة نوع البيانات تلقائياً
                    if target_dtype == np.float32:
                        img_array = img_array.astype(np.float32) / 255.0
                    else:
                        img_array = img_array.astype(target_dtype)

                    img_array = np.expand_dims(img_array, axis=0)

                    # 3. تشغيل الفحص
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                    
                    # تحويل المخرجات إلى احتمالات (داخلياً فقط)
                    probs = tf.nn.softmax(output_data).numpy()

                    # --- [نظام الفرز الثلاثي المستقل] ---
                    # تحديد الأكواد بناءً على النموذج (عدل الأرقام إذا لزم الأمر)
                    malignant_indices = [1, 4, 17] 
                    benign_indices = [2, 5, 23]
                    
                    p_malig = sum([probs[i] for i in malignant_indices if i < len(probs)])
                    p_benign = sum([probs[i] for i in benign_indices if i < len(probs)])

                    # --- منطق اتخاذ القرار (بدون عرض نسب مئوية) ---
                    # الأولوية 1: الاشتباه الخبيث (عتبة الأمان 25%)
                    if p_malig >= 0.25:
                        res_msg = "🚨 النتيجة: اشتباه ورم خبيث"
                        sub_msg = "تم رصد أنماط خلوية غير منتظمة تتطلب تقييماً طبياً متخصصاً."
                        bg_color, txt_color = "#fff1f0", "#cf1322"
                        advice = "يُنصح بمراجعة طبيب المختص فوراً لإجراء فحص سريري دقيق."
                    
                    # الأولوية 2: الورم الحميد (عتبة الثقة 60%)
                    elif p_benign >= 0.60:
                        res_msg = "🔍 النتيجة: ورم جلدي حميد"
                        sub_msg = "تشير التحليلات إلى أن هذه الآفة من النوع السليم وغير المقلق حالياً."
                        bg_color, txt_color = "#f6ffed", "#389e0d"
                        advice = "الحالة مستقرة، يفضل مراقبة أي تغير مفاجئ في الشكل أو اللون."
                    
                    # الأولوية 3: حالة جلدية عامة
                    else:
                        res_msg = "🩺 النتيجة: حالة جلدية عامة"
                        sub_msg = "التحليل يرجح وجود نمط جلدي غير ورمي (التهاب أو حساسية)."
                        bg_color, txt_color = "#e6f7ff", "#096dd9"
                        advice = "هذه الأعراض شائعة؛ استشر طبيب الأسرة لوصف العلاج الموضعي المناسب."

                    # عرض التقرير النهائي
                    st.markdown(f"""
                        <div class="report-card" style="background-color: {bg_color}; border-color: {txt_color}; color: {txt_color};">
                            <p class="result-title">{res_msg}</p>
                            <p class="result-desc">{sub_msg}</p>
                        </div>
                        <div class="advice-box">
                            <div class="advice-title">💡 توصية النظام:</div>
                            <div class="result-desc" style="color: #455a64;">{advice}</div>
                        </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"⚠️ حدث خطأ تقني أثناء الفحص: {e}")

    # قسم الأسئلة الشائعة
    st.write("---")
    with st.expander("❓ الأسئلة الشائعة"):
        st.markdown("""
        **- هل النتائج نهائية؟** لا، هذا فحص أولي بالذكاء الاصطناعي للمساعدة والفرز فقط.
        **- ماذا أفعل عند الاشتباه؟** يجب التوجه فوراً لطبيب جلدية مختص للقيام بالخزعة أو الفحص السريري.
        """)

else:
    st.error("❌ تعذر تهيئة نظام الفحص. تأكد من وجود ملف النموذج في المسار الصحيح.")

# التذييل الاحترافي
st.markdown("<br><hr><p style='text-align: center; color: #9e9e9e;'>نظام تقييم سلامة الجلد الذكي - كافة الحقوق محفوظة © 2026</p>", unsafe_allow_html=True)
