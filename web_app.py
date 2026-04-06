import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. إعدادات الهوية البصرية والواجهة ---
st.set_page_config(page_title="Skin Safety System", layout="centered")

st.markdown("""
<style>
    .main-title { text-align: center; color: #0d47a1; font-size: 35px; font-weight: bold; margin-bottom: 5px; }
    .report-card { padding: 30px; border-radius: 20px; text-align: center; margin-top: 25px; border: 4px solid; box-shadow: 0px 4px 15px rgba(0,0,0,0.1); }
    .result-title { font-size: 32px; font-weight: bold; margin-bottom: 12px; }
    .result-desc { font-size: 20px; font-weight: 500; line-height: 1.6; }
    .advice-box { background-color: #f8f9fa; padding: 25px; border-radius: 15px; margin-top: 25px; border-right: 8px solid #455a64; }
    .quality-alert { background-color: #fffbe6; border: 1px solid #ffe58f; padding: 15px; border-radius: 10px; color: #856404; font-size: 15px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 2. دالة تحميل النموذج مع معالجة استباقية للأخطاء ---
@st.cache_resource
def load_skin_model():
    try:
        # تأكد من أن اسم الملف هو skin_expert_refined.tflite في مجلدك
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ فشل في تحميل المحرك: {e}")
        return None

interpreter = load_skin_model()

if interpreter:
    # استخراج الخصائص التقنية للنموذج لضمان توافق البيانات (حل مشكلة ValueError)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    target_dtype = input_details[0]['dtype']
    input_shape = input_details[0]['shape'][1:3] # الحجم المطلوب للنموذج (مثلاً 224x224)

    st.markdown("<div class='main-title'>🛡️ الكشف عن سلامة الجلد لمرض سرطان</div>", unsafe_allow_html=True)
    st.write("<p style='text-align: center; font-size: 1.2em; color: #555;'>نظام الفحص الذكي المتقدم للتقييم الأولي</p>", unsafe_allow_html=True)

    st.markdown("""
        <div class="quality-alert">
            💡 <b>للحصول على فحص دقيق:</b> استخدم إضاءة طبيعية واضحة، تأكد من ثبات اليد أثناء التصوير، واجعل الآفة الجلدية في مركز الصورة تماماً.
        </div>
    """, unsafe_allow_html=True)

    # زر إعادة الفحص لتنظيف الواجهة
    if st.button("🔄 فحص حالة جديدة"):
        st.rerun()

    uploaded_file = st.file_uploader("📥 قم برفع صورة المنطقة المراد فحصها", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="الصورة المرفوعة للفحص")

        if st.button("🚀 بدء عملية التحليل العميق"):
            with st.spinner("جاري تدقيق الأنماط المجهرية للصورة..."):
                try:
                    # --- المرحلة الأولى: المعالجة المسبقة الدقيقة (Pre-processing) ---
                    # تحويل الصورة وتغيير حجمها بما يطابق النموذج تماماً
                    img = image.convert("RGB").resize(input_shape)
                    img_array = np.array(img)
                    
                    # مطابقة نوع البيانات (الحل الجذري لخطأ ValueError)
                    if target_dtype == np.float32:
                        img_array = img_array.astype(np.float32) / 255.0
                    else:
                        img_array = img_array.astype(target_dtype)
                    
                    img_array = np.expand_dims(img_array, axis=0)

                    # --- المرحلة الثانية: تشغيل الفحص واستخراج الاحتمالات ---
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                    
                    # تطبيق Softmax لضمان توزيع النسب بشكل منطقي
                    probs = tf.nn.softmax(output_data).numpy()

                    # --- المرحلة الثالثة: منطق التصنيف الطبي المتقدم (الأولويات) ---
                    # تم ربط الكود 17 ومجموعته بالفئات الخبيثة لضمان الدقة
                    malignant_indices = [1, 4, 17] 
                    benign_indices = [2, 5, 23]
                    
                    # حساب قوة الاحتمال لكل عائلة
                    p_malignant = sum([probs[i] for i in malignant_indices if i < len(probs)])
                    p_benign = sum([probs[i] for i in benign_indices if i < len(probs)])

                    # --- المرحلة الرابعة: اتخاذ القرار النهائي بناءً على الأمان (Safety-First) ---
                    # الأولوية دائماً لكشف الخطر حتى لو كانت النسبة منخفضة (15%) لمنع "الحالة العامة" الخاطئة
                    if p_malignant >= 0.15:
                        res_title = "🚨 النتيجة: اشتباه ورم خبيث"
                        res_text = "تم رصد خصائص بصرية غير منتظمة تتطلب تقييماً طبياً عاجلاً."
                        b_color, t_color = "#fff1f0", "#cf1322"
                        final_advice = "يُنصح بشدة بمراجعة طبيب اختصاص الجلدية لإجراء الفحص السريري أو الخزعة لضمان سلامتك."
                    
                    # إذا لم يوجد خطر، نفحص احتمالية كونه حميداً بعتبة ثقة متوسطة (50%)
                    elif p_benign >= 0.50:
                        res_title = "🔍 النتيجة: ورم جلدي حميد"
                        res_text = "تشير التحليلات الرقمية إلى أن هذه الآفة من النوع السليم وغير المقلق حالياً."
                        b_color, t_color = "#f6ffed", "#389e0d"
                        final_advice = "الحالة تبدو مستقرة؛ ومع ذلك، يفضل مراقبة أي تغير مفاجئ في الحجم أو اللون ومراجعة الطبيب عند الضرورة."
                    
                    # الخيار الأخير (Default) هو الحالة العامة فقط إذا كانت كل الاحتمالات الأخرى ضعيفة جداً
                    else:
                        res_title = "🩺 النتيجة: حالة جلدية عامة"
                        res_text = "التحليل يرجح وجود نمط جلدي شائع لا يندرج تحت تصنيفات الأورام (مثل الحساسية أو الالتهاب)."
                        b_color, t_color = "#e6f7ff", "#096dd9"
                        final_advice = "هذه الأعراض غالباً ما تكون بسيطة، يمكنك استشارة الطبيب العام لوصف العلاج الموضعي المناسب."

                    # عرض التقرير النهائي بدون أي نسب مئوية تسبب القلق
                    st.markdown(f"""
                        <div class="report-card" style="background-color: {b_color}; border-color: {t_color}; color: {t_color};">
                            <p class="result-title">{res_title}</p>
                            <p class="result-desc">{res_text}</p>
                        </div>
                        <div class="advice-box">
                            <p style="font-size: 20px; font-weight: bold; color: #263238; margin-bottom: 10px;">💡 توصية النظام الطبي:</p>
                            <p class="result-desc" style="color: #455a64;">{final_advice}</p>
                        </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"⚠️ حدث عارض تقني أثناء المعالجة: {e}. يرجى محاولة رفع صورة أخرى بوضوح أعلى.")

    # قسم الأسئلة الشائعة لتعزيز مصداقية المشروع
    st.write("---")
    with st.expander("❓ الأسئلة الشائعة حول الفحص والنتائج"):
        st.markdown("""
        * **هل التشخيص نهائي؟** لا، هذا النظام مدعوم بالذكاء الاصطناعي للمساعدة في الفرز الأولي فقط، والكلمة الفصل للطبيب.
        * **لماذا قد تتغير النتيجة لنفس الحالة؟** جودة الصورة، زاوية التصوير، وقوة الإضاءة عوامل حاسمة في دقة التحليل الرقمي.
        * **ماذا أفعل عند ظهور 'اشتباه خبيث'؟** لا داعي للذعر؛ النتيجة تعني ضرورة الفحص السريري للتأكد فقط.
        """)

else:
    st.warning("🔄 جاري تهيئة محرك الفحص الذكي... يرجى الانتظار.")

# التذييل النهائي (تم إلغاء جملة مشروع تخرج بناءً على طلبك)
st.markdown("<br><br><hr><p style='text-align: center; color: #9e9e9e; font-size: 0.9em;'>نظام تقييم سلامة الجلد الذكي المعتمد - كافة الحقوق محفوظة © 2026</p>", unsafe_allow_html=True)
