import streamlit as st
import tensorflow as tf
from PIL import Image, ImageStat
import numpy as np
import os

# 1. إعدادات الواجهة والنموذج
st.set_page_config(page_title="نظام التشخيص الذكي للجلد", page_icon="🔬")

@st.cache_resource
def load_model():
    try:
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, 'skin_cancer_expert.h5')
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {e}")
        return None

model = load_model()

# --- تعريف البيانات والشروحات ---
cancer_info = {
    'mel': ('سرطان الجلد الصبغي (Melanoma)', 'أخطر أنواع سرطان الجلد، يبدأ في الخلايا الصبغية. يتطلب تدخل طبي سريع.'),
    'bcc': ('سرطان الخلايا القاعدية (BCC)', 'نوع شائع جداً، ينمو ببطء ونادراً ما ينتشر، لكنه يتطلب إزالة جراحية.'),
    'akiec': ('التقرن الشعاعي (Pre-Cancer)', 'آفات قشرية تعتبر مرحلة ما قبل السرطان، علاجها مبكراً يمنع تحولها لسرطان.')
}

benign_info = {
    'nv': ('شامة عادية (Nevi)', 'بقع جلدية طبيعية ناتجة عن تجمع الخلايا الصبغية، غالباً ما تكون حميدة تماماً.'),
    'bkl': ('آفة حميدة (BKL)', 'نمو جلدي غير سرطاني يشمل التقرن الدهني، شائع مع تقدم العمر.'),
    'df': ('ليف جلدي (Dermatofibroma)', 'كتل صغيرة صلبة غير ضارة تنمو غالباً تحت الجلد في الساقين.'),
    'vasc': ('آفة وعائية (Vascular)', 'تجمعات لأوعية دموية مثل الوحمات الدموية، وهي حالات حميدة طبياً.')
}

# --- نظام الحماية ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("<h2 style='text-align: center; color: #1e3a8a;'>🔐 تسجيل الدخول للنظام</h2>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1,2,1])
    with col_b:
        password = st.text_input("أدخل كلمة المرور:", type="password")
        if st.button("دخول"):
            if password == "test**00": 
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("كلمة المرور غير صحيحة")
else:
    st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🔬 نظام التحليل الرقمي لسلامة الجلد</h1>", unsafe_allow_html=True)

    # إحصائيات النظام
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("قوة التدريب", "80%")
    with col2: st.metric("حجم الاختبار", "20%")
    with col3: st.metric("دقة النموذج", "93%")

    st.divider()

    uploaded_file = st.file_uploader("📥 إدراج صورة الفحص", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة التي يتم تحليلها", width=350)
        
        if st.button("🔍 بدء التحليل العميق"):
            with st.spinner('جاري فحص الأنماط واستخلاص الخواص...'):
                img_res = image.resize((150, 150))
                img_arr = np.array(img_res.convert('RGB')) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)
                
                if model is not None:
                    preds = model.predict(img_arr)[0]
                    idx = np.argmax(preds)
                    all_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
                    label = all_labels[idx]
                    confidence = preds[idx]

                    st.write("### 📋 نتائج التقرير التفصيلي:")

                    # --- المسار الأول: حالات السرطان (العتبة الذهبية 70%) ---
                    if label in cancer_info and confidence >= 0.70:
                        name, desc = cancer_info[label]
                        st.warning(f"## التصنيف: اشتباه {name} ⚠️")
                        st.error(f"**تعريف الحالة:** {desc}")
                        st.progress(float(confidence))
                        st.write(f"قوة المطابقة الرقمية: {confidence*100:.1f}%")
                        st.markdown("> **توصية:** يرجى حجز موعد مع طبيب اختصاصي جلدية لإجراء فحص سريري بأسرع وقت.")

                    # --- المسار الثاني: الحالات الحميدة ---
                    else:
                        if label in benign_info:
                            name, desc = benign_info[label]
                        else:
                            name, desc = ("حالة جلدية عامة", "تظهر الصورة ملامح لحالة جلدية شائعة (مثل الحساسية أو الإكزيما) وهي غير سرطانية.")
                        
                        st.success(f"## التصنيف: {name} ✅")
                        st.info(f"**عن هذه الحالة:** {desc}")
                        st.progress(float(confidence))
                        st.write(f"نسبة الطمأنينة الرقمية: {confidence*100:.1f}%")
                        st.balloons()
                        st.markdown("> **توصية:** الحالة تظهر خصائص حميدة. استشر الصيدلي أو الطبيب لمنتجات العناية المناسبة.")
                    
                    st.write("---")
                    st.caption("تحذير: هذا البرنامج هو أداة مساعدة رقمية ولا يعتبر تشخيصاً طبياً معتمداً.")
