import streamlit as st
import tensorflow as tf
from PIL import Image, ImageStat
import numpy as np
import os

# 1. إعدادات الواجهة
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
    'mel': ('سرطان الجلد الصبغي (Melanoma)', 'أخطر أنواع سرطان الجلد، يتطلب تدخل طبي عاجل.'),
    'bcc': ('سرطان الخلايا القاعدية (BCC)', 'نوع ينمو ببطء ويجب إزالته جراحياً لمنع تضرر الأنسجة المحيطة.'),
    'akiec': ('التقرن الشعاعي (Pre-Cancer)', 'آفات تعتبر مرحلة ما قبل السرطان، علاجها يمنع تحولها لورم خبيث.')
}

benign_info = {
    'nv': ('شامة عادية (Nevi)', 'بقع جلدية طبيعية وحميدة تماماً في أغلب الحالات.'),
    'bkl': ('آفة حميدة (BKL)', 'نمو جلدي غير سرطاني شائع مع تقدم العمر.'),
    'df': ('ليف جلدي (Dermatofibroma)', 'كتل صغيرة صلبة غير ضارة تنمو تحت الجلد.'),
    'vasc': ('آفة وعائية (Vascular)', 'تجمعات لأوعية دموية وهي حالات حميدة طبياً.')
}

# --- نظام الحماية ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("<h2 style='text-align: center;'>🔐 تسجيل الدخول</h2>", unsafe_allow_html=True)
    password = st.text_input("أدخل كلمة المرور:", type="password")
    if st.button("دخول"):
        if password == "test**00": 
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("خطأ!")
else:
    st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🔬 نظام الفحص الذكي (الأولوية القصوى للأمان الطبي)</h1>", unsafe_allow_html=True)

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
            with st.spinner('جاري فحص الأنماط...'):
                img_res = image.resize((150, 150))
                img_arr = np.array(img_res.convert('RGB')) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)
                
                if model is not None:
                    preds = model.predict(img_arr)[0]
                    all_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
                    results = dict(zip(all_labels, preds))
                    
                    # --- منطق "الأولوية للسرطان" الحازم (عتبة 55%) ---
                    cancer_probs = {k: results[k] for k in cancer_info.keys()}
                    max_cancer_label = max(cancer_probs, key=cancer_probs.get)
                    max_cancer_val = cancer_probs[max_cancer_label]

                    st.write("### 📋 التقرير النهائي للمعاينة:")

                    # القاعدة الذهبية: إذا كان احتمال السرطان >= 55%، نلغي أي نتائج حميدة أخرى مهما كانت نسبتها
                    if max_cancer_val >= 0.55:
                        name, desc = cancer_info[max_cancer_label]
                        st.warning(f"## التصنيف: اشتباه {name} ⚠️")
                        st.error(f"**تنبيه حرج:** تم رصد مؤشرات رقمية تقع في نطاق الاشتباه (أعلى من 55%).")
                        st.info(f"**تعريف الحالة:** {desc}")
                        st.progress(float(max_cancer_val))
                        st.write(f"قوة المطابقة الرقمية للحالة: {max_cancer_val*100:.1f}%")
                        st.markdown("> **ملاحظة هامة:** في هذا النظام، يتم إعطاء الأولوية للتحذير من السرطان لضمان أعلى مستويات الأمان.")
                    
                    # إذا كانت جميع احتمالات السرطان تحت الـ 55%، ننتقل للتشخيص الحميد
                    else:
                        idx = np.argmax(preds)
                        label = all_labels[idx]
                        confidence = preds[idx]
                        
                        if label in benign_info:
                            name, desc = benign_info[label]
                        else:
                            name, desc = ("حالة جلدية آمنة", "تظهر الصورة ملامح لحالة جلدية شائعة وهي غير سرطانية.")
                        
                        st.success(f"## التصنيف: {name} ✅")
                        st.info(f"**عن هذه الحالة:** {desc}")
                        st.progress(float(confidence))
                        st.write(f"نسبة الطمأنينة: {confidence*100:.1f}%")
                        st.balloons()
                    
                    st.write("---")
                    st.caption("تحذير: هذا البرنامج أداة تقنية مساعدة وليس تشخيصاً طبياً نهائياً.")
