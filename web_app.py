import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الواجهة
st.set_page_config(page_title="نظام الأمان القصوى للجلد", page_icon="🛡️")

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

# شروحات الحالات
cancer_info = {
    'mel': ('سرطان الجلد الصبغي (Melanoma)', 'أخطر أنواع سرطان الجلد، يتطلب تدخل طبي عاجل.'),
    'bcc': ('سرطان الخلايا القاعدية (BCC)', 'نوع ينمو ببطء ويجب إزالته جراحياً لمنع تضرر الأنسجة.'),
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
    st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🛡️ نظام الفحص (بروتوكول الاستبعاد الطبي)</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📥 إدراج صورة الفحص", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة تحت المجهر الرقمي", width=350)
        
        if st.button("🔍 فحص شامل"):
            with st.spinner('جاري تطبيق فحص الأمان...'):
                img_res = image.resize((150, 150))
                img_arr = np.array(img_res.convert('RGB')) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)
                
                if model is not None:
                    preds = model.predict(img_arr)[0]
                    all_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
                    results = dict(zip(all_labels, preds))

                    # ---------------------------------------------------------
                    # الحل الجذري: بروتوكول الاستبعاد (Rule-based Override)
                    # ---------------------------------------------------------
                    # سنبحث عن أعلى نسبة بين السرطانات
                    cancer_probs = {k: results[k] for k in cancer_info.keys()}
                    max_cancer_label = max(cancer_probs, key=cancer_probs.get)
                    max_cancer_val = cancer_probs[max_cancer_label]

                    # إذا اكتشف النظام أي مؤشر سرطان يتجاوز 15% فقط (عتبة حساسة جداً)
                    # وكان هذا المؤشر هو الأقوى بين احتمالات الخطر، سنعطيه الأولوية
                    if max_cancer_val > 0.15: 
                        name, desc = cancer_info[max_cancer_label]
                        st.warning(f"## تحذير: تم رصد مؤشرات اشتباه {name} ⚠️")
                        st.error(f"**قرار النظام:** إعطاء الأولوية للتحذير لوجود سمات بصرية مقلقة.")
                        st.info(f"**عن الحالة:** {desc}")
                        st.progress(float(max_cancer_val))
                        st.write(f"قوة المطابقة الرقمية: {max_cancer_val*100:.1f}%")
                        st.markdown("> **تنبيه:** تم تفعيل بروتوكول الأمان لضمان عدم إهمال أي اشتباه سرطاني.")

                    else:
                        # إذا كانت احتمالات السرطان شبه منعدمة (أقل من 15%)
                        idx = np.argmax(preds)
                        label = all_labels[idx]
                        confidence = preds[idx]
                        
                        if label in benign_info:
                            name, desc = benign_info[label]
                        else:
                            name, desc = ("جلد سليم", "الحالة تظهر خصائص بصرية آمنة تماماً.")
                        
                        st.success(f"## النتيجة: {name} ✅")
                        st.info(f"**عن الحالة:** {desc}")
                        st.progress(float(confidence))
                        st.balloons()

                    st.write("---")
                    st.caption("تنبيه: هذا النظام مصمم لتقليل الأخطاء الطبية عبر التحذير المبكر.")
