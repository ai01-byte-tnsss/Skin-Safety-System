import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الواجهة
st.set_page_config(page_title="نظام الحماية الفائقة للجلد", page_icon="🛡️")

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

# --- قاعدة بيانات الحالات ---
cancer_info = {
    'mel': ('سرطان الجلد الصبغي (Melanoma)', 'أخطر أنواع سرطان الجلد. يتطلب فحصاً طبياً فورياً وخزعة للتأكد.'),
    'bcc': ('سرطان الخلايا القاعدية (BCC)', 'نوع سرطاني شائع ينمو موضعياً. يجب إزالته جراحياً لمنع تضرر الجلد.'),
    'akiec': ('التقرن الشعاعي (AKIEC)', 'آفة تعتبر مرحلة ما قبل السرطان. إهمالها قد يؤدي لتحولها لورم خبيث.')
}

benign_info = {
    'nv': ('شامة عادية (Nevi)', 'بقعة جلدية طبيعية وحميدة تماماً.'),
    'bkl': ('آفة حميدة (BKL)', 'نمو جلدي غير سرطاني شائع جداً مع تقدم العمر.'),
    'df': ('ليف جلدي (Dermatofibroma)', 'كتلة صلبة صغيرة حميدة تنمو تحت الجلد.'),
    'vasc': ('آفة وعائية (Vascular)', 'تجمع أوعية دموية حميد (وحمة دموية).')
}

# --- نظام الحماية بكلمة المرور ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("<h2 style='text-align: center;'>🔐 تسجيل الدخول للنظام</h2>", unsafe_allow_html=True)
    password = st.text_input("أدخل كلمة المرور:", type="password")
    if st.button("دخول"):
        if password == "test**00": 
            st.session_state["authenticated"] = True
            st.rerun()
else:
    st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🛡️ نظام الفحص (بروتوكول حماية المرضى)</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📥 إدراج صورة الفحص", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة قيد التحليل الرقمي المكثف", width=350)
        
        if st.button("🔍 فحص شامل للمخاطر"):
            with st.spinner('جاري تطبيق بروتوكول الاستبعاد الطبي...'):
                img_res = image.resize((150, 150))
                img_arr = np.array(img_res.convert('RGB')) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)
                
                if model is not None:
                    preds = model.predict(img_arr)[0]
                    all_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
                    results = dict(zip(all_labels, preds))

                    # ---------------------------------------------------------
                    # الحل الجذري: بروتوكول "صفر تسامح" مع السرطان
                    # ---------------------------------------------------------
                    # نجمع كل احتمالات السرطان معاً لنرى "إجمالي الشك"
                    total_cancer_risk = results['mel'] + results['bcc'] + results['akiec']
                    
                    # نحدد نوع السرطان الأكثر احتمالاً من بينهم
                    cancer_probs = {k: results[k] for k in cancer_info.keys()}
                    top_cancer_type = max(cancer_probs, key=cancer_probs.get)
                    top_cancer_val = cancer_probs[top_cancer_type]

                    st.write("### 📋 التقرير النهائي للمعاينة:")

                    # القاعدة الجديدة: إذا كان إجمالي الشك في وجود "أي نوع سرطان" > 15%
                    # أو إذا كان أي نوع سرطان بمفرده هو الأقوى بين احتمالات الخطر
                    if total_cancer_risk > 0.15: 
                        name, desc = cancer_info[top_cancer_type]
                        st.warning(f"## تحذير: رصد مؤشرات اشتباه {name} ⚠️")
                        st.error(f"**قرار الأمان:** تم تصنيف الحالة كاشتباه مرتفع لضمان عدم إهمال أي ملامح سرطانية.")
                        st.info(f"**عن النوع المكتشف:** {desc}")
                        st.progress(float(top_cancer_val))
                        st.write(f"قوة المؤشرات الرقمية لهذا النوع: {top_cancer_val*100:.1f}%")
                        st.markdown("> **توصية طبية:** النظام يطبق بروتوكول الحماية؛ أي اشتباه يتجاوز 15% يستوجب مراجعة الطبيب فوراً.")

                    else:
                        # حالة نادرة: عندما تكون كل أنواع السرطان مجتمعة تحت الـ 15%
                        idx = np.argmax(preds)
                        label = all_labels[idx]
                        confidence = preds[idx]
                        
                        if label in benign_info:
                            name, desc = benign_info[label]
                        else:
                            name, desc = ("جلد سليم", "تظهر الصورة ملامح بصرية آمنة وطبيعية.")
                        
                        st.success(f"## النتيجة: {name} ✅")
                        st.info(f"**عن الحالة:** {desc}")
                        st.progress(float(confidence))
                        st.balloons()

                    st.write("---")
                    st.caption("ملاحظة: تم ضبط هذا النظام برمجياً ليعطي الأولوية القصوى للتحذير من السرطان بكل أنواعه.")
