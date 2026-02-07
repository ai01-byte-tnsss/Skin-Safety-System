import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الواجهة
st.set_page_config(page_title="نظام التشخيص الدقيق للجلد", page_icon="🛡️")

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

# --- قاعدة بيانات الحالات مع الشروحات المفصلة لكل نوع ---
cancer_info = {
    'mel': ('سرطان الجلد الصبغي (Melanoma)', 'يظهر غالباً كشامة غير منتظمة الشكل أو متغيرة اللون. هو النوع الأكثر خطورة ويتطلب فحصاً طبياً عاجلاً.'),
    'bcc': ('سرطان الخلايا القاعدية (BCC)', 'يظهر غالباً كبقعة لؤلؤية أو وردية لامعة. ينمو ببطء ولكن يجب إزالته جراحياً لحماية الأنسجة.'),
    'akiec': ('التقرن الشعاعي (AKIEC)', 'يظهر كبقع قشرية خشنة. يعتبر مرحلة ما قبل السرطان ويجب علاجه لمنع تطوره لورم خبيث.')
}

benign_info = {
    'nv': ('شامة عادية (Nevi)', 'بقعة جلدية طبيعية متناسقة الشكل واللون، وهي حميدة تماماً.'),
    'bkl': ('آفة حميدة (BKL)', 'نمو جلدي غير سرطاني، يشمل التقرن الدهني الذي يظهر مع تقدم العمر.'),
    'df': ('ليف جلدي (Dermatofibroma)', 'كتلة صلبة صغيرة تحت الجلد، غالباً ما تكون نتيجة إصابة بسيطة سابقة.'),
    'vasc': ('آفة وعائية (Vascular)', 'تجمع أوعية دموية حميد مثل الشامات الدموية.')
}

# --- نظام الدخول ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("<h2 style='text-align: center;'>🔐 تسجيل الدخول</h2>", unsafe_allow_html=True)
    password = st.text_input("كلمة المرور:", type="password")
    if st.button("دخول"):
        if password == "test**00": 
            st.session_state["authenticated"] = True
            st.rerun()
else:
    st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🔬 الفحص الرقمي الدقيق لسلامة الجلد</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📥 إدراج صورة الفحص", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة قيد المعالجة التشخيصية", width=350)
        
        if st.button("🔍 تحليل نوع الإصابة"):
            with st.spinner('جاري تدقيق نوع الخلايا وتحديد المسمى الطبي...'):
                img_res = image.resize((150, 150))
                img_arr = np.array(img_res.convert('RGB')) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)
                
                if model is not None:
                    preds = model.predict(img_arr)[0]
                    all_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
                    results = dict(zip(all_labels, preds))

                    # 1. حساب إجمالي خطر السرطان (لضمان عدم التغاضي عن أي حالة)
                    total_cancer_risk = results['mel'] + results['bcc'] + results['akiec']
                    
                    # 2. تحديد نوع السرطان "الأكثر دقة" من بين الاحتمالات السرطانية
                    cancer_candidates = {k: results[k] for k in cancer_info.keys()}
                    exact_cancer_type = max(cancer_candidates, key=cancer_candidates.get)
                    exact_val = cancer_candidates[exact_cancer_type]

                    st.write("### 📋 التقرير التشخيصي النهائي:")

                    # منطق القرار: إذا كان هناك خطر إجمالي، نحدد الاسم الدقيق لنوع السرطان
                    if total_cancer_risk > 0.15: 
                        name, desc = cancer_info[exact_cancer_type]
                        st.warning(f"## الحالة المكتشفة: {name} ⚠️")
                        st.error(f"**التشخيص الرقمي:** تم تحديد ملامح بصرية تطابق نوع ({exact_cancer_type.upper()}).")
                        st.info(f"**معلومات عن هذا النوع:** {desc}")
                        st.progress(float(exact_val))
                        st.write(f"دقة المطابقة لهذا النوع تحديداً: {exact_val*100:.1f}%")
                        st.markdown("> **توصية:** يجب عرض هذا التقرير على طبيب مختص للفحص السريري.")

                    else:
                        # إذا كان المسار آمناً تماماً
                        idx = np.argmax(preds)
                        label = all_labels[idx]
                        confidence = preds[idx]
                        
                        if label in benign_info:
                            name, desc = benign_info[label]
                        else:
                            name, desc = ("بنية جلدية سليمة", "لا توجد مؤشرات بصرية لأي آفات مقلقة.")
                        
                        st.success(f"## النتيجة: {name} ✅")
                        st.info(f"**عن الحالة:** {desc}")
                        st.progress(float(confidence))
                        st.balloons()

                    st.write("---")
                    st.caption("ملاحظة: هذا النظام يعتمد على تحليل الأنماط الرقمية لزيادة دقة الكشف المبكر.")

