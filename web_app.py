import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الواجهة
st.set_page_config(page_title="نظام التشخيص الذكي الشامل", page_icon="🛡️")

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
    'mel': ('سرطان الجلد الصبغي (Melanoma)', 'أخطر أنواع سرطان الجلد، يتطلب تدخل طبي عاجل.'),
    'bcc': ('سرطان الخلايا القاعدية (BCC)', 'نوع سرطاني شائع ينمو موضعياً ويجب إزالته جراحياً.'),
    'akiec': ('التقرن الشعاعي (AKIEC)', 'آفات تعتبر ما قبل سرطانية، علاجها يمنع تحولها لورم خبيث.')
}

benign_info = {
    'nv': ('شامة عادية (Nevi)', 'بقعة جلدية طبيعية وحميدة تماماً.'),
    'bkl': ('آفة حميدة (BKL)', 'نمو جلدي غير سرطاني شائع مع تقدم العمر.'),
    'df': ('ليف جلدي (Dermatofibroma)', 'كتلة صلبة صغيرة حميدة تنمو تحت الجلد.'),
    'vasc': ('آفة وعائية (Vascular)', 'تجمع أوعية دموية حميد (وحمة دموية).')
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
    st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🛡️ نظام الفحص الشامل لسلامة الجلد</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📥 إدراج صورة الفحص (سرطان، التهابات، أو آفات أخرى)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة قيد التحليل الرقمي", width=350)
        
        if st.button("🔍 تحليل الحالة"):
            with st.spinner('جاري فحص الأنماط ومقارنتها بقاعدة البيانات...'):
                img_res = image.resize((150, 150))
                img_arr = np.array(img_res.convert('RGB')) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)
                
                if model is not None:
                    preds = model.predict(img_arr)[0]
                    all_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
                    results = dict(zip(all_labels, preds))

                    # 1. منطق تحليل الحالات خارج النطاق (مثل حب الشباب أو أنواع سرطان نادرة)
                    total_cancer_risk = results['mel'] + results['bcc'] + results['akiec']
                    max_confidence = np.max(preds) 
                    
                    st.write("### 📋 التقرير التشخيصي:")

                    # أ- إذا كان اليقين العام منخفضاً جداً (حالة غير معروفة للنظام)
                    if max_confidence < 0.35:
                        st.info("## النتيجة: حالة غير نمطية أو غير معروفة ⚠️")
                        st.warning("تحليل الأنماط يشير إلى أن هذه الحالة قد تكون خارج تخصص النموذج الأساسي (مثل حب الشباب، ساركوما، أو التهاب جلدي حاد).")
                        st.error("توصية: الأشكال غير المنتظمة تستوجب فحصاً سريرياً فورياً لاستبعاد أنواع السرطان النادرة.")

                    # ب- إذا وجد النظام أي مؤشر خطر (حتى لو الصورة غير واضحة)
                    elif total_cancer_risk > 0.15: 
                        cancer_candidates = {k: results[k] for k in cancer_info.keys()}
                        top_cancer = max(cancer_candidates, key=cancer_candidates.get)
                        
                        st.warning(f"## اشتباه إصابة: {cancer_info[top_cancer][0]} ⚠️")
                        st.error(f"تنبيه: تم رصد خصائص بصرية تتقاطع مع معايير الخطر الرقمية.")
                        st.info(f"وصف الحالة المحتملة: {cancer_info[top_cancer][1]}")
                        st.progress(float(results[top_cancer]))
                        st.write(f"قوة المطابقة مع هذا النوع: {results[top_cancer]*100:.1f}%")

                    # ج- الحالات التي يثق النظام أنها حميدة
                    else:
                        idx = np.argmax(preds)
                        label = all_labels[idx]
                        if label in benign_info:
                            st.success(f"## النتيجة: {benign_info[label][0]} ✅")
                            st.info(benign_info[label][1])
                        else:
                            st.success("## النتيجة: جلد سليم ✅")
                        st.balloons()

                    st.write("---")
                    st.caption("ملاحظة: تم تطوير هذا النظام ليعطي الأولوية للأمان الطبي والتحذير من أي نمط غير طبيعي.")


