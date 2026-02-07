import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الواجهة
st.set_config = st.set_page_config(page_title="نظام التشخيص المتقدم", page_icon="🛡️")

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
    st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🛡️ نظام الفحص المطور (كشف الحالات الخارجية)</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📥 إدراج صورة الفحص", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة قيد التحليل", width=350)
        
        if st.button("🔍 تحليل ذكي"):
            with st.spinner('جاري تدقيق الخصائص...'):
                img_res = image.resize((150, 150))
                img_arr = np.array(img_res.convert('RGB')) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)
                
                if model is not None:
                    preds = model.predict(img_arr)[0]
                    all_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
                    results = dict(zip(all_labels, preds))

                    # --- خوارزمية كشف الحالات الغريبة (مثل حب الشباب) ---
                    # 1. ترتيب النتائج من الأعلى للأقل
                    sorted_preds = sorted(preds, reverse=True)
                    top1_val = sorted_preds[0]
                    top2_val = sorted_preds[1]
                    
                    # 2. حساب الفرق بين أعلى احتمالين (Confidence Margin)
                    # إذا كان الفرق صغيراً جداً، يعني أن النموذج "مرتبك" وغير متأكد
                    margin = top1_val - top2_val

                    st.write("### 📋 التقرير التشخيصي:")

                    # أ- كشف الحالات المشكوك في هويتها (مثل حب الشباب والساركوما)
                    if margin < 0.20: 
                        st.info("## حالة غير نمطية / غير واضحة ⚠️")
                        st.warning("النموذج يظهر ارتباكاً في تحديد النوع (تداخل الخصائص).")
                        st.error("قد تكون هذه الحالة (حب شباب، التهاب، أو نوع نادر من السرطان) خارج النطاق المباشر للنموذج.")
                        st.markdown("**يُنصح بالتشخيص السريري الفوري لأن الملامح البصرية غير حاسمة رقمياً.**")

                    # ب- المسار الوقائي للسرطان
                    elif any(results[k] > 0.30 for k in cancer_info.keys()):
                        top_cancer = max({k: results[k] for k in cancer_info.keys()}, key=lambda x: results[x])
                        st.warning(f"## اشتباه: {cancer_info[top_cancer][0]} ⚠️")
                        st.error("تم رصد ملامح تطابق الأنماط السرطانية المعروفة لدى النظام.")
                        st.progress(float(results[top_cancer]))

                    # ج- الحالات السليمة الواضحة
                    else:
                        st.success("## النتيجة: ملامح بصرية سليمة ✅")
                        st.balloons()

                    st.write("---")
                    st.caption("ملاحظة: تم تحديث المنطق لتمييز حالات 'ارتباك النموذج' الناتجة عن صور خارج التخصص.")
