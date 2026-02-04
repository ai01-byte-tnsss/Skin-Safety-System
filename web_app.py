import streamlit as st
import tensorflow as tf
from PIL import Image, ImageStat
import numpy as np
import os

# 1. إعدادات الواجهة والنموذج
st.set_page_config(page_title="نظام الكشف عن سلامة الجلد", page_icon="🛡️")

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
all_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
malignant_types = ['mel', 'bcc', 'akiec'] 

# --- نظام الحماية بكلمة المرور ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("<h2 style='text-align: center; color: #1e3a8a;'>🔐 تسجيل الدخول للنظام</h2>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1,2,1])
    with col_b:
        password = st.text_input("أدخل كلمة المرور الخاصة بالنظام:", type="password")
        if st.button("دخول"):
            if password == "test**00": 
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("كلمة المرور غير صحيحة")
else:
    # --- الواجهة الرئيسية بعد تسجيل الدخول ---
    st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🛡️ النظام الذكي للكشف عن سلامة الجلد</h1>", unsafe_allow_html=True)

    # 2. لوحة الإحصائيات الفنية (80/20 والدقة 93%)
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("نسبة التدريب", "80%")
    with col2: st.metric("نسبة الاختبار", "20%")
    with col3: st.metric("الدقة الإجمالية", "93%")

    st.divider()

    # 3. إدراج الصورة
    uploaded_file = st.file_uploader("📥 إدراج صورة الفحص", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة قيد التحليل الرقمي", width=350)
        
        if st.button("🔍 اختبار سرطان أم لا"):
            stat = ImageStat.Stat(image.convert('L'))
            if stat.var[0] < 80: # تقليل صرامة فحص الوضوح للسماح بمزيد من الصور
                st.error("❌ عذراً، الصورة غير واضحة. يرجى إعادة التصوير بوضوح أكبر.")
            else:
                with st.spinner('جاري معالجة البيانات الرقمية...'):
                    try:
                        # المعالجة الرقمية
                        img_res = image.resize((150, 150))
                        img_arr = np.array(img_res.convert('RGB')) / 255.0
                        img_arr = np.expand_dims(img_arr, axis=0)
                        
                        if model is not None:
                            preds = model.predict(img_arr)[0]
                            idx = np.argmax(preds)
                            label = all_classes[idx]
                            confidence = preds[idx]

                            st.write("### 📋 نتيجة التقرير النهائي:")

                            # --- تعديل العتبة لزيادة الحساسية (0.50 بدلاً من 0.92) ---
                            if confidence < 0.50: 
                                # الحالة: لا (يقين منخفض جداً)
                                st.success("## النتيجة: لا")
                                st.info("### الحالة: مرض جلدي غير سرطاني")
                                st.markdown("""
                                <div style="background-color: #f0f9ff; padding: 20px; border-radius: 12px; border-right: 6px solid #0284c7; text-align: right;">
                                    <p style="color: #0369a1; font-weight: bold; font-size: 18px; margin: 0;">
                                        (ملاحظة: الحالة تندرج ضمن الأمراض الجلدية الشائعة مثل الإكزيما أو الصدفية..)
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                # الحالة: نعم (سرطان) لأن اليقين تجاوز 50%
                                is_malignant = label in malignant_types
                                res_type = "خبيث ⚠️" if is_malignant else "حميد ✅"
                                
                                st.warning("## النتيجة: نعم (سرطان)")
                                st.error(f"### الحالة المكتشفة: {res_type}")
                                st.info(f"درجة يقين النظام في هذا التشخيص: {confidence*100:.2f}%")
                                
                                st.markdown(":red[**تم رصد خصائص بصرية تستوجب المتابعة الطبية الفورية.**]")
                            
                            st.error("⚠️ تنبيه: هذا التقرير هو تحليل أولي رقمي؛ يرجى مراجعة دكتور مختص لتأكيد التشخيص.")
                        else:
                            st.error("خطأ: لم يتم تحميل ملف النموذج h5.")
                    except Exception as e:
                        st.error(f"حدث خطأ أثناء المعالجة: {e}")

    st.markdown("---")
    st.write(":grey[**الدقة الكاملة للنظام المعتمدة: 93%**]")


