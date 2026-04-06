import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. إعدادات الواجهة الاحترافية ---
st.set_page_config(page_title="Skin Health System", layout="wide")

st.markdown("""
<style>
    .report-card { padding: 30px; border-radius: 20px; text-align: center; border: 4px solid; box-shadow: 0px 8px 20px rgba(0,0,0,0.1); }
    .result-title { font-size: 28px; font-weight: bold; }
    .preview-box { border: 2px dashed #0d47a1; padding: 10px; border-radius: 15px; background-color: #f0f4f8; margin-bottom: 20px; }
    
    /* تنسيق دليل الأمراض المطور */
    .guide-section { background-color: #ffffff; padding: 20px; border-radius: 15px; border: 1px solid #e0e0e0; }
    .malig-header { color: #cf1322; border-bottom: 2px solid #cf1322; padding-bottom: 5px; margin-top: 20px; }
    .benign-header { color: #389e0d; border-bottom: 2px solid #389e0d; padding-bottom: 5px; margin-top: 20px; }
    .disease-info { font-size: 15px; line-height: 1.6; color: #444; }
    .symptom-tag { background-color: #f0f2f5; padding: 2px 8px; border-radius: 5px; font-size: 13px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. محرك تحميل النموذج ---
@st.cache_resource
def load_expert_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except: return None

interpreter = load_expert_model()

# --- 3. الواجهة الرئيسية ---
st.markdown("<h1 style='text-align: center; color: #0d47a1;'>🛡️ الكشف عن سلامة الجلد لمرض سرطان</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("🔍 منطقة الفحص والتحليل")
    source = st.radio("مصدر الصورة:", ("رفع ملف", "كاميرا فورية"))
    uploaded_file = st.file_uploader("📥 ارفع الصورة", type=["jpg", "png"]) if source == "رفع ملف" else st.camera_input("📸 التقط صورة")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.markdown('<div class="preview-box">', unsafe_allow_html=True)
        st.image(image, caption="معاينة الصورة المرفوعة", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 بدء التحليل الرقمي"):
            if interpreter:
                # عمليات المعالجة والتحليل (كما في الأكواد السابقة)
                input_details = interpreter.get_input_details()
                target_dtype = input_details[0]['dtype']
                input_shape = input_details[0]['shape'][1:3]
                img = image.convert("RGB").resize(input_shape)
                img_array = np.array(img)
                img_array = (img_array.astype(np.float32) / 255.0) if target_dtype == np.float32 else img_array.astype(target_dtype)
                img_array = np.expand_dims(img_array, axis=0)
                
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
                idx = np.argmax(output)

                if idx in [1, 4, 17]:
                    res, sub, color = "🚨 اشتباه ورم خبيث", "تم رصد علامات نمو غير طبيعي.", "#cf1322"
                elif idx in [2, 5, 23]:
                    res, sub, color = "🔍 ورم جلدي حميد", "الآفة تبدو من النوع السليم.", "#389e0d"
                else:
                    res, sub, color = "🩺 حالة جلدية عامة", "النمط الجلدي يرجح وجود حالة غير ورمية.", "#096dd9"

                st.markdown(f'<div class="report-card" style="border-color: {color}; color: {color};"><p class="result-title">{res}</p><p>{sub}</p></div>', unsafe_allow_html=True)

# --- 4. الدليل الشامل (الجهة اليسرى) ---
with col2:
    st.markdown('<div class="guide-section">', unsafe_allow_html=True)
    st.subheader("📚 الدليل الطبي الشامل للأورام")
    
    tab1, tab2 = st.tabs(["🔴 الأورام الخبيثة", "🟢 الأورام الحميدة"])

    with tab1:
        st.markdown("<h4 class='malig-header'>أنواع سرطان الجلد</h4>", unsafe_allow_html=True)
        
        with st.expander("1. سرطان الخلايا القاعدية (BCC)"):
            st.write("**الوصف:** الأكثر شيوعاً، يظهر كبقعة لؤلؤية أو وردية.")
            st.write("**الأعراض:** نزيف بسيط، تقرح لا يلتئم، عروق دموية مرئية.")
            st.image("https://www.mayoclinic.org/-/media/kcms/gpts/2013/08/26/10/33/ds00925_ds00039_im01991_r7_bcc_armthu_jpg.jpg", caption="مثال لسرطان الخلايا القاعدية")

        with st.expander("2. سرطان الخلايا الحرشفية (SCC)"):
            st.write("**الوصف:** ينمو في المناطق المعرضة للشمس.")
            st.write("**الأعراض:** نتوء صلب أحمر، بقعة قشرية مسطحة.")
            st.image("https://www.mayoclinic.org/-/media/kcms/gpts/2013/08/26/10/33/ds00924_ds00039_im01993_r7_sccthu_jpg.jpg", caption="مثال لسرطان الخلايا الحرشفية")

        with st.expander("3. الميلانوما (Melanoma)"):
            st.write("**الوصف:** الأخطر؛ يبدأ في خلايا الصبغة.")
            st.write("**الأعراض:** تغير في لون/شكل شامة قديمة، حدود غير منتظمة.")
            st.image("https://www.skincancer.org/wp-content/uploads/melanoma-example.jpg", caption="مثال للميلانوما")

        with st.expander("4. سرطان خلايا ميركل"):
            st.write("**الوصف:** نادر وعنيف جداً.")
            st.write("**الأعراض:** نتوء ثابت غير مؤلم بلون أحمر أو أرجواني.")

        with st.expander("5. ساركوما كابوزي"):
            st.write("**الوصف:** يظهر في الأوعية الدموية للجلد.")
            st.write("**الأعراض:** بقع أرجوانية أو حمراء على الجلد أو الأغشية المخاطية.")

    with tab2:
        st.markdown("<h4 class='benign-header'>أورام الجلد الحميدة</h4>", unsafe_allow_html=True)
        
        with st.expander("1. الشامات (Moles)"):
            st.write("**الوصف:** بقع ملونة منتظمة الشكل.")
            st.image("https://www.aad.org/Images/Public/Diseases/Skin-Cancer/moles-atypical-mole.jpg", caption="شامة طبيعية")

        with st.expander("2. التقران الدهني (Seborrheic Keratosis)"):
            st.write("**الوصف:** نمو جلدي يشبه " "الشمع" " الملصق بالجلد.")
            st.write("**الأعراض:** لون بني أو أسود، ملمس خشن أو ناعم.")

        with st.expander("3. الأورام الشحمية (Lipomas)"):
            st.write("**الوصف:** كتل دهنية تحت الجلد مباشرة.")
            st.write("**الأعراض:** لينة الملمس، تتحرك بسهولة عند لمسها.")

        with st.expander("4. التقران الشعاعي (Actinic Keratosis)"):
            st.write("**تنبيه:** حالة ما قبل السرطان!")
            st.write("**الوصف:** بقع خشنة قشرية ناتجة عن التعرض الطويل للشمس.")

    st.markdown('</div>', unsafe_allow_html=True)

# التذييل
st.markdown("<br><hr><p style='text-align: center; color: #9e9e9e;'>نظام تقييم سلامة الجلد الذكي المعتمد © 2026</p>", unsafe_allow_html=True)
