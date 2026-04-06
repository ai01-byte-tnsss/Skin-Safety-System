import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.parse

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Global Skin Guard AI", layout="centered")

# --- 2. القاموس اللغوي (ثابت كما هو) ---
LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "🛡️ نظام الكشف عن سلامة الجلد", "upload": "📥 ارفع صورة الفحص", "camera": "📸 صورة فورية", "analyze": "🚀 بدء التحليل", "guide": "📚 الدليل الطبي الشامل", "malig": "🔴 الأورام الخبيثة", "benign": "🟢 الأورام الحميدة", "more": "التفاصيل الطبية", "res_m": "🚨 اشتباه ورم خبيث", "res_b": "🔍 ورم حميد", "res_g": "🩺 حالة عامة", "advice": "يرجى مراجعة المختص لضمان السلامة.", "share": "مشاركة"},
    "English": {"dir": "ltr", "title": "🛡️ Skin Safety AI System", "upload": "📥 Upload Image", "camera": "📸 Take Photo", "analyze": "🚀 Analyze", "guide": "📚 Medical Guide", "malig": "🔴 Malignant", "benign": "🟢 Benign", "more": "Medical Details", "res_m": "🚨 Malignant Suspect", "res_b": "🔍 Benign", "res_g": "🩺 General", "advice": "Please consult a specialist.", "share": "Share"},
    # ... اللغات الأخرى مخزنة داخلياً بنفس الترتيب
}

# --- 3. التنسيق البصري (ثابت) ---
selected_lang = st.sidebar.selectbox("🌐 اختر اللغة / Language", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

st.markdown(f"""
<style>
    div[dir='{t['dir']}'] {{ text-align: {'right' if t['dir']=='rtl' else 'left'}; }}
    .report-card {{ padding: 25px; border-radius: 15px; text-align: center; border: 5px solid; margin-top: 20px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
    .disease-card {{ border-right: 5px solid #0d47a1; border-left: 1px solid #eee; padding: 15px; background: #f9f9f9; margin-bottom: 15px; border-radius: 10px; }}
    .disease-title {{ color: #0d47a1; font-weight: bold; font-size: 1.1em; }}
</style>
""", unsafe_allow_html=True)

# --- 4. المحرك مع معالجة الـ Float التلقائية ---
@st.cache_resource
def load_expert_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except: return None

interpreter = load_expert_model()

def prepare_image_auto(image, interpreter):
    # الحل التلقائي: البرنامج يسأل النموذج "ماذا تريد؟" وينفذ فوراً
    input_details = interpreter.get_input_details()
    _, height, width, _ = input_details[0]['shape']
    target_dtype = input_details[0]['dtype'] # اكتشاف float16 أو float32 تلقائياً
    
    img_rgb = image.convert("RGB")
    img_resized = img_rgb.resize((width, height))
    img_array = np.array(img_resized).astype(target_dtype) # تحويل تلقائي
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# --- 5. واجهة الفحص ---
st.markdown(f"<div dir='{t['dir']}'>", unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align: center; color: #0d47a1;'>{t['title']}</h1>", unsafe_allow_html=True)

choice = st.radio("", (t['upload'], t['camera']))
file = st.file_uploader(t['upload'], type=["jpg", "png", "jpeg"]) if choice == t['upload'] else st.camera_input(t['camera'])

if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)
    
    if st.button(t['analyze']):
        if interpreter:
            with st.spinner("AI Analyzing..."):
                try:
                    final_input = prepare_image_auto(img, interpreter)
                    in_idx = interpreter.get_input_details()[0]['index']
                    interpreter.set_tensor(in_idx, final_input)
                    interpreter.invoke()
                    
                    out_idx = interpreter.get_output_details()[0]['index']
                    output = interpreter.get_tensor(out_idx)[0]
                    idx = np.argmax(output)

                    if idx in [1, 4, 17]:
                        res_msg, color = t['res_m'], "#cf1322"
                    elif idx in [2, 5, 23]:
                        res_msg, color = t['res_b'], "#389e0d"
                    else:
                        res_msg, color = t['res_g'], "#096dd9"

                    st.markdown(f'<div class="report-card" style="border-color: {color}; color: {color};"><h2>{res_msg}</h2><p>{t["advice"]}</p></div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"خطأ في الاستدلال: {e}")

st.write("---")

# --- 6. الدليل الطبي (وصف ومعلومات) ---
with st.expander(f"📖 {t['guide']}"):
    tab_m, tab_b = st.tabs([t['malig'], t['benign']])
    
    with tab_m:
        st.markdown("""<div class="disease-card"><span class="disease-title">🔴 Melanoma</span><br><b>ماهو:</b> سرطان الخلايا الصبغية.<br><b>كيف يتكون:</b> طفرات جينية بسبب الشمس.<br><b>الأعراض:</b> تغير مفاجئ في شكل الشامات.</div>""", unsafe_allow_html=True)
        # أضف بقية الأنواع بنفس التنسيق
    with tab_b:
        st.markdown("""<div class="disease-card" style="border-right-color:#389e0d;"><span class="disease-title">🟢 Lipoma</span><br><b>ماهو:</b> ورم شحمي حميد.<br><b>كيف يتكون:</b> تجمع خلايا دهنية تحت الجلد.<br><b>الأعراض:</b> كتلة طرية تتحرك باللمس.</div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
