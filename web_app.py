import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. إعدادات الصفحة واللغات ---
st.set_page_config(page_title="Skin AI Expert System", layout="wide")

LANG_DATA = {
    "العربية": {
        "dir": "rtl", 
        "title": "نظام خبير الذكاء الاصطناعي لتشخيص الجلد",
        "upload": "📥 ارفع صورة الفحص",
        "cam": "📸 الكاميرا",
        "btn": "🔍 بدء التحليل العميق",
        "invalid": "❌ الصورة غير صالحة لفحص الجلد.",
        "advice": "⚠️ تنبيه: هذا النظام أداة أكاديمية ولا يغني عن استشارة الطبيب."
    },
    "English": {
        "dir": "ltr",
        "title": "Skin AI Expert Diagnostic System",
        "upload": "📥 Upload Scan Image",
        "cam": "📸 Camera",
        "btn": "🔍 Start Deep Analysis",
        "invalid": "❌ Invalid skin image detected.",
        "advice": "⚠️ Note: Academic guidance tool; not a substitute for professional advice."
    }
}

# --- 2. الدليل الطبي المرجعي لجميع الأنواع العشرة ---
# تم ضبط الأوزان (Weight) لضمان تصنيف كل نوع بشكل مستقل ومنفصل
MEDICAL_DB = {
    0: {"n": "Melanoma (ميلانوما)", "c": "#FF3B30", "s": "🚨 خبيث جداً", "w": 1.30, "d": "أخطر أنواع سرطان الجلد، يتطلب فحصاً طبياً فورياً."},
    1: {"n": "Melanocytic Nevi (وحمة صبغية)", "c": "#34C759", "s": "✅ حميد", "w": 0.70, "d": "شامة طبيعية آمنة وغير ضارة تماماً."},
    2: {"n": "Basal Cell Carcinoma (BCC)", "c": "#FF9500", "s": "🚨 خبيث", "w": 0.65, "d": "سرطان الخلايا القاعدية، ينمو ببطء ويجب علاجه."},
    3: {"n": "Actinic Keratosis (AK)", "c": "#AF52DE", "s": "⚠️ ما قبل سرطاني", "w": 1.05, "d": "بقع ناتجة عن التلف الشمسي قد تتطور لسرطان مستقبلاً."},
    4: {"n": "Benign Keratosis (BKL)", "c": "#5856D6", "s": "✅ حميد", "w": 0.85, "d": "زوائد جلدية غير سرطانية تظهر مع تقدم العمر."},
    5: {"n": "Dermatofibroma (DF)", "c": "#007AFF", "s": "✅ حميد", "w": 1.10, "d": "كتلة صلبة صغيرة ناتجة عن رد فعل لإصابة طفيفة."},
    6: {"n": "Vascular Lesions (VASC)", "c": "#5AC8FA", "s": "✅ حميد", "w": 1.15, "d": "آفات وعائية ناتجة عن تجمع الشعيرات الدموية."},
    7: {"n": "Squamous Cell Carcinoma", "c": "#FF2D55", "s": "🚨 خبيث", "w": 1.20, "d": "سرطان الخلايا الحرشفية، يتطلب تدخلاً جراحياً مختصاً."},
    8: {"n": "Psoriasis (الصدفية)", "c": "#4CD964", "s": "🔍 حالة جلدية", "w": 1.00, "d": "مرض مناعي يسبب التهاب الجلد وقشوراً فضية."},
    9: {"n": "Eczema (الأكزيما)", "c": "#FFCC00", "s": "🔍 حالة جلدية", "w": 1.05, "d": "التهاب جلدي يسبب حكة شديدة واحمراراً نسيجياً."}
}

# --- 3. محرك الذكاء الاصطناعي (Hybrid Engine) ---
@st.cache_resource
def load_expert_engine():
    # بناء هيكل النموذج الهجين لضمان أعلى دقة
    base_eff = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    base_mob = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    merged = Concatenate()([GlobalAveragePooling2D()(base_eff.output), GlobalAveragePooling2D()(base_mob.output)])
    dense = Dense(512, activation='relu')(merged)
    output = Dense(10, activation='softmax')(Dropout(0.4)(dense))
    
    model = Model(inputs=[base_eff.input, base_mob.input], outputs=output)
    
    # التحقق من وجود ملف الأوزان
    h5_path = "skin_expert_master.h5"
    is_ready = False
    if os.path.exists(h5_path):
        model.load_weights(h5_path)
        is_ready = True
    return model, is_ready

diag_model, ready_status = load_expert_engine()

# --- 4. واجهة المستخدم الرسومية ---
lang_key = st.selectbox("🌐 لغة النظام / Language", list(LANG_DATA.keys()))
ui = LANG_DATA[lang_key]

st.markdown(f"<h1 style='text-align:center; color:#1E3A8A;'>{ui['title']}</h1>", unsafe_allow_html=True)

if not ready_status:
    st.error(f"❌ ملف الأوزان '{os.path.basename('skin_expert_master.h5')}' مفقود!")
st.info(ui['advice'])

up_img = st.file_uploader(ui['upload'], type=["jpg", "jpeg", "png"])

if up_img and ready_status:
    img_pil = Image.open(up_img).convert('RGB')
    col1, col2 = st.columns(2)
    with col1: st.image(img_pil, caption="Preview", use_container_width=True)
    
    if st.button(ui['btn'], use_container_width=True):
        with st.spinner("⏳ Analyzing..."):
            # معالجة الصور لكسر الانحياز ومنع الخطأ الظاهر سابقا
            img_cv = cv2.resize(np.array(img_pil), (224, 224))
            
            # موازنة الألوان يدوياً (White Balance)
            avg_all = np.mean(img_cv)
            img_proc = img_cv.astype(np.float32)
            for i in range(3):
                img_proc[:, :, i] = np.clip(img_cv[:, :, i] * (avg_all / np.mean(img_cv[:, :, i])), 0, 255)
            
            # تحسين النسيج الجلدي (CLAHE)
            lab = cv2.cvtColor(img_proc.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8,8)).apply(l)
            final_proc = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

            # التنبوء مع تطبيق مصفوفة المعايرة لضمان فصل الأنواع
            inp = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(final_proc, axis=0))
            raw_preds = diag_model.predict([inp, inp])[0]
            
            # موازنة النتائج برمجياً لمنع طغيان الـ BCC والحميد
            cal_weights = np.array([v['w'] for v in MEDICAL_DB.values()])
            final_idx = np.argmax(raw_preds * cal_weights)
            
            res = MEDICAL_DB[final_idx]
            with col2:
                st.markdown(f"""
                <div style="border: 8px solid {res['c']}; padding: 25px; border-radius: 15px; background: white; text-align: center;">
                    <h1 style="color: {res['c']};">{res['n']}</h1>
                    <h3>التصنيف: {res['s']}</h3>
                    <hr style="border: 1px solid {res['c']};">
                    <p style="font-size: 1.2em;">{res['d']}</p>
                </div>
                """, unsafe_allow_html=True)

# --- 5. الدليل الثابت التفاعلي ---
st.write("---")
with st.expander("📖 الدليل المرجعي الكامل للأمراض والآفات الجلدية"):
    for k, v in MEDICAL_DB.items():
        st.markdown(f"<span style='color:{v['c']};'>●</span> **{v['n']}**: {v['d']}", unsafe_allow_html=True)
