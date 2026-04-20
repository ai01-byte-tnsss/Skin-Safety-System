import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. إعدادات الصفحة واللغات (20 لغة) ---
st.set_page_config(page_title="Skin AI Expert System", layout="wide")

LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "نظام التشخيص الذكي لأمراض الجلد", "upload": "📥 ارفع صورة الفحص", "btn": "🔍 بدء التحليل", "advice": "⚠️ تنبيه: هذا النظام أداة أكاديمية استرشادية فقط."},
    "English": {"dir": "ltr", "title": "Skin AI Expert Diagnostic System", "upload": "📥 Upload Image", "btn": "🔍 Start Analysis", "advice": "⚠️ Note: Academic tool only."},
    "Kurdî": {"dir": "rtl", "title": "ژیری پێست", "upload": "📥 وێنە", "btn": "🔍 شیکاری", "advice": "⚠️ پزیشک ببینە."}
    # ... يمكن إضافة اللغات الأخرى هنا
}

# --- 2. الدليل الطبي (7 أصناف للتشخيص + 3 إضافية للشرح) ---
DIAGNOSTIC_DB = {
    0: {"n": "Melanoma (ميلانوما)", "c": "#D32F2F", "w": 1.45, "d": "أخطر أنواع سرطان الجلد، يتطلب فحصاً فورياً."},
    1: {"n": "Melanocytic Nevi (وحمة)", "c": "#388E3C", "w": 0.55, "d": "شامات طبيعية وحميدة، تظهر بشكل منتظم."},
    2: {"n": "Basal Cell Carcinoma (BCC)", "c": "#F57C00", "w": 0.50, "d": "سرطان جلدي ينمو ببطء ونادراً ما ينتشر."},
    3: {"n": "Actinic Keratosis (AK)", "c": "#7B1FA2", "w": 1.15, "d": "بقع ناتجة عن ضرر الشمس قد تتطور مستقبلاً."},
    4: {"n": "Benign Keratosis (BKL)", "c": "#1976D2", "w": 0.85, "d": "زوائد جلدية غير سرطانية تظهر مع العمر."},
    5: {"n": "Dermatofibroma (DF)", "c": "#00796B", "w": 1.20, "d": "كتلة صلبة صغيرة تظهر غالباً في الساقين."},
    6: {"n": "Vascular Lesions (VASC)", "c": "#C2185B", "w": 1.25, "d": "آفات وعائية ناتجة عن تجمع الشعيرات الدموية."}
}

EXTRA_INFO = {
    7: {"n": "Squamous Cell Carcinoma", "c": "#E64A19", "d": "سرطان ينشأ في طبقات الجلد السطحية."},
    8: {"n": "Psoriasis (الصدفية)", "c": "#512DA8", "d": "التهاب مناعي يسبب قشوراً فضية وحكة."},
    9: {"n": "Eczema (الأكزيما)", "c": "#FFA000", "d": "تهيج وتحسس جلدي يسبب جفافاً واحمراراً."}
}

# --- 3. محرك الذكاء الاصطناعي المعدل (حل مشكلة ValueError) ---
@st.cache_resource
def build_final_model():
    # مدخل واحد موحد يغذي المسارين لضمان استقرار الأوزان
    master_in = Input(shape=(224, 224, 3))
    
    # بناء المسارات الهجينة
    base_eff = EfficientNetB0(weights=None, include_top=False)(master_in)
    base_mob = MobileNetV2(weights=None, include_top=False)(master_in)
    
    # دمج المخرجات
    merged = Concatenate()([GlobalAveragePooling2D()(base_eff), GlobalAveragePooling2D()(base_mob)])
    
    # الطبقات النهائية: 7 مخرجات فقط لتطابق ملف skin_expert_master.h5
    x = Dense(512, activation='relu')(merged)
    x = Dropout(0.5)(x)
    final_out = Dense(7, activation='softmax')(x)
    
    model = Model(inputs=master_in, outputs=final_out)
    
    # تحميل الأوزان مع التحقق من المسار
    h5_path = "skin_expert_master.h5"
    ready = False
    if os.path.exists(h5_path):
        try:
            model.load_weights(h5_path)
            ready = True
        except:
            st.error("⚠️ خطأ في توافق الأوزان مع الهيكلية!")
    return model, ready

model_ai, is_ready = build_final_model()

# --- 4. واجهة المستخدم والمنطق ---
sel_lang = st.selectbox("🌐 اختر اللغة / Language", list(LANG_DATA.keys()))
ui = LANG_DATA[sel_lang]

st.markdown(f"<h1 style='text-align:center;'>{ui['title']}</h1>", unsafe_allow_html=True)
st.warning(ui['advice'])

if not is_ready:
    st.error(f"❌ ملف الأوزان مفقود في: {os.getcwd()}")

file = st.file_uploader(ui['upload'], type=["jpg", "png", "jpeg"])

if file and is_ready:
    img = Image.open(file).convert('RGB')
    st.image(img, width=300, caption="الصورة المرفوعة")
    
    if st.button(ui['btn'], use_container_width=True):
        with st.spinner("⏳ جاري التحليل النسيجي..."):
            img_res = cv2.resize(np.array(img), (224, 224))
            
            # تحسين الصورة (CLAHE) لزيادة الدقة
            lab = cv2.cvtColor(img_res, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
            final_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

            # التنبؤ والمعايرة بالأوزان
            inp = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(final_img, axis=0))
            preds = model_ai.predict(inp)[0]
            
            # مصفوفة الأوزان لكسر الانحياز
            cal_w = np.array([v['w'] for v in DIAGNOSTIC_DB.values()])
            idx = np.argmax(preds * cal_w)
            
            res = DIAGNOSTIC_DB[idx]
            st.markdown(f"""
                <div style="border: 8px solid {res['c']}; padding: 25px; border-radius: 15px; text-align: center; background: white;">
                    <h1 style="color: {res['c']};">{res['n']}</h1>
                    <h3>{res['s']}</h3>
                    <p>{res['d']}</p>
                    <strong>نسبة المطابقة: {preds[idx]*100:.2f}%</strong>
                </div>
            """, unsafe_allow_html=True)

# --- 5. الدليل الطبي المرجعي الكامل (10 أصناف) ---
st.write("---")
st.subheader("📚 الدليل الطبي المرجعي الكامل (10 حالات)")
all_info = {**DIAGNOSTIC_DB, **EXTRA_INFO}
cols = st.columns(2)
for i, (k, v) in enumerate(all_info.items()):
    with cols[i % 2]:
        st.markdown(f"<span style='color:{v['c']};'>●</span> **{v['n']}**: {v['d']}", unsafe_allow_html=True)
