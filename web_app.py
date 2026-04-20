import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. إعدادات الصفحة والقاموس اللغوي ---
st.set_page_config(page_title="Skin AI Expert System", layout="wide")

LANG_DATA = {
    "العربية": {
        "dir": "rtl", 
        "title": "النظام الخبير لتشخيص أمراض الجلد الذكي", 
        "upload": "📥 ارفع صورة الفحص الجلدي", 
        "btn": "🔍 بدء التحليل الطبي",
        "result_head": "النتيجة المتوقعة:",
        "advice": "⚠️ ملاحظة: هذا التشخيص استرشادي، يرجى مراجعة الطبيب المختص."
    },
    "English": {
        "dir": "ltr", 
        "title": "Smart Skin Disease Expert System", 
        "upload": "📥 Upload Skin Image", 
        "btn": "🔍 Start Diagnosis",
        "result_head": "Predicted Result:",
        "advice": "⚠️ Note: This is an academic tool; consult a professional."
    }
}

# --- 2. البيانات الطبية (7 للتشخيص + 3 للشرح فقط) ---
# ملاحظة: الموديل سيعمل على أول 7 فئات ليطابق ملف skin_expert_master.h5
DIAGNOSTIC_CLASSES = {
    0: {"n": "Melanoma (ميلانوما)", "c": "#D32F2F", "w": 1.45, "d": "أخطر أنواع سرطان الجلد، يتطلب تدخل طبي سريع."},
    1: {"n": "Melanocytic Nevi (وحمة)", "c": "#388E3C", "w": 0.55, "d": "شامة طبيعية وحميدة، غالباً ما تكون آمنة."},
    2: {"n": "Basal Cell Carcinoma (سرطان قاعدي)", "c": "#F57C00", "w": 0.50, "d": "نوع من سرطان الجلد ينمو ببطء ونادراً ما ينتشر."},
    3: {"n": "Actinic Keratosis (تقرن شمسى)", "c": "#7B1FA2", "w": 1.15, "d": "بقع قشرية ناتجة عن ضرر الشمس، قد تتحول لسرطان."},
    4: {"n": "Benign Keratosis (تقرن حميد)", "c": "#1976D2", "w": 0.85, "d": "آفات جلدية غير سرطانية شائعة مع تقدم العمر."},
    5: {"n": "Dermatofibroma (ليفي جلدي)", "c": "#00796B", "w": 1.20, "d": "كتل جلدية حميدة وصغيرة، تظهر عادة بعد إصابة طفيفة."},
    6: {"n": "Vascular Lesions (آفات وعائية)", "c": "#C2185B", "w": 1.25, "d": "تشمل الأورام الوعائية وتجمعات الشعيرات الدموية."}
}

EXTRA_GUIDE = {
    7: {"n": "Squamous Cell Carcinoma (سرطان حرشفي)", "c": "#E64A19", "d": "ثاني أكثر أنواع سرطان الجلد شيوعاً."},
    8: {"n": "Psoriasis (الصدفية)", "c": "#512DA8", "d": "مرض جلدي مناعي يسبب قشوراً فضية."},
    9: {"n": "Eczema (الأكزيما)", "c": "#FFA000", "d": "التهاب جلدي يسبب الحكة والاحمرار والجفاف."}
}

# --- 3. محرك الذكاء الاصطناعي (7 مخرجات ليطابق ملف الأوزان) ---
@st.cache_resource
def load_skin_ai_model():
    # مدخل موحد لحل خطأ ValueError
    img_input = Input(shape=(224, 224, 3), name="main_input")
    
    # دمج الموديلين (Ensemble)
    base_1 = EfficientNetB0(weights=None, include_top=False)(img_input)
    base_2 = MobileNetV2(weights=None, include_top=False)(img_input)
    
    # دمج المخرجات
    merged = Concatenate()([GlobalAveragePooling2D()(base_1), GlobalAveragePooling2D()(base_2)])
    
    # الطبقة النهائية: 7 مخرجات فقط لتطابق ملفك البرمجي
    x = Dense(512, activation='relu')(merged)
    x = Dropout(0.4)(x)
    output = Dense(7, activation='softmax')(x)
    
    model = Model(inputs=img_input, outputs=output)
    
    # تحميل الأوزان
    weights_path = "skin_expert_master.h5"
    ready = False
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            ready = True
        except:
            st.error("❌ تضارب: ملف الأوزان يحتوي على عدد طبقات مختلف عن الكود.")
    return model, ready

model, is_ready = load_skin_ai_model()

# --- 4. واجهة المستخدم ---
lang_choice = st.sidebar.selectbox("🌐 اختر اللغة / Language", ["العربية", "English"])
ui = LANG_DATA[lang_choice]

st.markdown(f"<h1 style='text-align:center;'>{ui['title']}</h1>", unsafe_allow_html=True)

if not is_ready:
    st.error(f"❌ لم يتم العثور على 'skin_expert_master.h5' في المجلد الحالي: {os.getcwd()}")

file = st.file_uploader(ui['upload'], type=["jpg", "png", "jpeg"])

if file and is_ready:
    img_pil = Image.open(file).convert('RGB')
    st.image(img_pil, width=300)
    
    if st.button(ui['btn'], use_container_width=True):
        with st.spinner("⏳ جاري تحليل النسيج..."):
            # 1. معالجة الصورة يدوياً (موازنة الألوان + تحسين التباين)
            img_np = np.array(img_pil)
            img_res = cv2.resize(img_np, (224, 224))
            
            # موازنة الألوان (Gray World)
            avg = np.mean(img_res)
            proc = img_res.astype(np.float32)
            for i in range(3):
                proc[:,:,i] = np.clip(img_res[:,:,i] * (avg / (np.mean(img_res[:,:,i]) + 1e-6)), 0, 255)
            
            # تحسين التباين CLAHE
            lab = cv2.cvtColor(proc.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
            final_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

            # 2. إجراء التشخيص (باستخدام مصفوفة الـ 7 أصناف)
            inp = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(final_img, axis=0))
            preds = model.predict(inp)[0]
            
            # موازنة الأوزان لكسر الانحياز
            weights = np.array([v['w'] for v in DIAGNOSTIC_CLASSES.values()])
            win_idx = np.argmax(preds * weights)
            
            # 3. عرض النتيجة
            res = DIAGNOSTIC_CLASSES[win_idx]
            st.markdown(f"""
                <div style="border: 10px solid {res['c']}; padding: 20px; border-radius: 15px; background: white; text-align: center;">
                    <h2 style="color: {res['c']};">{ui['result_head']} {res['n']}</h2>
                    <p style="font-size: 1.2em;">{res['d']}</p>
                    <strong>الدقة التقنية للمطابقة: {preds[win_idx]*100:.2f}%</strong>
                </div>
            """, unsafe_allow_html=True)
            st.info(ui['advice'])

# --- 5. الدليل المرجعي الشامل (10 أصناف) ---
st.write("---")
st.subheader("📚 الدليل الطبي المرجعي الكامل")
full_guide = {**DIAGNOSTIC_CLASSES, **EXTRA_GUIDE}
cols = st.columns(2)
for i, (k, v) in enumerate(full_guide.items()):
    with cols[i % 2]:
        st.markdown(f"<span style='color:{v.get('c', '#333')};'>●</span> **{v['n']}**: {v['d']}", unsafe_allow_html=True)
