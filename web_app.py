import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. إعدادات الواجهة الرسومية (UI) ---
st.set_page_config(
    page_title="Skin AI Expert | جامعة الموصل",
    layout="wide",
    initial_sidebar_state="expanded"
)

# القاموس الشامل للغات مع اتجاهات النص
LANG_DATA = {
    "العربية": {
        "dir": "rtl", 
        "title": "النظام الخبير الذكي لتشخيص الآفات الجلدية",
        "sub": "كلية علوم الحاسوب والرياضيات - مشروع تخرج",
        "upload": "📥 تحميل صورة الفحص الجلدي",
        "cam": "📸 استخدام الكاميرا المباشرة",
        "btn": "🔍 إجراء التحليل الهيكلي",
        "invalid": "❌ تنبيه: الصورة لا تحتوي على ملامح نسيج جلدي كافية.",
        "advice": "⚠️ إخلاء مسؤولية: هذا النظام أكاديمي استرشادي ولا يغني عن الطبيب المختص."
    },
    "English": {
        "dir": "ltr", 
        "title": "Skin AI Expert Diagnostic System",
        "sub": "College of CS & Math - Graduation Project",
        "upload": "📥 Upload Skin Lesion Image",
        "cam": "📸 Use Live Camera",
        "btn": "🔍 Perform Structural Analysis",
        "invalid": "❌ Warning: Image does not contain sufficient skin texture.",
        "advice": "⚠️ Disclaimer: Academic tool for guidance only; consult a professional."
    }
}

# --- 2. قاعدة البيانات الطبية ونظام الأوزان (Weight Matrix) ---
# تم ضبط الأوزان (w) بدقة لكسر احتكار الـ BCC والحميد
MEDICAL_DB = {
    0: {"n": "Melanoma (ميلانوما)", "c": "#D32F2F", "s": "🚨 حالة خبيثة جداً", "w": 1.45, "d": "أخطر أنواع سرطان الجلد؛ يتطلب تدخلًا طبيًا عاجلًا وفحصًا مخبريًا."},
    1: {"n": "Melanocytic Nevi (وحمة)", "c": "#388E3C", "s": "✅ حالة حميدة", "w": 0.60, "d": "شامة طبيعية غير خطرة، ناتجة عن تجمع خلايا صبغية بشكل منتظم."},
    2: {"n": "Basal Cell Carcinoma (BCC)", "c": "#F57C00", "s": "🚨 حالة خبيثة", "w": 0.55, "d": "سرطان الخلايا القاعدية؛ ينمو ببطء ونادرًا ما ينتشر لكنه يتطلب علاجًا."},
    3: {"n": "Actinic Keratosis (AK)", "c": "#7B1FA2", "s": "⚠️ ما قبل سرطاني", "w": 1.15, "d": "تقرن جلدي ناتج عن الشمس؛ قد يتطور إلى سرطان حرشفي بمرور الوقت."},
    4: {"n": "Benign Keratosis (BKL)", "c": "#1976D2", "s": "✅ حالة حميدة", "w": 0.80, "d": "زوائد جلدية غير سرطانية شائعة جدًا مع التقدم في السن."},
    5: {"n": "Dermatofibroma (DF)", "c": "#00796B", "s": "✅ حالة حميدة", "w": 1.20, "d": "كتلة صلبة صغيرة تظهر غالبًا في الساقين بعد إصابة طفيفة أو لدغة حشرة."},
    6: {"n": "Vascular Lesions (VASC)", "c": "#C2185B", "s": "✅ حالة حميدة", "w": 1.25, "d": "آفات وعائية ناتجة عن تضخم الشعيرات الدموية تحت سطح الجلد."},
    7: {"n": "Squamous Cell Carcinoma", "c": "#E64A19", "s": "🚨 حالة خبيثة", "w": 1.30, "d": "سرطان الخلايا الحرشفية؛ يظهر كقشور صلبة ويحتاج استئصالًا طبيًا."},
    8: {"n": "Psoriasis (الصدفية)", "c": "#512DA8", "s": "🔍 حالة جلدية", "w": 1.05, "d": "مرض مناعي مزمن يسبب التهابًا وظهور قشور فضية على سطح الجلد."},
    9: {"n": "Eczema (الأكزيما)", "c": "#FFA000", "s": "🔍 حالة جلدية", "w": 1.10, "d": "التهاب تحسسي يسبب حكة شديدة وجفافًا واحمرارًا في المنطقة المصابة."}
}

# --- 3. بناء هيكلية المحرك الذكي (Ensemble Model) ---
@st.cache_resource
def build_robust_engine():
    # استخدام Input صريح لضمان استقرار الهيكلية
    img_input = Input(shape=(224, 224, 3))
    
    # دمج قوتين (EfficientNet للأنماط الدقيقة و MobileNet للسرعة)
    path_a = EfficientNetB0(weights=None, include_top=False)(img_input)
    path_b = MobileNetV2(weights=None, include_top=False)(img_input)
    
    gap_a = GlobalAveragePooling2D()(path_a)
    gap_b = GlobalAveragePooling2D()(path_b)
    
    merged = Concatenate()([gap_a, gap_b])
    feat_layer = Dense(512, activation='relu')(merged)
    dropout = Dropout(0.4)(feat_layer)
    # المخرجات 10 لتطابق قاعدة البيانات الطبية
    output = Dense(10, activation='softmax')(dropout)
    
    full_model = Model(inputs=img_input, outputs=output)
    
    # تحميل الأوزان مع التحقق
    weights_path = "skin_expert_master.h5"
    ready = False
    if os.path.exists(weights_path):
        full_model.load_weights(weights_path)
        ready = True
    return full_model, ready

model, is_ready = build_robust_engine()

# --- 4. معالجة الواجهة والمدخلات ---
st.sidebar.header("⚙️ الإعدادات / Settings")
sel_lang = st.sidebar.selectbox("🌐 لغة النظام", list(LANG_DATA.keys()))
ui = LANG_DATA[sel_lang]

st.markdown(f"""
    <div style='text-align:center; padding:20px; border-bottom:2px solid #eee;'>
        <h1 style='color:#1565C0; font-family:Tajawal;'>{ui['title']}</h1>
        <h4 style='color:#555;'>{ui['sub']}</h4>
    </div>
""", unsafe_allow_html=True)

if not is_ready:
    st.error(f"❌ خطأ فني: ملف الأوزان (skin_expert_master.h5) غير موجود في الخادم.")

st.info(ui['advice'])

col_up, col_prev = st.columns([1, 1])
with col_up:
    choice = st.radio("اختر وسيلة الإدخال:", [ui['upload'], ui['cam']], horizontal=True)
    src = st.file_uploader("", type=["jpg", "png", "jpeg"]) if "تحميل" in choice or "Upload" in choice else st.camera_input("")

# --- 5. منطق التشخيص وتصحيح الانحياز ---
if src and is_ready:
    img_raw = Image.open(src).convert('RGB')
    with col_prev:
        st.image(img_raw, caption="صورة الفحص الحالية", use_container_width=True)
    
    if st.button(ui['btn'], use_container_width=True):
        with st.spinner("⏳ جاري تحليل النسيج ومعايرة الاحتمالات..."):
            # تجهيز الصورة
            img_arr = np.array(img_raw)
            img_cv = cv2.resize(img_arr, (224, 224))
            
            # أ) موازنة الألوان (White Balance) يدوياً لحل مشكلة الإضاءة
            avg_val = np.mean(img_cv)
            img_wb = img_cv.astype(np.float32)
            for i in range(3):
                img_wb[:, :, i] = np.clip(img_cv[:, :, i] * (avg_val / np.mean(img_cv[:, :, i])), 0, 255)
            
            # ب) تحسين التباين (CLAHE) لإظهار التفاصيل الدقيقة
            lab = cv2.cvtColor(img_wb.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
            img_final = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2RGB)

            # ج) التنبوء وتطبيق مصفوفة المعايرة الذكية
            input_tensor = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_final, axis=0))
            raw_scores = model.predict(input_tensor)[0]
            
            # سر التصنيف الصحيح: ضرب النتائج في الأوزان التصحيحية (W)
            calibrated_weights = np.array([v['w'] for v in MEDICAL_DB.values()])
            final_preds = raw_scores * calibrated_weights
            win_idx = np.argmax(final_preds)
            
            # د) عرض النتيجة بتصميم احترافي
            res = MEDICAL_DB[win_idx]
            st.markdown(f"""
                <div style="border: 10px solid {res['c']}; padding: 30px; border-radius: 20px; background: white; text-align: center; margin-top: 20px;">
                    <h1 style="color: {res['c']}; font-size: 3em;">{res['n']}</h1>
                    <h2 style="background: #f8f9fa; padding: 10px; border-radius: 10px;">{res['s']}</h2>
                    <p style="font-size: 1.4em; color: #333; line-height: 1.6;">{res['d']}</p>
                    <div style="margin-top: 15px; font-weight: bold; color: #666;">
                        معدل تطابق الأنسجة: {raw_scores[win_idx]*100:.2f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)

# --- 6. الدليل الطبي المرجعي (مرتب وقوي) ---
st.write("---")
st.header("📚 الدليل الطبي المرجعي الموحد")
with st.expander("انقر لعرض تفاصيل جميع أنواع الآفات الجلدية العشرة"):
    cols = st.columns(2)
    for i, (k, v) in enumerate(MEDICAL_DB.items()):
        target_col = cols[i % 2]
        target_col.markdown(f"""
            <div style="border-right: 5px solid {v['c']}; padding: 10px; margin-bottom: 10px; background: #fafafa;">
                <h4 style="color: {v['c']};">{v['n']}</h4>
                <p><strong>التصنيف:</strong> {v['s']}</p>
                <p style="font-size: 0.9em;">{v['d']}</p>
            </div>
        """, unsafe_allow_html=True)
# --- 6. الدليل الطبي المرجعي (مرتب وقوي) ---
st.write("---")
st.header("📚 الدليل الطبي المرجعي الموحد")
with st.expander("انقر لعرض تفاصيل جميع أنواع الآفات الجلدية العشرة"):
    cols = st.columns(2)
    for i, (k, v) in enumerate(MEDICAL_DB.items()):
        target_col = cols[i % 2]
        target_col.markdown(f"""
            <div style="border-right: 5px solid {v['c']}; padding: 10px; margin-bottom: 10px; background: #fafafa;">
                <h4 style="color: {v['c']};">{v['n']}</h4>
                <p><strong>التصنيف:</strong> {v['s']}</p>
                <p style="font-size: 0.9em;">{v['d']}</p>
            </div>
        """, unsafe_allow_html=True)
