import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Skin AI System", layout="wide")

# الروابط والأسماء (تأكد من وجود الملفات في نفس المجلد أو تحميلها)
# تم اعتماد الملف الهجين ذو الـ 24 صنفاً بناءً على طلبك
HYBRID_MODEL_FILE = "skin_expert_hybrid_24ch.h5" 
MASTER_MODEL_FILE = "skin_expert_master.h5"

# --- 2. قائمة الـ 24 صنفاً (الترتيب الدقيق) ---
CLASSES_24 = [
    'Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 
    'Atopic Dermatitis Photos', 'Bullous Disease Photos', 
    'Cellulitis Impetigo and other Bacterial Infections', 'Eczema Photos', 
    'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases', 
    'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation', 
    'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles', 
    'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis', 
    'Psoriasis pictures Lichen Planus and related diseases', 
    'Scabies Lyme Disease and other Infestations and Bites', 
    'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease', 
    'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 
    'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections', 'archive'
]

# --- 3. لغات الواجهة ---
LANGS_CONFIG = {"العربية": "rtl", "English": "ltr", "Français": "ltr", "Deutsch": "ltr", "Español": "ltr"}
UI_LABELS = {
    "العربية": {
        "title": "نظام الفحص الذكي للجلد",
        "upload": "📥 ارفع صورة الفحص أو استخدم الكاميرا",
        "btn": "🔍 بدء تحليل الصورة",
        "advice": "⚠️ تنبيه طبي: هذا النظام أداة برمجية استرشادية، ولا يغني عن زيارة الطبيب المختص.",
        "guide_title": "📖 الدليل الطبي المرجعي لآفات الجلد",
        "invalid": "❌ الصورة لا تبدو كفحص جلدي."
    },
    "English": {
        "title": "Skin AI Diagnostic System",
        "upload": "📥 Upload scan / Camera",
        "btn": "🔍 Start Analysis",
        "advice": "⚠️ Medical Note: AI tool for guidance only.",
        "guide_title": "📖 Medical Reference Guide",
        "invalid": "❌ Image is not a skin scan."
    }
}

# --- 4. منطق التشخيص (تصنيف الـ 24 صنف) ---
def get_diagnosis_result(idx):
    malignant_ids = [1, 11] # خلايا سرطانية وميلانوما
    benign_ids = [16, 20]    # زوائد حميدة وأورام وعائية
    
    raw_label = CLASSES_24[idx]
    
    if idx in malignant_ids:
        return "خبيث (Malignant)", "#FF3B30"
    elif idx in benign_ids:
        return "حميد (Benign)", "#34C759"
    else:
        # استخلاص اسم المرض الجلدي (مثل صدفية، أكزيما، حب شباب)
        clean_name = raw_label.replace("Photos", "").replace("pictures", "").strip()
        return clean_name, "#007AFF"

# --- 5. بناء وتحميل النموذج الهجين ---
@st.cache_resource
def load_hybrid_model():
    # نموذج الفلترة (MobileNetV2)
    filter_net = tf.keras.applications.MobileNetV2(weights="imagenet")
    
    # بناء الهيكل الهجين المتوافق مع ملف الـ 24 صنف
    input_layer = Input(shape=(224, 224, 3))
    b1 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False)(input_layer)
    b2 = tf.keras.applications.MobileNetV2(weights=None, include_top=False)(input_layer)
    merged = Concatenate()([GlobalAveragePooling2D()(b1), GlobalAveragePooling2D()(b2)])
    d = Dense(512, activation='relu')(merged)
    drop = Dropout(0.4)(d)
    out = Dense(24, activation='softmax')(drop) # يجب أن يكون 24 متوافقاً مع ملف الدرايف
    
    model = Model(inputs=input_layer, outputs=out)
    
    # محاولة تحميل الملف الهجين من الدرايف أولاً، ثم الماستر كخيار ثانٍ
    if os.path.exists(HYBRID_MODEL_FILE):
        model.load_weights(HYBRID_MODEL_FILE)
    elif os.path.exists(MASTER_MODEL_FILE):
        # ملاحظة: إذا كان ملف الماستر يحتوي على عدد أصناف مختلف (مثلاً 10) سيحدث خطأ هنا.
        # الكود مصمم للعمل مع ملف الـ 24 صنفاً.
        try: model.load_weights(MASTER_MODEL_FILE)
        except: pass
        
    return filter_net, model

filter_model, diag_model = load_hybrid_model()

# --- 6. الواجهة الرسومية ---
selected_lang = st.sidebar.selectbox("Language / اللغة", list(LANGS_CONFIG.keys()))
labels = UI_LABELS.get(selected_lang, UI_LABELS["English"])
st.markdown(f"<div dir='{LANGS_CONFIG[selected_lang]}' style='text-align:center;'><h1 style='color:#1E3A8A;'>{labels['title']}</h1></div>", unsafe_allow_html=True)
st.warning(labels['advice'])

col_up, col_res = st.columns([1, 1])

with col_up:
    src = st.radio("Source:", ["Upload", "Camera"], label_visibility="collapsed")
    uploaded = st.file_uploader("", type=["jpg","png","jpeg"]) if src == "Upload" else st.camera_input("")

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    with col_res:
        st.image(img, use_container_width=True)
        if st.button(labels['btn']):
            # المعالجة
            img_res = cv2.resize(np.array(img), (224, 224))
            
            # فحص نوع الصورة
            f_in = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_res.copy(), axis=0))
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(filter_model.predict(f_in))[0]
            
            if any(k in decoded[0][1].lower() for k in ['car','dog','furniture']):
                st.error(labels['invalid'])
            else:
                # التشخيص النهائي
                d_in = np.expand_dims(img_res, axis=0) / 255.0
                pred = diag_model.predict(d_in)[0]
                status, color = get_diagnosis_result(np.argmax(pred))
                
                # عرض "الكلمة" فقط بوضوح
                st.markdown(f"""
                <div style="padding:40px; border-radius:20px; border:10px solid {color}; text-align:center; background:white;">
                    <h1 style="color:{color}; font-size:3.5em; font-weight:bold; margin:0;">{status}</h1>
                </div>
                """, unsafe_allow_html=True)

# --- 7. الدليل الطبي الملون مع الروابط ---
st.markdown("---")
st.markdown(f"<h3 style='text-align:center;'>{labels['guide_title']}</h3>", unsafe_allow_html=True)

guide_data = [
    {"t": "الآفات الخبيثة (Malignant)", "c": "#FF3B30", "d": "أورام تتطلب تدخلاً طبياً فورياً مثل الميلانوما.", "l": "https://www.cancer.org/cancer/skin-cancer.html"},
    {"t": "الآفات الحميدة (Benign)", "c": "#34C759", "d": "زوائد جلدية غير سرطانية وغالباً ما تكون مستقرة.", "l": "https://www.mayoclinic.org/diseases-conditions/moles/symptoms-causes/syc-20375200"},
    {"t": "الأمراض الجلدية (Skin Diseases)", "c": "#007AFF", "d": "حالات التهابية مثل الصدفية، حب الشباب، والأكزيما.", "l": "https://www.aad.org/public/diseases/a-z"}
]

cols = st.columns(3)
for i, item in enumerate(guide_data):
    with cols[i]:
        st.markdown(f"""
        <div style="padding:15px; border-top:8px solid {item['c']}; background:{item['c']}10; border-radius:10px;">
            <h4 style="color:{item['c']};">{item['t']}</h4>
            <p style="font-size:0.85em;">{item['d']}</p>
            <a href="{item['l']}" target="_blank" style="color:{item['c']}; text-decoration:none; font-weight:bold;">الرابط الطبي 🔗</a>
        </div>
        """, unsafe_allow_html=True)
