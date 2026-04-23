import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. إعدادات الصفحة العامة ---
st.set_page_config(page_title="Skin AI System - نظام فحص الجلد", layout="wide")

# أسماء الملفات كما تم الاتفاق عليها
HYBRID_FILE = "skin_expert_hybrid_24ch.h5"  # الملف المرتبط بـ 24 صنف (في درايف)
MASTER_FILE = "skin_expert_master.h5"       # الملف الموجود على جيت هب
DRIVE_LINK = "https://drive.google.com/file/d/135lZpgsipHNk2IZBo6H4lZZ9WzVizLqb/view?usp=sharing"

# --- 2. تعريف اللغات (بدون تغيير في الترتيب أو الواجهة) ---
LANGS_CONFIG = {
    "العربية": "rtl", "English": "ltr", "Français": "ltr", "Deutsch": "ltr",
    "Español": "ltr", "Türkçe": "ltr", "Русский": "ltr", "中文": "ltr",
    "हिन्दी": "ltr", "Kurdî": "rtl"
}

UI_LABELS = {
    "العربية": {
        "title": "نظام الفحص الذكي للجلد",
        "upload": "📥 ارفع صورة الفحص أو استخدم الكاميرا",
        "btn": "🔍 بدء تحليل الصورة",
        "advice": "⚠️ تنبيه طبي: هذا النظام أداة برمجية استرشادية تعتمد على الذكاء الاصطناعي، ولا يغني عن زيارة الطبيب المختص.",
        "guide_title": "📖 الدليل الطبي المرجعي للآفات والنوع",
        "invalid": "❌ الصورة المرفوعة لا تبدو كفحص جلدي، يرجى التأكد من الصورة."
    },
    "English": {
        "title": "Skin AI Diagnostic System",
        "upload": "📥 Upload scan or use camera",
        "btn": "🔍 Start Analysis",
        "advice": "⚠️ Medical Note: This AI tool is for guidance only and is not a substitute for a doctor.",
        "guide_title": "📖 Medical Reference Guide",
        "invalid": "❌ The image does not appear to be a skin scan."
    }
}

# --- 3. قائمة الأصناف الـ 24 (الترتيب الدقيق لملف الدرايف) ---
CLASSES_24 = [
    'Acne and Rosacea Photos', 
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 
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

def get_diagnosis_info(idx):
    # التصنيف بناءً على الرقم المستخرج من الموديل
    malignant_indices = [1, 11] # سرطان الخلايا القاعدية والميلانوما
    benign_indices = [16, 20]    # الأورام الحميدة والوعائية
    
    label_raw = CLASSES_24[idx]
    
    if idx in malignant_indices:
        return "خبيث (Malignant)", "#FF3B30"
    elif idx in benign_indices:
        return "حميد (Benign)", "#34C759"
    else:
        # حالات أخرى مثل الصدفية، حب الشباب، إلخ
        clean_name = label_raw.replace("Photos", "").replace("pictures", "").strip()
        return f"{clean_name}", "#007AFF"

# --- 4. تحميل النماذج (الهجين) ---
@st.cache_resource
def load_ai_model():
    filter_net = tf.keras.applications.MobileNetV2(weights="imagenet")
    
    # بناء هيكل الـ Hybrid 24ch
    input_layer = Input(shape=(224, 224, 3))
    b1 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False)(input_layer)
    b2 = tf.keras.applications.MobileNetV2(weights=None, include_top=False)(input_layer)
    merged = Concatenate()([GlobalAveragePooling2D()(b1), GlobalAveragePooling2D()(b2)])
    d = Dense(512, activation='relu')(merged)
    drop = Dropout(0.4)(d)
    out = Dense(24, activation='softmax')(drop)
    
    model = Model(inputs=input_layer, outputs=out)
    
    # تحميل الأوزان (الأولوية لملف الـ 24 صنف)
    if os.path.exists(HYBRID_FILE):
        model.load_weights(HYBRID_FILE)
    elif os.path.exists(MASTER_FILE):
        try: model.load_weights(MASTER_FILE)
        except: pass
        
    return filter_net, model

filter_model, diag_model = load_ai_model()

# --- 5. واجهة المستخدم ---
st.sidebar.markdown("### 🌐 Settings / الإعدادات")
selected_lang = st.sidebar.selectbox("Language / اللغة", list(LANGS_CONFIG.keys()))
current_dir = LANGS_CONFIG[selected_lang]
labels = UI_LABELS.get(selected_lang, UI_LABELS["English"])

st.markdown(f"<div dir='{current_dir}' style='text-align:center;'><h1 style='color:#1E3A8A;'>{labels['title']}</h1></div>", unsafe_allow_html=True)
st.warning(labels['advice'])

col_up, col_res = st.columns([1, 1])

with col_up:
    st.markdown(f"<div dir='{current_dir}'><strong>{labels['upload']}</strong></div>", unsafe_allow_html=True)
    src_opt = st.radio("", ["Upload Image", "Use Camera"], label_visibility="collapsed")
    uploaded = st.file_uploader("", type=["jpg", "png", "jpeg"]) if src_opt == "Upload Image" else st.camera_input("")

if uploaded:
    img_in = Image.open(uploaded).convert('RGB')
    with col_res:
        st.image(img_in, use_container_width=True)
        if st.button(labels['btn']):
            with st.spinner("⏳ جاري التحليل..."):
                img_arr = np.array(img_in)
                img_res = cv2.resize(img_arr, (224, 224))
                
                # التحقق من أن الصورة للجلد
                f_in = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_res.copy(), axis=0))
                preds = filter_model.predict(f_in)
                decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]
                
                if any(k in decoded[0][1].lower() for k in ['car', 'dog', 'furniture', 'building']):
                    st.error(labels['invalid'])
                else:
                    # التشخيص (عرض الكلمة فقط)
                    d_in = np.expand_dims(img_res, axis=0) / 255.0
                    prediction = diag_model.predict(d_in)[0]
                    res_idx = np.argmax(prediction)
                    status, color = get_diagnosis_info(res_idx)
                    
                    st.markdown(f"""
                    <div style="padding:40px; border-radius:20px; border:10px solid {color}; text-align:center; background:white; box-shadow: 0px 4px 15px rgba(0,0,0,0.1);">
                        <h1 style="color:{color}; font-size:3.5em; font-weight:900; margin:0;">{status}</h1>
                    </div>
                    """, unsafe_allow_html=True)

# --- 6. الدليل الطبي الملون مع الروابط ---
st.markdown("---")
st.markdown(f"<h3 style='text-align:center;'>{labels['guide_title']}</h3>", unsafe_allow_html=True)

guide_data = [
    {"t": "الأورام الخبيثة (Malignant)", "c": "#FF3B30", "d": "تنمو بسرعة وتتطلب فحصاً طبياً فورياً. تشمل الميلانوما والسرطان القاعدي.", "l": "https://www.cancer.org/cancer/skin-cancer.html"},
    {"t": "الأورام الحميدة (Benign)", "c": "#34C759", "d": "نمو غير سرطاني مثل الشامات العادية والزوائد الجلدية.", "l": "https://www.mayoclinic.org/diseases-conditions/moles/symptoms-causes/syc-20375200"},
    {"t": "أمراض جلدية (Psoriasis/Acne)", "c": "#007AFF", "d": "حالات مثل الصدفية وحب الشباب، وهي التهابات جلدية شائعة.", "l": "https://www.aad.org/public/diseases/a-z"}
]

g_cols = st.columns(3)
for i, item in enumerate(guide_data):
    with g_cols[i]:
        st.markdown(f"""
        <div style="padding:20px; border-top:10px solid {item['c']}; background:{item['c']}10; border-radius:10px; height:200px;">
            <h4 style="color:{item['c']};">{item['t']}</h4>
            <p style="font-size:0.9em;">{item['d']}</p>
            <a href="{item['l']}" target="_blank" style="color:{item['c']}; font-weight:bold; text-decoration:none;">المصدر العالمي 🔗</a>
        </div>
        """, unsafe_allow_html=True)

st.markdown(f"<p style='text-align:center; color:grey;'>رابط الملف الكبير: <a href='{DRIVE_LINK}'>تحميل skin_expert_hybrid_24ch.h5</a></p>", unsafe_allow_html=True)
