import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2

# --- 1. إعدادات الصفحة واللغات (كاملة لضمان الاحترافية) ---
st.set_page_config(page_title="Skin AI Expert System", layout="wide", initial_sidebar_state="collapsed")

LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "نظام الخبير الذكي لتشخيص الجلد", "upload": "📥 ارفع صورة الفحص", "cam": "📸 استخدام الكاميرا", "btn": "🔍 بدء عملية الفحص الدقيق", "invalid": "❌ الصورة غير صالحة لفحص الجلد.", "advice": "⚠️ تنبيه: هذا النظام أداة أكاديمية استرشادية ولا يغني عن استشارة الطبيب المختص."},
    "English": {"dir": "ltr", "title": "Skin AI Expert Diagnostic System", "upload": "📥 Upload Scan Image", "cam": "📸 Use Camera", "btn": "🔍 Start Deep Analysis", "invalid": "❌ Invalid skin image.", "advice": "⚠️ Note: This is an academic guidance tool and not a substitute for a professional doctor."}
}

# --- 2. الدليل الطبي المرجعي لجميع الأنواع (10 فئات) ---
# تم ضبط الألوان والبيانات لضمان عدم التداخل البصري
MEDICAL_INFO = {
    0: {"n": "Melanoma (ميلانوما)", "c": "#FF3B30", "s": "🚨 خبيث جداً", "d": "ورم صبغي عدواني يتطلب استئصالاً فورياً ومتابعة طبية دقيقة."},
    1: {"n": "Melanocytic Nevi (وحمة صبغية)", "c": "#34C759", "s": "✅ حميد", "d": "شامة طبيعية شائعة، آمنة ومستقرة، ولا تشكل خطراً صحياً."},
    2: {"n": "Basal Cell Carcinoma (BCC)", "c": "#FF9500", "s": "🚨 خبيث", "d": "سرطان الخلايا القاعدية، ينمو موضعياً ببطء ويجب علاجه لمنع تضرر الأنسجة."},
    3: {"n": "Actinic Keratosis (AK)", "c": "#AF52DE", "s": "⚠️ ما قبل سرطاني", "d": "آفات قشرية ناتجة عن التلف الضوئي، قد تتحول لسرطان إذا أُهملت."},
    4: {"n": "Benign Keratosis (BKL)", "c": "#5856D6", "s": "✅ حميد", "d": "تقرن جلدي غير سرطاني، يظهر غالباً مع التقدم في العمر."},
    5: {"n": "Dermatofibroma (DF)", "c": "#007AFF", "s": "✅ حميد", "d": "عقيدات جلدية صلبة وصغيرة، ناتجة غالباً عن رد فعل لإصابة طفيفة."},
    6: {"n": "Vascular Lesions (VASC)", "c": "#5AC8FA", "s": "✅ حميد", "d": "آفات وعائية مثل الأورام الوعائية، ناتجة عن تجمع الشعيرات الدموية."},
    7: {"n": "Squamous Cell Carcinoma", "c": "#FF2D55", "s": "🚨 خبيث", "d": "سرطان الخلايا الحرشفية، ثاني أكثر الأنواع شيوعاً ويتطلب تدخلاً جراحياً."},
    8: {"n": "Psoriasis (الصدفية)", "c": "#4CD964", "s": "🔍 حالة جلدية", "d": "اضطراب مناعي مزمن يسبب تراكم خلايا الجلد على شكل قشور فضية."},
    9: {"n": "Eczema (الأكزيما)", "c": "#FFCC00", "s": "🔍 حالة جلدية", "d": "التهاب جلدي يسبب احمراراً وحكة وجفافاً، وهو غير معدٍ تماماً."}
}

# --- 3. بناء وتحميل محركات الذكاء الاصطناعي (Hybrid Ensemble) ---
@st.cache_resource
def load_expert_system():
    # محرك التصفية لمنع المدخلات الخاطئة
    f_mod = tf.keras.applications.MobileNetV2(weights="imagenet")
    
    # بناء هيكل التشخيص الهجين
    base1 = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    base2 = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    merged = Concatenate()([GlobalAveragePooling2D()(base1.output), GlobalAveragePooling2D()(base2.output)])
    dense = Dense(512, activation='relu')(merged)
    drop = Dropout(0.4)(dense)
    output = Dense(10, activation='softmax')(drop) # 10 فئات لتشخيص شامل
    
    model = Model(inputs=[base1.input, base2.input], outputs=output)
    try:
        model.load_weights("skin_expert_master.h5")
    except:
        st.error("⚠️ ملف الأوزان 'skin_expert_master.h5' غير موجود!")
    
    return f_mod, model

filter_net, diag_net = load_expert_system()

# --- 4. واجهة المستخدم الرسومية ---
lang_choice = st.selectbox("🌐 لغة النظام / System Language", list(LANG_DATA.keys()))
ui = LANG_DATA[lang_choice]

st.markdown(f"<h1 style='text-align:center; color:#1E3A8A; font-family:Arial;'>{ui['title']}</h1>", unsafe_allow_html=True)
st.info(ui['advice'])

col1, col2 = st.columns([1, 1])

with col1:
    mode = st.radio("", [ui['upload'], ui['cam']], horizontal=True)
    up_file = st.file_uploader("", type=["jpg", "png", "jpeg"]) if "ارفع" in mode or "Upload" in mode else st.camera_input("")

if up_file:
    raw_img = Image.open(up_file).convert('RGB')
    with col2:
        st.image(raw_img, caption="Image Preview", use_container_width=True)
    
    if st.button(ui['btn'], use_container_width=True):
        with st.spinner("⏳ جاري تحليل الأنسجة الجلدية..."):
            # المعالجة المسبقة الفنية
            img_array = np.array(raw_img)
            img_resized = cv2.resize(img_array, (224, 224))
            
            # الخطوة 1: التحقق من أن الصورة هي "جلد" فعلاً
            check_inp = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_resized, axis=0))
            f_preds = filter_net.predict(check_inp)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(f_preds, top=3)[0]
            
            is_valid_skin = True
            invalid_objects = ['car', 'dog', 'cat', 'flower', 'laptop', 'screen', 'furniture']
            for _, label, score in decoded:
                if any(obj in label.lower() for obj in invalid_objects) and score > 0.45:
                    is_valid_skin = False
            
            if not is_valid_skin:
                st.error(ui['invalid'])
            else:
                # الخطوة 2: تصحيح الانحياز اللوني (Hybrid Preprocessing)
                # توازن اللون الأبيض اليدوي لكسر انحياز "لون البشرة"
                avg_val = np.mean(img_resized)
                balanced = img_resized.astype(np.float32)
                for i in range(3):
                    c_avg = np.mean(img_resized[:, :, i])
                    balanced[:, :, i] = np.clip(img_resized[:, :, i] * (avg_val / c_avg), 0, 255)
                
                # تحسين التباين (CLAHE) لإبراز حدود الإصابة
                lab = cv2.cvtColor(balanced.astype(np.uint8), cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8,8))
                final_proc = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)
                
                # الخطوة 3: التشخيص مع كسر الانحياز (Calibration Logic)
                final_inp = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(final_proc, axis=0))
                preds = diag_net.predict([final_inp, final_inp])[0]
                
                # مصفوفة الأوزان التصحيحية لضمان عدم طغيان فئة واحدة (BCC أو الحميد)
                # قمنا بتقليل وزن الفئات التي تظهر بكثرة وزيادة وزن الفئات النادرة
                calibration_weights = np.array([1.2, 0.75, 0.70, 1.0, 0.9, 1.1, 1.1, 1.2, 1.0, 1.0])
                balanced_preds = preds * calibration_weights
                
                final_idx = np.argmax(balanced_preds)
                res = MEDICAL_INFO[final_idx]
                
                # عرض النتيجة بتصميم احترافي
                st.markdown(f"""
                <div style="border: 8px solid {res['c']}; padding: 30px; border-radius: 20px; background-color: #fcfcfc; text-align: center; margin-top: 20px;">
                    <h1 style="color: {res['c']}; font-size: 2.8em; margin-bottom: 10px;">{res['n']}</h1>
                    <h2 style="color: #333;">التصنيف: {res['s']}</h2>
                    <hr style="border: 1px solid {res['c']}; width: 50%; margin: 20px auto;">
                    <p style="font-size: 1.4em; color: #444; line-height: 1.6;">{res['d']}</p>
                </div>
                """, unsafe_allow_html=True)

# --- 5. الدليل المرجعي التفاعلي ---
st.write("---")
with st.expander("📖 الدليل الطبي الشامل لسرطان وآفات الجلد"):
    sel = st.selectbox("اختر نوع المرض لعرض تفاصيله:", [v['n'] for v in MEDICAL_INFO.values()])
    for k, v in MEDICAL_INFO.items():
        if v['n'] == sel:
            st.markdown(f"""
            <div style="padding: 20px; border-right: 10px solid {v['c']}; background-color: {v['c']}10;">
                <h3 style="color: {v['c']};">{v['n']}</h3>
                <p><strong>حالة الخطورة:</strong> {v['s']}</p>
                <p><strong>الوصف السريري:</strong> {v['d']}</p>
            </div>
            """, unsafe_allow_html=True)
