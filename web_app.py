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
    page_title="Skin AI Expert | Mosul University",
    layout="wide",
    initial_sidebar_state="expanded"
)

# القاموس الشامل للغات (يدعم 20 لغة)
LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "نظام التشخيص العالمي الذكي للجلد", "upload": "📥 ارفع صورة", "cam": "📸 كاميرا", "btn": "🔍 تحليل الأنسجة", "invalid": "❌ الصورة ليست فحصاً جلدياً.", "advice": "⚠️ تنبيه: استشر الطبيب فوراً."},
    "English": {"dir": "ltr", "title": "Global AI Skin Diagnostic", "upload": "📥 Upload", "cam": "📸 Camera", "btn": "🔍 Analyze Tissue", "invalid": "❌ Invalid Image.", "advice": "⚠️ Note: Consult a doctor."},
    "Français": {"dir": "ltr", "title": "Diagnostic Cutané IA", "upload": "📥 Charger", "cam": "📸 Caméra", "btn": "🔍 Analyser", "invalid": "❌ Invalide.", "advice": "⚠️ Consultez un médecin."},
    "Español": {"dir": "ltr", "title": "IA Diagnóstico de Piel", "upload": "📥 Subir", "cam": "📸 Cámara", "btn": "🔍 Analizar", "invalid": "❌ Imagen inválida.", "advice": "⚠️ Consulte a un médico."},
    "Deutsch": {"dir": "ltr", "title": "KI Hautdiagnose", "upload": "📥 Hochladen", "cam": "📸 Kamera", "btn": "🔍 Analyse", "invalid": "❌ Ungültig.", "advice": "⚠️ Arzt aufsuchen."},
    "中文": {"dir": "ltr", "title": "皮肤人工智能诊断", "upload": "📥 上传", "cam": "📸 相机", "btn": "🔍 分析", "invalid": "❌ 无效图像。", "advice": "⚠️ 请咨询医生。"},
    "हिन्दी": {"dir": "ltr", "title": "त्वचा एआई निदान", "upload": "📥 अपलोड", "cam": "📸 कैमरा", "btn": "🔍 विश्लेषण", "invalid": "❌ अमान्य।", "advice": "⚠️ डॉक्टर से मिलें।"},
    "Русский": {"dir": "ltr", "title": "ИИ диагностика кожи", "upload": "📥 Загрузить", "cam": "📸 Камера", "btn": "🔍 Анализ", "invalid": "❌ Ошибка.", "advice": "⚠️ Обратитесь к врачу."},
    "日本語": {"dir": "ltr", "title": "皮膚AI診断", "upload": "📥 アップロード", "cam": "📸 カメラ", "btn": "🔍 解析", "invalid": "❌ 無効。", "advice": "⚠️ 医師に相談。"},
    "Português": {"dir": "ltr", "title": "IA Pele", "upload": "📥 Carregar", "cam": "📸 Câmera", "btn": "🔍 Analisar", "invalid": "❌ Inválido.", "advice": "⚠️ Consulte médico."},
    "Türkçe": {"dir": "ltr", "title": "Cilt AI", "upload": "📥 Yükle", "cam": "📸 Kamera", "btn": "🔍 Analiz", "invalid": "❌ Geçersiz.", "advice": "⚠️ Doktora danışın."},
    "한국어": {"dir": "ltr", "title": "피부 AI", "upload": "📥 업로드", "cam": "📸 카메라", "btn": "🔍 분석", "invalid": "❌ 무효.", "advice": "⚠️ 의사 상담."},
    "Italiano": {"dir": "ltr", "title": "IA Pelle", "upload": "📥 Carica", "cam": "📸 Camera", "btn": "🔍 Analizza", "invalid": "❌ Invalido.", "advice": "⚠️ Consulti medico."},
    "اردو": {"dir": "rtl", "title": "جلد کی تشخیص", "upload": "📥 اپلوڈ", "cam": "📸 کیمرہ", "btn": "🔍 معائنہ", "invalid": "❌ تصویر درست نہیں۔", "advice": "⚠️ ڈاکٹر سے ملیں۔"},
    "فارسي": {"dir": "rtl", "title": "هوش مصنوعی پوست", "upload": "📥 بارگذاری", "cam": "📸 دوربین", "btn": "🔍 آنالیز", "invalid": "❌ نامعتبر.", "advice": "⚠️ پزشک بروید."},
    "Tiếng Việt": {"dir": "ltr", "title": "AI Da liễu", "upload": "📥 Tải lên", "cam": "📸 Máy ảnh", "btn": "🔍 Phân tích", "invalid": "❌ Lỗi.", "advice": "⚠️ Gặp bác sĩ."},
    "Bahasa Indonesia": {"dir": "ltr", "title": "AI Kulit", "upload": "📥 Unggah", "cam": "📸 Kamera", "btn": "🔍 Analisis", "invalid": "❌ Gagal.", "advice": "⚠️ Hubungi dokter."},
    "Nederlands": {"dir": "ltr", "title": "Huid AI", "upload": "📥 Upload", "cam": "📸 Camera", "btn": "🔍 Analyse", "invalid": "❌ Ongeldig.", "advice": "⚠️ Raadpleeg arts."},
    "Polski": {"dir": "ltr", "title": "AI Skóry", "upload": "📥 Prześlij", "cam": "📸 Kamera", "btn": "🔍 Analiza", "invalid": "❌ Błąd.", "advice": "⚠️ Idź do lekarزا."},
    "Kurdî": {"dir": "rtl", "title": "ژیری پێست", "upload": "📥 وێنە", "cam": "📸 کامێرا", "btn": "🔍 شیکاری", "invalid": "❌ هەڵە.", "advice": "⚠️ پزیشک ببینە."}
}

# --- 2. الدليل الطبي الملون ونظام الأوزان ---
MEDICAL_INFO = {
    0: {"n": "Melanoma (ميلانوما)", "c": "#D32F2F", "s": "🚨 خبيث جداً", "w": 1.40, "d": "أخطر سرطان جلدي، يتطلب تدخلاً طبياً عاجلاً."},
    1: {"n": "Melanocytic Nevi (وحمة)", "c": "#388E3C", "s": "✅ حميد", "w": 0.65, "d": "شامة طبيعية آمنة وغير خطيرة."},
    2: {"n": "Basal Cell Carcinoma (BCC)", "c": "#F57C00", "s": "🚨 خبيث", "w": 0.60, "d": "سرطان قاعدي ينمو ببطء ويحتاج استئصالاً."},
    3: {"n": "Actinic Keratosis (AK)", "c": "#7B1FA2", "s": "⚠️ ما قبل سرطاني", "w": 1.10, "d": "تلف شمسي قد يتطور لسرطان بمرور الوقت."},
    4: {"n": "Benign Keratosis (BKL)", "c": "#1976D2", "s": "✅ حميد", "w": 0.85, "d": "زوائد غير سرطانية شائعة مع تقدم العمر."},
    5: {"n": "Dermatofibroma (DF)", "c": "#00796B", "s": "✅ حميد", "w": 1.20, "d": "كتلة صلبة صغيرة تظهر بعد إصابة طفيفة."},
    6: {"n": "Vascular Lesions (VASC)", "c": "#C2185B", "s": "✅ حميد", "w": 1.25, "d": "آفات وعائية ناتجة عن تمدد الشعيرات."},
    7: {"n": "Squamous Cell Carcinoma", "c": "#E64A19", "s": "🚨 خبيث", "w": 1.30, "d": "سرطان حرشفي يحتاج تدخلاً جراحياً."},
    8: {"n": "Psoriasis (الصدفية)", "c": "#512DA8", "s": "🔍 حالة جلدية", "w": 1.00, "d": "مرض مناعي يسبب التهاباً وقشوراً فضية."},
    9: {"n": "Eczema (الأكزيما)", "c": "#FFA000", "s": "🔍 حالة جلدية", "w": 1.10, "d": "التهاب تحسسي يسبب جفافاً وحكة شديدة."}
}

# --- 3. بناء هيكلية النموذج (Ensemble Architecture) ---
@st.cache_resource
def load_full_system():
    # بناء هيكلية قوية لحل مشكلة ValueError
    input_layer = Input(shape=(224, 224, 3))
    
    # دمج موديلين للوصول لأعلى دقة
    base_1 = EfficientNetB0(weights=None, include_top=False)(input_layer)
    base_2 = MobileNetV2(weights=None, include_top=False)(input_layer)
    
    gap_1 = GlobalAveragePooling2D()(base_1)
    gap_2 = GlobalAveragePooling2D()(base_2)
    
    merged = Concatenate()([gap_1, gap_2])
    dense = Dense(512, activation='relu')(merged)
    dropout = Dropout(0.4)(dense)
    output = Dense(10, activation='softmax')(dropout)
    
    full_model = Model(inputs=input_layer, outputs=output)
    
    # تحميل الأوزان
    weights_file = "skin_expert_master.h5"
    ready = False
    if os.path.exists(weights_file):
        full_model.load_weights(weights_file)
        ready = True
    
    # موديل إضافي لفلترة الصور الخارجية
    filter_model = tf.keras.applications.MobileNetV2(weights="imagenet")
    
    return full_model, filter_model, ready

main_model, filter_m, is_ready = load_full_system()

# --- 4. واجهة المستخدم والتصميم ---
sel_lang = st.sidebar.selectbox("🌐 لغة النظام / Language", list(LANG_DATA.keys()))
ui = LANG_DATA[sel_lang]

st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
        * {{ direction: {ui['dir']}; font-family: 'Tajawal', sans-serif; }}
        .title {{ text-align: center; color: #0d47a1; padding: 20px; font-size: 2.5em; }}
    </style>
""", unsafe_allow_html=True)

st.markdown(f"<div class='title'>{ui['title']}</div>", unsafe_allow_html=True)
st.info(ui['advice'])

col_a, col_b = st.columns(2)
with col_a:
    method = st.radio("", [ui['upload'], ui['cam']], horizontal=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"]) if "ارفع" in method or "Upload" in method else st.camera_input("")

if uploaded_file and is_ready:
    original_image = Image.open(uploaded_file).convert('RGB')
    with col_b:
        st.image(original_image, caption="الصورة المرفوعة", use_container_width=True)
    
    if st.button(ui['btn'], use_container_width=True):
        with st.spinner("⏳ جاري التحليل..."):
            # تجهيز ومعالجة الصور
            img_np = np.array(original_image)
            img_res = cv2.resize(img_np, (224, 224))
            
            # 1. فلترة الصور الخارجية (المنع)
            xf = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_res, axis=0))
            f_preds = filter_m.predict(xf)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(f_preds, top=3)[0]
            
            is_skin = True
            for _, label, score in decoded:
                if any(x in label.lower() for x in ['car', 'wheel', 'dog', 'flower', 'screen', 'laptop']) and score > 0.35:
                    is_skin = False

            if not is_skin:
                st.error(ui['invalid'])
            else:
                # 2. تحسين الصورة (White Balance & CLAHE) لكسر الانحياز
                avg_gray = np.mean(img_res)
                wb_img = img_res.astype(np.float32)
                for i in range(3):
                    wb_img[:, :, i] = np.clip(img_res[:, :, i] * (avg_gray / np.mean(img_res[:, :, i])), 0, 255)
                
                lab = cv2.cvtColor(wb_img.astype(np.uint8), cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                l_en = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
                final_img = cv2.cvtColor(cv2.merge((l_en, a, b)), cv2.COLOR_LAB2RGB)

                # 3. التشخيص وتطبيق الأوزان
                inp_tensor = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(final_img, axis=0))
                raw_preds = main_model.predict(inp_tensor)[0]
                
                # تطبيق مصفوفة المعايرة لمنع الانحياز
                cal_w = np.array([v['w'] for v in MEDICAL_INFO.values()])
                final_idx = np.argmax(raw_preds * cal_w)
                
                res = MEDICAL_INFO[final_idx]
                st.markdown(f"""
                    <div style="border: 8px solid {res['c']}; padding: 30px; border-radius: 20px; background: white; text-align: center;">
                        <h1 style="color: {res['c']};">{res['n']}</h1>
                        <h2 style="background: #f0f0f0; border-radius: 10px;">{res['s']}</h2>
                        <hr>
                        <p style="font-size: 1.3em;">{res['d']}</p>
                        <p>نسبة التأكد: {raw_preds[final_idx]*100:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)

# --- 5. الدليل المرجعي الكامل ---
st.write("---")
st.subheader("📖 الدليل الطبي المرجعي")
selected_name = st.selectbox("اختر نوع الإصابة لعرض تفاصيلها:", [v['n'] for v in MEDICAL_INFO.values()])

for k, v in MEDICAL_INFO.items():
    if v['n'] == selected_name:
        st.markdown(f"""
            <div style="background-color:{v['c']}10; padding:20px; border-right:10px solid {v['c']}; border-radius:10px;">
                <h3 style="color:{v['c']};">{v['n']}</h3>
                <p><strong>التصنيف:</strong> {v['s']}</p>
                <p><strong>التشخيص:</strong> {v['d']}</p>
            </div>
        """, unsafe_allow_html=True)

if not is_ready:
    st.error("❌ تحذير: ملف الأوزان 'skin_expert_master.h5' مفقود. يرجى رفعه بجانب الكود.")
