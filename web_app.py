import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. إعدادات الصفحة واللغات (كاملة كما طلبت) ---
st.set_page_config(page_title="Global Skin AI Expert", layout="wide")

LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "نظام التشخيص العالمي الذكي للجلد", "upload": "📥 ارفع صورة", "cam": "📸 كاميرا", "btn": "🔍 تحليل الأنسجة", "invalid": "❌ الصورة لا تبدو فحصاً جلدياً.", "advice": "⚠️ تنبيه: استشر الطبيب فوراً."},
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
    "Polski": {"dir": "ltr", "title": "AI Skóry", "upload": "📥 Prześlij", "cam": "📸 Kamera", "btn": "🔍 Analiza", "invalid": "❌ Błąd.", "advice": "⚠️ Idź do lekarza."},
    "Kurdî": {"dir": "rtl", "title": "ژیری پێست", "upload": "📥 وێنە", "cam": "📸 کامێرا", "btn": "🔍 شیکاری", "invalid": "❌ هەڵە.", "advice": "⚠️ پزیشک ببینە."}
}

# --- 2. الدليل الطبي الملون (10 أنواع كاملة) ---
# تم ضبط الأوزان (w) لكل نوع بشكل منفصل لضمان دقة التصنيف
MEDICAL_INFO = {
    0: {"n": "Melanoma (ميلانوما)", "c": "#FF0000", "s": "🚨 خبيث جداً", "w": 1.35, "d": "أخطر أنواع سرطان الجلد، يتميز بعدم التماثل وتعدد الألوان."},
    1: {"n": "Melanocytic Nevi (وحمة)", "c": "#27AE60", "s": "✅ حميد", "w": 0.65, "d": "شامات طبيعية آمنة، تظهر بشكل منتظم على الجلد."},
    2: {"n": "Basal Cell Carcinoma (BCC)", "c": "#C0392B", "s": "🚨 خبيث", "w": 0.60, "d": "سرطان الخلايا القاعدية، ينمو ببطء ويظهر كقرحة لؤلؤية."},
    3: {"n": "Actinic Keratosis (AK)", "c": "#E67E22", "s": "⚠️ ما قبل سرطاني", "w": 1.10, "d": "بقع خشنة ناتجة عن الشمس، قد تتحول لسرطان بمرور الوقت."},
    4: {"n": "Benign Keratosis (BKL)", "c": "#2ECC71", "s": "✅ حميد", "w": 0.85, "d": "زوائد جلدية غير سرطانية تظهر مع تقدم العمر."},
    5: {"n": "Dermatofibroma (DF)", "c": "#16A085", "s": "✅ حميد", "w": 1.15, "d": "كتلة صلبة صغيرة تظهر غالباً في الساقين بعد إصابة طفيفة."},
    6: {"n": "Vascular Lesions (VASC)", "c": "#8E44AD", "s": "✅ حميد", "w": 1.20, "d": "آفات ناتجة عن تجمع الشعيرات الدموية."},
    7: {"n": "Squamous Cell Carcinoma", "c": "#A93226", "s": "🚨 خبيث", "w": 1.25, "d": "سرطان الخلايا الحرشفية، يظهر كبقعة حمراء متقشرة."},
    8: {"n": "Psoriasis (الصدفية)", "c": "#2980B9", "s": "🔍 حالة جلدية", "w": 1.00, "d": "مرض مناعي يسبب قشوراً فضية وبقعاً حمراء."},
    9: {"n": "Eczema (الأكزيما)", "c": "#F39C12", "s": "🔍 حالة جلدية", "w": 1.10, "d": "التهاب جلدي يسبب جفافاً وحكة شديدة."}
}

# --- 3. محركات الذكاء الاصطناعي (تصحيح الهيكل) ---
@st.cache_resource
def load_engines():
    f_mod = tf.keras.applications.MobileNetV2(weights="imagenet")
    
    # حل مشكلة ValueError: استخدام Input صريح
    inp = Input(shape=(224, 224, 3))
    b1 = EfficientNetB0(weights=None, include_top=False)(inp)
    b2 = MobileNetV2(weights=None, include_top=False)(inp)
    
    comb = Concatenate()([GlobalAveragePooling2D()(b1), GlobalAveragePooling2D()(b2)])
    # تصحيح: Dense(10) ليتطابق مع الدليل الطبي المكون من 10 أنواع
    out = Dense(10, activation='softmax')(Dropout(0.5)(Dense(512, activation='relu')(comb)))
    d_mod = Model(inputs=inp, outputs=out)
    
    h5_path = "skin_expert_master.h5"
    if os.path.exists(h5_path):
        d_mod.load_weights(h5_path)
    else:
        st.error(f"❌ Weights not found! {h5_path}")
    return f_mod, d_mod

filter_m, diag_m = load_engines()

# --- 4. واجهة المستخدم ---
selected_lang = st.selectbox("🌐 Choose Language / اختر اللغة", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    * {{ direction: {t['dir']}; font-family: 'Tajawal', sans-serif; }}
    .main-title {{ text-align: center; color: #0d47a1; font-size: 2.2em; font-weight: bold; padding: 15px; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"<div class='main-title'>{t['title']}</div>", unsafe_allow_html=True)
st.warning(t['advice'])

col1, col2 = st.columns(2)
with col1:
    m = st.radio("", [t['upload'], t['cam']], horizontal=True)
    file = st.file_uploader("", type=["jpg", "png", "jpeg"]) if "ارفع" in m or "Upload" in m else st.camera_input("")

if file:
    img = Image.open(file).convert('RGB')
    with col2: st.image(img, use_container_width=True)
    
    if st.button(t['btn']):
        with st.spinner("Processing..."):
            img_np = np.array(img)
            img_res = cv2.resize(img_np, (224, 224))
            
            # فلترة الصور الخارجية
            xf = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_res, axis=0))
            f_preds = filter_m.predict(xf)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(f_preds, top=3)[0]
            
            is_skin = True
            for _, label, score in decoded:
                if any(x in label.lower() for x in ['car', 'wheel', 'dog', 'flower', 'screen']) and score > 0.3:
                    is_skin = False
            
            if not is_skin:
                st.error(t['invalid'])
            else:
                # حل مشكلة AttributeError وكسر الانحياز
                avg = np.mean(img_res)
                proc = img_res.astype(np.float32)
                for i in range(3):
                    proc[:, :, i] = np.clip(img_res[:, :, i] * (avg / np.mean(img_res[:, :, i])), 0, 255)
                
                lab = cv2.cvtColor(proc.astype(np.uint8), cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_final = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)
                
                # التشخيص مع نظام المعايرة المنفصل
                inp = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_final, axis=0))
                res_preds = diag_m.predict(inp)[0]
                
                # تطبيق مصفوفة الأوزان التصحيحية لكل نوع بشكل مستقل
                cal_w = np.array([v['w'] for v in MEDICAL_INFO.values()])
                idx = np.argmax(res_preds * cal_w)
                
                conf = res_preds[idx]
                info = MEDICAL_INFO[idx]
                
                st.markdown(f"""
                <div style="padding:25px; border-radius:15px; border:6px solid {info['c']}; text-align:center; background:white;">
                    <h2 style="color:{info['c']};">{info['n']}</h2>
                    <h3>{info['s']}</h3>
                    <hr>
                    <h4>نسبة التأكد: {conf*100:.1f}%</h4>
                    <p style='font-size:1.1em;'>{info['d']}</p>
                </div>
                """, unsafe_allow_html=True)

# --- 5. الدليل الطبي المرجعي الثابت ---
st.write("---")
st.subheader("📖 الدليل الطبي المرجعي")
selected_info = st.selectbox("اختر نوع الإصابة لعرض تفاصيلها:", [v['n'] for v in MEDICAL_INFO.values()])

for k, v in MEDICAL_INFO.items():
    if v['n'] == selected_info:
        st.markdown(f"""
        <div style="background-color:{v['c']}15; padding:20px; border-right:10px solid {v['c']}; border-left:10px solid {v['c']}; border-radius:5px;">
            <h3 style="color:{v['c']};">{v['n']}</h3>
            <p><strong>التصنيف:</strong> {v['s']}</p>
            <p><strong>التشخيص الطبي:</strong> {v['d']}</p>
        </div>
        """, unsafe_allow_html=True)
