import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. قاعدة بيانات اللغات (20 لغة كاملة ومثبتة) ---
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
    "Italiano": {"dir": "ltr", "title": "IA Pelle", " carica": "📥 Carica", "cam": "📸 Camera", "btn": "🔍 Analizza", "invalid": "❌ Invalido.", "advice": "⚠️ Consulti medico."},
    "اردو": {"dir": "rtl", "title": "جلد کی تشخیص", "upload": "📥 اپلوڈ", "cam": "📸 کیمرہ", "btn": "🔍 معائنہ", "invalid": "❌ تصویر درست نہیں۔", "advice": "⚠️ ڈاکٹر سے ملیں۔"},
    "فارسي": {"dir": "rtl", "title": "هوش مصنوعی پوست", "upload": "📥 بارگذاری", "cam": "📸 دوربین", "btn": "🔍 آناليز", "invalid": "❌ نامعتبر.", "advice": "⚠️ پزشک بروید."},
    "Tiếng Việt": {"dir": "ltr", "title": "AI Da liễu", "upload": "📥 Tải lên", "cam": "📸 Máy ảnh", "btn": "🔍 Phân tích", "invalid": "❌ Lỗi.", "advice": "⚠️ Gặp bác sĩ."},
    "Bahasa Indonesia": {"dir": "ltr", "title": "AI Kulit", "upload": "📥 Unggah", "cam": "📸 Kamera", "btn": "🔍 Analisis", "invalid": "❌ Gagal.", "advice": "⚠️ Hubungi dokter."},
    "Nederlands": {"dir": "ltr", "title": "Huid AI", "upload": "📥 Upload", "cam": "📸 Camera", "btn": "🔍 Analyse", "invalid": "❌ Ongeldig.", "advice": "⚠️ Raadpleeg arts."},
    "Polski": {"dir": "ltr", "title": "AI Skóry", "upload": "📥 Prześlij", "cam": "📸 Kamera", "btn": "🔍 Analiza", "invalid": "❌ Błąd.", "advice": "⚠️ Idź do lekarza."},
    "Kurdî": {"dir": "rtl", "title": "ژیری پێست", "upload": "📥 وێنە", "cam": "📸 کامێرا", "btn": "🔍 شیکاری", "invalid": "❌ هەڵە.", "advice": "⚠️ پزیشک ببینە."}
}

# --- 2. الدليل الطبي المرجعي (10 أنواع ثابتة مع أوزان المعايرة) ---
MEDICAL_INFO = {
    0: {"n": "Melanoma (ميلانوما)", "c": "#FF0000", "s": "🚨 خبيث جداً", "w": 1.45, "d": "أخطر أنواع سرطان الجلد، يتطلب تدخلاً طبياً فورياً."},
    1: {"n": "Melanocytic Nevi (وحمة)", "c": "#27AE60", "s": "✅ حميد", "w": 0.65, "d": "شامات طبيعية آمنة، تظهر بشكل منتظم على الجلد."},
    2: {"n": "Basal Cell Carcinoma (BCC)", "c": "#C0392B", "s": "🚨 خبيث", "w": 1.25, "d": "سرطان الخلايا القاعدية، ينمو ببطء كقرحة لؤلؤية."},
    3: {"n": "Actinic Keratosis (AK)", "c": "#E67E22", "s": "⚠️ ما قبل سرطاني", "w": 1.15, "d": "بقع خشنة ناتجة عن الشمس، قد تسبق السرطان."},
    4: {"n": "Benign Keratosis (BKL)", "c": "#2ECC71", "s": "✅ حميد", "w": 0.85, "d": "زوائد جلدية غير سرطانية مرتبطة بتقدم السن."},
    5: {"n": "Dermatofibroma (DF)", "c": "#16A085", "s": "✅ حميد", "w": 1.10, "d": "كتلة صلبة صغيرة تظهر غالباً في الساقين."},
    6: {"n": "Vascular Lesions (VASC)", "c": "#8E44AD", "s": "✅ حميد", "w": 1.20, "d": "آفات ناتجة عن تجمعات الأوعية الدموية."},
    7: {"n": "Squamous Cell Carcinoma", "c": "#A93226", "s": "🚨 خبيث", "w": 1.30, "d": "سرطان الخلايا الحرشفية، يظهر كبقعة حمراء متقشرة."},
    8: {"n": "Psoriasis (الصدفية)", "c": "#2980B9", "s": "🔍 حالة جلدية", "w": 1.00, "d": "مرض مناعي يسبب قشوراً فضية وبقعاً حمراء."},
    9: {"n": "Eczema (الأكزيما)", "c": "#F39C12", "s": "🔍 حالة جلدية", "w": 1.10, "d": "التهاب يسبب جفافاً وحكة شديدة بالجلد."}
}

# --- 3. محركات الذكاء الاصطناعي (الحل الجذري للـ ValueError) ---
@st.cache_resource
def load_engines():
    f_mod = tf.keras.applications.MobileNetV2(weights="imagenet")
    
    inp = Input(shape=(224, 224, 3), name="main_input")
    
    # تفادي تضارب الأسماء بتمييز كل فرع
    base_eff = EfficientNetB0(weights=None, include_top=False, input_tensor=inp)
    for layer in base_eff.layers: layer._name = f"eff_{layer.name}"
    
    base_mob = MobileNetV2(weights=None, include_top=False, input_tensor=inp)
    for layer in base_mob.layers: layer._name = f"mob_{layer.name}"
    
    gap_eff = GlobalAveragePooling2D(name="gap_eff")(base_eff.output)
    gap_mob = GlobalAveragePooling2D(name="gap_mob")(base_mob.output)
    comb = Concatenate(name="fusion_layer")([gap_eff, gap_mob])
    
    x = Dense(512, activation='relu', name="fc_mid")(comb)
    x = Dropout(0.5, name="dropout_mid")(x)
    out = Dense(10, activation='softmax', name="final_output")(x)
    
    d_mod = Model(inputs=inp, outputs=out)
    
    h5_path = "skin_expert_master.h5"
    if os.path.exists(h5_path):
        try:
            d_mod.load_weights(h5_path, by_name=False)
            st.sidebar.success("✅ Weights Ready")
        except: st.sidebar.error("⚠️ Weights Mismatch")
    else: st.sidebar.warning("❌ Missing .h5 File")
    
    return f_mod, d_mod

filter_m, diag_m = load_engines()

# --- 4. واجهة المستخدم ---
selected_lang = st.sidebar.selectbox("🌐 Choose Language / اختر اللغة", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    * {{ direction: {t['dir']}; font-family: 'Tajawal', sans-serif; text-align: {'right' if t['dir']=='rtl' else 'left'}; }}
    .main-title {{ text-align: center; color: #1a237e; font-size: 2.5em; font-weight: bold; padding: 20px; }}
    .result-box {{ padding:25px; border-radius:20px; text-align:center; background:white; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"<div class='main-title'>{t['title']}</div>", unsafe_allow_html=True)
st.warning(t['advice'])

col1, col2 = st.columns(2)
with col1:
    choice = st.radio("", [t['upload'], t['cam']], horizontal=True)
    file = st.file_uploader("", type=["jpg", "png", "jpeg"]) if "ارفع" in choice or "Upload" in choice else st.camera_input("")

if file:
    img = Image.open(file).convert('RGB')
    with col2: st.image(img, use_container_width=True)
    
    if st.button(t['btn']):
        with st.spinner("Analyzing..."):
            img_np = np.array(img)
            img_res = cv2.resize(img_np, (224, 224))
            
            # المعالجة والفلترة
            avg = np.mean(img_res)
            proc = img_res.astype(np.float32)
            for i in range(3):
                proc[:, :, i] = np.clip(img_res[:, :, i] * (avg / np.mean(img_res[:, :, i])), 0, 255)
            
            lab = cv2.cvtColor(proc.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            img_final = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)
            
            # التنبؤ والمعايرة
            inp = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_final, axis=0))
            raw_preds = diag_m.predict(inp)[0]
            calibrated = raw_preds * np.array([v['w'] for v in MEDICAL_INFO.values()])
            calibrated /= calibrated.sum()
            
            idx = np.argmax(calibrated)
            info = MEDICAL_INFO[idx]
            
            # تحديد الألوان
            status_color = "#E74C3C" if "خبيث" in info['s'] else "#27AE60" if "حميد" in info['s'] else "#F1C40F"
            bg_color = status_color + "15" # شفافية 15%

            st.markdown(f"""
            <div class="result-box" style="border: 6px solid {status_color}; background-color: {bg_color};">
                <h1 style="color:{status_color};">{info['n']}</h1>
                <h2>{info['s']}</h2>
                <hr style="border: 1px solid {status_color}; opacity:0.3;">
                <h3 style="margin: 10px 0;">ثقة النظام: {calibrated[idx]*100:.1f}%</h3>
                <p style="font-size:1.2em;">{info['d']}</p>
            </div>
            """, unsafe_allow_html=True)

# --- 5. الدليل الطبي المرجعي الثابت (في أسفل الصفحة) ---
st.write("---")
st.subheader("📖 " + ("الدليل المرجعي" if selected_lang == "العربية" else "Medical Reference Guide"))
with st.expander("اضغط لعرض كافة أنواع الإصابات الجلدية المدعومة"):
    for k, v in MEDICAL_INFO.items():
        st.markdown(f"**{v['n']}** | {v['s']}  \n*{v['d']}*")
        st.write("---")
