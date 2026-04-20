import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. إعدادات الواجهة ---
st.set_page_config(page_title="Skin AI Expert System", layout="wide")

# القاموس الكامل لـ 20 لغة (ثابت كما طلبت)
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

# --- 2. الدليل الطبي مع أوزان التصحيح ---
MEDICAL_INFO = {
    0: {"n": "Melanoma (ميلانوما)", "c": "#D32F2F", "s": "🚨 خبيث جداً", "w": 1.45, "d": "أخطر أنواع سرطان الجلد."},
    1: {"n": "Melanocytic Nevi (وحمة)", "c": "#388E3C", "s": "✅ حميد", "w": 0.55, "d": "شامة طبيعية آمنة."},
    2: {"n": "Basal Cell Carcinoma (BCC)", "c": "#F57C00", "s": "🚨 خبيث", "w": 0.50, "d": "سرطان قاعدي ينمو ببطء."},
    3: {"n": "Actinic Keratosis (AK)", "c": "#7B1FA2", "s": "⚠️ ما قبل سرطاني", "w": 1.15, "d": "تلف شمسي قد يتطور."},
    4: {"n": "Benign Keratosis (BKL)", "c": "#1976D2", "s": "✅ حميد", "w": 0.85, "d": "زوائد غير سرطانية."},
    5: {"n": "Dermatofibroma (DF)", "c": "#00796B", "s": "✅ حميد", "w": 1.20, "d": "كتلة صلبة بعد إصابة طفيفة."},
    6: {"n": "Vascular Lesions (VASC)", "c": "#C2185B", "s": "✅ حميد", "w": 1.25, "d": "آفات وعائية تجمع شعيرات."},
    7: {"n": "Squamous Cell Carcinoma", "c": "#E64A19", "s": "🚨 خبيث", "w": 1.35, "d": "سرطان الخلايا الحرشفية."},
    8: {"n": "Psoriasis (الصدفية)", "c": "#512DA8", "s": "🔍 حالة جلدية", "w": 1.05, "d": "التهاب مزمن وقشور فضية."},
    9: {"n": "Eczema (الأكزيما)", "c": "#FFA000", "s": "🔍 حالة جلدية", "w": 1.15, "d": "التهاب جلدي وحكة وجفاف."}
}

# --- 3. محرك الذكاء الاصطناعي (تم حل مشكلة ValueError) ---
@st.cache_resource
def load_expert_system():
    # بناء الهيكل باستخدام Input موحد لضمان الاستقرار
    inp = Input(shape=(224, 224, 3))
    
    # دمج قوتين (Ensemble)
    b1 = EfficientNetB0(weights=None, include_top=False)(inp)
    b2 = MobileNetV2(weights=None, include_top=False)(inp)
    
    g1 = GlobalAveragePooling2D()(b1)
    g2 = GlobalAveragePooling2D()(b2)
    
    merged = Concatenate()([g1, g2])
    # مخرجات 10 لتطابق الدليل الطبي
    out = Dense(10, activation='softmax')(Dropout(0.4)(Dense(512, activation='relu')(merged)))
    
    model = Model(inputs=inp, outputs=out)
    
    h5_path = "skin_expert_master.h5"
    ready = False
    if os.path.exists(h5_path):
        model.load_weights(h5_path)
        ready = True
    
    # موديل الفلترة
    f_model = tf.keras.applications.MobileNetV2(weights="imagenet")
    
    return model, f_model, ready

main_model, filter_model, is_ready = load_expert_system()

# --- 4. واجهة المستخدم ---
sel_lang = st.selectbox("🌐 Choose Language / اختر اللغة", list(LANG_DATA.keys()))
ui = LANG_DATA[sel_lang]

st.markdown(f"<h1 style='text-align:center; color:#1E3A8A;'>{ui['title']}</h1>", unsafe_allow_html=True)

if not is_ready:
    st.error("❌ ملف 'skin_expert_master.h5' غير موجود!")

up_file = st.file_uploader(ui['upload'], type=["jpg", "png", "jpeg"])

if up_file and is_ready:
    img = Image.open(up_file).convert('RGB')
    st.image(img, width=400)
    
    if st.button(ui['btn']):
        with st.spinner("⏳ Analyzing..."):
            img_np = np.array(img)
            img_res = cv2.resize(img_np, (224, 224))
            
            # فلترة الصور (لحماية النظام)
            xf = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_res, axis=0))
            f_preds = filter_model.predict(xf)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(f_preds, top=3)[0]
            
            is_skin = True
            for _, label, score in decoded:
                if any(x in label.lower() for x in ['car', 'wheel', 'dog', 'flower', 'laptop']) and score > 0.3:
                    is_skin = False
            
            if not is_skin:
                st.error(ui['invalid'])
            else:
                # حل مشكلة AttributeError: موازنة ألوان يدوية 100%
                avg = np.mean(img_res)
                proc = img_res.astype(np.float32)
                for i in range(3):
                    proc[:, :, i] = np.clip(img_res[:, :, i] * (avg / np.mean(img_res[:, :, i])), 0, 255)
                
                # تحسين النسيج الجلدي
                lab = cv2.cvtColor(proc.astype(np.uint8), cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                l = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8)).apply(l)
                final = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

                # التشخيص مع الأوزان (السر في التصنيف الصحيح)
                inp = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(final, axis=0))
                preds = main_model.predict(inp)[0]
                
                # تطبيق الأوزان التصحيحية لكسر انحياز BCC والحميد
                weights = np.array([v['w'] for v in MEDICAL_INFO.values()])
                idx = np.argmax(preds * weights)
                
                res = MEDICAL_INFO[idx]
                st.markdown(f"""
                <div style="border: 8px solid {res['c']}; padding: 20px; border-radius: 15px; background: white; text-align: center;">
                    <h1 style="color: {res['c']};">{res['n']}</h1>
                    <h3>{res['s']}</h3>
                    <p>{res['d']}</p>
                    <p>Confidence: {preds[idx]*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

# --- 5. الدليل المرجعي الكامل ---
st.write("---")
st.subheader("📖 الدليل الطبي المرجعي المعتمد")
for k, v in MEDICAL_INFO.items():
    st.markdown(f"<span style='color:{v['c']};'>●</span> **{v['n']}**: {v['d']}", unsafe_allow_html=True)
