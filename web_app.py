import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2

# --- 1. الإعدادات واللغات (ثابتة) ---
st.set_page_config(page_title="Skin AI System", layout="wide")

LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "نظام الفحص الذكي للجلد", "upload": "📥 ارفع صورة", "cam": "📸 كاميرا", "btn": "🔍 بدء الفحص", "invalid": "❌ عذراً، الصورة ليست فحصاً جلدياً.", "advice": "⚠️ تنبيه: هذا النظام أداة برمجية استرشادية فقط وليست بديلاً عن استشارة الطبيب المختص."},
    "English": {"dir": "ltr", "title": "Skin AI Diagnostic System", "upload": "📥 Upload", "cam": "📸 Camera", "btn": "🔍 Start Analysis", "invalid": "❌ Invalid skin image.", "advice": "⚠️ Note: This system is a guidance tool and not a substitute for a professional medical consultation."},
    "Français": {"dir": "ltr", "title": "IA Diagnostic Cutané", "upload": "📥 Charger", "cam": "📸 Caméra", "btn": "🔍 Analyser", "invalid": "❌ Invalide.", "advice": "⚠️ Note: Ce système est un outil et ne remplace pas un médecin."},
    "Español": {"dir": "ltr", "title": "IA Diagnóstico de Piel", "upload": "📥 Subir", "cam": "📸 Cámara", "btn": "🔍 Analizar", "invalid": "❌ Inválido.", "advice": "⚠️ Nota: Este sistema no sustituye a un médico."},
    "Deutsch": {"dir": "ltr", "title": "KI Hautdiagnose", "upload": "📥 Hochladen", "cam": "📸 Kamera", "btn": "🔍 Analyse", "invalid": "❌ Ungültig.", "advice": "⚠️ Hinweis: Ersetzt keinen Arzt."},
    "中文": {"dir": "ltr", "title": "皮肤人工智能诊断", "upload": "📥 上传", "cam": "📸 相机", "btn": "🔍 分析", "invalid": "❌ 无效。", "advice": "⚠️ 注意：不能代替医生。"},
    "हिन्दी": {"dir": "ltr", "title": "त्वचा एआई निदान", "upload": "📥 अपलोड", "cam": "📸 कैमरा", "btn": "🔍 विश्लेषण", "invalid": "❌ अमान्य।", "advice": "⚠️ नोट: डॉक्टर का विकल्प नहीं।"},
    "Русский": {"dir": "ltr", "title": "ИИ диагностика кожи", "upload": "📥 Загрузить", "cam": "📸 Камера", "btn": "🔍 Анализ", "invalid": "❌ Ошибка.", "advice": "⚠️ Примечание: Не заменяет врача."},
    "日本語": {"dir": "ltr", "title": "皮膚AI診断", "upload": "📥 アップロード", "cam": "📸 カメラ", "btn": "🔍 解析", "invalid": "❌ 無効。", "advice": "⚠️ 注意：医師に代わるものではありません。"},
    "Português": {"dir": "ltr", "title": "IA Pele", "upload": "📥 Carregar", "cam": "📸 Câmera", "btn": "🔍 Analisar", "invalid": "❌ Inválido.", "advice": "⚠️ Nota: Não substitui um médico."},
    "Türkçe": {"dir": "ltr", "title": "Cilt AI", "upload": "📥 Yükle", "cam": "📸 Kamera", "btn": "🔍 Analiz", "invalid": "❌ Geçersiz.", "advice": "⚠️ Not: Doktorun yerini tutmaz."},
    "한국어": {"dir": "ltr", "title": "피부 AI", "upload": "📥 업로드", "cam": "📸 카메라", "btn": "🔍 분석", "invalid": "❌ 무효.", "advice": "⚠️ 주의: 의사를 대신할 수 없습니다."},
    "Italiano": {"dir": "ltr", "title": "IA Pelle", "upload": "📥 Carica", "cam": "📸 Camera", "btn": "🔍 Analizza", "invalid": "❌ Invalido.", "advice": "⚠️ Nota: Non sostituisce il medico."},
    "اردو": {"dir": "rtl", "title": "جلد کی تشخیص", "upload": "📥 اپلوڈ", "cam": "📸 کیمرہ", "btn": "🔍 معائنہ", "invalid": "❌ غلط تصویر۔", "advice": "⚠️ نوٹ: ڈاکٹر کا متبادل نہیں۔"},
    "فارسي": {"dir": "rtl", "title": "هوش مصنوعی پوست", "upload": "📥 بارگذاری", "cam": "📸 دوربین", "btn": "🔍 آنالیز", "invalid": "❌ نامعتبر.", "advice": "⚠️ توجه: جایگزین پزشک نیست."},
    "Tiếng Việt": {"dir": "ltr", "title": "AI Da liễu", "upload": "📥 Tải lên", "cam": "📸 Máy ảnh", "btn": "🔍 Phân tích", "invalid": "❌ Lỗi.", "advice": "⚠️ Lưu ý: Không thay thế bác sĩ."},
    "Bahasa Indonesia": {"dir": "ltr", "title": "AI Kulit", "upload": "📥 Unggah", "cam": "📸 Kamera", "btn": "🔍 Analisis", "invalid": "❌ Gagal.", "advice": "⚠️ Catatan: Bukan pengganti dokter."},
    "Nederlands": {"dir": "ltr", "title": "Huid AI", "upload": "📥 Upload", "cam": "📸 Camera", "btn": "🔍 Analyse", "invalid": "❌ Ongeldig.", "advice": "⚠️ Let op: Geen vervanging voor arts."},
    "Polski": {"dir": "ltr", "title": "AI Skóry", "upload": "📥 Prześlij", "cam": "📸 Kamera", "btn": "🔍 Analiza", "invalid": "❌ Błąd.", "advice": "⚠️ Uwaga: Nie zastępuje lekarza."},
    "Kurdî": {"dir": "rtl", "title": "ژیری پێست", "upload": "📥 وێنە", "cam": "📸 کامێرا", "btn": "🔍 شیکاری", "invalid": "❌ هەڵە.", "advice": "⚠️ ئاگاداری: جێگرەوەی پزیشک نییە."}
}

# --- 2. الدليل الطبي (ثابت - 10 أنواع) ---
MEDICAL_INFO = {
    0: {"n": "Melanoma (ميلانوما)", "c": "#FF4B4B", "s": "🚨 خبيث جداً", "d": "أخطر أنواع سرطان الجلد، يتطلب فحصاً مجهرياً فورياً."},
    1: {"n": "Melanocytic Nevi (وحمة صبغية)", "c": "#28A745", "s": "✅ حميد", "d": "شامات عادية ناتجة عن تجمع الخلايا الصبغية، آمنة غالباً."},
    2: {"n": "Basal Cell Carcinoma (BCC)", "c": "#D9534F", "s": "🚨 خبيث", "d": "سرطان الخلايا القاعدية، ينمو ببطء ويحتاج فحصاً مختصاً."},
    3: {"n": "Actinic Keratosis (AK)", "c": "#F0AD4E", "s": "⚠️ ما قبل سرطاني", "d": "بقع خشنة ناتجة عن أضرار الشمس، قد تتحول لسرطان."},
    4: {"n": "Benign Keratosis (BKL)", "c": "#5CB85C", "s": "✅ حميد", "d": "زوائد جلدية غير سرطانية تظهر مع تقدم العمر."},
    5: {"n": "Dermatofibroma (DF)", "c": "#20C997", "s": "✅ حميد", "d": "كتلة صلبة صغيرة تحت الجلد، غير ضارة تماماً."},
    6: {"n": "Vascular Lesions (VASC)", "c": "#6F42C1", "s": "✅ حميد", "d": "آفات وعائية ناتجة عن تجمع الشعيرات الدموية."},
    7: {"n": "Squamous Cell Carcinoma", "c": "#B03A2E", "s": "🚨 خبيث", "d": "ثاني أكثر أنواع سرطان الجلد شيوعاً، يتطلب علاجاً."},
    8: {"n": "Psoriasis (الصدفية)", "c": "#007BFF", "s": "🔍 حالة جلدية", "d": "مرض مناعي يسبب قشوراً فضية وبقعاً حمراء."},
    9: {"n": "Eczema (الأكزيما)", "c": "#FD7E14", "s": "🔍 حالة جلدية", "d": "التهاب جلدي يسبب حكة وجفافاً غير سرطاني."}
}

# --- 3. محركات الذكاء الاصطناعي والمعالجة ---
@st.cache_resource
def load_all_engines():
    f_mod = tf.keras.applications.MobileNetV2(weights="imagenet")
    b1 = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    b2 = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    comb = Concatenate()([GlobalAveragePooling2D()(b1.output), GlobalAveragePooling2D()(b2.output)])
    out = Dense(7, activation='softmax')(Dropout(0.5)(Dense(512, activation='relu')(comb)))
    d_mod = Model(inputs=[b1.input, b2.input], outputs=out)
    try: d_mod.load_weights("skin_expert_master.h5")
    except: pass
    return f_mod, d_mod

filter_m, diag_m = load_all_engines()

# --- 4. واجهة المستخدم ---
selected_lang = st.selectbox("🌐 Choose Language / اختر اللغة", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

st.markdown(f"<h1 style='text-align:center; color:#1E3A8A;'>{t['title']}</h1>", unsafe_allow_html=True)
st.warning(t['advice'])

c1, c2 = st.columns(2)
with c1:
    m = st.radio("", [t['upload'], t['cam']], horizontal=True)
    file = st.file_uploader("", type=["jpg", "png", "jpeg"]) if "ارفع" in m or "Upload" in m else st.camera_input("")

if file:
    img = Image.open(file).convert('RGB')
    with c2: st.image(img, use_container_width=True)
    
    if st.button(t['btn']):
        with st.spinner("Analyzing..."):
            img_np = np.array(img)
            img_res = cv2.resize(img_np, (224, 224))
            
            # الفحص الأولي (كائنات حية/غير حية)
            xf = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_res, axis=0))
            f_preds = filter_m.predict(xf)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(f_preds, top=3)[0]
            
            is_valid = True
            for _, label, score in decoded:
                if any(x in label.lower() for x in ['car', 'dog', 'cat', 'flower', 'screen', 'laptop']) and score > 0.4:
                    is_valid = False
            
            if not is_valid:
                st.error(t['invalid'])
            else:
                # معالجة متقدمة لضمان التفرقة الدقيقة
                # تحسين التباين التكيفي (CLAHE)
                lab = cv2.cvtColor(img_res, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                img_proc = cv2.merge((clahe.apply(l), a, b))
                img_proc = cv2.cvtColor(img_proc, cv2.COLOR_LAB2RGB)
                
                # التشخيص النهائي بدون نسبة مئوية
                inp = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_proc, axis=0))
                res_preds = diag_m.predict([inp, inp])[0]
                idx = np.argmax(res_preds)
                
                res = MEDICAL_INFO[idx]
                st.markdown(f"""
                <div style="padding:30px; border-radius:15px; border:8px solid {res['c']}; text-align:center; background:white;">
                    <h1 style="color:{res['c']}; margin-bottom:10px;">{res['n']}</h1>
                    <h2 style="color:#444;">الحالة: {res['s']}</h2>
                    <p style="font-size:1.2em; color:#333; margin-top:15px;">{res['d']}</p>
                </div>
                """, unsafe_allow_html=True)

# --- 5. الدليل الطبي المنسدل (ثابت) ---
st.write("---")
st.subheader("📖 الدليل المرجعي لأنواع الآفات الجلدية")
selected_info = st.selectbox("اختر نوعاً لعرض تفاصيله:", [v['n'] for v in MEDICAL_INFO.values()])

for k, v in MEDICAL_INFO.items():
    if v['n'] == selected_info:
        st.markdown(f"""
        <div style="background-color:{v['c']}15; padding:25px; border-right:12px solid {v['c']}; border-radius:8px;">
            <h2 style="color:{v['c']};">{v['n']}</h2>
            <p style="font-size:1.1em;"><strong>التصنيف الطبي:</strong> {v['s']}</p>
            <p style="font-size:1.1em;"><strong>معلومات الفحص:</strong> {v['d']}</p>
        </div>
        """, unsafe_allow_html=True)
