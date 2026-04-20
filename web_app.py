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
    "العربية": {"dir": "rtl", "title": "نظام الفحص الذكي للجلد", "upload": "📥 ارفع صورة", "cam": "📸 كاميرا", "btn": "🔍 بدء الفحص", "invalid": "❌ عذراً، الصورة ليست فحصاً جلدياً.", "advice": "⚠️ تنبيه: هذا النظام أداة برمجية استرشادية فقط وليس بديلاً عن الطبيب المختص."},
    "English": {"dir": "ltr", "title": "Skin AI Diagnostic System", "upload": "📥 Upload", "cam": "📸 Camera", "btn": "🔍 Start Analysis", "invalid": "❌ Invalid skin image.", "advice": "⚠️ Note: This system is a guidance tool and not a substitute for a professional doctor."},
    "Français": {"dir": "ltr", "title": "IA Diagnostic Cutané", "upload": "📥 Charger", "cam": "📸 Caméra", "btn": "🔍 Analyser", "invalid": "❌ Invalide.", "advice": "⚠️ Note: Ce système ne remplace pas un médecin."},
    "Español": {"dir": "ltr", "title": "IA Diagnóstico de Piel", "upload": "📥 Subir", "cam": "📸 Cámara", "btn": "🔍 Analizar", "invalid": "❌ Inválido.", "advice": "⚠️ Nota: No sustituye a un médico."},
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
    "فارسي": {"dir": "rtl", "title": "هوش مصنوعی پوست", "upload": "📥 بارگذاری", "cam": "📸 دوربین", "btn": "🔍 آناليز", "invalid": "❌ نامعتبر.", "advice": "⚠️ توجه: جایگزین پزشک نیست."},
    "Tiếng Việt": {"dir": "ltr", "title": "AI Da liễu", "upload": "📥 Tải lên", "cam": "📸 Máy ảnh", "btn": "🔍 Phân tích", "invalid": "❌ Lỗi.", "advice": "⚠️ Lưu ý: Không thay thế bác sĩ."},
    "Bahasa Indonesia": {"dir": "ltr", "title": "AI Kulit", "upload": "📥 Unggah", "cam": "📸 Kamera", "btn": "🔍 Analisis", "invalid": "❌ Gagal.", "advice": "⚠️ Catatan: Bukan pengganti dokter."},
    "Nederlands": {"dir": "ltr", "title": "Huid AI", "upload": "📥 Upload", "cam": "📸 Camera", "btn": "🔍 Analyse", "invalid": "❌ Ongeldig.", "advice": "⚠️ Let op: Geen vervanging voor arts."},
    "Polski": {"dir": "ltr", "title": "AI Skóry", "upload": "📥 Prześlij", "cam": "📸 Kamera", "btn": "🔍 Analiza", "invalid": "❌ Błąd.", "advice": "⚠️ Uwaga: Nie zastępuje lekarza."},
    "Kurdî": {"dir": "rtl", "title": "ژیری پێست", "upload": "📥 وێنە", "cam": "📸 کامێرا", "btn": "🔍 شیکاری", "invalid": "❌ هەڵە.", "advice": "⚠️ ئاگاداري: جێگرەوەی پزیشک نییە."}
}

# --- 2. الدليل الطبي (ثابت - 10 أنواع) ---
MEDICAL_INFO = {
    0: {"n": "Melanoma (ميلانوما)", "c": "#FF3B30", "s": "🚨 خبيث جداً", "d": "سرطان جلدي خطير يتطلب تدخل طبي فوري."},
    1: {"n": "Melanocytic Nevi (وحمة صبغية)", "c": "#34C759", "s": "✅ حميد", "d": "شامة طبيعية، آمنة ومستقرة غالباً."},
    2: {"n": "Basal Cell Carcinoma (BCC)", "c": "#FF9500", "s": "🚨 خبيث", "d": "سرطان الخلايا القاعدية، ينمو ببطء ويجب علاجه."},
    3: {"n": "Actinic Keratosis (AK)", "c": "#AF52DE", "s": "⚠️ ما قبل سرطاني", "d": "بقع ناتجة عن الشمس قد تتحول لسرطان."},
    4: {"n": "Benign Keratosis (BKL)", "c": "#5856D6", "s": "✅ حميد", "d": "زوائد جلدية غير سرطانية تظهر مع العمر."},
    5: {"n": "Dermatofibroma (DF)", "c": "#007AFF", "s": "✅ حميد", "d": "كتلة صلبة صغيرة، غير ضارة تماماً."},
    6: {"n": "Vascular Lesions (VASC)", "c": "#5AC8FA", "s": "✅ حميد", "d": "آفات وعائية ناتجة عن تجمع الشعيرات."},
    7: {"n": "Squamous Cell Carcinoma", "c": "#FF2D55", "s": "🚨 خبيث", "d": "سرطان الخلايا الحرشفية، يتطلب استئصال."},
    8: {"n": "Psoriasis (الصدفية)", "c": "#4CD964", "s": "🔍 حالة جلدية", "d": "مرض مناعي يسبب قشور فضية."},
    9: {"n": "Eczema (الأكزيما)", "c": "#FFCC00", "s": "🔍 حالة جلدية", "d": "التهاب جلدي يسبب حكة واحمرار."}
}

# --- 3. تحميل المحركات ---
@st.cache_resource
def load_all_engines():
    f_mod = tf.keras.applications.MobileNetV2(weights="imagenet")
    b1 = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    b2 = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    comb = Concatenate()([GlobalAveragePooling2D()(b1.output), GlobalAveragePooling2D()(b2.output)])
    out = Dense(7, activation='softmax')(Dropout(0.4)(Dense(512, activation='relu')(comb)))
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
        with st.spinner("⏳ Analyzing..."):
            img_np = np.array(img)
            img_res = cv2.resize(img_np, (224, 224))
            
            # فلترة الكائنات (الحل لمنع تداخل الأنواع غير الحية)
            xf = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_res, axis=0))
            f_preds = filter_m.predict(xf)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(f_preds, top=3)[0]
            
            is_valid = True
            for _, label, score in decoded:
                if any(x in label.lower() for x in ['car', 'wheel', 'dog', 'cat', 'flower', 'screen', 'laptop']) and score > 0.4:
                    is_valid = False
            
            if not is_valid:
                st.error(t['invalid'])
            else:
                # --- حل مشكلة الانحياز (Preprocessing Optimization) ---
                # 1. تحسين التباين (CLAHE) لإظهار التفاصيل الخبيثة
                lab = cv2.cvtColor(img_res, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)) # رفع العتبة لزيادة الوضوح
                l = clahe.apply(l)
                img_proc = cv2.merge((l,a,b))
                img_proc = cv2.cvtColor(img_proc, cv2.COLOR_LAB2RGB)
                
                # 2. موازنة اللون الأبيض (White Balance) لمنع تأثير لون الجلد
                result = cv2.xphoto.createSimpleWB()
                img_proc = result.balanceWhite(img_proc)
                
                # التشخيص
                inp = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_proc, axis=0))
                res_preds = diag_m.predict([inp, inp])[0]
                idx = np.argmax(res_preds)
                
                res = MEDICAL_INFO[idx]
                st.markdown(f"""
                <div style="padding:30px; border-radius:15px; border:10px solid {res['c']}; text-align:center; background:white; margin-top:20px;">
                    <h1 style="color:{res['c']};">{res['n']}</h1>
                    <h2 style="color:#555;">الحالة: {res['s']}</h2>
                    <hr style="border:1px solid {res['c']}; width:40%; margin:auto;">
                    <p style="font-size:1.3em; color:#333; margin-top:15px; font-weight:bold;">{res['d']}</p>
                </div>
                """, unsafe_allow_html=True)

# --- 5. الدليل المرجعي (ثابت) ---
st.write("---")
st.subheader("📖 الدليل المرجعي")
selected_info = st.selectbox("اختر فئة لعرض التفاصيل:", [v['n'] for v in MEDICAL_INFO.values()])

for k, v in MEDICAL_INFO.items():
    if v['n'] == selected_info:
        st.markdown(f"""
        <div style="background-color:{v['c']}10; padding:25px; border-right:15px solid {v['c']}; border-radius:10px;">
            <h2 style="color:{v['c']};">{v['n']}</h2>
            <p><strong>التصنيف:</strong> {v['s']}</p>
            <p><strong>الوصف:</strong> {v['d']}</p>
        </div>
        """, unsafe_allow_html=True)
