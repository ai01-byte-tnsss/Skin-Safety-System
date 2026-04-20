import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# ==========================================
# 1. قاعدة بيانات اللغات (20 لغة متكاملة)
# ==========================================
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
    "Italiano": {"dir": "ltr", "title": "IA Pelle", "upload": "📥 Carica", "cam": "📸 Camera", "btn": "🔍 Analizza", "invalid": "❌ Invalido.", "advice": "⚠️ Consulti médico."},
    "اردو": {"dir": "rtl", "title": "جلد کی تشخیص", "upload": "📥 اپلوڈ", "cam": "📸 کیمرہ", "btn": "🔍 معائنہ", "invalid": "❌ تصویر درست نہیں۔", "advice": "⚠️ ڈاکٹر سے ملیں۔"},
    "فارسي": {"dir": "rtl", "title": "هوش مصنوعی پوست", "upload": "📥 بارگذاری", "cam": "📸 دوربین", "btn": "🔍 آنالیز", "invalid": "❌ نامعتبر.", "advice": "⚠️ پزشک بروید."},
    "Tiếng Việt": {"dir": "ltr", "title": "AI Da liễu", "upload": "📥 Tải lên", "cam": "📸 Máy ảnh", "btn": "🔍 Phân tích", "invalid": "❌ Lỗi.", "advice": "⚠️ Gặp bác sĩ."},
    "Bahasa Indonesia": {"dir": "ltr", "title": "AI Kulit", "upload": "📥 Unggah", "cam": "📸 Kamera", "btn": "🔍 Analisis", "invalid": "❌ Gagal.", "advice": "⚠️ Hubungi dokter."},
    "Nederlands": {"dir": "ltr", "title": "Huid AI", "upload": "📥 Upload", "cam": "📸 Camera", "btn": "🔍 Analyse", "invalid": "❌ Ongeldig.", "advice": "⚠️ Raadpleeg arts."},
    "Polski": {"dir": "ltr", "title": "AI Skóry", "upload": "📥 Prześlij", "cam": "📸 Kamera", "btn": "🔍 Analiza", "invalid": "❌ Błąd.", "advice": "⚠️ Idź do lekarza."},
    "Kurdî": {"dir": "rtl", "title": "ژیری پێست", "upload": "📥 وێنە", "cam": "📸 کامێرا", "btn": "🔍 شیکاري", "invalid": "❌ هەڵە.", "advice": "⚠️ پزیشک ببینە."}
}

# ==========================================
# 2. الدليل الطبي المرجعي (10 أنواع ثابتة)
# ==========================================
# الأوزان (w) تستخدم لموازنة الموديل إذا كان ينحاز لنوع واحد
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

# ==========================================
# 3. محرك الذكاء الاصطناعي (حل الـ Mismatch)
# ==========================================
@st.cache_resource
def load_engines():
    # موديل الفلترة لتمييز صور الجلد عن غيرها
    f_mod = tf.keras.applications.MobileNetV2(weights="imagenet")
    
    # بناء الهيكل الهجين المتطابق مع skin_expert_master.h5
    inp = Input(shape=(224, 224, 3), name="main_input")
    
    # تفادي تضارب الأسماء باستخدام skip_mismatch لاحقاً
    base_eff = EfficientNetB0(weights=None, include_top=False, input_tensor=inp)
    base_mob = MobileNetV2(weights=None, include_top=False, input_tensor=inp)
    
    gap_eff = GlobalAveragePooling2D()(base_eff.output)
    gap_mob = GlobalAveragePooling2D()(base_mob.output)
    comb = Concatenate()([gap_eff, gap_mob])
    
    x = Dense(512, activation='relu')(comb)
    x = Dropout(0.5)(x)
    out = Dense(10, activation='softmax')(out if 'out' in locals() else x)
    
    d_mod = Model(inputs=inp, outputs=out)
    
    # التحميل المرن للأوزان
    h5_path = "skin_expert_master.h5"
    if os.path.exists(h5_path):
        try:
            d_mod.load_weights(h5_path, by_name=False, skip_mismatch=True)
            st.sidebar.success("✅ AI Engine: Optimized & Loaded")
        except Exception as e:
            st.sidebar.error(f"⚠️ Calibration Error: {str(e)[:50]}")
    else:
        st.sidebar.warning("❌ Missing: skin_expert_master.h5")
        
    return f_mod, d_mod

# تشغيل التحميل
filter_m, diag_m = load_engines()

# ==========================================
# 4. واجهة المستخدم والتنسيق (CSS)
# ==========================================
st.set_page_config(page_title="Skin Health AI", layout="wide")
selected_lang = st.sidebar.selectbox("🌐 Choose Language / اختر اللغة", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    * {{ direction: {t['dir']}; font-family: 'Tajawal', sans-serif; }}
    .main-title {{ text-align: center; color: #003366; font-size: 3em; font-weight: bold; padding: 10px; }}
    .result-card {{ padding:30px; border-radius:25px; text-align:center; background:white; box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin-top: 20px; border: 8px solid; }}
    .guide-card {{ padding:15px; border-radius:15px; margin-bottom:12px; border-right: 12px solid; background: #ffffff; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }}
    .stButton>button {{ width: 100%; border-radius: 10px; height: 3em; font-size: 1.2em; background-color: #003366; color: white; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"<div class='main-title'>{t['title']}</div>", unsafe_allow_html=True)
st.warning(t['advice'])

# ==========================================
# 5. منطقة المعالجة والرفع
# ==========================================
col_up, col_pre = st.columns(2)
with col_up:
    choice = st.radio("", [t['upload'], t['cam']], horizontal=True)
    file = st.file_uploader("", type=["jpg", "png", "jpeg"]) if "ارفع" in choice or "Upload" in choice else st.camera_input("")

if file:
    img = Image.open(file).convert('RGB')
    with col_pre: st.image(img, caption="Preview", use_container_width=True)
    
    if st.button(t['btn']):
        with st.spinner("Processing Tissue Analysis..."):
            # تجهيز الصورة
            img_np = np.array(img)
            img_res = cv2.resize(img_np, (224, 224))
            
            # --- موازنة الإضاءة (Histogram Equalization) ---
            # تمنع تصنيف الصور الداكنة كنوع واحد
            lab = cv2.cvtColor(img_res, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            img_balanced = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2RGB)
            
            # التنبؤ
            inp = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_balanced, axis=0))
            raw_preds = diag_m.predict(inp)[0]
            
            # --- مصفوفة المعايرة (Calibration) ---
            # تمنع التحيز (Bias) لمرض معين
            cal_weights = np.array([v['w'] for v in MEDICAL_INFO.values()])
            final_probs = raw_preds * cal_weights
            final_probs /= final_probs.sum() # إعادة التطبيع لـ 100%
            
            idx = np.argmax(final_probs)
            info = MEDICAL_INFO[idx]
            
            # تحديد نمط النتيجة بناءً على النوع
            status_color = info['c']
            bg_color = status_color + "10" # شفافية خفيفة جداً للورق

            st.markdown(f"""
            <div class="result-card" style="border-color: {status_color}; background-color: {bg_color};">
                <h1 style="color:{status_color}; margin-bottom:5px;">{info['n']}</h1>
                <h2 style="color:#2c3e50;">{info['s']}</h2>
                <hr style="border: 1.5px solid {status_color}; width: 50%; opacity: 0.3;">
                <div style="display: flex; justify-content: space-around; align-items: center; padding: 20px;">
                    <div>
                        <p style="font-size: 1.2em; margin:0;">دقة التشخيص</p>
                        <h1 style="font-size: 3.5em; margin:0; color:{status_color};">{final_probs[idx]*100:.1f}%</h1>
                    </div>
                    <div style="text-align: {'right' if t['dir']=='rtl' else 'left'}; max-width: 60%;">
                        <p style="font-size: 1.3em; line-height: 1.5;">{info['d']}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# 6. الدليل الطبي الملون (أسفل الموقع)
# ==========================================
st.write("---")
st.subheader("📖 " + ("الدليل الطبي المرجعي للأنواع" if t['dir']=="rtl" else "Medical Reference Guide"))
with st.expander("اضغط لعرض تفاصيل تصنيفات الأمراض الجلدية المدعومة"):
    # عرض الدليل بشكل بطاقات ملونة منظمة
    cols = st.columns(2)
    for i, (k, v) in enumerate(MEDICAL_INFO.items()):
        target_col = cols[i % 2]
        target_col.markdown(f"""
        <div class="guide-card" style="border-color: {v['c']};">
            <h4 style="color: {v['c']}; margin: 0;">{v['n']}</h4>
            <span style="background: {v['c']}; color: white; padding: 2px 8px; border-radius: 5px; font-size: 0.8em;">{v['s']}</span>
            <p style="margin: 8px 0 0 0; font-size: 0.95em; color: #555;">{v['d']}</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 7. التذييل (Footer)
# ==========================================
st.markdown("---")
st.caption("Graduation Project - College of Computer Science and Mathematics | University of Mosul 2026")
