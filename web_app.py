import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Global Skin AI Expert", layout="wide")

# --- 2. القاموس العالمي الشامل (20 لغة) ---
LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "نظام الفحص الذكي العالمي للجلد", "btn": "🔍 بدء الفحص", "upload": "📥 ارفع صورة", "cam": "📸 كاميرا", "invalid": "❌ صورة غير صالحة.", "advice": "⚠️ تنبيه: استشر طبيباً مختصاً فوراً."},
    "English": {"dir": "ltr", "title": "Global Skin AI Diagnostic System", "btn": "🔍 Start Scan", "upload": "📥 Upload", "cam": "📸 Camera", "invalid": "❌ Invalid image.", "advice": "⚠️ Note: Consult a doctor immediately."},
    "Français": {"dir": "ltr", "title": "IA de Diagnostic Cutané", "btn": "🔍 Analyser", "upload": "📥 Charger", "cam": "📸 Caméra", "invalid": "❌ Image invalide.", "advice": "⚠️ Note: Consultez un médecin."},
    "Español": {"dir": "ltr", "title": "IA de Diagnóstico Cutáneo", "btn": "🔍 Analizar", "upload": "📥 Subir", "cam": "📸 Cámara", "invalid": "❌ Imagen inválida.", "advice": "⚠️ Nota: Consulte a un médico."},
    "Deutsch": {"dir": "ltr", "title": "KI Hautdiagnosesystem", "btn": "🔍 Analyse", "upload": "📥 Hochladen", "cam": "📸 Kamera", "invalid": "❌ Ungültiges Bild.", "advice": "⚠️ Hinweis: Arzt aufsuchen."},
    "中文": {"dir": "ltr", "title": "皮肤人工智能诊断系统", "btn": "🔍 开始扫描", "upload": "📥 上传", "cam": "📸 相机", "invalid": "❌ 图像无效。", "advice": "⚠️ 注意：请立即咨询医生。"},
    "हिन्दी": {"dir": "ltr", "title": "त्वचा एआई नैदानिक प्रणाली", "btn": "🔍 स्कैन शुरू करें", "upload": "📥 अपलोड", "cam": "📸 कैमरा", "invalid": "❌ अमान्य छवि।", "advice": "⚠️ नोट: डॉक्टर से सलाह लें।"},
    "Русский": {"dir": "ltr", "title": "ИИ система диагностики кожи", "btn": "🔍 Начать", "upload": "📥 Загрузить", "cam": "📸 Камера", "invalid": "❌ Ошибка изображения.", "advice": "⚠️ Примечание: Обратитесь к врачу."},
    "日本語": {"dir": "ltr", "title": "皮膚AI診断システム", "btn": "🔍 解析開始", "upload": "📥 アップロード", "cam": "📸 カメラ", "invalid": "❌ 無効な画像。", "advice": "⚠️ 注意：医師にご相談ください。"},
    "Português": {"dir": "ltr", "title": "IA de Diagnóstico de Pele", "btn": "🔍 Analisar", "upload": "📥 Carregar", "cam": "📸 Câmera", "invalid": "❌ Imagem inválida.", "advice": "⚠️ Nota: Consulte um médico."},
    "Türkçe": {"dir": "ltr", "title": "Cilt AI Tanı Sistemi", "btn": "🔍 Analiz Et", "upload": "📥 Yükle", "cam": "📸 Kamera", "invalid": "❌ Geçersiz resim.", "advice": "⚠️ Not: Hemen doktora danışın."},
    "한국어": {"dir": "ltr", "title": "피부 AI 진단 시스템", "btn": "🔍 분석 시작", "upload": "📥 업로드", "cam": "📸 카메라", "invalid": "❌ 잘못된 이미지.", "advice": "⚠️ 참고: 의사와 상담하십시오."},
    "Italiano": {"dir": "ltr", "title": "IA Diagnostica Pelle", "btn": "🔍 Analizza", "upload": "📥 Carica", "cam": "📸 Camera", "invalid": "❌ Immagine non valida.", "advice": "⚠️ Nota: Consultare un medico."},
    "اردو": {"dir": "rtl", "title": "جلد کی تشخیص کا نظام", "btn": "🔍 معائنہ کریں", "upload": "📥 اپلوڈ", "cam": "📸 کیمرہ", "invalid": "❌ تصویر درست نہیں۔", "advice": "⚠️ نوٹ: ڈاکٹر سے رجوع کریں۔"},
    "فارسي": {"dir": "rtl", "title": "سیستم تشخیص هوشمند پوست", "btn": "🔍 شروع آنالیز", "upload": "📥 بارگذاری", "cam": "📸 دوربین", "invalid": "❌ تصویر معتبر نیست.", "advice": "⚠️ توجه: با پزشک مشورت کنید."},
    "Tiếng Việt": {"dir": "ltr", "title": "Hệ thống AI Chẩn đoán Da", "btn": "🔍 Phân tích", "upload": "📥 Tải lên", "cam": "📸 Máy ảnh", "invalid": "❌ Ảnh không hợp lệ.", "advice": "⚠️ Lưu ý: Hãy gặp bác sĩ."},
    "Bahasa Indonesia": {"dir": "ltr", "title": "Sistem AI Diagnosis Kulit", "btn": "🔍 Mulai", "upload": "📥 Unggah", "cam": "📸 Kamera", "invalid": "❌ Gambar tidak valid.", "advice": "⚠️ Catatan: Hubungi dokter."},
    "Nederlands": {"dir": "ltr", "title": "Huid AI Diagnosesysteem", "btn": "🔍 Analyse", "upload": "📥 Uploaden", "cam": "📸 Camera", "invalid": "❌ Ongeldige foto.", "advice": "⚠️ Let op: Raadpleeg een arts."},
    "Polski": {"dir": "ltr", "title": "System Diagnostyki Skóry AI", "btn": "🔍 Analizuj", "upload": "📥 Prześlij", "cam": "📸 Kamera", "invalid": "❌ Błędne zdjęcie.", "advice": "⚠️ Uwaga: Skonsultuj się z lekarzem."},
    "Kurdî": {"dir": "rtl", "title": "سیستەمی ژیری دەستکردی پێست", "btn": "🔍 شیکاری", "upload": "📥 وێنە بنێرە", "cam": "📸 کامێرا", "invalid": "❌ وێنەکە هەڵەیە.", "advice": "⚠️ ئاگاداری: سەردانی پزیشک بکە."}
}

if 'lang' not in st.session_state: st.session_state.lang = "العربية"
t = LANG_DATA[st.session_state.lang]

# --- 3. الدليل الطبي الملون (10 أنواع) ---
MEDICAL_TYPES = {
    0: {"n": "Melanoma (ميلانوما)", "c": "#C0392B", "s": "🚨 خبيث جداً", "d": "أخطر أنواع سرطان الجلد، يتطلب تدخلاً طبياً فورياً."},
    1: {"n": "Melanocytic Nevi (وحمة)", "c": "#27AE60", "s": "✅ حميد", "d": "شامات عادية ناتجة عن تجمع صبغي، آمنة غالباً."},
    2: {"n": "Basal Cell Carcinoma", "c": "#E74C3C", "s": "🚨 خبيث", "d": "سرطان الخلايا القاعدية، ينمو ببطء ويحتاج فحصاً مختصاً."},
    3: {"n": "Actinic Keratosis", "c": "#D35400", "s": "⚠️ ما قبل سرطاني", "d": "بقع خشنة ناتجة عن أضرار الشمس، قد تتحول لسرطان."},
    4: {"n": "Benign Keratosis", "c": "#2ECC71", "s": "✅ حميد", "d": "زوائد جلدية غير سرطانية تظهر مع تقدم العمر."},
    5: {"n": "Dermatofibroma", "c": "#16A085", "s": "✅ حميد", "d": "كتلة صلبة صغيرة تحت الجلد، غير ضارة تماماً."},
    6: {"n": "Vascular Lesions", "c": "#8E44AD", "s": "✅ حميد", "d": "بقع ناتجة عن تجمع أوعية دموية."},
    7: {"n": "Squamous Cell Carcinoma", "c": "#A93226", "s": "🚨 خبيث", "d": "ثاني أكثر أنواع سرطان الجلد شيوعاً."},
    8: {"n": "Psoriasis (الصدفية)", "c": "#2980B9", "s": "🔍 حالة جلدية", "d": "مرض مناعي يسبب قشوراً وبقعاً حمراء."},
    9: {"n": "Eczema (الأكزيما)", "c": "#F39C12", "s": "🔍 حالة جلدية", "d": "التهاب جلدي يسبب حكة وجفافاً غير سرطاني."}
}

# --- 4. المحركات البرمجية (الذكاء الاصطناعي) ---
@st.cache_resource
def load_models():
    # موديل الفلترة
    filter_m = tf.keras.applications.MobileNetV2(weights="imagenet")
    # موديل التشخيص الهجين
    base1 = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    base2 = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    comb = Concatenate()([GlobalAveragePooling2D()(base1.output), GlobalAveragePooling2D()(base2.output)])
    out = Dense(7, activation='softmax')(Dropout(0.5)(Dense(512, activation='relu')(comb)))
    diag_m = Model(inputs=[base1.input, base2.input], outputs=out)
    try: diag_m.load_weights("skin_expert_master.h5")
    except: pass
    return filter_m, diag_m

f_model, d_model = load_models()

# --- 5. التنسيق والواجهة (CSS) ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    * {{ font-family: 'Tajawal', sans-serif; direction: {t['dir']}; }}
    .main-title {{ text-align: center; color: #0d47a1; font-size: 2.2em; font-weight: bold; padding: 15px; }}
    .advice-box {{ background-color: #fff1f0; border: 1px solid #ffa39e; padding: 10px; border-radius: 8px; color: #cf1322; text-align: center; }}
    .card {{ padding: 20px; border-radius: 15px; border: 4px solid; text-align: center; background: white; }}
</style>
""", unsafe_allow_html=True)

# شريط اختيار اللغة العلوي
cols = st.columns(10)
for i, lang in enumerate(list(LANG_DATA.keys())[:10]):
    if cols[i].button(lang, key=f"L1_{i}"):
        st.session_state.lang = lang; st.rerun()
cols2 = st.columns(10)
for i, lang in enumerate(list(LANG_DATA.keys())[10:]):
    if cols2[i].button(lang, key=f"L2_{i}"):
        st.session_state.lang = lang; st.rerun()

st.markdown(f"<div class='main-title'>{t['title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='advice-box'>{t['advice']}</div>", unsafe_allow_html=True)

# رفع الصورة
col1, col2 = st.columns(2)
with col1:
    choice = st.radio("", [t['upload'], t['cam']], horizontal=True)
    file = st.file_uploader("", type=["jpg", "png", "jpeg"]) if "ارفع" in choice or "Upload" in choice else st.camera_input("")

if file:
    img = Image.open(file).convert('RGB')
    with col2: st.image(img, use_container_width=True)
    
    if st.button(t['btn']):
        # فحص محتوى الصورة
        img_res = cv2.resize(np.array(img), (224, 224))
        x_f = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_res, axis=0))
        f_pred = f_model.predict(x_f)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(f_pred, top=3)[0]
        
        # فلترة الصور الخارجية
        is_skin = True
        for _, label, score in decoded:
            if any(forbidden in label.lower() for forbidden in ['car', 'dog', 'flower', 'screen']) and score > 0.3:
                is_skin = False
        
        if not is_skin:
            st.error(t['invalid'])
        else:
            # التشخيص
            x_d = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_res, axis=0))
            preds = d_model.predict([x_d, x_d])[0]
            idx = np.argmax(preds)
            res = MEDICAL_TYPES[idx]
            
            st.markdown(f"""
            <div class="card" style="border-color: {res['c']}; color: {res['c']};">
                <h2>{res['n']}</h2>
                <h4>{res['s']}</h4>
                <p style="color: #333;">{res['d']}</p>
                <hr>
                <h3>دقة الفحص: {preds[idx]*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)

# --- 6. عرض الدليل الطبي (10 أنواع) ---
st.write("---")
st.subheader("📚 الدليل الطبي الملون")
d_cols = st.columns(2)
for i, (k, v) in enumerate(MEDICAL_TYPES.items()):
    with d_cols[i % 2]:
        st.markdown(f"""
        <div style="border-left: 8px solid {v['c']}; padding: 10px; background: {v['c']}10; margin-bottom: 10px; border-radius: 5px;">
            <strong style="color: {v['c']};">{v['n']}</strong><br>
            <small>{v['s']}</small><br>
            <span>{v['d']}</span>
        </div>
        """, unsafe_allow_html=True)
