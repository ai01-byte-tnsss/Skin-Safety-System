import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2

# --- 1. إعدادات الصفحة الأساسية ---
st.set_page_config(
    page_title="Skin Safety AI System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. قاموس اللغات الشامل (20 لغة) ---
LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "نظام سلامة الجلد الذكي", "upload": "📥 ارفع صورة الفحص", "camera": "📸 صورة فورية", "analyze": "🚀 بدء التحليل المتقدم", "res_m": "🚨 اشتباه ورم خبيث", "res_b": "🔍 حالة حميدة", "advice": "⚠️ تنبيه طبي: هذا النظام هو أداة ذكاء اصطناعي للاسترشاد فقط، وليس بديلاً عن التشخيص الطبي المهني. يرجى استشارة الطبيب فوراً عند الشك.", "guide": "📖 الدليل الطبي الشامل لسرطان الجلد", "links": "🔗 مراجع طبية عالمية موثوقة"},
    "English": {"dir": "ltr", "title": "Smart Skin Safety AI System", "upload": "📥 Upload Scan", "camera": "📸 Instant Photo", "analyze": "🚀 Start Analysis", "res_m": "🚨 Malignant Suspect", "res_b": "🔍 Benign Condition", "advice": "⚠️ Medical Note: This AI tool is for guidance only and is NOT a substitute for professional medical advice. Consult a doctor immediately.", "guide": "📖 Full Skin Cancer Guide", "links": "🔗 Trusted Global Medical Links"},
    "Français": {"dir": "ltr", "title": "IA Sécurité Cutanée", "upload": "📥 Charger l'image", "camera": "📸 Caméra", "analyze": "🚀 Analyser", "res_m": "🚨 Suspect Malin", "res_b": "🔍 État Bénin", "advice": "⚠️ Note: Ce système est une aide IA et ne remplace pas un avis médical professionnel.", "guide": "📖 Guide Médical Complet", "links": "🔗 Liens Médicaux"},
    "Español": {"dir": "ltr", "title": "IA de Seguridad de la Piel", "upload": "📥 Subir imagen", "camera": "📸 Cámara", "analyze": "🚀 Analizar", "res_m": "🚨 Sospecha Maligna", "res_b": "🔍 Estado Benigno", "advice": "⚠️ Nota: Esta IA es solo para orientación y no sustituye al diagnóstico médico.", "guide": "📖 Guía Médica Completa", "links": "🔗 Enlaces Médicos"},
    "Deutsch": {"dir": "ltr", "title": "KI-Hautschutzsystem", "upload": "📥 Bild hochladen", "camera": "📸 Kamera", "analyze": "🚀 Analysieren", "res_m": "🚨 Krebsverdacht", "res_b": "🔍 Gutartig", "advice": "⚠️ Hinweis: Diese KI dient der Orientierung und ersetzt keinen Arztbesuch.", "guide": "📖 Medizinischer Leitfaden", "links": "🔗 Medizinische Links"},
    "中文": {"dir": "ltr", "title": "智能皮肤安全AI系统", "upload": "📥 上传图片", "camera": "📸 相机", "analyze": "🚀 开始分析", "res_m": "🚨 疑似恶性", "res_b": "🔍 良性状态", "advice": "⚠️ 注意：此AI工具仅供参考，不能替代专业的医疗建议。", "guide": "📖 完整医学指南", "links": "🔗 全球医学链接"},
    "हिन्दी": {"dir": "ltr", "title": "स्मार्ट त्वचा सुरक्षा AI", "upload": "📥 अपलोड करें", "camera": "📸 कैमरा", "analyze": "🚀 विश्लेषण करें", "res_m": "🚨 घातक संदिग्ध", "res_b": "🔍 सौम्य स्थिति", "advice": "⚠️ नोट: यह AI उपकरण केवल मार्गदर्शन के लिए है, पेशेवर चिकित्सा सलाह का विकल्प नहीं है।", "guide": "📖 चिकित्सा गाइड", "links": "🔗 चिकित्सा लिंक"},
    "Русский": {"dir": "ltr", "title": "ИИ-система кожи", "upload": "📥 Загрузить", "camera": "📸 Камера", "analyze": "🚀 Анализировать", "res_m": "🚨 Подозрение на рак", "res_b": "🔍 Доброкачественное", "advice": "⚠️ Примечание: Этот ИИ не заменяет профессиональную медицинскую консультацию.", "guide": "📖 Справочник", "links": "🔗 Ссылки"},
    "日本語": {"dir": "ltr", "title": "皮膚安全AIシステム", "upload": "📥 アップロード", "camera": "📸 カメラ", "analyze": "🚀 解析開始", "res_m": "🚨 悪性の疑い", "res_b": "🔍 良性", "advice": "⚠️ 注意：このAIツールは専門的な医学的診断の代わりにはなりません。", "guide": "📖 医学ガイド", "links": "🔗 関連リンク"},
    "Português": {"dir": "ltr", "title": "IA de Segurança da Pele", "upload": "📥 Carregar", "camera": "📸 Câmera", "analyze": "🚀 Analisar", "res_m": "🚨 Suspeita Maligna", "res_b": "🔍 Benigno", "advice": "⚠️ Nota: Esta IA não substitui o conselho médico profissional.", "guide": "📖 Guia Médico", "links": "🔗 Links Médicos"},
    "Türkçe": {"dir": "ltr", "title": "Cilt Güvenliği AI", "upload": "📥 Yükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz Et", "res_m": "🚨 Kötü Huylu", "res_b": "🔍 İyi Huylu", "advice": "⚠️ Not: Bu AI aracı tıbbi teşhis yerine geçmez.", "guide": "📖 Tıbbi Rehber", "links": "🔗 Bağlantılar"},
    "한국어": {"dir": "ltr", "title": "피부 안전 AI 시스템", "upload": "📥 업로드", "camera": "📸 카메라", "analyze": "🚀 분석", "res_m": "🚨 악성 의심", "res_b": "🔍 양성", "advice": "⚠️ 참고: 이 AI는 전문적인 의학적 조언을 대신할 수 없습니다.", "guide": "📖 의학 가이드", "links": "🔗 의료 링크"},
    "Italiano": {"dir": "ltr", "title": "IA Sicurezza Pelle", "upload": "📥 Carica", "camera": "📸 Camera", "analyze": "🚀 Analizza", "res_m": "🚨 Sospetto Maligno", "res_b": "🔍 Benigno", "advice": "⚠️ Nota: Questa IA non sostituisce il parere di un medico.", "guide": "📖 Guida Medica", "links": "🔗 Link Medici"},
    "اردو": {"dir": "rtl", "title": "اسکن سیفٹی AI", "upload": "📥 اپلوڈ کریں", "camera": "📸 کیمرہ", "analyze": "🚀 تجزیہ کریں", "res_m": "🚨 کینسر کا شبہ", "res_b": "🔍 بے ضرر", "advice": "⚠️ نوٹ: یہ پروگرام ڈاکٹر کا متبادل نہیں ہے۔", "guide": "📖 طبی گائیڈ", "links": "🔗 طبی روابط"},
    "فارسي": {"dir": "rtl", "title": "هوش مصنوعی ایمنی پوست", "upload": "📥 بارگذاری", "camera": "📸 دوربین", "analyze": "🚀 آنالیز", "res_m": "🚨 مشکوک بدخیم", "res_b": "🔍 خوش‌خیم", "advice": "⚠️ توجه: این ابزار جایگزین تشخیص پزشک نیست.", "guide": "📖 راهنمای کامل", "links": "🔗 پیوندها"},
    "Tiếng Việt": {"dir": "ltr", "title": "AI An toàn Da liễu", "upload": "📥 Tải lên", "camera": "📸 Máy ảnh", "analyze": "🚀 Phân tích", "res_m": "🚨 Nghi ngờ ác tính", "res_b": "🔍 Lành tính", "advice": "⚠️ Lưu ý: AI này không thay thế lời khuyên của bác sĩ.", "guide": "📖 Hướng dẫn y tế", "links": "🔗 Liên kết"},
    "Bahasa Indonesia": {"dir": "ltr", "title": "AI Keamanan Kulit", "upload": "📥 Unggah", "camera": "📸 Kamera", "analyze": "🚀 Analisis", "res_m": "🚨 Kecurigaan Ganas", "res_b": "🔍 Kondisi Jinak", "advice": "⚠️ Catatan: AI ini bukan pengganti saran medis profesional.", "guide": "📖 Panduan Medis", "links": "🔗 Tautan Medis"},
    "Nederlands": {"dir": "ltr", "title": "Huidveiligheid AI", "upload": "📥 Uploaden", "camera": "📸 Camera", "analyze": "🚀 Analyseren", "res_m": "🚨 Kwaadaardig", "res_b": "🔍 Goedaardig", "advice": "⚠️ Let op: Deze AI is geen vervanging voor medisch advies.", "guide": "📖 Medische Gids", "links": "🔗 Links"},
    "Polski": {"dir": "ltr", "title": "System AI skóry", "upload": "📥 Prześlij", "camera": "📸 Kamera", "analyze": "🚀 Analizuj", "res_m": "🚨 Podejrzenie raka", "res_b": "🔍 Stan łagodny", "advice": "⚠️ Uwaga: AI nie zastępuje profesjonalnej porady medycznej.", "guide": "📖 Przewodnik", "links": "🔗 Linki"},
    "کوردی": {"dir": "rtl", "title": "ژیری دەستکردی پێست", "upload": "📥 وێنە بنێرە", "camera": "📸 کامێرا", "analyze": "🚀 شیکاری", "res_m": "🚨 گومانی خراپ", "res_b": "🔍 بێ زیان", "advice": "⚠️ ئاگاداری: ئەم بەرنامەیە جێگەی پزیشک ناگرێتەوە.", "guide": "📖 ڕێبەری پزیشکی", "links": "🔗 بەستەرەکان"}
}

# --- 3. إدارة الجلسة واللغة ---
if 'lang' not in st.session_state:
    st.session_state.lang = "العربية"
t = LANG_DATA[st.session_state.lang]

# --- 4. واجهة المستخدم والتنسيق (CSS) لإلغاء التداخل ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    * {{ font-family: 'Tajawal', sans-serif; direction: {t['dir']}; }}
    .main-title {{ text-align: center; color: #0d47a1; font-size: 2.3em; font-weight: bold; padding: 20px; }}
    .medical-warning {{ background-color: #fff2f0; border: 1px solid #ffccc7; padding: 15px; border-radius: 12px; color: #a8071a; text-align: center; margin-bottom: 25px; }}
    .stButton>button {{ width: 100%; border-radius: 10px; height: 3.5em; font-weight: bold; background-color: #0d47a1; color: white; border: none; }}
    .stButton>button:hover {{ background-color: #1565c0; }}
    /* منع تداخل النصوص في مكونات Streamlit */
    .stRadio > div {{ gap: 20px; padding: 10px 0; }}
    div[data-testid="stFileUploader"] {{ padding: 5px; }}
    .report-card {{ padding: 25px; border-radius: 15px; border: 4px solid; text-align: center; background: #f9f9f9; }}
</style>
""", unsafe_allow_html=True)

# --- 5. تحميل المحرك الهجين (Ensemble) ---
@st.cache_resource
def load_expert_model():
    base1 = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    base2 = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    combined = Concatenate()([GlobalAveragePooling2D()(base1.output), GlobalAveragePooling2D()(base2.output)])
    x = Dense(512, activation='relu')(combined)
    x = Dropout(0.5)(x)
    preds = Dense(7, activation='softmax')(x)
    model = Model(inputs=[base1.input, base2.input], outputs=preds)
    try:
        model.load_weights("skin_expert_master.h5") # ملف أوزانك الشخصي
    except:
        pass
    return model

model = load_expert_model()

# --- 6. معالجة الصور (CLAHE) والتنبؤ ---
def medical_process(image):
    # استخدام تقنية CLAHE لتحسين تباين الصور الطبية
    img = np.array(image.convert('RGB'))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    img_resized = cv2.resize(enhanced_rgb, (224, 224))
    img_prep = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_resized, axis=0))
    return img_prep

# --- 7. بناء الواجهة التفاعلية ---
st.markdown(f"<div class='main-title'>{t['title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='medical-warning'>{t['advice']}</div>", unsafe_allow_html=True)

# اختيار اللغة (منع التداخل اللغوي)
with st.popover(f"🌐 {st.session_state.lang}"):
    cols = st.columns(2)
    for i, lang_name in enumerate(LANG_DATA.keys()):
        with cols[i % 2]:
            if st.button(lang_name, key=f"btn_{lang_name}"):
                st.session_state.lang = lang_name
                st.rerun()

st.write("---")

col_ui1, col_ui2 = st.columns(2)

with col_ui1:
    st.markdown(f"<div dir='{t['dir']}'>", unsafe_allow_html=True)
    mode = st.radio("", [t['upload'], t['camera']], horizontal=True, label_visibility="collapsed")
    file = st.file_uploader("", type=["jpg", "png", "jpeg"]) if mode == t['upload'] else st.camera_input("")
    st.markdown("</div>", unsafe_allow_html=True)

if file:
    img_input = Image.open(file)
    with col_ui2:
        st.image(img_input, use_container_width=True, caption="Scan Image")
    
    if st.button(t['analyze']):
        with st.spinner("🚀 Hybrid AI Analyzing..."):
            processed = medical_process(img_input)
            preds = model.predict([processed, processed])[0]
            # عتبة أمان حساسة 0.3 لاكتشاف الحالات الخبيثة مبكراً
            is_m = (preds[0] + preds[1] + preds[4]) >= 0.3
            
            color = "#cf1322" if is_m else "#389e0d"
            res_text = t['res_m'] if is_m else t['res_b']
            conf = np.max(preds) * 100

            st.markdown(f"""
            <div class="report-card" style="border-color: {color}; color: {color};">
                <h1 style="margin:0;">{res_text}</h1>
                <hr style="border: 1px solid {color}">
                <h3>ثقة النظام: {conf:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)

# --- 8. الدليل الطبي والروابط العالمية ---
st.write("---")
with st.expander(t['guide']):
    st.markdown(f"""
    <div dir="{t['dir']}">
    <h3>أنواع سرطان الجلد والأمراض المشابهة:</h3>
    <ul>
        <li><b>الميلانوما (Melanoma):</b> أخطر أنواع سرطان الجلد، يظهر غالباً كشامة غير منتظمة الشكل أو اللون.</li>
        <li><b>سرطان الخلايا القاعدية (BCC):</b> يظهر كبقعة لؤلؤية أو جرح لا يلتئم، وهو الأكثر شيوعاً.</li>
        <li><b>سرطان الخلايا الحرشفية (SCC):</b> يظهر كبقعة حمراء متقشرة، وقد ينتشر إذا لم يعالج.</li>
        <li><b>التقران السفعي (AK):</b> حالة ما قبل سرطانية تنتج عن التعرض الطويل للشمس.</li>
        <li><b>الأورام الحميدة (Benign):</b> تشمل الشامات العادية، التقران الدهني، والوحمات.</li>
    </ul>
    <hr>
    <h3>{t['links']}</h3>
    <ul>
        <li><a href="https://www.mayoclinic.org/diseases-conditions/skin-cancer/symptoms-causes/syc-20377605" target="_blank">Mayo Clinic - Skin Cancer Guide</a></li>
        <li><a href="https://www.cancer.org/cancer/skin-cancer.html" target="_blank">American Cancer Society</a></li>
        <li><a href="https://www.skincancer.org/" target="_blank">Skin Cancer Foundation</a></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
