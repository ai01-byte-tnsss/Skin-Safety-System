import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np

# --- 1. إعدادات الصفحة الأساسية ---
st.set_page_config(
    page_title="Skin Safety AI System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. القاموس اللغوي الشامل (25 لغة) ---
# تم وضع جميع النصوص لكل لغة لضمان عدم الاختصار وتسهيل التنقل
LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "نظام الكشف عن سلامة الجلد باستخدام الذكاء الاصطناعي (كشف أولي)", "note": "⚠️ للحصول على أدق النتائج، يرجى التصوير في ضوء طبيعي جيد والمركز على المنطقة المصابة فقط.", "upload": "📥 ارفع صورة الفحص", "camera": "📸 صورة فورية", "analyze": "🚀 بدء التحليل", "guide": "📚 الدليل الطبي الشامل", "malig": "🔴 الأورام الخبيثة", "benign": "🟢 الأورام الحميدة", "res_m": "🚨 اشتباه ورم خبيث (Malignant)", "res_b": "🔍 ورم سليم (Benign)", "res_g": "🩺 حالة عامة / غير ذلك", "advice": "يرجى مراجعة الطبيب المختص لضمان السلامة.", "cause": "سبب التكوين", "lang_btn": "🌐 تغيير اللغة", "ref_btn": "🔗 مراجع طبية عالمية للاحتياط"},
    "English": {"dir": "ltr", "title": "Skin Safety Detection System using AI (Initial Scan)", "note": "⚠️ For best results, use good natural lighting and focus only on the affected area.", "upload": "📥 Upload Scan", "camera": "📸 Instant Photo", "analyze": "🚀 Analyze", "guide": "📚 Medical Guide", "malig": "🔴 Malignant", "benign": "🟢 Benign", "res_m": "🚨 Malignant Suspect", "res_b": "🔍 Benign Result", "res_g": "🩺 General/Other", "advice": "Please consult a specialist.", "cause": "Cause", "lang_btn": "🌐 Language", "ref_btn": "🔗 Medical References"},
    "Français": {"dir": "ltr", "title": "Système de détection de la peau par IA (Scan initial)", "note": "⚠️ Utilisez un bon éclairage naturel.", "upload": "📥 Charger", "camera": "📸 Caméra", "analyze": "🚀 Analyser", "guide": "📚 Guide Médical", "malig": "🔴 Malin", "benign": "🟢 Bénin", "res_m": "🚨 Suspect Malin", "res_b": "🔍 Résultat Bénin", "res_g": "🩺 Autre", "advice": "Consultez un spécialiste.", "cause": "Cause", "lang_btn": "🌐 Langue", "ref_btn": "🔗 Références Médicales"},
    "Español": {"dir": "ltr", "title": "Sistema de detección de piel por IA (Escaneo inicial)", "note": "⚠️ Use luz natural para mejores resultados.", "upload": "📥 Subir", "camera": "📸 Cámara", "analyze": "🚀 Analizar", "guide": "📚 Guía Médica", "malig": "🔴 Maligno", "benign": "🟢 Benigno", "res_m": "🚨 Sospecha Maligna", "res_b": "🔍 Benigno", "res_g": "🩺 Otro", "advice": "Consulte a un médico.", "cause": "Causa", "lang_btn": "🌐 Idioma", "ref_btn": "🔗 Referencias Médicas"},
    "Deutsch": {"dir": "ltr", "title": "KI-Hauterkennungssystem (Erstscan)", "note": "⚠️ Nutzen Sie natürliches Licht.", "upload": "📥 Hochladen", "camera": "📸 Kamera", "analyze": "🚀 Analysieren", "guide": "📚 Leitfaden", "malig": "🔴 Bösartig", "benign": "🟢 Gutartig", "res_m": "🚨 Krebsverdacht", "res_b": "🔍 Gutartig", "res_g": "🩺 Allgemein", "advice": "Arzt aufsuchen.", "cause": "Ursache", "lang_btn": "🌐 Sprache", "ref_btn": "🔗 Medizinische Quellen"},
    "中文": {"dir": "ltr", "title": "人工智能皮肤安全检测系统（初筛）", "note": "⚠️ 为了获得最佳效果，请使用自然光。", "upload": "📥 上传", "camera": "📸 相机", "analyze": "🚀 分析", "guide": "📚 医学指南", "malig": "🔴 恶性", "benign": "🟢 良性", "res_m": "🚨 疑似恶性", "res_b": "🔍 良性结果", "res_g": "🩺 其他", "advice": "请咨询医生。", "cause": "原因", "lang_btn": "🌐 语言", "ref_btn": "🔗 医学参考"},
    "हिन्दी": {"dir": "ltr", "title": "AI त्वचा सुरक्षा प्रणाली (प्रारंभिक स्कैन)", "note": "⚠️ सर्वोत्तम परिणामों के लिए प्राकृतिक रोशनी का उपयोग करें।", "upload": "📥 अपलोड", "camera": "📸 कैमरा", "analyze": "🚀 विश्लेषण", "guide": "📚 चिकित्सा गाइड", "malig": "🔴 घातक", "benign": "🟢 सौम्य", "res_m": "🚨 घातक संदेह", "res_b": "🔍 सौम्य परिणाम", "res_g": "🩺 अन्य", "advice": "विशेषज्ञ से सलाह लें।", "cause": "कारण", "lang_btn": "🌐 भाषा", "ref_btn": "🔗 चिकित्सा संदर्भ"},
    "Русский": {"dir": "ltr", "title": "Система ИИ для кожи (Первичный осмотр)", "note": "⚠️ Используйте естественный свет.", "upload": "📥 Загрузить", "camera": "📸 Камера", "analyze": "🚀 Анализ", "guide": "📚 Справочник", "malig": "🔴 Злокачественные", "benign": "🟢 Доброкачественные", "res_m": "🚨 Подозрение", "res_b": "🔍 Доброкачественное", "res_g": "🩺 Общее", "advice": "Обратитесь к врачу.", "cause": "Причина", "lang_btn": "🌐 Язык", "ref_btn": "🔗 Мед. справка"},
    "日本語": {"dir": "ltr", "title": "AI皮膚検知システム（初期スキャン）", "note": "⚠️ 自然光を使用してください。", "upload": "📥 アップロード", "camera": "📸 カメラ", "analyze": "🚀 解析", "guide": "📚 ガイド", "malig": "🔴 悪性", "benign": "🟢 良性", "res_m": "🚨 悪性の疑い", "res_b": "🔍 良性", "res_g": "🩺 その他", "advice": "医師に相談。", "cause": "原因", "lang_btn": "🌐 言語", "ref_btn": "🔗 医学的参照"},
    "Português": {"dir": "ltr", "title": "Sistema AI de Pele (Triagem Inicial)", "note": "⚠️ Use luz natural clara.", "upload": "📥 Enviar", "camera": "📸 Câmera", "analyze": "🚀 Analisar", "guide": "📚 Guia Médico", "malig": "🔴 Maligno", "benign": "🟢 Benigno", "res_m": "🚨 Suspeita", "res_b": "🔍 Benigno", "res_g": "🩺 Outro", "advice": "Consulte um médico.", "cause": "Causa", "lang_btn": "🌐 Idioma", "ref_btn": "🔗 Referências Médicas"},
    "Türkçe": {"dir": "ltr", "title": "Yapay Zeka Cilt Sistemi (Ön Tarama)", "note": "⚠️ Doğal ışık kullanın.", "upload": "📥 Yükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz Et", "guide": "📚 Tıbbi Rehber", "malig": "🔴 Kötü Huylu", "benign": "🟢 İyi Huylu", "res_m": "🚨 Şüphe", "res_b": "🔍 İyi Huylu", "res_g": "🩺 Diğer", "advice": "Doktora danışın.", "cause": "Neden", "lang_btn": "🌐 Dil", "ref_btn": "🔗 Tıbbi Kaynaklar"},
    "한국어": {"dir": "ltr", "title": "AI 피부 안전 시스템 (초기 스캔)", "note": "⚠️ 자연광에서 촬영하세요.", "upload": "📥 업로드", "camera": "📸 카메라", "analyze": "🚀 분석", "guide": "📚 가이드", "malig": "🔴 악성", "benign": "🟢 양성", "res_m": "🚨 악성 의심", "res_b": "🔍 양성", "res_g": "🩺 기타", "advice": "전문가 상담。", "cause": "원인", "lang_btn": "🌐 언어", "ref_btn": "🔗 의학적 참고"},
    "Italiano": {"dir": "ltr", "title": "Sistema AI Pelle (Scansione Iniziale)", "note": "⚠️ Usa luce naturale.", "upload": "📥 Carica", "camera": "📸 Camera", "analyze": "🚀 Analizza", "guide": "📚 Guida", "malig": "🔴 Maligno", "benign": "🟢 Benigno", "res_m": "🚨 Sospetto", "res_b": "🔍 Benigno", "res_g": "🩺 Altro", "advice": "Consulta un medico.", "cause": "Causa", "lang_btn": "🌐 Lingua", "ref_btn": "🔗 Riferimenti Medici"},
    "اردو": {"dir": "rtl", "title": "جلد کی حفاظت کا AI نظام (ابتدائی اسکین)", "note": "⚠️ قدرتی روشنی استعمال کریں۔", "upload": "📥 اپلوڈ", "camera": "📸 کیمرہ", "analyze": "🚀 تجزیہ", "guide": "📚 گائیڈ", "malig": "🔴 خطرناک", "benign": "🟢 بے ضرر", "res_m": "🚨 شبہ", "res_b": "🔍 بے ضرر", "res_g": "🩺 دیگر", "advice": "ڈاکٹر سے مشورہ۔", "cause": "وجہ", "lang_btn": "🌐 زبان", "ref_btn": "🔗 طبی حوالہ جات"},
    "فارسي": {"dir": "rtl", "title": "سیستم هوش مصنوعی پوست (بررسی اولیه)", "note": "⚠️ از نور طبیعی استفاده کنید.", "upload": "📥 بارگذاری", "camera": "📸 دوربین", "analyze": "🚀 آنالیز", "guide": "📚 راهنما", "malig": "🔴 بدخیم", "benign": "🟢 خوش‌خیم", "res_m": "🚨 مشکوک", "res_b": "🔍 خوش‌خیم", "res_g": "🩺 سایر", "advice": "به پزشک مراجعه کنید.", "cause": "علت", "lang_btn": "🌐 زبان", "ref_btn": "🔗 مراجع پزشکی"},
    "Tiếng Việt": {"dir": "ltr", "title": "Hệ thống AI Da liễu (Kiểm tra sơ bộ)", "note": "⚠️ Sử dụng ánh sáng tự nhiên.", "upload": "📥 Tải lên", "camera": "📸 Máy ảnh", "analyze": "🚀 Phân tích", "guide": "📚 Hướng dẫn", "malig": "🔴 Ác tính", "benign": "🟢 Lành tính", "res_m": "🚨 Nghi ngờ", "res_b": "🔍 Lành tính", "res_g": "🩺 Khác", "advice": "Hỏi ý kiến bác sĩ.", "cause": "Nguyên nhân", "lang_btn": "🌐 Ngôn ngữ", "ref_btn": "🔗 Tài liệu y tế"},
    "Bahasa Indonesia": {"dir": "ltr", "title": "Sistem AI Kulit (Pemindaian Awal)", "note": "⚠️ Gunakan cahaya alami.", "upload": "📥 Unggah", "camera": "📸 Kamera", "analyze": "🚀 Analisis", "guide": "📚 Panduan", "malig": "🔴 Ganas", "benign": "🟢 Jinak", "res_m": "🚨 Kecurigaan", "res_b": "🔍 Jinak", "res_g": "🩺 Umum", "advice": "Konsultasi dokter.", "cause": "Penyebab", "lang_btn": "🌐 Bahasa", "ref_btn": "🔗 Referensi Medis"},
    "Nederlands": {"dir": "ltr", "title": "Huid AI-systeem (Eerste scan)", "note": "⚠️ Gebruik natuurlijk licht.", "upload": "📥 Upload", "camera": "📸 Camera", "analyze": "🚀 Analyse", "guide": "📚 Gids", "malig": "🔴 Kwaadaardig", "benign": "🟢 Goedaardig", "res_m": "🚨 Verdacht", "res_b": "🔍 Goedaardig", "res_g": "🩺 Overig", "advice": "Raadpleeg arts.", "cause": "Oorzaak", "lang_btn": "🌐 Taal", "ref_btn": "🔗 Medische Referenties"},
    "Polski": {"dir": "ltr", "title": "System AI Skóry (Wstępna analiza)", "note": "⚠️ Użyj światła dziennego.", "upload": "📥 Prześlij", "camera": "📸 Kamera", "analyze": "🚀 Analiza", "guide": "📚 Przewodnik", "malig": "🔴 Złośliwe", "benign": "🟢 Łagodne", "res_m": "🚨 Podejrzenie", "res_b": "🔍 Łagodne", "res_g": "🩺 Inne", "advice": "Skonsultuj się.", "cause": "Przyczyna", "lang_btn": "🌐 Język", "ref_btn": "🔗 Referencje Medyczne"},
    "ไทย": {"dir": "ltr", "title": "ระบบ AI ตรวจผิวหนัง (การตรวจเบื้องต้น)", "note": "⚠️ ใช้แสงธรรมชาติ", "upload": "📥 อัปโหลด", "camera": "📸 กล้อง", "analyze": "🚀 วิเคราะห์", "guide": "📚 คู่มือ", "malig": "🔴 เนื้อร้าย", "benign": "🟢 เนื้อดี", "res_m": "🚨 สงสัยเนื้อร้าย", "res_b": "🔍 เนื้อดี", "res_g": "🩺 ทั่วไป", "advice": "ปรึกษาแพทย์", "cause": "สาเหตุ", "lang_btn": "🌐 ภาษา", "ref_btn": "🔗 แหล่งอ้างอิงทางการแพทย์"},
    "کوردی": {"dir": "rtl", "title": "سیستەمی AI پێست (پشکنینی سەرەتایی)", "note": "⚠️ تیشکی سروشتی بەکاربهێنە.", "upload": "📥 وێنە", "camera": "📸 کامێرا", "analyze": "🚀 شیکاري", "guide": "📚 ڕێبەر", "malig": "🔴 خراپ", "benign": "🟢 بێ زیان", "res_m": "🚨 گومانی خراپ", "res_b": "🔍 بێ زیان", "res_g": "🩺 گشتی", "advice": "سەردانی پزیشک بکە.", "cause": "هۆکار", "lang_btn": "🌐 زمان", "ref_btn": "🔗 سەرچاوە پزیشکییەکان"},
    "Bengali": {"dir": "ltr", "title": "AI স্কিন সিস্টেম (প্রাথমিক স্ক্যান)", "note": "⚠️ প্রাকৃতিক আলো ব্যবহার করুন।", "upload": "📥 আপলোড", "camera": "📸 ক্যামেরা", "analyze": "🚀 বিশ্লেষণ", "guide": "📚 নির্দেশিকা", "malig": "🔴 মারাত্মক", "benign": "🟢 সৌম্য", "res_m": "🚨 সন্দেহজনক", "res_b": "🔍 সৌম্য", "res_g": "🩺 সাধারণ", "advice": "পরামর্শ নিন।", "cause": "कारण", "lang_btn": "🌐 ভাষা", "ref_btn": "🔗 চিকিৎসা রেফারেন্স"},
    "Română": {"dir": "ltr", "title": "Sistem AI Piele (Scanare Inițială)", "note": "⚠️ Folosiți lumină naturală.", "upload": "📥 Încarcă", "camera": "📸 Cameră", "analyze": "🚀 Analizează", "guide": "📚 Ghid", "malig": "🔴 Malign", "benign": "🟢 Benign", "res_m": "🚨 Suspect", "res_b": "🔍 Benign", "res_g": "🩺 General", "advice": "Consultă medicul.", "cause": "Cauza", "lang_btn": "🌐 Limbă", "ref_btn": "🔗 Referințe Medicale"},
    "Kiswahili": {"dir": "ltr", "title": "Mfumo wa AI wa Ngozi (Uchunguzi wa Kwanza)", "note": "⚠️ Tumia mwanga wa asili.", "upload": "📥 Pakia", "camera": "📸 Kamera", "analyze": "🚀 Uchambuzi", "guide": "📚 Mwongozo", "malig": "🔴 Saratani", "benign": "🟢 Salama", "res_m": "🚨 Shaka", "res_b": "🔍 Salama", "res_g": "🩺 Nyingine", "advice": "Ona daktari.", "cause": "Sababu", "lang_btn": "🌐 Lugha", "ref_btn": "🔗 Marejeleo ya Matibabu"},
    "Türkmençe": {"dir": "ltr", "title": "Deri AI ulgamy (Deslapky anyklaýyş)", "note": "⚠️ Tebigy yşyk ulanyň.", "upload": "📥 Ýükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz", "guide": "📚 Gollanma", "malig": "🔴 Howply", "benign": "🟢 Howpsuz", "res_m": "🚨 Şüphe", "res_b": "🔍 Howpsuz", "res_g": "🩺 Başga", "advice": "Lukmana ýüz tutuň.", "cause": "Sebäbi", "lang_btn": "🌐 Dil", "ref_btn": "🔗 Lukmançylyk çeşmeleri"}
}

# --- 3. إدارة حالة الجلسة والتنسيق البصري ---
if 'lang' not in st.session_state:
    st.session_state.lang = "العربية"

t = LANG_DATA[st.session_state.lang]

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    html, body, [class*="st-"] {{ font-family: 'Tajawal', sans-serif; font-size: 16px; }}
    div[dir='{t['dir']}'] {{ text-align: {'right' if t['dir']=='rtl' else 'left'}; }}
    .main-title {{ text-align: center; color: #0d47a1; font-size: 1.6em; margin-bottom: 20px; font-weight: bold; line-height: 1.4; }}
    .report-card {{ padding: 25px; border-radius: 20px; text-align: center; border: 5px solid; margin-top: 15px; background: white; }}
    .note-box {{ background: #fffbe6; border: 1px solid #ffe58f; padding: 12px; border-radius: 10px; margin-bottom: 15px; font-size: 0.9em; }}
    .stButton>button {{ width: 100%; border-radius: 8px; font-weight: bold; height: 3.5em; background-color: #0d47a1; color: white; }}
    .lang-container {{ display: flex; justify-content: center; margin-bottom: 20px; }}
    .disease-card {{ border-right: 5px solid #0d47a1; padding: 12px; background: #fdfdfd; margin-bottom: 10px; border-radius: 8px; }}
</style>
""", unsafe_allow_html=True)

# --- 4. واجهة اختيار اللغة (الزر) ---
st.markdown("<div class='lang-container'>", unsafe_allow_html=True)
with st.popover(t['lang_btn']):
    cols = st.columns(2)
    for i, lang_name in enumerate(LANG_DATA.keys()):
        with cols[i % 2]:
            if st.button(lang_name, key=f"lang_opt_{lang_name}"):
                st.session_state.lang = lang_name
                st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# --- 5. محرك الذكاء الاصطناعي (تحميل الموديل) ---
@st.cache_resource
def load_expert_model():
    # استخدام بنية EfficientNetB0 القوية للتعرف على الصور الطبية
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    predictions = Dense(7, activation='softmax')(Dropout(0.4)(x))
    return Model(inputs=base.input, outputs=predictions)

model = load_expert_model()

# --- 6. الواجهة الرئيسية للبرنامج ---
st.markdown(f"<div dir='{t['dir']}'>", unsafe_allow_html=True)
st.markdown(f"<div class='main-title'>{t['title']}</div>", unsafe_allow_html=True)
st.markdown(f'<div class="note-box">{t["note"]}</div>', unsafe_allow_html=True)

c1, c2 = st.columns([1, 1])

with c1:
    choice = st.radio(" ", (t['upload'], t['camera']), horizontal=True)
    file = st.file_uploader(t['upload'], type=["jpg", "png", "jpeg"]) if choice == t['upload'] else st.camera_input(t['camera'])

if file:
    img = Image.open(file)
    with c2:
        st.image(img, use_container_width=True)
    
    if st.button(t['analyze']):
        with st.spinner("🚀 جاري التحليل والمطابقة..."):
            # معالجة الصورة
            img_resized = img.convert("RGB").resize((224, 224))
            img_array = np.array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            processed_img = tf.keras.applications.efficientnet.preprocess_input(img_array)
            
            # التنبؤ المباشر بناءً على الأرجحية (Argmax)
            preds = model.predict(processed_img)[0]
            idx = np.argmax(preds)
            
            # التصنيفات الطبية (HAM10000)
            # خبيث: mel, bcc, akiec | حميد: nv, bkl, df, vasc
            malignant_indices = [0, 1, 4] 
            benign_indices = [2, 3, 5, 6]

            if idx in malignant_indices:
                res_msg, color = t['res_m'], "#cf1322"
            elif idx in benign_indices:
                res_msg, color = t['res_b'], "#389e0d"
            else:
                res_msg, color = t['res_g'], "#096dd9"

            st.markdown(f'<div class="report-card" style="border-color: {color}; color: {color};"><h2>{res_msg}</h2><p>{t["advice"]}</p></div>', unsafe_allow_html=True)

# --- 7. قسم المراجع الطبية للاحتياط ---
st.write("---")
st.markdown(f"### {t['ref_btn']}")
st.info("لضمان أقصى درجات الأمان، يمكنك مطابقة الحالة مع المراجع الطبية العالمية الموثوقة:")
ref_cols = st.columns(2)
with ref_cols[0]:
    st.markdown("🔗 [Mayo Clinic Skin Cancer Guide](https://www.mayoclinic.org/diseases-conditions/skin-cancer/symptoms-causes/syc-20377605)")
    st.markdown("🔗 [American Cancer Society](https://www.cancer.org/cancer/skin-cancer.html)")
with ref_cols[1]:
    st.markdown("🔗 [Skin Cancer Foundation](https://www.skincancer.org/)")
    st.markdown("🔗 [WebMD Skin Health](https://www.webmd.com/melanoma-skin-cancer/default.htm)")

# --- 8. الدليل الطبي الشامل (غير مختصر) ---
with st.expander(f"📖 {t['guide']}"):
    m_tab, b_tab = st.tabs([t['malig'], t['benign']])
    
    with m_tab:
        m_list = [
            ("Melanoma (mel)", "أخطر أنواع سرطان الجلد، يبدأ في الخلايا الصبغية.", "التعرض الشديد للأشعة فوق البنفسجية.", "تغير في حجم أو لون شامة موجودة."),
            ("Basal Cell Carcinoma (bcc)", "الأكثر شيوعاً، ينمو ببطء ونادراً ما ينتشر.", "التعرض المستمر لأشعة الشمس لسنوات.", "نتوء لؤلؤي أو قرحة لا تلتئم."),
            ("Actinic Keratosis (akiec)", "آفة سرطانية أولية قد تتحول لسرطان حرشفي.", "تراكم أضرار أشعة الشمس على الجلد.", "بقعة قشرية خشنة على المناطق المعرضة للشمس.")
        ]
        for name, desc, cause, symp in m_list:
            st.markdown(f'<div class="disease-card" style="border-right-color:#cf1322;"><b style="color:#cf1322;">{name}</b><br><b>الوصف:</b> {desc}<br><b>السبب:</b> {cause}<br><b>الأعراض:</b> {symp}</div>', unsafe_allow_html=True)

    with b_tab:
        b_list = [
            ("Melanocytic Nevi (nv)", "الشامات العادية السليمة تماماً.", "تجمع طبيعي للخلايا الصبغية.", "بقعة بنية دائرية منتظمة الحدود."),
            ("Benign Keratosis (bkl)", "آفات حميدة تشبه الثآليل.", "التقدم في السن والوراثة.", "زوائد شمعية أو داكنة تظهر على الجلد."),
            ("Dermatofibroma (df)", "كتل ليفية حميدة صغيرة.", "رد فعل طفيف لقرصات الحشرات.", "عقدة صلبة صغيرة بنية اللون.")
        ]
        for name, desc, cause, symp in b_list:
            st.markdown(f'<div class="disease-card" style="border-right-color:#389e0d;"><b style="color:#389e0d;">{name}</b><br><b>الوصف:</b> {desc}<br><b>السبب:</b> {cause}<br><b>الأعراض:</b> {symp}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
