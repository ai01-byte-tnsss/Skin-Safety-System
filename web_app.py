import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.parse

# --- 1. إعدادات الصفحة (ثابتة) ---
st.set_page_config(page_title="Global Skin Guard AI", layout="centered")

# --- 2. القاموس اللغوي (ثابت كما هو) ---
LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "🛡️ نظام الكشف عن سلامة الجلد", "upload": "📥 ارفع صورة الفحص", "camera": "📸 صورة فورية", "analyze": "🚀 بدء التحليل", "guide": "📚 الدليل الطبي الشامل", "malig": "الأورام الخبيثة", "benign": "الأورام الحميدة", "more": "تفاصيل وصور", "res_m": "🚨 اشتباه ورم خبيث", "res_b": "🔍 ورم حميد", "res_g": "🩺 حالة عامة", "advice": "يرجى مراجعة المختص لضمان السلامة.", "share": "مشاركة"},
    "English": {"dir": "ltr", "title": "🛡️ Skin Safety AI System", "upload": "📥 Upload Image", "camera": "📸 Take Photo", "analyze": "🚀 Analyze", "guide": "📚 Medical Guide", "malig": "Malignant", "benign": "Benign", "more": "Details", "res_m": "🚨 Malignant Suspect", "res_b": "🔍 Benign", "res_g": "🩺 General", "advice": "Please consult a specialist.", "share": "Share"},
    "کوردی": {"dir": "rtl", "title": "🛡️ سیستەمی پشکنینی پێست", "upload": "📥 وێنە دابنێ", "camera": "📸 وێنە بگرە", "analyze": "🚀 شیکاری", "guide": "📚 ڕێبەری پزیشکی", "malig": "گرێی خراپ", "benign": "گرێی بێ زیان", "more": "زانیاری", "res_m": "🚨 گومانی گرێی خراپ", "res_b": "🔍 گرێی بێ زیان", "res_g": "🩺 باری گشتی", "advice": "سەردانی پزیشک بکە.", "share": "ناردن"},
    "Türkçe": {"dir": "ltr", "title": "🛡️ Cilt Güvenliği AI", "upload": "📥 Resim Yükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz Et", "guide": "📚 Tıbbi Rehber", "malig": "Kötü Huylu", "benign": "İyi Huylu", "more": "Detaylar", "res_m": "🚨 Kötü Huylu Şübhesi", "res_b": "🔍 İyi Huylu", "res_g": "🩺 Genel Durum", "advice": "Doktora danışın.", "share": "Paylaş"},
    "Français": {"dir": "ltr", "title": "🛡️ IA de Sécurité Cutanée", "upload": "📥 Charger l'image", "camera": "📸 Caméra", "analyze": "🚀 Analyser", "guide": "📚 Guide Médical", "malig": "Malin", "benign": "Bénin", "more": "Détails", "res_m": "🚨 Suspect Malin", "res_b": "🔍 Bénin", "res_g": "🩺 État Général", "advice": "Consultez un médecin.", "share": "Partager"},
    "Español": {"dir": "ltr", "title": "🛡️ IA de Seguridad Cutánea", "upload": "📥 Subir imagen", "camera": "📸 Cámara", "analyze": "🚀 Analizar", "guide": "📚 Guía Médica", "malig": "Maligno", "benign": "Benigno", "more": "Detalles", "res_m": "🚨 Sospecha Maligna", "res_b": "🔍 Benigno", "res_g": "🩺 General", "advice": "Consulte a un médico.", "share": "Compartir"},
    "Português": {"dir": "ltr", "title": "🛡️ IA de Segurança da Pele", "upload": "📥 Enviar foto", "camera": "📸 Câmera", "analyze": "🚀 Analisar", "guide": "📚 Guia Médico", "malig": "Maligno", "benign": "Benigno", "more": "Detalhes", "res_m": "🚨 Suspeita Maligna", "res_b": "🔍 Benigno", "res_g": "🩺 Geral", "advice": "Consulte um médico.", "share": "Compartilhar"},
    "Deutsch": {"dir": "ltr", "title": "🛡️ Hautsicherheits-KI", "upload": "📥 Bild hochladen", "camera": "📸 Kamera", "analyze": "🚀 Analysieren", "guide": "📚 Med. Leitfaden", "malig": "Bösartig", "benign": "Gutartig", "more": "Details", "res_m": "🚨 Krebsverdacht", "res_b": "🔍 Gutartig", "res_g": "🩺 Allgemein", "advice": "Arzt aufsuchen.", "share": "Teilen"},
    "Русский": {"dir": "ltr", "title": "🛡️ ИИ Безопасности Кожи", "upload": "📥 Загрузить", "camera": "📸 Камера", "analyze": "🚀 Анализ", "guide": "📚 Мед. справочник", "malig": "Злокачественные", "benign": "Доброкачественные", "more": "Подробнее", "res_m": "🚨 Подозрение на рак", "res_b": "🔍 Доброкачественное", "res_g": "🩺 Общее", "advice": "Обратитесь к врачу.", "share": "Поделиться"},
    "中文": {"dir": "ltr", "title": "🛡️ 皮肤安全人工智能", "upload": "📥 上传图片", "camera": "📸 相机", "analyze": "🚀 开始分析", "guide": "📚 医学指南", "malig": "恶性肿瘤", "benign": "良性肿瘤", "more": "详情", "res_m": "🚨 疑似恶性", "res_b": "🔍 良性", "res_g": "🩺 一般情况", "advice": "请咨询医生。", "share": "分享"},
    "हिन्दी": {"dir": "ltr", "title": "🛡️ त्वचा सुरक्षा एआई", "upload": "📥 छवि अपलोड करें", "camera": "📸 कैमरा", "analyze": "🚀 विश्लेषण करें", "guide": "📚 चिकित्सा गाइड", "malig": "घातक ट्यूمر", "benign": "सौم्य ट्यूمر", "more": "विवरण", "res_m": "🚨 घातक संदेह", "res_b": "🔍 सौم्य", "res_g": "🩺 सामान्य", "advice": "डॉक्टर से सलाह लें।", "share": "साझा करें"},
    "Italiano": {"dir": "ltr", "title": "🛡️ IA Sicurezza Pelle", "upload": "📥 Carica immagine", "camera": "📸 Fotocamera", "analyze": "🚀 Analizza", "guide": "📚 Guida Medica", "malig": "Maligno", "benign": "Benigno", "more": "Dettagli", "res_m": "🚨 Sospetto Maligno", "res_b": "🔍 Benigno", "res_g": "🩺 Generale", "advice": "Consultare un medico.", "share": "Condividi"},
    "日本語": {"dir": "ltr", "title": "🛡️ 皮膚安全AI", "upload": "📥 画像をアップロード", "camera": "📸 カメラ", "analyze": "🚀 解析開始", "guide": "📚 医療ガイド", "malig": "悪性", "benign": "良性", "more": "詳細", "res_m": "🚨 悪性の疑い", "res_b": "🔍 良性", "res_g": "🩺 一般的な状態", "advice": "医師に相談してください。", "share": "共有"},
    "한국어": {"dir": "ltr", "title": "🛡️ 피부 안전 AI", "upload": "📥 이미지 업로드", "camera": "📸 카메라", "analyze": "🚀 분석 시작", "guide": "📚 의료 가이드", "malig": "악성", "benign": "양성", "more": "상세 정보", "res_m": "🚨 악성 의심", "res_b": "🔍 양성 종양", "res_g": "🩺 일반 상태", "advice": "전문가와 상담하세요.", "share": "공유하기"},
    "Tiếng Việt": {"dir": "ltr", "title": "🛡️ AI An Toàn Da", "upload": "📥 Tải ảnh lên", "camera": "📸 Máy ảnh", "analyze": "🚀 Phân tích", "guide": "📚 Hướng dẫn Y tế", "malig": "Ác tính", "benign": "Lành tính", "more": "Chi tiết", "res_m": "🚨 Nghi ngờ Ác tính", "res_b": "🔍 Lành tính", "res_g": "🩺 Trạng thái Chung", "advice": "Vui lòng tham khảo ý kiến bác sĩ.", "share": "Chia sẻ"},
    "فارسی": {"dir": "rtl", "title": "🛡️ هوش مصنوعی سلامت پوست", "upload": "📥 بارگذاری تصویر", "camera": "📸 دوربین", "analyze": "🚀 شروع آنالیز", "guide": "📚 راهنمای پزشکی", "malig": "بدخیم", "benign": "خوش‌خیم", "more": "جزئیات", "res_m": "🚨 مشکوک به بدخیم", "res_b": "🔍 خوش‌خیم", "res_g": "🩺 وضعیت عمومی", "advice": "به پزشک مراجعه کنید.", "share": "اشتراك‌گذاری"},
    "اردو": {"dir": "rtl", "title": "🛡️ جلد کی حفاظت کا AI", "upload": "📥 تصویر اپلوڈ کریں", "camera": "📸 کیمرہ", "analyze": "🚀 تجزیہ شروع کریں", "guide": "📚 طبی گائیڈ", "malig": "خطرناک", "benign": "بے ضرر", "more": "تفصیلات", "res_m": "🚨 خطرناک شبہ", "res_b": "🔍 بے ضرر رسولی", "res_g": "🩺 عام صورتحال", "advice": "ڈاکٹر سے مشورہ کریں۔", "share": "شیئر کریں"},
    "Kiswahili": {"dir": "ltr", "title": "🛡️ AI ya Usalama wa Ngozi", "upload": "📥 Pakia picha", "camera": "📸 Kamera", "analyze": "🚀 Uchambuzi", "guide": "📚 Mwongozo wa Matibabu", "malig": "Saratani", "benign": "Sio Saratani", "more": "Maelezo", "res_m": "🚨 Shaka ya Saratani", "res_b": "🔍 Sio Saratani", "res_g": "🩺 Hali ya Jumla", "advice": "Wasiliana na dكتari.", "share": "Shiriki"},
    "Nederlands": {"dir": "ltr", "title": "🛡️ Huidveiligheid AI", "upload": "📥 Upload afbeelding", "camera": "📸 Camera", "analyze": "🚀 Analyseer", "guide": "📚 Medische Gids", "malig": "Kwaadaardig", "benign": "Goedaardig", "more": "Details", "res_m": "🚨 Kwaadaardig", "res_b": "🔍 Goedaardig", "res_g": "🩺 Algemeen", "advice": "Raadpleeg een arts.", "share": "Delen"},
    "Bahasa Indonesia": {"dir": "ltr", "title": "🛡️ AI Keamanan Kulit", "upload": "📥 Unggah Gambar", "camera": "📸 Kamera", "analyze": "🚀 Analisis", "guide": "📚 Panduan Medis", "malig": "Ganas", "benign": "Jinak", "more": "Detail", "res_m": "🚨 Kecurigaan Ganas", "res_b": "🔍 Jinak", "res_g": "🩺 Kondisi Umum", "advice": "Konsultasikan dengan dokter.", "share": "Bagikan"},
    "Polski": {"dir": "ltr", "title": "🛡️ AI Bezpieczeństwa Skóry", "upload": "📥 Prześlij obraz", "camera": "📸 Kamera", "analyze": "🚀 Analizuj", "guide": "📚 Przewodnik Medyczny", "malig": "Złośliwe", "benign": "Łagodne", "more": "Szczegóły", "res_m": "🚨 Podejrzenie Zmiany", "res_b": "🔍 Zmiana Łagodna", "res_g": "🩺 Stan Ogólny", "advice": "Skonsultuj się z lekarzem.", "share": "Udostępnij"},
    "Türkmençe": {"dir": "ltr", "title": "🛡️ Deri Saglygy AI", "upload": "📥 Suraty ýükle", "camera": "📸 Kamera", "analyze": "🚀 Analizi başlat", "guide": "📚 Lukmançylyk Gollanmasy", "malig": "Howply Çişler", "benign": "Howpsuz Çişler", "more": "Maglumat", "res_m": "🚨 Howply Çiş Şübhessi", "res_b": "🔍 Howpsuz Çiş", "res_g": "🩺 Umumy Ýagdaý", "advice": "Hünärmen lukmana ýüz tutuň.", "share": "Paýlaş"},
    "বাংলা": {"dir": "ltr", "title": "🛡️ স্কিন সেফটি AI", "upload": "📥 ছবি আপলোড", "camera": "📸 ক্যামেরা", "analyze": "🚀 বিশ্লেষণ শুরু", "guide": "📚 চিকিৎসা নির্দেশিকা", "malig": "মারাত্মক", "benign": "সৌম্য", "more": "বিস্তারিত", "res_m": "🚨 মারাত্মক সন্দেহ", "res_b": "🔍 সৌম্য টিউমার", "res_g": "🩺 সাধারণ অবস্থা", "advice": "ডাক্তারের পরামর্শ নিন।", "share": "শেয়ার"},
    "ไทย": {"dir": "ltr", "title": "🛡️ AI ตรวจสอบผิวหนัง", "upload": "📥 อัปโหลดรูปภาพ", "camera": "📸 กล้อง", "analyze": "🚀 เริ่มวิเคราะห์", "guide": "📚 คู่มือการแพทย์", "malig": "เนื้อร้าย", "benign": "เนื้องอกธรรมดา", "more": "รายละเอียด", "res_m": "🚨 สงสัยว่าเป็นเนื้อร้าย", "res_b": "🔍 เนื้องอกธรรมดา", "res_g": "🩺 สภาวะทั่วไป", "advice": "โปรดปรึกษาแพทย์", "share": "แชร์"},
    "Română": {"dir": "ltr", "title": "🛡️ AI Siguranța Pielii", "upload": "📥 Încarcă imaginea", "camera": "📸 Cameră", "analyze": "🚀 Analizează", "guide": "📚 Ghid Medical", "malig": "Malign", "benign": "Benign", "more": "Detalii", "res_m": "🚨 Suspiciune Malignă", "res_b": "🔍 Benign", "res_g": "🩺 Stare Generală", "advice": "Consultați un medic.", "share": "Distribuiți"}
}

# --- 3. التنسيق البصري (ثابت) ---
selected_lang = st.sidebar.selectbox("🌐 اختر اللغة / Language", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

st.markdown(f"""
<style>
    div[dir='{t['dir']}'] {{ text-align: {'right' if t['dir']=='rtl' else 'left'}; }}
    .report-card {{ padding: 25px; border-radius: 15px; text-align: center; border: 5px solid; margin-top: 20px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
    .disease-item {{ border-right: 5px solid #0d47a1; border-left: 1px solid #eee; padding: 12px; background: #fff; margin-bottom: 8px; border-radius: 8px; font-size: 14px; }}
    .link-btn {{ display: inline-block; padding: 6px 12px; background: #1a73e8; color: white !important; text-decoration: none; border-radius: 5px; font-weight: bold; margin-top: 5px; }}
</style>
""", unsafe_allow_html=True)

# --- 4. معالجة الصور و Float (تحديث لحل المشكلة) ---
@st.cache_resource
def load_expert_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except: return None

interpreter = load_expert_model()

def prepare_image(image, interpreter):
    # استقبال الأبعاد المطلوبة آلياً من ملف الـ TFLite نفسه لضمان التوافق
    input_details = interpreter.get_input_details()
    batch, height, width, channels = input_details[0]['shape']
    
    # 1. تحويل الصورة لـ RGB لضمان وجود 3 قنوات ألوان دائماً
    img_rgb = image.convert("RGB")
    
    # 2. تغيير المقاس للمقاس الذي يطلبه النموذج (مثلاً 224x224) مهما كان حجم الأصل
    img_resized = img_rgb.resize((width, height))
    
    # 3. تحويل لنوع float32 العام والمعتمد في أغلب موديلات TFLite
    img_array = np.array(img_resized).astype(np.float32)
    
    # 4. التطبيع (Normalization) لجعل القيم بين 0 و 1
    img_array = img_array / 255.0
    
    # 5. إضافة بعد الـ Batch لتصبح [1, H, W, C]
    return np.expand_dims(img_array, axis=0)

# --- 5. واجهة الفحص (ثابتة مع المعالجة الجديدة) ---
st.markdown(f"<div dir='{t['dir']}'>", unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align: center; color: #0d47a1;'>{t['title']}</h1>", unsafe_allow_html=True)

choice = st.radio("", (t['upload'], t['camera']))
file = st.file_uploader(t['upload'], type=["jpg", "png", "jpeg"]) if choice == t['upload'] else st.camera_input(t['camera'])

if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)
    
    if st.button(t['analyze']):
        if interpreter:
            with st.spinner("AI Analysis..."):
                try:
                    final_input = prepare_image(img, interpreter)
                    in_idx = interpreter.get_input_details()[0]['index']
                    interpreter.set_tensor(in_idx, final_input)
                    interpreter.invoke()
                    
                    out_idx = interpreter.get_output_details()[0]['index']
                    output = interpreter.get_tensor(out_idx)[0]
                    idx = np.argmax(output)

                    # منطق التصنيف (خبيث: 1,4,17 | حميد: 2,5,23)
                    if idx in [1, 4, 17]:
                        res_msg, color = t['res_m'], "#cf1322"
                    elif idx in [2, 5, 23]:
                        res_msg, color = t['res_b'], "#389e0d"
                    else:
                        res_msg, color = t['res_g'], "#096dd9"

                    st.markdown(f'<div class="report-card" style="border-color: {color}; color: {color};"><h2>{res_msg}</h2><p>{t["advice"]}</p></div>', unsafe_allow_html=True)
                    
                    share_txt = urllib.parse.quote(f"{res_msg} - {t['advice']}")
                    st.markdown(f'<div style="text-align: center; margin-top: 15px;"><a href="https://wa.me/?text={share_txt}" target="_blank" style="background:#25D366; color:white; padding:10px 20px; border-radius:10px; text-decoration:none; font-weight:bold;">WhatsApp</a></div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Inference Error: {e}")

st.write("---")

# --- 6. الدليل الطبي (تحديث الروابط الصحيحة 100%) ---
with st.expander(f"📖 {t['guide']}"):
    tab_m, tab_b = st.tabs([t['malig'], t['benign']])
    
    with tab_m: # الأورام الخبيثة - روابط Mayo Clinic المباشرة
        m_list = [
            ("Basal Cell Carcinoma (BCC)", "https://www.mayoclinic.org/diseases-conditions/basal-cell-carcinoma/symptoms-causes/syc-20354487"),
            ("Squamous Cell Carcinoma (SCC)", "https://www.mayoclinic.org/diseases-conditions/squamous-cell-carcinoma/symptoms-causes/syc-20352480"),
            ("Melanoma / الميلانوما", "https://www.mayoclinic.org/diseases-conditions/melanoma/symptoms-causes/syc-20374884"),
            ("Merkel Cell Carcinoma", "https://www.mayoclinic.org/diseases-conditions/merkel-cell-carcinoma/symptoms-causes/syc-20351030"),
            ("Kaposi Sarcoma", "https://www.mayoclinic.org/diseases-conditions/kaposi-sarcoma/symptoms-causes/syc-20353140"),
            ("Sebaceous Carcinoma", "https://www.mayoclinic.org/diseases-conditions/sebaceous-carcinoma/symptoms-causes/syc-20352957"),
            ("Dermatofibrosarcoma Protuberans", "https://www.mayoclinic.org/diseases-conditions/dermatofibrosarcoma-protuberans/symptoms-causes/syc-20352949"),
            ("Cutaneous T-cell Lymphoma", "https://www.mayoclinic.org/diseases-conditions/cutaneous-t-cell-lymphoma/symptoms-causes/syc-20351034")
        ]
        for n, l in m_list:
            st.markdown(f'<div class="disease-item"><strong>{n}</strong><br><a href="{l}" target="_blank" class="link-btn">{t["more"]}</a></div>', unsafe_allow_html=True)

    with tab_b: # الأورام الحميدة - روابط Mayo Clinic المباشرة
        b_list = [
            ("Nevi / Moles (الشامات)", "https://www.mayoclinic.org/diseases-conditions/moles/symptoms-causes/syc-20375200"),
            ("Seborrheic Keratosis", "https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878"),
            ("Lipoma (الأورام الشحمية)", "https://www.mayoclinic.org/diseases-conditions/lipoma/symptoms-causes/syc-20374470"),
            ("Hemangioma (الأورام الوعائية)", "https://www.mayoclinic.org/diseases-conditions/infantile-hemangioma/symptoms-causes/syc-20353177"),
            ("Dermatofibroma", "https://my.clevelandclinic.org/health/diseases/22643-dermatofibroma"), # Cleveland Clinic المصدر الأدق هنا
            ("Sebaceous Cyst", "https://www.healthline.com/health/sebaceous-cyst"), 
            ("Skin Tags", "https://www.healthline.com/health/skin-tag"),
            ("Actinic Keratosis", "https://www.mayoclinic.org/diseases-conditions/actinic-keratosis/symptoms-causes/syc-20354969")
        ]
        for n, l in b_list:
            st.markdown(f'<div class="disease-item" style="border-right-color:#389e0d;"><strong>{n}</strong><br><a href="{l}" target="_blank" class="link-btn">{t["more"]}</a></div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey; font-size: 0.8em;'>Global Skin Guard AI © 2026</p>", unsafe_allow_html=True)
