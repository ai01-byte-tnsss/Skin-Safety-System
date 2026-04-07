import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np

# --- 1. إعدادات الصفحة ---
st.set_page_config(
    page_title="Skin Safety AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. القاموس اللغوي الشامل (25 لغة) ---
LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "نظام الكشف عن سلامة الجلد باستخدام AI", "note": "⚠️ للحصول على أدق النتائج، يرجى التصوير في ضوء طبيعي جيد والتركيز على المنطقة المصابة فقط.", "upload": "📥 ارفع صورة الفحص", "camera": "📸 صورة فورية", "analyze": "🚀 بدء التحليل", "guide": "📚 الدليل الطبي الشامل", "malig": "🔴 الأورام الخبيثة", "benign": "🟢 الأورام الحميدة", "res_m": "🚨 اشتباه ورم خبيث", "res_b": "🔍 ورم سليم (حميد)", "res_g": "🩺 حالة عامة / غير ذلك", "advice": "يرجى مراجعة المختص لضمان السلامة.", "cause": "سبب التكوين"},
    "English": {"dir": "ltr", "title": "Skin Safety Detection System using AI", "note": "⚠️ For best results, use good natural lighting and focus only on the affected area.", "upload": "📥 Upload Scan", "camera": "📸 Instant Photo", "analyze": "🚀 Analyze", "guide": "📚 Medical Guide", "malig": "🔴 Malignant", "benign": "🟢 Benign", "res_m": "🚨 Malignant Suspect", "res_b": "🔍 Benign Result", "res_g": "🩺 General/Other", "advice": "Please consult a specialist.", "cause": "Cause of Formation"},
    "Français": {"dir": "ltr", "title": "Système de détection de la sécurité cutanée utilisant l'IA", "note": "⚠️ Pour de meilleurs résultats, utilisez un bon éclairage naturel.", "upload": "📥 Charger", "camera": "📸 Caméra", "analyze": "🚀 Analyser", "guide": "📚 Guide Médical", "malig": "🔴 Malin", "benign": "🟢 Bénin", "res_m": "🚨 Suspect Malin", "res_b": "🔍 Résultat Bénin", "res_g": "🩺 Autre/Général", "advice": "Consultez un spécialiste.", "cause": "Cause de formation"},
    "Español": {"dir": "ltr", "title": "Sistema de detección de seguridad cutánea mediante IA", "note": "⚠️ Use luz natural para mejores resultados.", "upload": "📥 Subir", "camera": "📸 Cámara", "analyze": "🚀 Analizar", "guide": "📚 Guía Médica", "malig": "🔴 Maligno", "benign": "🟢 Benigno", "res_m": "🚨 Sospecha Maligna", "res_b": "🔍 Benigno", "res_g": "🩺 Otro/General", "advice": "Consulte a un médico.", "cause": "Causa de formación"},
    "Deutsch": {"dir": "ltr", "title": "Haut-Sicherheits-Erkennungssystem mit KI", "note": "⚠️ Nutzen Sie natürliches Licht.", "upload": "📥 Hochladen", "camera": "📸 Kamera", "analyze": "🚀 Analysieren", "guide": "📚 Leitfaden", "malig": "🔴 Bösartig", "benign": "🟢 Gutartig", "res_m": "🚨 Krebsverdacht", "res_b": "🔍 Gutartig", "res_g": "🩺 Allgemeين/Andere", "advice": "Arzt aufsuchen.", "cause": "Ursache der Entstehung"},
    "中文": {"dir": "ltr", "title": "利用人工智能检测皮肤安全系统", "note": "⚠️ 为了获得最佳效果，请使用自然光。", "upload": "📥 上传", "camera": "📸 相机", "analyze": "🚀 分析", "guide": "📚 医学指南", "malig": "🔴 恶性", "benign": "🟢 良性", "res_m": "🚨 疑似恶性", "res_b": "🔍 良性结果", "res_g": "🩺 其他/一般", "advice": "请咨询医生。", "cause": "形成原因"},
    "हिन्दी": {"dir": "ltr", "title": "AI का उपयोग करके त्वचा सुरक्षा पहचान प्रणाली", "note": "⚠️ सर्वोत्तम परिणामों के लिए प्राकृतिक रोशनी का उपयोग करें।", "upload": "📥 अपलोड", "camera": "📸 कैमरा", "analyze": "🚀 विश्लेषण", "guide": "📚 चिकित्सा गाइड", "malig": "🔴 घातक", "benign": "🟢 सौम्य", "res_m": "🚨 घातक संदेह", "res_b": "🔍 सौम्य परिणाम", "res_g": "🩺 अन्य/सामान्य", "advice": "विशेषज्ञ से सलाह लें।", "cause": "गठन का कारण"},
    "Русский": {"dir": "ltr", "title": "Система контроля состояния кожи с использованием ИИ", "note": "⚠️ Используйте естественный свет.", "upload": "📥 Загрузить", "camera": "📸 Камера", "analyze": "🚀 Анализ", "guide": "📚 Справочник", "malig": "🔴 Злокачественные", "benign": "🟢 Доброкачественные", "res_m": "🚨 Подозрение", "res_b": "🔍 Доброкачественное", "res_g": "🩺 Общее/Другое", "advice": "Обратитесь к врачу.", "cause": "Причина образования"},
    "日本語": {"dir": "ltr", "title": "AIを活用した皮膚安全検知システム", "note": "⚠️ 自然光を使用してください。", "upload": "📥 アップロード", "camera": "📸 カメラ", "analyze": "🚀 解析", "guide": "📚 ガイド", "malig": "🔴 悪性", "benign": "🟢 良性", "res_m": "🚨 悪性の疑い", "res_b": "🔍 良性", "res_g": "🩺 その他/一般", "advice": "医師に相談。", "cause": "形成の原因"},
    "Português": {"dir": "ltr", "title": "Sistema de detecção de segurança de pele usando IA", "note": "⚠️ Use luz natural clara.", "upload": "📥 Enviar", "camera": "📸 Câmera", "analyze": "🚀 Analisar", "guide": "📚 Guia Médico", "malig": "🔴 Maligno", "benign": "🟢 Benigno", "res_m": "🚨 Suspeita", "res_b": "🔍 Benigno", "res_g": "🩺 Outro/Geral", "advice": "Consulte um médico.", "cause": "Causa da formação"},
    "Türkçe": {"dir": "ltr", "title": "Yapay Zeka Kullanarak Cilt Güvenliği Tespit Sistemi", "note": "⚠️ Doğal ışık kullanın.", "upload": "📥 Yükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz Et", "guide": "📚 Tıbbi Rehber", "malig": "🔴 Kötü Huylu", "benign": "🟢 İyi Huylu", "res_m": "🚨 Şüphe", "res_b": "🔍 İyi Huylu", "res_g": "🩺 Diğer/Genel", "advice": "Doktora danışın.", "cause": "Oluşum Nedeni"},
    "한국어": {"dir": "ltr", "title": "AI를 활용한 피부 안전 감지 시스템", "note": "⚠️ 자연광에서 촬영하세요.", "upload": "📥 업로드", "camera": "📸 카메라", "analyze": "🚀 분석", "guide": "📚 가이드", "malig": "🔴 악성", "benign": "🟢 양성", "res_m": "🚨 악성 의심", "res_b": "🔍 양성", "res_g": "🩺 기타/일반", "advice": "전문가 상담。", "cause": "형성 원인"},
    "Italiano": {"dir": "ltr", "title": "Sistema di rilevamento della sicurezza cutanea tramite IA", "note": "⚠️ Usa luce naturale.", "upload": "📥 Carica", "camera": "📸 Camera", "analyze": "🚀 Analizza", "guide": "📚 Guida", "malig": "🔴 Maligno", "benign": "🟢 Benigno", "res_m": "🚨 Sospetto", "res_b": "🔍 Benigno", "res_g": "🩺 Altro/Generale", "advice": "Consulta un medico.", "cause": "Causa della formazione"},
    "اردو": {"dir": "rtl", "title": "AI کا استعمال کرتے ہوئے جلد کی حفاظت کا پتہ لگانے والا نظام", "note": "⚠️ قدرتی روشنی استعمال کریں۔", "upload": "📥 اپلوڈ", "camera": "📸 کیمرہ", "analyze": "🚀 تجزیہ", "guide": "📚 گائیڈ", "malig": "🔴 خطرناک", "benign": "🟢 بے ضرر", "res_m": "🚨 شبہ", "res_b": "🔍 بے ضرر", "res_g": "🩺 دیگر/عام", "advice": "ڈاکٹر سے مشورہ۔", "cause": "بننے کی وجہ"},
    "فارسي": {"dir": "rtl", "title": "سیستم تشخیص سلامت پوست با استفاده از هوش مصنوعی", "note": "⚠️ از نور طبیعی استفاده کنید.", "upload": "📥 بارگذاری", "camera": "📸 دوربین", "analyze": "🚀 آنالیز", "guide": "📚 راهنما", "malig": "🔴 بدخیم", "benign": "🟢 خوش‌خیم", "res_m": "🚨 مشکوک", "res_b": "🔍 خوش‌خیم", "res_g": "🩺 سایر/عمومی", "advice": "به پزشک مراجعه کنید.", "cause": "علت ایجاد"},
    "Tiếng Việt": {"dir": "ltr", "title": "Hệ thống phát hiện an toàn da bằng AI", "note": "⚠️ Sử dụng ánh sáng tự nhiên.", "upload": "📥 Tải lên", "camera": "📸 Máy ảnh", "analyze": "🚀 Phân tích", "guide": "📚 Hướng dẫn", "malig": "🔴 Ác tính", "benign": "🟢 Lành tính", "res_m": "🚨 Nghi ngờ", "res_b": "🔍 Lành tính", "res_g": "🩺 Khác/Tổng quát", "advice": "Hỏi ý kiến bác sĩ.", "cause": "Nguyên nhân hình thành"},
    "Bahasa Indonesia": {"dir": "ltr", "title": "Sistem Deteksi Keamanan Kulit menggunakan AI", "note": "⚠️ Gunakan cahaya alami.", "upload": "📥 Unggah", "camera": "📸 Kamera", "analyze": "🚀 Analisis", "guide": "📚 Panduan", "malig": "🔴 Ganas", "benign": "🟢 Jinak", "res_m": "🚨 Kecurigaan", "res_b": "🔍 Jinak", "res_g": "🩺 Lainnya/Umum", "advice": "Konsultasi dokter.", "cause": "Penyebab pembentukan"},
    "Nederlands": {"dir": "ltr", "title": "Huidsysteem voor veiligheidsdetectية met behulp van AI", "note": "⚠️ Gebruik natuurlijk licht.", "upload": "📥 Upload", "camera": "📸 Camera", "analyze": "🚀 Analyse", "guide": "📚 Gids", "malig": "🔴 Kwaadaardig", "benign": "🟢 Goedaardig", "res_m": "🚨 Verdacht", "res_b": "🔍 Goedaardig", "res_g": "🩺 Overig/Algemeen", "advice": "Raadpleeg arts.", "cause": "Oorzaak van vorming"},
    "Polski": {"dir": "ltr", "title": "System wykrywania bezpieczeństwa skóry przy użyciu AI", "note": "⚠️ Użyj światła dziennego.", "upload": "📥 Prześlij", "camera": "📸 Kamera", "analyze": "🚀 Analiza", "guide": "📚 Przewodnik", "malig": "🔴 Złośliwe", "benign": "🟢 Łagodne", "res_m": "🚨 Podejrzenie", "res_b": "🔍 Łagodne", "res_g": "🩺 Inne/Ogólne", "advice": "Skonsultuj się.", "cause": "Przyczyna powstania"},
    "ไทย": {"dir": "ltr", "title": "ระบบตรวจจับความปลอดภัยของผิวหนังโดยใช้ AI", "note": "⚠️ ใช้แสงธรรมชาติ", "upload": "📥 อัปโหลด", "camera": "📸 กล้อง", "analyze": "🚀 วิเคราะห์", "guide": "📚 คู่มือ", "malig": "🔴 เนื้อร้าย", "benign": "🟢 เนื้อดี", "res_m": "🚨 สงสัยเนื้อร้าย", "res_b": "🔍 เนื้อดี", "res_g": "🩺 อื่นๆ/ทั่วไป", "advice": "ปรึกษาแพทย์", "cause": "สาเหตุการเกิด"},
    "کوردی": {"dir": "rtl", "title": "سیستەمی دەستنیشانکردنی سەلامەتی پێست بە بەکارهێنانی AI", "note": "⚠️ تیشکی سروشتی بەکاربهێنە.", "upload": "📥 وێنە", "camera": "📸 کامێرا", "analyze": "🚀 شیکاري", "guide": "📚 ڕێبەر", "malig": "🔴 خراپ", "benign": "🟢 بێ زیان", "res_m": "🚨 گومانی خراپ", "res_b": "🔍 بێ زیان", "res_g": "🩺 جۆری تر/گشتی", "advice": "سەردانی پزیشک بکە.", "cause": "هۆکاری دروستبوون"},
    "Bengali": {"dir": "ltr", "title": "AI ব্যবহার করে ত্বক সুরক্ষা সনাক্তকরণ সিস্টেম", "note": "⚠️ প্রাকৃতিক আলো ব্যবহার করুন।", "upload": "📥 আপলোড", "camera": "📸 ক্যামেরা", "analyze": "🚀 विश्लेषण", "guide": "📚 নির্দেশিকা", "malig": "🔴 মারাত্মক", "benign": "🟢 সৌম্য", "res_m": "🚨 সন্দেহজনক", "res_b": "🔍 সৌম্য", "res_g": "🩺 অন্যান্য/সাধারণ", "advice": "পরামর্শ নিন।", "cause": "গঠনের কারণ"},
    "Română": {"dir": "ltr", "title": "Sistem de detectare a siguranței pielii folosind AI", "note": "⚠️ Folosiți lumină naturală.", "upload": "📥 Încarcă", "camera": "📸 Cameră", "analyze": "🚀 Analizează", "guide": "📚 Ghid", "malig": "🔴 Malign", "benign": "🟢 Benign", "res_m": "🚨 Suspect", "res_b": "🔍 Benign", "res_g": "🩺 Altele/General", "advice": "Consultă medicul.", "cause": "Cauza formării"},
    "Kiswahili": {"dir": "ltr", "title": "Mfumo wa utambuzi wa usalama wa ngozi kwa kutumia AI", "note": "⚠️ Tumia mwanga wa asili.", "upload": "📥 Pakia", "camera": "📸 Kamera", "analyze": "🚀 Uchambuzi", "guide": "📚 Mwongozo", "malig": "🔴 Saratani", "benign": "🟢 Salama", "res_m": "🚨 Shaka", "res_b": "🔍 Salama", "res_g": "🩺 Nyingine/Jumla", "advice": "Ona daktari.", "cause": "Sababu ya kuunda"},
    "Türkmençe": {"dir": "ltr", "title": "AI ulanyp deri howpsuzlygyny anyklaýyş ulgamy", "note": "⚠️ Tebigy yşyk ulanyň.", "upload": "📥 Ýükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz", "guide": "📚 Gollanma", "malig": "🔴 Howply", "benign": "🟢 Howpsuz", "res_m": "🚨 Şüphe", "res_b": "🔍 Howpsuz", "res_g": "🩺 Başga/Umumy", "advice": "Lukmana ýüz tutuň.", "cause": "Emele gelmeginiň sebäbi"}
}

# --- 3. التنسيق البصري المحسن ---
selected_lang = st.sidebar.selectbox("🌐 اختر اللغة / Language", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    html, body, [class*="st-"] {{ font-family: 'Tajawal', sans-serif; font-size: 16px; }}
    div[dir='{t['dir']}'] {{ text-align: {'right' if t['dir']=='rtl' else 'left'}; }}
    .main-title {{ text-align: center; color: #0d47a1; font-size: 1.8em; margin-bottom: 20px; font-weight: bold; }}
    .report-card {{ padding: 25px; border-radius: 20px; text-align: center; border: 5px solid; margin-top: 15px; background: white; }}
    .note-box {{ background: #fffbe6; border: 1px solid #ffe58f; padding: 12px; border-radius: 10px; margin-bottom: 15px; font-size: 0.9em; }}
    .disease-card {{ border-right: 5px solid #0d47a1; border-left: 1px solid #ddd; padding: 12px; background: #fdfdfd; margin-bottom: 10px; border-radius: 8px; font-size: 0.95em; }}
    .stButton>button {{ width: 100%; border-radius: 8px; font-weight: bold; background-color: #0d47a1; color: white; height: 3em; }}
</style>
""", unsafe_allow_html=True)

# --- 4. المحرك البرمجي (التصحيح الدقيق للفئات) ---
@st.cache_resource
def load_expert_model():
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    predictions = Dense(7, activation='softmax')(Dropout(0.4)(x))
    return Model(inputs=base.input, outputs=predictions)

model = load_expert_model()

# --- 5. واجهة المستخدم الرئيسية ---
st.markdown(f"<div dir='{t['dir']}'>", unsafe_allow_html=True)
st.markdown(f"<div class='main-title'>{t['title']}</div>", unsafe_allow_html=True)
st.markdown(f'<div class="note-box">{t["note"]}</div>', unsafe_allow_html=True)

choice = st.radio(" ", (t['upload'], t['camera']), horizontal=True)
file = st.file_uploader(t['upload'], type=["jpg", "png", "jpeg"]) if choice == t['upload'] else st.camera_input(t['camera'])

if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)
    
    if st.button(t['analyze']):
        with st.spinner("..."):
            # معالجة الصورة للأبعاد الصحيحة
            img_resized = img.convert("RGB").resize((224, 224))
            img_array = np.array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            processed_img = tf.keras.applications.efficientnet.preprocess_input(img_array)
            
            # التنبؤ
            preds = model.predict(processed_img)[0]
            idx = np.argmax(preds)
            confidence = np.max(preds)
            
            # --- منطق التصنيف المصحح والمطور ---
            # 0: akiec (خبيث), 1: bcc (خبيث), 2: bkl (حميد), 3: df (حميد), 4: mel (خبيث), 5: nv (حميد), 6: vasc (حميد)
            malignant_indices = [0, 1, 4] 
            benign_indices = [2, 3, 5, 6]

            # التأكد من صحة التصنيف بناءً على عتبة ثقة (تلقائياً)
            if confidence < 0.30:
                res_msg, color = t['res_g'], "#096dd9"
            elif idx in malignant_indices:
                res_msg, color = t['res_m'], "#cf1322"
            elif idx in benign_indices:
                res_msg, color = t['res_b'], "#389e0d"
            else:
                res_msg, color = t['res_g'], "#096dd9"

            st.markdown(f'<div class="report-card" style="border-color: {color}; color: {color};"><h2>{res_msg}</h2><p>{t["advice"]}</p></div>', unsafe_allow_html=True)

st.write("---")

# --- 6. الدليل الطبي الشامل (بدون اختصار) ---
with st.expander(f"📖 {t['guide']}"):
    m_tab, b_tab = st.tabs([t['malig'], t['benign']])
    with m_tab:
        m_diseases = [
            ("Melanoma (mel)", "سرطان الخلايا الصبغية الأخطر.", "طفرات في الميلانين بسبب الأشعة.", "تغير مفاجئ في لون وحجم الشامات."),
            ("Basal Cell Carcinoma (bcc)", "سرطان الخلايا القاعدية الشائع.", "التعرض الطويل لأشعة الشمس.", "نتوء لؤلؤي أو قشرة تنزف ولا تشفى."),
            ("Actinic Keratosis (akiec)", "التقران السفعي / حرشفي موضعي.", "تضرر الـ DNA في طبقات الجلد السطحية.", "كتلة حمراء صلبة ذات سطح متقشر."),
            ("Merkel Cell Carcinoma", "سرطان خلايا ميركل النادر.", "فيروس ميركل أو ضعف الجهاز المناعي.", "نتوءات صلبة غير مؤلمة سريعة النمو."),
            ("Kaposi Sarcoma", "ساركوما كابوزي الوعائي.", "عدوى فيروسية (HHV-8).", "بقع أو كتل أرجوانية/حمراء على الجلد."),
            ("Sebaceous Carcinoma", "سرطان الغدد الدهنية.", "نمو سرطاني في غدد الجفون والوجه.", "نتوء صلب يشبه 'شحاذ العين' المستمر."),
            ("Dermatofibrosarcoma", "ساركوما جلدية ليفية جاحظة.", "طفرة جينية نادرة في الأنسجة.", "ندبة صلبة تنمو ببطء شديد لسنوات."),
            ("Cutaneous Lymphoma", "ليمفوما جلدية (خلايا T).", "تكاثر غير طبيعي للخلايا الليمفاوية.", "بقع تشبه الإكزيما أو الصدفية.")
        ]
        for n, m, h, s in m_diseases:
            st.markdown(f'<div class="disease-card" style="border-right-color:#cf1322;"><span style="color:#cf1322; font-weight:bold;">🔴 {n}</span><br><b>الوصف:</b> {m}<br><b>{t["cause"]}:</b> {h}<br><b>الأعراض:</b> {s}</div>', unsafe_allow_html=True)
    
    with b_tab:
        b_diseases = [
            ("Nevi (nv)", "الشامات الطبيعية.", "تجمع سليم للخلايا الصبغية.", "بقع بنية منتظمة ومستقرة تماماً."),
            ("Benign Keratosis (bkl)", "التقران الحميد.", "تكاثر خلايا الكيراتين السطحية.", "زوائد شمعية بنية تشبه الملصقات الجلدية."),
            ("Dermatofibroma (df)", "الألياف الجلدية السليمة.", "رد فعل لقرصة حشرة أو جرح بسيط.", "عقدة صغيرة صلبة بنية تميل للداخل."),
            ("Lipoma", "الورم الشحمي السليم.", "تجمع كتل دهنية تحت الجلد.", "كتلة لينة تتحرك بسهولة عند لمسها."),
            ("Hemangioma", "الورم الوعائي (النقطة الكرزية).", "تجمع غير سرطاني للأوعية الدموية.", "بقع حمراء زاهية أو نتوءات دموية."),
            ("Skin Cyst", "الأكياس الجلدية/الدهنية.", "انسداد المسام أو التهاب الغدد.", "نتوء يحتوي على مادة كيراتينية بيضاء."),
            ("Skin Tags", "الزوائد الجلدية الشائعة.", "احتكاك الجلد المستمر أو الوراثة.", "قطع جلدية صغيرة متدلية من الرقبة."),
            ("Angiokeratoma", "التقرن الوعائي.", "توسع الشعيرات الدموية السطحية.", "نقاط حمراء أو زرقاء داكنة صلبة جداً.")
        ]
        for n, m, h, s in b_diseases:
            st.markdown(f'<div class="disease-card" style="border-right-color:#389e0d;"><span style="color:#389e0d; font-weight:bold;">🟢 {n}</span><br><b>الوصف:</b> {m}<br><b>{t["cause"]}:</b> {h}<br><b>الأعراض:</b> {s}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
