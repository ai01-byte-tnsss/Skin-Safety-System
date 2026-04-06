import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Global Skin Guard AI", layout="wide")

# --- 2. القاموس اللغوي (25 لغة كاملة) ---
LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "🛡️ نظام الكشف عن سلامة الجلد", "note": "⚠️ للحصول على أدق النتائج، يرجى التصوير في ضوء طبيعي جيد والتركيز على المنطقة المصابة فقط.", "upload": "📥 ارفع صورة الفحص", "camera": "📸 صورة فورية", "analyze": "🚀 بدء التحليل", "guide": "📚 الدليل الطبي الشامل (أكثر الأنواع شيوعاً)", "malig": "🔴 الأورام الخبيثة", "benign": "🟢 الأورام الحميدة", "res_m": "🚨 اشتباه ورم خبيث", "res_b": "🔍 ورم حميد", "res_g": "🩺 مرض غير ذلك / حالة عامة", "advice": "يرجى مراجعة المختص لضمان السلامة."},
    "English": {"dir": "ltr", "title": "🛡️ Skin Safety AI", "note": "⚠️ For best results, use good natural lighting and focus only on the affected area.", "upload": "📥 Upload Scan", "camera": "📸 Instant Photo", "analyze": "🚀 Analyze", "guide": "📚 Medical Guide (Most Common Types)", "malig": "🔴 Malignant", "benign": "🟢 Benign", "res_m": "🚨 Malignant Suspect", "res_b": "🔍 Benign Result", "res_g": "🩺 Other/General Condition", "advice": "Please consult a specialist."},
    "Français": {"dir": "ltr", "title": "🛡️ IA de Sécurité Cutanée", "note": "⚠️ Pour de meilleurs résultats, utilisez un bon éclairage naturel.", "upload": "📥 Charger", "camera": "📸 Caméra", "analyze": "🚀 Analyser", "guide": "📚 Guide Médical (Types communs)", "malig": "🔴 Malin", "benign": "🟢 Bénin", "res_m": "🚨 Suspect Malin", "res_b": "🔍 Résultat Bénin", "res_g": "🩺 Autre/Général", "advice": "Consultez un spécialiste."},
    "Español": {"dir": "ltr", "title": "🛡️ IA Seguridad Cutánea", "note": "⚠️ Use luz natural para mejores resultados.", "upload": "📥 Subir", "camera": "📸 Cámara", "analyze": "🚀 Analizar", "guide": "📚 Guía Médica (Tipos comunes)", "malig": "🔴 Maligno", "benign": "🟢 Benigno", "res_m": "🚨 Sospecha Maligna", "res_b": "🔍 Benigno", "res_g": "🩺 Otro/General", "advice": "Consulte a un médico."},
    "Deutsch": {"dir": "ltr", "title": "🛡️ Haut-KI", "note": "⚠️ Nutzen Sie natürliches Licht.", "upload": "📥 Hochladen", "camera": "📸 Kamera", "analyze": "🚀 Analysieren", "guide": "📚 Leitfaden (Häufige Typen)", "malig": "🔴 Bösartig", "benign": "🟢 Gutartig", "res_m": "🚨 Krebsverdacht", "res_b": "🔍 Gutartig", "res_g": "🩺 Allgemein/Andere", "advice": "Arzt aufsuchen."},
    "中文": {"dir": "ltr", "title": "🛡️ 皮肤安全AI", "note": "⚠️ 为了获得最佳效果，请使用自然光。", "upload": "📥 上传", "camera": "📸 相机", "analyze": "🚀 分析", "guide": "📚 医学指南 (最常见类型)", "malig": "🔴 恶性", "benign": "🟢 良性", "res_m": "🚨 疑似恶性", "res_b": "🔍 良性结果", "res_g": "🩺 其他/一般", "advice": "请咨询医生。"},
    "हिन्दी": {"dir": "ltr", "title": "🛡️ त्वचा सुरक्षा AI", "note": "⚠️ सर्वोत्तम परिणामों के लिए प्राकृतिक रोशनी का उपयोग करें।", "upload": "📥 अपलोड", "camera": "📸 कैमरा", "analyze": "🚀 विश्लेषण", "guide": "📚 चिकित्सा गाइड (सामान्य प्रकार)", "malig": "🔴 घातक", "benign": "🟢 सौम्य", "res_m": "🚨 घातक संदेह", "res_b": "🔍 सौम्य परिणाम", "res_g": "🩺 अन्य/सामान्य", "advice": "विशेषज्ञ से सलाह लें।"},
    "Русский": {"dir": "ltr", "title": "🛡️ ИИ Кожи", "note": "⚠️ Используйте естественный свет.", "upload": "📥 Загрузить", "camera": "📸 Камера", "analyze": "🚀 Анализ", "guide": "📚 Справочник (Частые типы)", "malig": "🔴 Злокачественные", "benign": "🟢 Доброкачественные", "res_m": "🚨 Подозрение", "res_b": "🔍 Доброкачественное", "res_g": "🩺 Общее/Другое", "advice": "Обратитесь к врачу."},
    "日本語": {"dir": "ltr", "title": "🛡️ 皮膚安全AI", "note": "⚠️ 自然光を使用してください。", "upload": "📥 アップロード", "camera": "📸 カメラ", "analyze": "🚀 解析", "guide": "📚 ガイド (一般的な種類)", "malig": "🔴 悪性", "benign": "🟢 良性", "res_m": "🚨 悪性の疑い", "res_b": "🔍 良性", "res_g": "🩺 その他/一般", "advice": "医師に相談。"},
    "Português": {"dir": "ltr", "title": "🛡️ IA de Pele", "note": "⚠️ Use luz natural clara.", "upload": "📥 Enviar", "camera": "📸 Câmera", "analyze": "🚀 Analisar", "guide": "📚 Guia Médico (Tipos comuns)", "malig": "🔴 Maligno", "benign": "🟢 Benigno", "res_m": "🚨 Suspeita", "res_b": "🔍 Benigno", "res_g": "🩺 Outro/Geral", "advice": "Consulte um médico."},
    "Türkçe": {"dir": "ltr", "title": "🛡️ Cilt Güvenliği AI", "note": "⚠️ Doğal ışık kullanın.", "upload": "📥 Yükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz Et", "guide": "📚 Tıbbi Rehber (Yaygın Türler)", "malig": "🔴 Kötü Huylu", "benign": "🟢 İyi Huylu", "res_m": "🚨 Şüphe", "res_b": "🔍 İyi Huylu", "res_g": "🩺 Diğer/Genel", "advice": "Doktora danışın."},
    "한국어": {"dir": "ltr", "title": "🛡️ 피부 안전 AI", "note": "⚠️ 자연광에서 촬영하세요.", "upload": "📥 업로드", "camera": "📸 카메라", "analyze": "🚀 분석", "guide": "📚 가이드 (일반적인 유형)", "malig": "🔴 악성", "benign": "🟢 양성", "res_m": "🚨 악성 의심", "res_b": "🔍 양성", "res_g": "🩺 기타/일반", "advice": "전문가 상담。"},
    "Italiano": {"dir": "ltr", "title": "🛡️ IA Pelle", "note": "⚠️ Usa luce naturale.", "upload": "📥 Carica", "camera": "📸 Camera", "analyze": "🚀 Analizza", "guide": "📚 Guida (Tipi comuni)", "malig": "🔴 Maligno", "benign": "🟢 Benigno", "res_m": "🚨 Sospetto", "res_b": "🔍 Benigno", "res_g": "🩺 Altro/Generale", "advice": "Consulta un medico."},
    "اردو": {"dir": "rtl", "title": "🛡️ جلد کی حفاظت AI", "note": "⚠️ قدرتی روشنی استعمال کریں۔", "upload": "📥 اپلوڈ", "camera": "📸 کیمرہ", "analyze": "🚀 تجزیہ", "guide": "📚 گائیڈ (عام اقسام)", "malig": "🔴 خطرناک", "benign": "🟢 بے ضرر", "res_m": "🚨 شبہ", "res_b": "🔍 بے ضرر", "res_g": "🩺 دیگر/عام", "advice": "ڈاکٹر سے مشورہ۔"},
    "فارسی": {"dir": "rtl", "title": "🛡️ هوش مصنوعی پوست", "note": "⚠️ از نور طبیعی استفاده کنید.", "upload": "📥 بارگذاری", "camera": "📸 دوربین", "analyze": "🚀 آنالیز", "guide": "📚 راهنما (انواع رایج)", "malig": "🔴 بدخیم", "benign": "🟢 خوش‌خیم", "res_m": "🚨 مشکوک", "res_b": "🔍 خوش‌خیم", "res_g": "🩺 سایر/عمومی", "advice": "به پزشک مراجعه کنید."},
    "Tiếng Việt": {"dir": "ltr", "title": "🛡️ AI Da Liễu", "note": "⚠️ Sử dụng ánh sáng tự nhiên.", "upload": "📥 Tải lên", "camera": "📸 Máy ảnh", "analyze": "🚀 Phân tích", "guide": "📚 Hướng dẫn (Loại phổ biến)", "malig": "🔴 Ác tính", "benign": "🟢 Lành tính", "res_m": "🚨 Nghi ngờ", "res_b": "🔍 Lành tính", "res_g": "🩺 Khác/Tổng quát", "advice": "Hỏi ý kiến bác sĩ."},
    "Bahasa Indonesia": {"dir": "ltr", "title": "🛡️ AI Kulit", "note": "⚠️ Gunakan cahaya alami.", "upload": "📥 Unggah", "camera": "📸 Kamera", "analyze": "🚀 Analisis", "guide": "📚 Panduan (Jenis Umum)", "malig": "🔴 Ganas", "benign": "🟢 Jinak", "res_m": "🚨 Kecurigaan", "res_b": "🔍 Jinak", "res_g": "🩺 Lainnya/Umum", "advice": "Konsultasi dokter."},
    "Nederlands": {"dir": "ltr", "title": "🛡️ Huid AI", "note": "⚠️ Gebruik natuurlijk licht.", "upload": "📥 Upload", "camera": "📸 Camera", "analyze": "🚀 Analyse", "guide": "📚 Gids (Veelvoorkomend)", "malig": "🔴 Kwaadaardig", "benign": "🟢 Goedaardig", "res_m": "🚨 Verdacht", "res_b": "🔍 Goedaardig", "res_g": "🩺 Overig/Algemeen", "advice": "Raadpleeg arts."},
    "Polski": {"dir": "ltr", "title": "🛡️ AI Skóry", "note": "⚠️ Użyj światła dziennego.", "upload": "📥 Prześlij", "camera": "📸 Kamera", "analyze": "🚀 Analiza", "guide": "📚 Przewodnik (Typowe)", "malig": "🔴 Złośliwe", "benign": "🟢 Łagodne", "res_m": "🚨 Podejrzenie", "res_b": "🔍 Łagodne", "res_g": "🩺 Inne/Ogólne", "advice": "Skonsultuj się."},
    "ไทย": {"dir": "ltr", "title": "🛡️ AI ตรวจผิว", "note": "⚠️ ใช้แสงธรรมชาติ", "upload": "📥 อัปโหลด", "camera": "📸 กล้อง", "analyze": "🚀 วิเคราะห์", "guide": "📚 คู่มือ (ชนิดที่พบบ่อย)", "malig": "🔴 เนื้อร้าย", "benign": "🟢 เนื้อดี", "res_m": "🚨 สงสัยเนื้อร้าย", "res_b": "🔍 เนื้อดี", "res_g": "🩺 อื่นๆ/ทั่วไป", "advice": "ปรึกษาแพทย์"},
    "کوردی": {"dir": "rtl", "title": "🛡️ پشکنینی پێست", "note": "⚠️ تیشکی سروشتی بەکاربهێنە.", "upload": "📥 وێنە", "camera": "📸 کامێرا", "analyze": "🚀 شیکاري", "guide": "📚 ڕێبەر (جۆرە باوەکان)", "malig": "🔴 خراپ", "benign": "🟢 بێ زیان", "res_m": "🚨 گومانی خراپ", "res_b": "🔍 بێ زیان", "res_g": "🩺 جۆری تر/گشتی", "advice": "سەردانی پزیشک بکە."},
    "Bengali": {"dir": "ltr", "title": "🛡️ স্কিন AI", "note": "⚠️ প্রাকৃতিক আলো ব্যবহার করুন।", "upload": "📥 আপলোড", "camera": "📸 ক্যামেরা", "analyze": "🚀 বিশ্লেষণ", "guide": "📚 নির্দেশিকা (সাধারণ ধরন)", "malig": "🔴 মারাত্মক", "benign": "🟢 সৌম্য", "res_m": "🚨 সন্দেহজনক", "res_b": "🔍 সৌম্য", "res_g": "🩺 অন্যান্য/সাধারণ", "advice": "পরামর্শ নিন।"},
    "Română": {"dir": "ltr", "title": "🛡️ AI Piele", "note": "⚠️ Folosiți lumină naturală.", "upload": "📥 Încarcă", "camera": "📸 Cameră", "analyze": "🚀 Analizează", "guide": "📚 Ghid (Tipuri comune)", "malig": "🔴 Malign", "benign": "🟢 Benign", "res_m": "🚨 Suspect", "res_b": "🔍 Benign", "res_g": "🩺 Altele/General", "advice": "Consultă medicul."},
    "Kiswahili": {"dir": "ltr", "title": "🛡️ AI ya Ngozi", "note": "⚠️ Tumia mwanga wa asili.", "upload": "📥 Pakia", "camera": "📸 Kamera", "analyze": "🚀 Uchambuzi", "guide": "📚 Mwongozo (Aina za kawaida)", "malig": "🔴 Saratani", "benign": "🟢 Salama", "res_m": "🚨 Shaka", "res_b": "🔍 Salama", "res_g": "🩺 Nyingine/Jumla", "advice": "Ona daktari."},
    "Türkmençe": {"dir": "ltr", "title": "🛡️ Deri AI", "note": "⚠️ Tebigy yşyk ulanyň.", "upload": "📥 Ýükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz", "guide": "📚 Gollanma (Adaty görnüşler)", "malig": "🔴 Howply", "benign": "🟢 Howpsuz", "res_m": "🚨 Şüphe", "res_b": "🔍 Howpsuz", "res_g": "🩺 Başga/Umumy", "advice": "Lukmana ýüz tutuň."}
}

# --- 3. التنسيق البصري وتأثير الليزر ---
selected_lang = st.sidebar.selectbox("🌐 Choose Language / اختر اللغة", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

st.markdown(f"""
<style>
    div[dir='{t['dir']}'] {{ text-align: {'right' if t['dir']=='rtl' else 'left'}; }}
    .report-card {{ padding: 30px; border-radius: 20px; text-align: center; border: 6px solid; margin-top: 25px; background: white; }}
    .note-box {{ background: #fffbe6; border: 1px solid #ffe58f; padding: 18px; border-radius: 12px; margin-bottom: 25px; color: #856404; }}
    
    /* تأثير النبض الليزري حول الصورة */
    .laser-scan {{
        position: relative;
        border: 4px solid #ff4b4b;
        border-radius: 15px;
        overflow: hidden;
        animation: pulse-red 1.5s infinite;
    }}
    @keyframes pulse-red {{
        0% {{ box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }}
        70% {{ box-shadow: 0 0 0 20px rgba(255, 75, 75, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }}
    }}
    .disease-card {{ border-right: 5px solid #0d47a1; border-left: 1px solid #eee; padding: 15px; background: #f9f9f9; margin-bottom: 12px; border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)

# --- 4. المحرك (Float ديناميكي) ---
@st.cache_resource
def load_expert_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_expert_refined.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except: return None

interpreter = load_expert_model()

def prepare_image_dynamic(image, interpreter):
    input_details = interpreter.get_input_details()
    _, height, width, _ = input_details[0]['shape']
    target_dtype = input_details[0]['dtype']
    img_rgb = image.convert("RGB")
    img_resized = img_rgb.resize((width, height), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized).astype(target_dtype)
    if np.issubdtype(target_dtype, np.floating):
        img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# --- 5. واجهة الفحص ---
st.markdown(f"<div dir='{t['dir']}'>", unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align: center; color: #0d47a1;'>{t['title']}</h1>", unsafe_allow_html=True)
st.markdown(f'<div class="note-box">{t["note"]}</div>', unsafe_allow_html=True)

choice = st.radio("", (t['upload'], t['camera']), horizontal=True)
file = st.file_uploader(t['upload'], type=["jpg", "png", "jpeg"]) if choice == t['upload'] else st.camera_input(t['camera'])

if file:
    img = Image.open(file)
    image_container = st.empty()
    image_container.image(img, use_container_width=True)
    
    if st.button(t['analyze'], use_container_width=True):
        if interpreter:
            # تفعيل تأثير الليزر أثناء التحليل
            image_container.markdown('<div class="laser-scan">', unsafe_allow_html=True)
            image_container.image(img, use_container_width=True)
            image_container.markdown('</div>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing..."):
                try:
                    final_input = prepare_image_dynamic(img, interpreter)
                    in_idx = interpreter.get_input_details()[0]['index']
                    interpreter.set_tensor(in_idx, final_input)
                    interpreter.invoke()
                    
                    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
                    idx = np.argmax(output) 
                    confidence = output[idx]

                    # مجموعات الـ idx الموزعة ديناميكياً
                    malignant_indices = {1, 4, 6, 8} 
                    benign_indices = {0, 2, 5, 7}

                    if idx in malignant_indices and confidence > 0.4:
                        res_msg, color = t['res_m'], "#cf1322"
                    elif idx in benign_indices and confidence > 0.4:
                        res_msg, color = t['res_b'], "#389e0d"
                    else:
                        res_msg, color = t['res_g'], "#096dd9"

                    st.markdown(f'<div class="report-card" style="border-color: {color}; color: {color};"><h2>{res_msg}</h2><p>{t["advice"]}</p></div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")

st.write("---")

# --- 6. الدليل الطبي الشامل (16 نوعاً) ---
with st.expander(f"📖 {t['guide']}"):
    m_tab, b_tab = st.tabs([t['malig'], t['benign']])
    with m_tab:
        m_diseases = [
            ("Melanoma", "سرطان الخلايا الصبغية.", "طفرات في الميلانين بسبب الأشعة.", "تغير لون وحجم الشامات."),
            ("Basal Cell Carcinoma", "سرطان الخلايا القاعدية.", "التعرض المزمن للشمس.", "نتوء لؤلؤي أو قشرة لا تشفى."),
            ("Squamous Cell Carcinoma", "سرطان الخلايا الحرشفية.", "تضرر الـ DNA في الطبقة السطحية.", "كتلة حمراء صلبة متقشرة."),
            ("Merkel Cell Carcinoma", "سرطان خلايا ميركل النادر.", "فيروس ميركل أو ضعف المناعة.", "نتوءات غير مؤلمة سريعة النمو."),
            ("Kaposi Sarcoma", "ساركوما كابوزي.", "فيروس HHV-8.", "بقع أرجوانية على الجلد أو المخاط."),
            ("Sebaceous Carcinoma", "سرطان الغدد الدهنية.", "نمو غير طبيعي في غدد الجفون.", "نتوء صلب يشبه شحاذ العين."),
            ("Dermatofibrosarcoma", "ساركوما جلدية ليفية.", "طفرة جينية في الأنسجة العميقة.", "ندبة صلبة تنمو ببطء."),
            ("Cutaneous Lymphoma", "ليمفوما جلدية.", "تكاثر خلايا T المناعية بالجلد.", "بقع تشبه الإكزيما لا تستجيب للعلاج.")
        ]
        for n, m, h, s in m_diseases:
            st.markdown(f'<div class="disease-card"><span style="color:#cf1322; font-weight:bold;">🔴 {n}</span><br><b>الوصف:</b> {m}<br><b>التكوين:</b> {h}<br><b>الأعراض:</b> {s}</div>', unsafe_allow_html=True)
    with b_tab:
        b_diseases = [
            ("Nevi", "الشامات الطبيعية.", "تجمع الخلايا الصبغية.", "بقع بنية منتظمة ومستقرة."),
            ("Lipoma", "الورم الشحمي.", "تجمع خلايا دهنية حميدة.", "كتلة لينة تتحرك تحت الجلد."),
            ("Seborrheic Keratosis", "التَقَرُّن المثي.", "تكاثر خلايا الكيراتين.", "زوائد شمعية بنية تشبه الملصقات."),
            ("Hemangioma", "الورم الوعائي.", "تكاثر الأوعية الدموية.", "بقع حمراء أو نتوءات دموية."),
            ("Dermatofibroma", "الألياف الجلدية.", "رد فعل لقرصة حشرة أو جرح.", "عقدة صغيرة صلبة بنية اللون."),
            ("Skin Cyst", "الأكياس الجلدية.", "انسداد المسام أو الغدد.", "نتوء يحتوي على مادة كيراتينية."),
            ("Skin Tags", "الزوائد الجلدية.", "احتكاك الجلد أو الوراثة.", "قطع جلدية صغيرة متدلية."),
            ("Angiokeratoma", "التقرن الوعائي.", "توسع الشعيرات الدموية سطحياً.", "نقاط حمراء صلبة صغيرة جداً.")
        ]
        for n, m, h, s in b_diseases:
            st.markdown(f'<div class="disease-card" style="border-right-color:#389e0d;"><span style="color:#389e0d; font-weight:bold;">🟢 {n}</span><br><b>الوصف:</b> {m}<br><b>التكوين:</b> {h}<br><b>الأعراض:</b> {s}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
