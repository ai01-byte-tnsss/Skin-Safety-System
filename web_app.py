import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Global Skin Guard AI", layout="wide")

# --- 2. القاموس اللغوي الشامل (25 لغة) ---
LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "🛡️ نظام الكشف عن سلامة الجلد", "note": "⚠️ للحصول على أدق النتائج، يرجى التصوير في ضوء طبيعي جيد والتركيز على المنطقة المصابة فقط.", "upload": "📥 ارفع صورة الفحص", "camera": "📸 صورة فورية", "analyze": "🚀 بدء التحليل", "guide": "📚 الدليل الطبي الشامل", "malig": "🔴 الأورام الخبيثة", "benign": "🟢 الأورام الحميدة", "res_m": "🚨 اشتباه ورم خبيث", "res_b": "🔍 ورم حميد", "res_g": "🩺 حالة عامة", "advice": "يرجى مراجعة المختص لضمان السلامة."},
    "English": {"dir": "ltr", "title": "🛡️ Skin Safety AI", "note": "⚠️ For best results, use good natural lighting and focus only on the affected area.", "upload": "📥 Upload Image", "camera": "📸 Camera", "analyze": "🚀 Analyze", "guide": "📚 Medical Guide", "malig": "🔴 Malignant", "benign": "🟢 Benign", "res_m": "🚨 Malignant Suspect", "res_b": "🔍 Benign", "res_g": "🩺 General", "advice": "Please consult a specialist."},
    "Français": {"dir": "ltr", "title": "🛡️ IA de Sécurité Cutanée", "note": "⚠️ Pour de meilleurs résultats, utilisez un bon éclairage naturel.", "upload": "📥 Charger", "camera": "📸 Caméra", "analyze": "🚀 Analyser", "guide": "📚 Guide Médical", "malig": "🔴 Malin", "benign": "🟢 Bénin", "res_m": "🚨 Suspect Malin", "res_b": "🔍 Bénin", "res_g": "🩺 État Général", "advice": "Consultez un médecin."},
    "Türkçe": {"dir": "ltr", "title": "🛡️ Cilt Güvenliği AI", "note": "⚠️ En iyi sonuçlar için doğal ışık kullanın.", "upload": "📥 Yükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz Et", "guide": "📚 Tıbbi Rehber", "malig": "🔴 Kötü Huylu", "benign": "🟢 İyi Huylu", "res_m": "🚨 Kötü Huylu Şübhesi", "res_b": "🔍 İyi Huylu", "res_g": "🩺 Genel Durum", "advice": "Doktora danışın."},
    "Español": {"dir": "ltr", "title": "🛡️ IA Seguridad Cutánea", "note": "⚠️ Para mejores resultados, use luz natural clara.", "upload": "📥 Subir", "camera": "📸 Cámara", "analyze": "🚀 Analizar", "guide": "📚 Guía Médica", "malig": "🔴 Maligno", "benign": "🟢 Benigno", "res_m": "🚨 Sospecha Maligna", "res_b": "🔍 Benigno", "res_g": "🩺 General", "advice": "Consulte a un médico."},
    "Deutsch": {"dir": "ltr", "title": "🛡️ Hautsicherheits-KI", "note": "⚠️ Für beste Ergebnisse bei natürlichem Licht fotografieren.", "upload": "📥 Hochladen", "camera": "📸 Kamera", "analyze": "🚀 Analysieren", "guide": "📚 Med. Leitfaden", "malig": "🔴 Bösartig", "benign": "🟢 Gutartig", "res_m": "🚨 Krebsverdacht", "res_b": "🔍 Gutartig", "res_g": "🩺 Allgemein", "advice": "Arzt aufsuchen."},
    "Русский": {"dir": "ltr", "title": "🛡️ ИИ Кожи", "note": "⚠️ Для лучших результатов используйте естественный свет.", "upload": "📥 Загрузить", "camera": "📸 Камера", "analyze": "🚀 Анализ", "guide": "📚 Справочник", "malig": "🔴 Злокачественные", "benign": "🟢 Доброкачественные", "res_m": "🚨 Подозрение на рак", "res_b": "🔍 Доброкачественное", "res_g": "🩺 Общее", "advice": "Обратитесь к врачу."},
    "中文": {"dir": "ltr", "title": "🛡️ 皮肤安全AI", "note": "⚠️ 为了获得最佳效果，请使用良好的自然光。", "upload": "📥 上传", "camera": "📸 相机", "analyze": "🚀 分析", "guide": "📚 指南", "malig": "🔴 恶性", "benign": "🟢 良性", "res_m": "🚨 疑似恶性", "res_b": "🔍 良性", "res_g": "🩺 一般", "advice": "请咨询医生。"},
    "हिन्दी": {"dir": "ltr", "title": "🛡️ त्वचा सुरक्षा एआई", "note": "⚠️ सर्वोत्तम परिणामों के लिए, अच्छी प्राकृतिक रोशनी का उपयोग करें।", "upload": "📥 अपलोड", "camera": "📸 कैमरा", "analyze": "🚀 विश्लेषण", "guide": "📚 गाइड", "malig": "🔴 घातक", "benign": "🟢 सौम्य", "res_m": "🚨 घातक संदेह", "res_b": "🔍 सौम्य", "res_g": "🩺 सामान्य", "advice": "डॉक्टर से मिलें।"},
    "日本語": {"dir": "ltr", "title": "🛡️ 皮膚安全AI", "note": "⚠️ 最良の結果を得るには、自然光を使用してください。", "upload": "📥 アップロード", "camera": "📸 カメラ", "analyze": "🚀 解析", "guide": "📚 ガイド", "malig": "🔴 悪性", "benign": "🟢 良性", "res_m": "🚨 悪性の疑い", "res_b": "🔍 良性", "res_g": "🩺 一般", "advice": "医師に相談。"},
    "한국어": {"dir": "ltr", "title": "🛡️ 피부 안전 AI", "note": "⚠️ 최상의 결과를 얻으려면 밝은 자연광에서 촬영하세요.", "upload": "📥 업로드", "camera": "📸 카메라", "analyze": "🚀 분석", "guide": "📚 가이드", "malig": "🔴 악성", "benign": "🟢 양성", "res_m": "🚨 악성 의심", "res_b": "🔍 양성", "res_g": "🩺 일반", "advice": "전문가 상담."},
    "Português": {"dir": "ltr", "title": "🛡️ IA Pele", "note": "⚠️ Para melhores resultados, use luz natural.", "upload": "📥 Enviar", "camera": "📸 Câmera", "analyze": "🚀 Analisar", "guide": "📚 Guia", "malig": "🔴 Maligno", "benign": "🟢 Benigno", "res_m": "🚨 Suspeita", "res_b": "🔍 Benigno", "res_g": "🩺 Geral", "advice": "Consulte médico."},
    "Italiano": {"dir": "ltr", "title": "🛡️ IA Pelle", "note": "⚠️ Per risultati ottimali, usa una buona luce naturale.", "upload": "📥 Carica", "camera": "📸 Camera", "analyze": "🚀 Analizza", "guide": "📚 Guida", "malig": "🔴 Maligno", "benign": "🟢 Benigno", "res_m": "🚨 Sospetto", "res_b": "🔍 Benigno", "res_g": "🩺 Generale", "advice": "Consulta medico."},
    "Tiếng Việt": {"dir": "ltr", "title": "🛡️ AI Da Liễu", "note": "⚠️ Để có kết quả tốt nhất, hãy sử dụng ánh sáng tự nhiên.", "upload": "📥 Tải lên", "camera": "📸 Máy ảnh", "analyze": "🚀 Phân tích", "guide": "📚 Hướng dẫn", "malig": "🔴 Ác tính", "benign": "🟢 Lành tính", "res_m": "🚨 Nghi ngờ", "res_b": "🔍 Lành tính", "res_g": "🩺 Tổng quát", "advice": "Hỏi ý kiến bác sĩ."},
    "فارسی": {"dir": "rtl", "title": "🛡️ هوش مصنوعی پوست", "note": "⚠️ برای بهترین نتیجه، از نور طبیعی خوب استفاده کنید.", "upload": "📥 بارگذاری", "camera": "📸 دوربین", "analyze": "🚀 آنالیز", "guide": "📚 راهنما", "malig": "🔴 بدخیم", "benign": "🟢 خوش‌خیم", "res_m": "🚨 مشکوک", "res_b": "🔍 خوش‌خیم", "res_g": "🩺 عمومی", "advice": "به پزشک بروید."},
    "اردو": {"dir": "rtl", "title": "🛡️ جلد کی حفاظت AI", "note": "⚠️ بہترین نتائج کے لیے قدرتی روشنی استعمال کریں۔", "upload": "📥 اپلوڈ", "camera": "📸 کیمرہ", "analyze": "🚀 تجزیہ", "guide": "📚 گائیڈ", "malig": "🔴 خطرناک", "benign": "🟢 بے ضرر", "res_m": "🚨 خطرناک شبہ", "res_b": "🔍 بے ضرر", "res_g": "🩺 عام", "advice": "ڈاکٹر سے مشورہ۔"},
    "کوردی": {"dir": "rtl", "title": "🛡️ پشکنینی پێست", "note": "⚠️ بۆ باشترین ئەنجام، تیشکی سروشتی بەکاربهێنە.", "upload": "📥 وێنە", "camera": "📸 کامێرا", "analyze": "🚀 شیکاری", "guide": "📚 ڕێبەر", "malig": "🔴 خراپ", "benign": "🟢 بێ زیان", "res_m": "🚨 گومانی خراپ", "res_b": "🔍 بێ زیان", "res_g": "🩺 گشتی", "advice": "سەردانی پزیشک."},
    "Kiswahili": {"dir": "ltr", "title": "🛡️ AI ya Ngozi", "note": "⚠️ Kwa matokeo bora, tumia mwanga wa asili.", "upload": "📥 Pakia", "camera": "📸 Kamera", "analyze": "🚀 Uchambuzi", "guide": "📚 Mwongozo", "malig": "🔴 Saratani", "benign": "🟢 Sio Saratani", "res_m": "🚨 Shaka", "res_b": "🔍 Salama", "res_g": "🩺 Hali ya Jumla", "advice": "Ona daktari."},
    "Nederlands": {"dir": "ltr", "title": "🛡️ Huid AI", "note": "⚠️ Gebruik natuurlijk licht voor het beste resultaat.", "upload": "📥 Upload", "camera": "📸 Camera", "analyze": "🚀 Analyseer", "guide": "📚 Gids", "malig": "🔴 Kwaadaardig", "benign": "🟢 Goedaardig", "res_m": "🚨 Verdacht", "res_b": "🔍 Goedaardig", "res_g": "🩺 Algemeen", "advice": "Raadpleeg arts."},
    "Polski": {"dir": "ltr", "title": "🛡️ AI Skóry", "note": "⚠️ Aby uzyskać najlepsze wyniki, użyj światła dziennego.", "upload": "📥 Prześlij", "camera": "📸 Kamera", "analyze": "🚀 Analizuj", "guide": "📚 Przewodnik", "malig": "🔴 Złośliwe", "benign": "🟢 Łagodne", "res_m": "🚨 Podejrzenie", "res_b": "🔍 Łagodne", "res_g": "🩺 Ogólne", "advice": "Skonsultuj lekarza."},
    "ไทย": {"dir": "ltr", "title": "🛡️ AI ตรวจผิว", "note": "⚠️ เพื่อผลลัพธ์ที่ดีที่สุด ควรใช้แสงธรรมชาติ", "upload": "📥 อัปโหลด", "camera": "📸 กล้อง", "analyze": "🚀 วิเคราะห์", "guide": "📚 คู่มือ", "malig": "🔴 เนื้อร้าย", "benign": "🟢 เนื้อดี", "res_m": "🚨 สงสัยเนื้อร้าย", "res_b": "🔍 เนื้อดี", "res_g": "🩺 ทั่วไป", "advice": "ปรึกษาแพทย์."},
    "বাংলা": {"dir": "ltr", "title": "🛡️ স্কিন AI", "note": "⚠️ সেরা ফলাফলের জন্য প্রাকৃতিক আলো ব্যবহার করুন।", "upload": "📥 আপলোড", "camera": "📸 ক্যামেরা", "analyze": "🚀 বিশ্লেষণ", "guide": "📚 নির্দেশিকা", "malig": "🔴 মারাত্মক", "benign": "🟢 সৌম্য", "res_m": "🚨 সন্দেহজনক", "res_b": "🔍 সৌম্য", "res_g": "🩺 সাধারণ", "advice": "পরামর্শ নিন।"},
    "Română": {"dir": "ltr", "title": "🛡️ AI Piele", "note": "⚠️ Pentru rezultate optime, folosiți lumină naturală.", "upload": "📥 Încarcă", "camera": "📸 Cameră", "analyze": "🚀 Analizează", "guide": "📚 Ghid", "malig": "🔴 Malign", "benign": "🟢 Benign", "res_m": "🚨 Suspiciune", "res_b": "🔍 Benign", "res_g": "🩺 General", "advice": "Consultă medic."},
    "Türkmençe": {"dir": "ltr", "title": "🛡️ Deri AI", "note": "⚠️ Iň gowy netijeler üçin tebigy yşyk ulanyň.", "upload": "📥 Ýükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz", "guide": "📚 Gollanma", "malig": "🔴 Howply", "benign": "🟢 Howpsuz", "res_m": "🚨 Howply şüphe", "res_b": "🔍 Howpsuz", "res_g": "🩺 Umumy", "advice": "Lukmana ýüz tutuň."},
    "Bahasa Indonesia": {"dir": "ltr", "title": "🛡️ AI Kulit", "note": "⚠️ Untuk hasil terbaik, gunakan pencahayaan alami.", "upload": "📥 Unggah", "camera": "📸 Kamera", "analyze": "🚀 Analisis", "guide": "📚 Panduan", "malig": "🔴 Ganas", "benign": "🟢 Jinak", "res_m": "🚨 Kecurigaan", "res_b": "🔍 Jinak", "res_g": "🩺 Umum", "advice": "Konsultasi dokter."}
}

# --- 3. التنسيق البصري (ثابت) ---
selected_lang = st.sidebar.selectbox("🌐 Choose Language / اختر اللغة", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

st.markdown(f"""
<style>
    div[dir='{t['dir']}'] {{ text-align: {'right' if t['dir']=='rtl' else 'left'}; }}
    .report-card {{ padding: 25px; border-radius: 15px; text-align: center; border: 5px solid; margin-top: 20px; }}
    .disease-card {{ border-right: 5px solid #0d47a1; border-left: 1px solid #eee; padding: 15px; background: #f9f9f9; margin-bottom: 10px; border-radius: 10px; }}
    .disease-title {{ font-weight: bold; font-size: 1.1em; margin-bottom: 5px; display: block; }}
    .note-box {{ background: #fffbe6; border: 1px solid #ffe58f; padding: 10px; border-radius: 8px; margin-bottom: 20px; color: #856404; font-size: 0.9em; }}
</style>
""", unsafe_allow_html=True)

# --- 4. المحرك ومعالجة الصور التلقائية (حل FLOAT) ---
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
    img_resized = img_rgb.resize((width, height))
    img_array = np.array(img_resized).astype(target_dtype)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# --- 5. واجهة الاستدلال ---
st.markdown(f"<div dir='{t['dir']}'>", unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align: center; color: #0d47a1;'>{t['title']}</h1>", unsafe_allow_html=True)
st.markdown(f'<div class="note-box">{t["note"]}</div>', unsafe_allow_html=True)

choice = st.radio("", (t['upload'], t['camera']))
file = st.file_uploader(t['upload'], type=["jpg", "png", "jpeg"]) if choice == t['upload'] else st.camera_input(t['camera'])

if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)
    
    if st.button(t['analyze']):
        if interpreter:
            with st.spinner("Analyzing..."):
                try:
                    final_input = prepare_image_dynamic(img, interpreter)
                    in_idx = interpreter.get_input_details()[0]['index']
                    interpreter.set_tensor(in_idx, final_input)
                    interpreter.invoke()
                    
                    out_idx = interpreter.get_output_details()[0]['index']
                    output = interpreter.get_tensor(out_idx)[0]
                    
                    idx = np.argmax(output)

                    # منطق التصنيف الصحيح
                    if idx in [1, 4, 17]:
                        res_msg, color = t['res_m'], "#cf1322"
                    elif idx in [2, 5, 23]:
                        res_msg, color = t['res_b'], "#389e0d"
                    else:
                        res_msg, color = t['res_g'], "#096dd9"

                    # تم إلغاء النسبة المئوية هنا
                    st.markdown(f'<div class="report-card" style="border-color: {color}; color: {color};"><h2>{res_msg}</h2><p>{t["advice"]}</p></div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")

st.write("---")

# --- 6. الدليل الطبي الكامل (16 نوعاً) ---
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
            st.markdown(f'<div class="disease-card"><span class="disease-title">🔴 {n}</span><b>الوصف:</b> {m}<br><b>التكوين:</b> {h}<br><b>الأعراض:</b> {s}</div>', unsafe_allow_html=True)

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
            st.markdown(f'<div class="disease-card" style="border-right-color:#389e0d;"><span class="disease-title">🟢 {n}</span><b>الوصف:</b> {m}<br><b>التكوين:</b> {h}<br><b>الأعراض:</b> {s}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
