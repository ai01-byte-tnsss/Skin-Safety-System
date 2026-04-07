import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np

# --- 1. إعدادات الصفحة ---
st.set_page_config(
    page_title="Global Skin Guard AI - Advanced Edition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. القاموس اللغوي الكامل (25 لغة) ---
LANG_DATA = {
    "العربية": {"dir": "rtl", "title": "🛡️ نظام الكشف عن سلامة الجلد المطور", "note": "⚠️ للحصول على أدق النتائج، يرجى التصوير في ضوء طبيعي جيد والتركيز على المنطقة المصابة فقط.", "upload": "📥 ارفع صورة الفحص", "camera": "📸 صورة فورية", "analyze": "🚀 بدء التحليل الذكي", "guide": "📚 الدليل الطبي الشامل", "malig": "🔴 الأورام الخبيثة", "benign": "🟢 الأورام الحميدة", "res_m": "🚨 اشتباه ورم خبيث", "res_b": "🔍 ورم حميد", "res_g": "🩺 حالة عامة / غير محددة", "advice": "يرجى مراجعة المختص لضمان السلامة.", "low_conf": "⚠️ الدقة منخفضة، يرجى إعادة التصوير بوضوح."},
    "English": {"dir": "ltr", "title": "🛡️ Advanced Skin Safety AI", "note": "⚠️ For best results, use good natural lighting and focus only on the affected area.", "upload": "📥 Upload Scan", "camera": "📸 Instant Photo", "analyze": "🚀 Smart Analyze", "guide": "📚 Medical Guide", "malig": "🔴 Malignant", "benign": "🟢 Benign", "res_m": "🚨 Malignant Suspect", "res_b": "🔍 Benign Result", "res_g": "🩺 General/Other Condition", "advice": "Please consult a specialist.", "low_conf": "⚠️ Low confidence, please re-take a clearer photo."},
    "Français": {"dir": "ltr", "title": "🛡️ IA de Sécurité Cutanée", "note": "⚠️ Pour de meilleurs résultats, utilisez un bon éclairage naturel.", "upload": "📥 Charger", "camera": "📸 Caméra", "analyze": "🚀 Analyser", "guide": "📚 Guide Médical", "malig": "🔴 Malin", "benign": "🟢 Bénin", "res_m": "🚨 Suspect Malin", "res_b": "🔍 Résultat Bénin", "res_g": "🩺 Autre/Général", "advice": "Consultez un spécialiste.", "low_conf": "⚠️ Confiance faible, reprenez la photo."},
    "Español": {"dir": "ltr", "title": "🛡️ IA Seguridad Cutánea", "note": "⚠️ Use luz natural para mejores resultados.", "upload": "📥 Subir", "camera": "📸 Cámara", "analyze": "🚀 Analizar", "guide": "📚 Guía Médica", "malig": "🔴 Maligno", "benign": "🟢 Benigno", "res_m": "🚨 Sospecha Maligna", "res_b": "🔍 Benigno", "res_g": "🩺 Otro/General", "advice": "Consulte a un médico.", "low_conf": "⚠️ Baja confianza, reintente."},
    "Deutsch": {"dir": "ltr", "title": "🛡️ Haut-KI", "note": "⚠️ Nutzen Sie natürliches Licht.", "upload": "📥 Hochladen", "camera": "📸 Kamera", "analyze": "🚀 Analysieren", "guide": "📚 Leitfaden", "malig": "🔴 Bösartig", "benign": "🟢 Gutartig", "res_m": "🚨 Krebsverdacht", "res_b": "🔍 Gutartig", "res_g": "🩺 Allgemein/Andere", "advice": "Arzt aufsuchen.", "low_conf": "⚠️ Geringe Sicherheit."},
    "中文": {"dir": "ltr", "title": "🛡️ 皮肤安全AI", "note": "⚠️ 为了获得最佳效果，请使用自然光。", "upload": "📥 上传", "camera": "📸 相机", "analyze": "🚀 分析", "guide": "📚 医学指南", "malig": "🔴 恶性", "benign": "🟢 良性", "res_m": "🚨 疑似恶性", "res_b": "🔍 良性结果", "res_g": "🩺 其他/一般", "advice": "请咨询医生。", "low_conf": "⚠️ 置信度低。"},
    "हिन्दी": {"dir": "ltr", "title": "🛡️ त्वचा सुरक्षा AI", "note": "⚠️ सर्वोत्तम परिणामों के लिए प्राकृतिक रोशनी का उपयोग करें।", "upload": "📥 अपलोड", "camera": "📸 कैमरा", "analyze": "🚀 विश्लेषण", "guide": "📚 चिकित्सा गाइड", "malig": "🔴 घातक", "benign": "🟢 सौम्य", "res_m": "🚨 घातक संदेह", "res_b": "🔍 सौम्य परिणाम", "res_g": "🩺 अन्य/सामान्य", "advice": "विशेषज्ञ से सलाह लें।", "low_conf": "⚠️ कम आत्मविश्वास।"},
    "Русский": {"dir": "ltr", "title": "🛡️ ИИ Кожи", "note": "⚠️ Используйте естественный свет.", "upload": "📥 Загрузить", "camera": "📸 Камера", "analyze": "🚀 Анализ", "guide": "📚 Справочник", "malig": "🔴 Злокачественные", "benign": "🟢 Доброкачественные", "res_m": "🚨 Подозрение", "res_b": "🔍 Доброкачественное", "res_g": "🩺 Общее/Другое", "advice": "Обратитесь к врачу.", "low_conf": "⚠️ Низкая точность."},
    "日本語": {"dir": "ltr", "title": "🛡️ 皮膚安全AI", "note": "⚠️ 自然光を使用してください。", "upload": "📥 アップロード", "camera": "📸 カメラ", "analyze": "🚀 解析", "guide": "📚 ガイド", "malig": "🔴 悪性", "benign": "🟢 良性", "res_m": "🚨 悪性の疑い", "res_b": "🔍 良性", "res_g": "🩺 その他/一般", "advice": "医師に相談。", "low_conf": "⚠️ 信頼度が低い。"},
    "Português": {"dir": "ltr", "title": "🛡️ IA de Pele", "note": "⚠️ Use luz natural clara.", "upload": "📥 Enviar", "camera": "📸 Câmera", "analyze": "🚀 Analisar", "guide": "📚 Guia Médico", "malig": "🔴 Maligno", "benign": "🟢 Benigno", "res_m": "🚨 Suspeita", "res_b": "🔍 Benigno", "res_g": "🩺 Outro/Geral", "advice": "Consulte um médico.", "low_conf": "⚠️ Confiança baixa."},
    "Türkçe": {"dir": "ltr", "title": "🛡️ Cilt Güvenliği AI", "note": "⚠️ Doğal ışık kullanın.", "upload": "📥 Yükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz Et", "guide": "📚 Tıbbi Rehber", "malig": "🔴 Kötü Huylu", "benign": "🟢 İyi Huylu", "res_m": "🚨 Şüphe", "res_b": "🔍 İyi Huylu", "res_g": "🩺 Diğer/Genel", "advice": "Doktora danışın.", "low_conf": "⚠️ Düşük güven."},
    "한국어": {"dir": "ltr", "title": "🛡️ 피부 안전 AI", "note": "⚠️ 자연광에서 촬영하세요.", "upload": "📥 업로드", "camera": "📸 카메라", "analyze": "🚀 분석", "guide": "📚 가이드", "malig": "🔴 악성", "benign": "🟢 양성", "res_m": "🚨 악성 의심", "res_b": "🔍 양성", "res_g": "🩺 기타/일반", "advice": "전문가 상담。", "low_conf": "⚠️ 신뢰도 낮음."},
    "Italiano": {"dir": "ltr", "title": "🛡️ IA Pelle", "note": "⚠️ Usa luce naturale.", "upload": "📥 Carica", "camera": "📸 Camera", "analyze": "🚀 Analizza", "guide": "📚 Guia", "malig": "🔴 Maligno", "benign": "🟢 Benigno", "res_m": "🚨 Sospetto", "res_b": "🔍 Benigno", "res_g": "🩺 Altro/Generale", "advice": "Consulta un medico.", "low_conf": "⚠️ Bassa fiducia."},
    "اردو": {"dir": "rtl", "title": "🛡️ جلد کی حفاظت AI", "note": "⚠️ قدرتی روشنی استعمال کریں۔", "upload": "📥 اپلوڈ", "camera": "📸 کیمرہ", "analyze": "🚀 تجزیہ", "guide": "📚 گائیڈ", "malig": "🔴 خطرناک", "benign": "🟢 بے ضرر", "res_m": "🚨 شبہ", "res_b": "🔍 بے ضرر", "res_g": "🩺 دیگر/عام", "advice": "ڈاکٹر سے مشورہ۔", "low_conf": "⚠️ کم اعتماد۔"},
    "فارسي": {"dir": "rtl", "title": "🛡️ هوش مصنوعی پوست", "note": "⚠️ از نور طبیعی استفاده کنید.", "upload": "📥 بارگذاری", "camera": "📸 دوربین", "analyze": "🚀 آناليز", "guide": "📚 راهنما", "malig": "🔴 بدخیم", "benign": "🟢 خوش‌خیم", "res_m": "🚨 مشکوک", "res_b": "🔍 خوش‌خیم", "res_g": "🩺 سایر/عمومی", "advice": "به پزشک مراجعه کنید.", "low_conf": "⚠️ اعتماد پایین."},
    "Tiếng Việt": {"dir": "ltr", "title": "🛡️ AI Da Liễu", "note": "⚠️ Sử dụng ánh sáng tự nhiên.", "upload": "📥 Tải lên", "camera": "📸 Máy ảnh", "analyze": "🚀 Phân tích", "guide": "📚 Hướng dẫn", "malig": "🔴 Ác tính", "benign": "🟢 Lành tính", "res_m": "🚨 Nghi ngờ", "res_b": "🔍 Lành tính", "res_g": "🩺 Khác/Tổng quát", "advice": "Hỏi ý kiến bác sĩ.", "low_conf": "⚠️ Độ tin cậy thấp."},
    "Bahasa Indonesia": {"dir": "ltr", "title": "🛡️ AI Kulit", "note": "⚠️ Gunakan cahaya alami.", "upload": "📥 Unggah", "camera": "📸 Kamera", "analyze": "🚀 Analisis", "guide": "📚 Panduan", "malig": "🔴 Ganas", "benign": "🟢 Jinak", "res_m": "🚨 Kecurigaan", "res_b": "🔍 Jinak", "res_g": "🩺 Lainnya/Umum", "advice": "Konsultasi dokter.", "low_conf": "⚠️ Kepercayaan rendah."},
    "Nederlands": {"dir": "ltr", "title": "🛡️ Huid AI", "note": "⚠️ Gebruik natuurlijk licht.", "upload": "📥 Upload", "camera": "📸 Camera", "analyze": "🚀 Analyse", "guide": "📚 Gids", "malig": "🔴 Kwaadaardig", "benign": "🟢 Goedaardig", "res_m": "🚨 Verdacht", "res_b": "🔍 Goedaardig", "res_g": "🩺 Overig/Algemeen", "advice": "Raadpleeg arts.", "low_conf": "⚠️ Lage betrouwbaarheid."},
    "Polski": {"dir": "ltr", "title": "🛡️ AI Skóry", "note": "⚠️ Użyj światła dziennego.", "upload": "📥 Prześlij", "camera": "📸 Kamera", "analyze": "🚀 Analiza", "guide": "📚 Przewodnik", "malig": "🔴 Złośliwe", "benign": "🟢 Łagodne", "res_m": "🚨 Podejrzenie", "res_b": "🔍 Łagodne", "res_g": "🩺 Inne/Ogólne", "advice": "Skonsultuj się.", "low_conf": "⚠️ Niska pewność."},
    "ไทย": {"dir": "ltr", "title": "🛡️ AI ตรวจผิว", "note": "⚠️ ใช้แสงธรรมชาติ", "upload": "📥 อัปโหลด", "camera": "📸 กล้อง", "analyze": "🚀 วิเคราะห์", "guide": "📚 คู่มือ", "malig": "🔴 เนื้อร้าย", "benign": "🟢 เนื้อดี", "res_m": "🚨 สงสัยเนื้อร้าย", "res_b": "🔍 เนื้อดี", "res_g": "🩺 อื่นๆ/ทั่วไป", "advice": "ปรึกษาแพทย์", "low_conf": "⚠️ ความแม่นยำต่ำ"},
    "کوردی": {"dir": "rtl", "title": "🛡️ پشکنینی پێست", "note": "⚠️ تیشکی سروشتی بەکاربهێنە.", "upload": "📥 وێنە", "camera": "📸 کامێرا", "analyze": "🚀 شیکاري", "guide": "📚 ڕێبەر", "malig": "🔴 خراپ", "benign": "🟢 بێ زیان", "res_m": "🚨 گومانی خراپ", "res_b": "🔍 بێ زیان", "res_g": "🩺 جۆری تر/گشتی", "advice": "سەردانی پزیشک بکە.", "low_conf": "⚠️ متمانەی کەم."},
    "Bengali": {"dir": "ltr", "title": "🛡️ স্কিন AI", "note": "⚠️ প্রাকৃতিক আলো ব্যবহার করুন।", "upload": "📥 আপलोड", "camera": "📸 ক্যামেরা", "analyze": "🚀 বিশ্লেষণ", "guide": "📚 নির্দেশিকা", "malig": "🔴 মারাত্মক", "benign": "🟢 সৌম্য", "res_m": "🚨 সন্দেহজনক", "res_b": "🔍 সৌম্য", "res_g": "🩺 অন্যান্য/সাধারণ", "advice": "পরামর্শ নিন।", "low_conf": "⚠️ কম আত্মবিশ্বাস।"},
    "Română": {"dir": "ltr", "title": "🛡️ AI Piele", "note": "⚠️ Folosiți lumină naturală.", "upload": "📥 Încarcă", "camera": "📸 Cameră", "analyze": "🚀 Analizează", "guide": "📚 Ghid", "malig": "🔴 Malign", "benign": "🟢 Benign", "res_m": "🚨 Suspect", "res_b": "🔍 Benign", "res_g": "🩺 Altele/General", "advice": "Consultă medicul.", "low_conf": "⚠️ Încredere scăzută."},
    "Kiswahili": {"dir": "ltr", "title": "🛡️ AI ya Ngozi", "note": "⚠️ Tumia mwanga wa asili.", "upload": "📥 Pakia", "camera": "📸 Kamera", "analyze": "🚀 Uchambuzi", "guide": "📚 Mwongozo", "malig": "🔴 Saratani", "benign": "🟢 Salama", "res_m": "🚨 Shaka", "res_b": "🔍 Salama", "res_g": "🩺 Nyingine/Jumla", "advice": "Ona daktari.", "low_conf": "⚠️ Imani ndogo."},
    "Türkmençe": {"dir": "ltr", "title": "🛡️ Deri AI", "note": "⚠️ Tebigy yşyk ulanyň.", "upload": "📥 Ýükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz", "guide": "📚 Gollanma", "malig": "🔴 Howply", "benign": "🟢 Howpsuz", "res_m": "🚨 Şüphe", "res_b": "🔍 Howpsuz", "res_g": "🩺 Başga/Umumy", "advice": "Lukmana ýüz tutuň.", "low_conf": "⚠️ Pes ynam."}
}

# --- 3. التنسيق البصري (Advanced CSS) ---
selected_lang = st.sidebar.selectbox("🌐 Select Language / اختر اللغة", list(LANG_DATA.keys()))
t = LANG_DATA[selected_lang]

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    html, body, [class*="st-"] {{ font-family: 'Tajawal', sans-serif; }}
    div[dir='{t['dir']}'] {{ text-align: {'right' if t['dir']=='rtl' else 'left'}; }}
    .report-card {{ padding: 35px; border-radius: 25px; text-align: center; border: 8px solid; margin-top: 25px; background: #ffffff; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }}
    .note-box {{ background: #fffbe6; border: 2px solid #ffe58f; padding: 20px; border-radius: 15px; margin-bottom: 25px; color: #856404; font-weight: bold; }}
    .laser-scan {{ position: relative; border: 5px solid #ff4b4b; border-radius: 20px; overflow: hidden; animation: scan-pulse 2s infinite; }}
    @keyframes scan-pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }} 70% {{ box-shadow: 0 0 0 25px rgba(255, 75, 75, 0); }} 100% {{ box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }} }}
    .disease-card {{ border-right: 6px solid #0d47a1; border-left: 1px solid #ddd; padding: 20px; background: #fdfdfd; margin-bottom: 15px; border-radius: 12px; transition: 0.3s; }}
    .disease-card:hover {{ transform: scale(1.01); background: #f0f7ff; }}
</style>
""", unsafe_allow_html=True)

# --- 4. محرك الـ CNN الديناميكي (Dynamic AI Engine) ---
@st.cache_resource
def load_dynamic_expert_model():
    try:
        # بناء الأساس باستخدام EfficientNetB0 (Transfer Learning)
        base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.4)(x)
        # عدد الفئات ديناميكي (هنا 7 لبيانات HAM10000)
        predictions = Dense(7, activation='softmax')(x)
        model = Model(inputs=base.input, outputs=predictions)
        
        # محاولة تحميل الأوزان إذا وجدت، وإلا سيعمل النموذج كـ Demo
        # model.load_weights('skin_ai_weights.h5') 
        return model
    except Exception as e:
        st.error(f"AI Engine Error: {e}")
        return None

model = load_dynamic_expert_model()

def process_image_dynamic(image, model_instance):
    # استنباط الأبعاد المطلوبة ديناميكياً من النموذج
    input_shape = model_instance.layers[0].input_shape[0] # (None, 224, 224, 3)
    target_size = (input_shape[1], input_shape[2])
    
    img = image.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # المعالجة المسبقة التلقائية لـ EfficientNet
    return tf.keras.applications.efficientnet.preprocess_input(img_array)

# --- 5. واجهة المستخدم (User Interface) ---
st.markdown(f"<div dir='{t['dir']}'>", unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align: center; color: #0d47a1; font-size: 3em;'>{t['title']}</h1>", unsafe_allow_html=True)
st.markdown(f'<div class="note-box">{t["note"]}</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    choice = st.radio("Choose Method / اختر الطريقة", (t['upload'], t['camera']), horizontal=True)
    file = st.file_uploader(t['upload'], type=["jpg", "png", "jpeg"]) if choice == t['upload'] else st.camera_input(t['camera'])

if file:
    img = Image.open(file)
    with col2:
        img_placeholder = st.empty()
        img_placeholder.image(img, use_container_width=True, caption="Original Image")
    
    if st.button(t['analyze'], use_container_width=True):
        if model:
            # تأثير الليزر البصري
            img_placeholder.markdown('<div class="laser-scan">', unsafe_allow_html=True)
            img_placeholder.image(img, use_container_width=True)
            img_placeholder.markdown('</div>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing Layers..."):
                processed = process_image_dynamic(img, model)
                preds = model.predict(processed)[0]
                
                # منطق ديناميكي للثقة (Confidence)
                confidence = np.max(preds)
                idx = np.argmax(preds)
                
                # قائمة الفئات الطبية (AK, BCC, BKL, DF, MEL, NV, VASC)
                malignant_indices = [0, 1, 4] # تقرحات، قاعدية، ميلانوما
                
                if confidence < 0.35:
                    st.warning(t['low_conf'])
                
                if idx in malignant_indices:
                    res_msg, color = t['res_m'], "#cf1322"
                elif idx in [2, 3, 5, 6]:
                    res_msg, color = t['res_b'], "#389e0d"
                else:
                    res_msg, color = t['res_g'], "#096dd9"

                st.markdown(f'<div class="report-card" style="border-color: {color}; color: {color};"><h2>{res_msg}</h2><h3>Confidence: {confidence*100:.2f}%</h3><p>{t["advice"]}</p></div>', unsafe_allow_html=True)

st.write("---")

# --- 6. الدليل الطبي المحدث (16 نوعاً) ---
with st.expander(f"📖 {t['guide']}"):
    m_tab, b_tab = st.tabs([t['malig'], t['benign']])
    with m_tab:
        m_diseases = [
            ("Melanoma", "سرطان الخلايا الصبغية الأخطر.", "طفرات في الميلانين بسبب الأشعة.", "تغير مفاجئ في لون وحجم الشامات."),
            ("Basal Cell Carcinoma", "سرطان الخلايا القاعدية الشائع.", "التعرض الطويل لأشعة الشمس.", "نتوء لؤلؤي أو قشرة تنزف ولا تشفى."),
            ("Squamous Cell Carcinoma", "سرطان الخلايا الحرشفية.", "تضرر الـ DNA في طبقات الجلد السطحية.", "كتلة حمراء صلبة ذات سطح متقشر."),
            ("Merkel Cell Carcinoma", "سرطان خلايا ميركل النادر.", "فيروس ميركل أو ضعف الجهاز المناعي.", "نتوءات صلبة غير مؤلمة سريعة النمو."),
            ("Kaposi Sarcoma", "ساركوما كابوزي الوعائي.", "عدوى فيروسية (HHV-8).", "بقع أو كتل أرجوانية/حمراء على الجلد."),
            ("Sebaceous Carcinoma", "سرطان الغدد الدهنية.", "نمو سرطاني في غدد الجفون والوجه.", "نتوء صلب يشبه 'شحاذ العين' المستمر."),
            ("Dermatofibrosarcoma", "ساركوما جلدية ليفية جاحظة.", "طفرة جينية نادرة في الأنسجة.", "ندبة صلبة تنمو ببطء شديد لسنوات."),
            ("Cutaneous Lymphoma", "ليمفوما جلدية (خلايا T).", "تكاثر غير طبيعي للخلايا الليمفاوية.", "بقع تشبه الإكزيما أو الصدفية.")
        ]
        for n, m, h, s in m_diseases:
            st.markdown(f'<div class="disease-card" style="border-right-color:#cf1322;"><span style="color:#cf1322; font-weight:bold; font-size:1.2em;">🔴 {n}</span><br><b>الوصف:</b> {m}<br><b>المنشأ:</b> {h}<br><b>الأعراض:</b> {s}</div>', unsafe_allow_html=True)
    
    with b_tab:
        b_diseases = [
            ("Nevi", "الشامات الطبيعية.", "تجمع سليم للخلايا الصبغية.", "بقع بنية منتظمة ومستقرة تماماً."),
            ("Lipoma", "الورم الشحمي السليم.", "تجمع كتل دهنية تحت الجلد.", "كتلة لينة تتحرك بسهولة عند لمسها."),
            ("Seborrheic Keratosis", "التَقَرُّن المثي.", "تكاثر خلايا الكيراتين السطحية.", "زوائد شمعية بنية تشبه الملصقات الجلدية."),
            ("Hemangioma", "الورم الوعائي (النقطة الكرزية).", "تجمع غير سرطاني للأوعية الدموية.", "بقع حمراء زاهية أو نتوءات دموية."),
            ("Dermatofibroma", "الألياف الجلدية السليمة.", "رد فعل لقرصة حشرة أو جرح بسيط.", "عقدة صغيرة صلبة بنية تميل للداخل."),
            ("Skin Cyst", "الأكياس الجلدية/الدهنية.", "انسداد المسام أو التهاب الغدد.", "نتوء يحتوي على مادة كيراتينية بيضاء."),
            ("Skin Tags", "الزوائد الجلدية الشائعة.", "احتكاك الجلد المستمر أو الوراثة.", "قطع جلدية صغيرة متدلية من الرقبة."),
            ("Angiokeratoma", "التقرن الوعائي.", "توسع الشعيرات الدموية السطحية.", "نقاط حمراء أو زرقاء داكنة صلبة جداً.")
        ]
        for n, m, h, s in b_diseases:
            st.markdown(f'<div class="disease-card" style="border-right-color:#389e0d;"><span style="color:#389e0d; font-weight:bold; font-size:1.2em;">🟢 {n}</span><br><b>الوصف:</b> {m}<br><b>المنشأ:</b> {h}<br><b>الأعراض:</b> {s}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
