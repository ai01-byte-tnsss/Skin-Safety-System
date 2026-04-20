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
    page_title="Skin Safety AI Expert System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. القاموس اللغوي الكامل (20 لغة) ---
LANG_DATA = {
    "العربية": {
        "dir": "rtl", "title": "نظام الفحص والتشخيص الذكي للجلد", "upload": "📥 ارفع صورة الفحص", "camera": "📸 التقاط صورة", "analyze": "🔍 بدء عملية الفحص",
        "advice": "⚠️ تنبيه طبي: هذا النظام هو أداة ذكاء اصطناعي للاسترشاد فقط، وليس بديلاً عن التشخيص الطبي المهني. يرجى استشارة الطبيب فوراً.",
        "guide": "📖 الدليل الطبي الشامل لسرطان الجلد", "links": "🔗 مراجع طبية عالمية موثوقة",
        "invalid_img": "❌ خطأ: هذه الصورة لا تبدو كفحص طبي للجلد (قد تكون سيارة، حيوان، أو لقطة شاشة). يرجى رفع صورة واضحة للجلد فقط.",
        "status_check": "⏳ جاري التحقق من هوية الصورة وفحص الأنسجة...",
        "classes": {
            0: {"name": "Melanoma (خبيث)", "desc": "اشتباه عالي بورم ميلانيني خبيث. يتطلب فحصاً طبياً عاجلاً."},
            1: {"name": "Melanocytic Nevi", "desc": "شامات ميلانينية عادية (تحتاج مراقبة دورية فقط)."},
            2: {"name": "Basal Cell Carcinoma (خبيث)", "desc": "سرطان الخلايا القاعدية. يحتاج لتدخل طبي مختص."},
            3: {"name": "Actinic Keratoses", "desc": "تقران سفعي (آفات ما قبل سرطانية بسبب الشمس)."},
            4: {"name": "Benign Keratosis", "desc": "تقران دهني حميد (زوائد جلدية غير ضارة)."},
            5: {"name": "Dermatofibroma", "desc": "ورم ليفي جلدي حميد."},
            6: {"name": "Vascular Lesions", "desc": "آفات وعائية حميدة (أورام وعائية)."}
        }
    },
    "English": {
        "dir": "ltr", "title": "Smart Skin Diagnostic AI System", "upload": "📥 Upload Scan", "camera": "📸 Capture Photo", "analyze": "🔍 Start Analysis",
        "advice": "⚠️ Medical Note: This AI tool is for guidance only and is NOT a substitute for professional medical advice.",
        "guide": "📖 Full Skin Cancer Guide", "links": "🔗 Trusted Global Medical Links",
        "invalid_img": "❌ Error: Image does not appear to be a skin scan (Car, Animal, or Screen). Please upload a clear skin photo.",
        "status_check": "⏳ Verifying image identity and analyzing tissue...",
        "classes": {
            0: {"name": "Melanoma (Malignant)", "desc": "High suspicion of malignant melanoma. Urgent check required."},
            1: {"name": "Melanocytic Nevi", "desc": "Common moles. Usually safe but monitor for changes."},
            2: {"name": "Basal Cell Carcinoma (Malignant)", "desc": "Common skin cancer. Requires specialist intervention."},
            3: {"name": "Actinic Keratoses", "desc": "Pre-cancerous lesions caused by sun damage."},
            4: {"name": "Benign Keratosis", "desc": "Safe, non-cancerous skin growth."},
            5: {"name": "Dermatofibroma", "desc": "Benign skin fibrous tumor."},
            6: {"name": "Vascular Lesions", "desc": "Benign blood vessel growths."}
        }
    },
    "Français": {"dir": "ltr", "title": "IA Sécurité Cutanée", "upload": "📥 Charger", "camera": "📸 Caméra", "analyze": "🚀 Analyser", "invalid_img": "❌ Image invalide.", "advice": "⚠️ Note: Aide IA uniquement.", "guide": "📖 Guide Médical", "links": "🔗 Liens", "classes": {0: {"name": "Mélanome", "desc": "Suspect Malin"}, 1: {"name": "Nævus", "desc": "Bénin"}, 2: {"name": "BCC", "desc": "Malin"}, 3: {"name": "AK", "desc": "Pré-cancéreux"}, 4: {"name": "BKL", "desc": "Bénin"}, 5: {"name": "DF", "desc": "Bénin"}, 6: {"name": "VASC", "desc": "Bénin"}}},
    "Español": {"dir": "ltr", "title": "IA Seguridad de la Piel", "upload": "📥 Subir", "camera": "📸 Cámara", "analyze": "🚀 Analizar", "invalid_img": "❌ Imagen inválida.", "advice": "⚠️ Nota: Solo para orientación.", "guide": "📖 Guía Médica", "links": "🔗 Enlaces", "classes": {0: {"name": "Melanoma", "desc": "Maligno"}, 1: {"name": "Nevo", "desc": "Benigno"}, 2: {"name": "BCC", "desc": "Maligno"}, 3: {"name": "AK", "desc": "Precanceroso"}, 4: {"name": "BKL", "desc": "Benigno"}, 5: {"name": "DF", "desc": "Benigno"}, 6: {"name": "VASC", "desc": "Benigno"}}},
    "Deutsch": {"dir": "ltr", "title": "KI-Hautschutz", "upload": "📥 Hochladen", "camera": "📸 Kamera", "analyze": "🚀 Analysieren", "invalid_img": "❌ Ungültiges Bild.", "advice": "⚠️ Hinweis: KI-Leitfaden.", "guide": "📖 Leitfaden", "links": "🔗 Links", "classes": {0: {"name": "Melanom", "desc": "Bösartig"}, 1: {"name": "Nävus", "desc": "Gutartig"}, 2: {"name": "BCC", "desc": "Bösartig"}, 3: {"name": "AK", "desc": "Vorkrebs"}, 4: {"name": "BKL", "desc": "Gutartig"}, 5: {"name": "DF", "desc": "Gutartig"}, 6: {"name": "VASC", "desc": "Gutartig"}}},
    "中文": {"dir": "ltr", "title": "皮肤安全AI", "upload": "📥 上传", "camera": "📸 相机", "analyze": "🚀 分析", "invalid_img": "❌ 图像无效。", "advice": "⚠️ 注意：仅供参考。", "guide": "📖 指南", "links": "🔗 链接", "classes": {0: {"name": "黑色素瘤", "desc": "恶性"}, 1: {"name": "痣", "desc": "良性"}, 2: {"name": "基底细胞癌", "desc": "恶性"}, 3: {"name": "日光性角化病", "desc": "癌前"}, 4: {"name": "良性角化病", "desc": "良性"}, 5: {"name": "皮肤纤维瘤", "desc": "良性"}, 6: {"name": "血管病变", "desc": "良性"}}},
    "हिन्दी": {"dir": "ltr", "title": "त्वचा सुरक्षा AI", "upload": "📥 अपलोड", "camera": "📸 कैमरा", "analyze": "🚀 विश्लेषण", "invalid_img": "❌ अमान्य छवि।", "advice": "⚠️ नोट: केवल मार्गदर्शन।", "guide": "📖 गाइड", "links": "🔗 लिंक", "classes": {0: {"name": "मेलानोमा", "desc": "घातक"}, 1: {"name": "नेवस", "desc": "सौम्य"}, 2: {"name": "बीसीसी", "desc": "घातक"}, 3: {"name": "एके", "desc": "कैंसर पूर्व"}, 4: {"name": "बीकेएल", "desc": "सौम्य"}, 5: {"name": "डीएफ", "desc": "सौम्य"}, 6: {"name": "वीएएससी", "desc": "सौम्य"}}},
    "Русский": {"dir": "ltr", "title": "ИИ кожи", "upload": "📥 Загрузить", "camera": "📸 Камера", "analyze": "🚀 Начать", "invalid_img": "❌ Ошибка.", "advice": "⚠️ Примечание: ИИ-помощник.", "guide": "📖 Справочник", "links": "🔗 Ссылки", "classes": {0: {"name": "Меланома", "desc": "Злокачественное"}, 1: {"name": "Невус", "desc": "Доброкачественное"}, 2: {"name": "БЦК", "desc": "Злокачественное"}, 3: {"name": "АК", "desc": "Предрак"}, 4: {"name": "БКЛ", "desc": "Доброкачественное"}, 5: {"name": "ДФ", "desc": "Доброкачественное"}, 6: {"name": "ВАСК", "desc": "Доброкачественное"}}},
    "日本語": {"dir": "ltr", "title": "皮膚安全AI", "upload": "📥 アップロード", "camera": "📸 カメラ", "analyze": "🚀 解析", "invalid_img": "❌ 無効な画像。", "advice": "⚠️ 注意：AI診断補助。", "guide": "📖 ガイド", "links": "🔗 リンク", "classes": {0: {"name": "メラノーマ", "desc": "悪性"}, 1: {"name": "母斑", "desc": "良性"}, 2: {"name": "BCC", "desc": "悪性"}, 3: {"name": "AK", "desc": "前癌"}, 4: {"name": "BKL", "desc": "良性"}, 5: {"name": "DF", "desc": "良性"}, 6: {"name": "VASC", "desc": "良性"}}},
    "Português": {"dir": "ltr", "title": "IA de Pele", "upload": "📥 Carregar", "camera": "📸 Câmera", "analyze": "🚀 Analisar", "invalid_img": "❌ Imagem inválida.", "advice": "⚠️ Nota: Apenas orientação.", "guide": "📖 Guia", "links": "🔗 Links", "classes": {0: {"name": "Melanoma", "desc": "Maligno"}, 1: {"name": "Nevo", "desc": "Benigno"}, 2: {"name": "BCC", "desc": "Maligno"}, 3: {"name": "AK", "desc": "Pré-cancerígeno"}, 4: {"name": "BKL", "desc": "Benigno"}, 5: {"name": "DF", "desc": "Benigno"}, 6: {"name": "VASC", "desc": "Benigno"}}},
    "Türkçe": {"dir": "ltr", "title": "Cilt Güvenliği AI", "upload": "📥 Yükle", "camera": "📸 Kamera", "analyze": "🚀 Analiz", "invalid_img": "❌ Geçersiz resim.", "advice": "⚠️ Not: Sadece rehberlik.", "guide": "📖 Rehber", "links": "🔗 Bağlantılar", "classes": {0: {"name": "Melanom", "desc": "Kötü Huylu"}, 1: {"name": "Nevüs", "desc": "İyi Huylu"}, 2: {"name": "BCC", "desc": "Kötü Huylu"}, 3: {"name": "AK", "desc": "Öncü"}, 4: {"name": "BKL", "desc": "İyi Huylu"}, 5: {"name": "DF", "desc": "İyi Huylu"}, 6: {"name": "VASC", "desc": "İyi Huylu"}}},
    "한국어": {"dir": "ltr", "title": "피부 안전 AI", "upload": "📥 업로드", "camera": "📸 카메라", "analyze": "🚀 분석", "invalid_img": "❌ 잘못된 이미지.", "advice": "⚠️ 참고: AI 도구입니다.", "guide": "📖 가이드", "links": "🔗 링크", "classes": {0: {"name": "흑색종", "desc": "악성"}, 1: {"name": "모반", "desc": "양성"}, 2: {"name": "BCC", "desc": "악성"}, 3: {"name": "AK", "desc": "전암"}, 4: {"name": "BKL", "desc": "양성"}, 5: {"name": "DF", "desc": "양성"}, 6: {"name": "VASC", "desc": "양성"}}},
    "Italiano": {"dir": "ltr", "title": "IA Pelle", "upload": "📥 Carica", "camera": "📸 Camera", "analyze": "🚀 Analizza", "invalid_img": "❌ Immagine non valida.", "advice": "⚠️ Nota: Solo per guida.", "guide": "📖 Guida", "links": "🔗 Link", "classes": {0: {"name": "Melanoma", "desc": "Maligno"}, 1: {"name": "Neo", "desc": "Benigno"}, 2: {"name": "BCC", "desc": "Maligno"}, 3: {"name": "AK", "desc": "Pre-canceroso"}, 4: {"name": "BKL", "desc": "Benigno"}, 5: {"name": "DF", "desc": "Benigno"}, 6: {"name": "VASC", "desc": "Benigno"}}},
    "اردو": {"dir": "rtl", "title": "اسکن سیفٹی AI", "upload": "📥 اپلوڈ", "camera": "📸 کیمرہ", "analyze": "🚀 تجزیہ", "invalid_img": "❌ غلط تصویر۔", "advice": "⚠️ نوٹ: صرف رہنمائی۔", "guide": "📖 طبی گائیڈ", "links": "🔗 روابط", "classes": {0: {"name": "کینسر", "desc": "خطرناک"}, 1: {"name": "تِل", "desc": "بے ضرر"}, 2: {"name": "BCC", "desc": "خطرناک"}, 3: {"name": "AK", "desc": "قبل از کینسر"}, 4: {"name": "BKL", "desc": "بے ضرر"}, 5: {"name": "DF", "desc": "بے ضرر"}, 6: {"name": "VASC", "desc": "بے ضرر"}}},
    "فارسي": {"dir": "rtl", "title": "هوش مصنوعی پوست", "upload": "📥 بارگذاری", "camera": "📸 دوربین", "analyze": "🚀 آنالیز", "invalid_img": "❌ تصویر نامعتبر.", "advice": "⚠️ توجه: ابزار راهنما.", "guide": "📖 راهنما", "links": "🔗 پیوندها", "classes": {0: {"name": "ملانوما", "desc": "بدخیم"}, 1: {"name": "خال", "desc": "خوش‌خیم"}, 2: {"name": "BCC", "desc": "بدخیم"}, 3: {"name": "AK", "desc": "پیش‌سرطانی"}, 4: {"name": "BKL", "desc": "خوش‌خیم"}, 5: {"name": "DF", "desc": "خوش‌خیم"}, 6: {"name": "VASC", "desc": "خوش‌خیم"}}},
    "Tiếng Việt": {"dir": "ltr", "title": "AI Da liễu", "upload": "📥 Tải lên", "camera": "📸 Máy ảnh", "analyze": "🚀 Phân tích", "invalid_img": "❌ Ảnh không hợp lệ.", "advice": "⚠️ Lưu ý: Chỉ tham khảo.", "guide": "📖 Hướng dẫn", "links": "🔗 Liên kết", "classes": {0: {"name": "U hắc tố", "desc": "Ác tính"}, 1: {"name": "Nốt ruồi", "desc": "Lành tính"}, 2: {"name": "BCC", "desc": "Ác tính"}, 3: {"name": "AK", "desc": "Tiền ung thư"}, 4: {"name": "BKL", "desc": "Lành tính"}, 5: {"name": "DF", "desc": "Lành tính"}, 6: {"name": "VASC", "desc": "Lành tính"}}},
    "Bahasa Indonesia": {"dir": "ltr", "title": "AI Kulit", "upload": "📥 Unggah", "camera": "📸 Kamera", "analyze": "🚀 Analisis", "invalid_img": "❌ Gambar tidak valid.", "advice": "⚠️ Catatan: Hanya panduan.", "guide": "📖 Panduan", "links": "🔗 Tautan", "classes": {0: {"name": "Melanoma", "desc": "Ganas"}, 1: {"name": "Nevi", "desc": "Jinak"}, 2: {"name": "BCC", "desc": "Ganas"}, 3: {"name": "AK", "desc": "Pra-kanker"}, 4: {"name": "BKL", "desc": "Jinak"}, 5: {"name": "DF", "desc": "Jinak"}, 6: {"name": "VASC", "desc": "Jinak"}}},
    "Nederlands": {"dir": "ltr", "title": "Huid AI", "upload": "📥 Uploaden", "camera": "📸 Camera", "analyze": "🚀 Analyse", "invalid_img": "❌ Ongeldige foto.", "advice": "⚠️ Let op: AI-gids.", "guide": "📖 Gids", "links": "🔗 Links", "classes": {0: {"name": "Melanoom", "desc": "Kwaadaardig"}, 1: {"name": "Moedervlek", "desc": "Goedaardig"}, 2: {"name": "BCC", "desc": "Kwaadaardig"}, 3: {"name": "AK", "desc": "Voorstadium"}, 4: {"name": "BKL", "desc": "Goedaardig"}, 5: {"name": "DF", "desc": "Goedaardig"}, 6: {"name": "VASC", "desc": "Goedaardig"}}},
    "Polski": {"dir": "ltr", "title": "AI Skóry", "upload": "📥 Prześlij", "camera": "📸 Kamera", "analyze": "🚀 Analizuj", "invalid_img": "❌ Błędne zdjęcie.", "advice": "⚠️ Uwaga: Tylko pomoc.", "guide": "📖 Przewodnik", "links": "🔗 Linki", "classes": {0: {"name": "Czerniak", "desc": "Złośliwy"}, 1: {"name": "Znamię", "desc": "Łagodne"}, 2: {"name": "BCC", "desc": "Złośliwy"}, 3: {"name": "AK", "desc": "Stan przedrakowy"}, 4: {"name": "BKL", "desc": "Łagodne"}, 5: {"name": "DF", "desc": "Łagodne"}, 6: {"name": "VASC", "desc": "Łagodne"}}},
    "کوردی": {"dir": "rtl", "title": "ژیری دەستکردی پێست", "upload": "📥 وێنە بنێرە", "camera": "📸 کامێرا", "analyze": "🚀 شیکاری", "invalid_img": "❌ وێنەکە هەڵەیە.", "advice": "⚠️ ئاگاداری: تەنها ڕێبەرە.", "guide": "📖 ڕێبەری پزیشکی", "links": "🔗 بەستەرەکان", "classes": {0: {"name": "میلانۆما", "desc": "خراپ"}, 1: {"name": "خاڵ", "desc": "بێ زیان"}, 2: {"name": "BCC", "desc": "خراپ"}, 3: {"name": "AK", "desc": "پێش شێرپەنجە"}, 4: {"name": "BKL", "desc": "بێ زیان"}, 5: {"name": "DF", "desc": "بێ زیان"}, 6: {"name": "VASC", "desc": "بێ زیان"}}}
}

if 'lang' not in st.session_state:
    st.session_state.lang = "العربية"
t = LANG_DATA[st.session_state.lang]

# --- 3. نظام الفلترة (MobileNetV2 Imagenet) ---
@st.cache_resource
def load_filter_engine():
    return tf.keras.applications.MobileNetV2(weights="imagenet")

filter_engine = load_filter_engine()

def validate_image_content(image):
    img = image.resize((224, 224))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(np.array(img), axis=0))
    preds = filter_engine.predict(x)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]
    
    # قائمة بالأشياء المرفوضة
    forbidden_labels = ['car', 'wheel', 'motor', 'dog', 'cat', 'flower', 'screen', 'monitor', 'website', 'text']
    for _, label, score in decoded:
        if any(fl in label.lower() for fl in forbidden_labels) and score > 0.2:
            return False
    return True

# --- 4. محرك التشخيص الهجين (Ensemble) ---
@st.cache_resource
def load_diagnostic_engine():
    base1 = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    base2 = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    combined = Concatenate()([GlobalAveragePooling2D()(base1.output), GlobalAveragePooling2D()(base2.output)])
    x = Dense(512, activation='relu')(combined)
    x = Dropout(0.5)(x)
    outputs = Dense(7, activation='softmax')(x)
    model = Model(inputs=[base1.input, base2.input], outputs=outputs)
    try:
        model.load_weights("skin_expert_master.h5")
    except:
        pass
    return model

diag_engine = load_diagnostic_engine()

# --- 5. التنسيق ومنع التداخل (CSS) ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    * {{ font-family: 'Tajawal', sans-serif; direction: {t['dir']}; }}
    .main-title {{ text-align: center; color: #0d47a1; font-size: 2.2em; font-weight: bold; padding: 15px; }}
    .medical-warning {{ background-color: #fff1f0; border: 1px solid #ffa39e; padding: 12px; border-radius: 10px; color: #cf1322; text-align: center; margin-bottom: 20px; font-weight: bold; }}
    .stButton>button {{ width: 100%; border-radius: 12px; height: 3.8em; font-weight: bold; background-color: #0d47a1; color: white; border: none; }}
    .res-card {{ padding: 25px; border-radius: 15px; border: 5px solid; background: white; text-align: center; }}
</style>
""", unsafe_allow_html=True)

# --- 6. واجهة المستخدم والتفاعل ---
st.markdown(f"<div class='main-title'>{t['title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='medical-warning'>{t['advice']}</div>", unsafe_allow_html=True)

# اختيار اللغة
with st.popover(f"🌐 {st.session_state.lang}"):
    cols = st.columns(2)
    for i, lang_name in enumerate(LANG_DATA.keys()):
        with cols[i % 2]:
            if st.button(lang_name, key=f"L_{lang_name}"):
                st.session_state.lang = lang_name
                st.rerun()

st.write("---")

col_ui1, col_ui2 = st.columns(2, gap="large")

with col_ui1:
    mode = st.radio("", [t['upload'], t['camera']], horizontal=True)
    file = st.file_uploader("", type=["jpg", "png", "jpeg"]) if mode == t['upload'] else st.camera_input("")

if file:
    img_input = Image.open(file).convert('RGB')
    with col_ui2:
        st.image(img_input, use_container_width=True, caption="Scan Input")
    
    if st.button(t['analyze']):
        with st.spinner(t['status_check']):
            # 1. فلترة الصور الخارجية
            if not validate_image_content(img_input):
                st.error(t['invalid_img'])
            else:
                # 2. التشخيص الطبي
                img_cv = cv2.resize(np.array(img_input), (224, 224))
                img_proc = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_cv, axis=0))
                
                preds = diag_engine.predict([img_proc, img_proc])[0]
                idx = np.argmax(preds)
                conf = preds[idx]
                
                # منع النتائج الضعيفة (أقل من 40%)
                if conf < 0.40:
                    st.warning("⚠️ نعتذر، جودة الصورة لا تسمح بتشخيص دقيق. يرجى إعادة التصوير.")
                else:
                    res = t['classes'][idx]
                    color = "#cf1322" if idx in [0, 2] else "#389e0d"
                    
                    st.markdown(f"""
                    <div class="res-card" style="border-color: {color}; color: {color};">
                        <h1 style="margin:0;">{res['name']}</h1>
                        <p style="font-size: 1.2em; color: #444;">{res['desc']}</p>
                        <hr style="border: 1px solid {color}">
                        <h3>نسبة التأكد: {conf*100:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)

# --- 7. الدليل الطبي والروابط العالمية ---
st.write("---")
with st.expander(t['guide']):
    st.markdown(f"""
    <div dir="{t['dir']}">
    <h3>أنواع سرطان الجلد والأمراض المشمولة:</h3>
    <ul>
        <li><b>Melanoma (ميلانوما):</b> أخطر أنواع سرطان الجلد. يبدأ في الخلايا الصبغية.</li>
        <li><b>BCC (سرطان الخلايا القاعدية):</b> ورم جلدي شائع ينمو ببطء ونادراً ما ينتشر.</li>
        <li><b>AK (تقران سفعي):</b> آفات قشرية ناتجة عن التعرض للشمس، قد تتحول لسرطان.</li>
        <li><b>الأمراض الحميدة (BKL, DF, VASC):</b> شامات وزوائد جلدية لا تشكل خطراً على الحياة.</li>
    </ul>
    <hr>
    <h3>{t['links']}</h3>
    <ul>
        <li><a href="https://www.mayoclinic.org/skin-cancer" target="_blank">Mayo Clinic Medical Center</a></li>
        <li><a href="https://www.skincancer.org/" target="_blank">Skin Cancer Foundation</a></li>
        <li><a href="https://www.cancer.org/" target="_blank">American Cancer Society</a></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
