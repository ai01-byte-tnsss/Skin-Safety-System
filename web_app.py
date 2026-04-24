import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os
import zipfile

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Skin AI Expert System", page_icon="🧬", layout="wide")

# --- 2. الدليل المرجعي (نفس القائمة السابقة) ---
MEDICAL_INFO = {
    0: {"n": "Acne and Rosacea", "s": "✅ حالة شائعة", "c": "#34C759", "d": "حب الشباب والوردية؛ حالات تتعلق بانسداد المسام والتهاب الغدد الدهنية."},
    1: {"n": "Actinic Keratosis / BCC", "s": "🚨 خبيث / ما قبل سرطاني", "c": "#FF3B30", "d": "تقرن ضوئي أو سرطان الخلايا القاعدية؛ يتطلب استشارة طبية فورية."},
    2: {"n": "Atopic Dermatitis", "s": "🔍 حالة جلدية", "c": "#5856D6", "d": "التهاب الجلد التأتبي؛ نوع مزمن من الحساسية يسبب حكة وطفحاً."},
    3: {"n": "Bullous Disease", "s": "🚨 حالة حرجة", "c": "#FF3B30", "d": "أمراض فقاعية؛ تسبب بثوراً مائية كبيرة وتطلب رعاية متخصصة."},
    4: {"n": "Bacterial Infections", "s": "🦠 عدوى بكتيرية", "c": "#FF9500", "d": "التهابات بكتيرية مثل القوباء؛ تتطلب عادةً مضادات حيوية."},
    5: {"n": "Eczema", "s": "🔍 حالة جلدية", "c": "#5856D6", "d": "الإكزيما؛ تهيج جلدي يسبب احمراراً وجفافاً وحكة شديدة."},
    6: {"n": "Exanthems and Drug Eruptions", "s": "⚠️ طارئ طبي", "c": "#FF9500", "d": "طفح دوائي؛ رد فعل تحسسي مفاجئ ناتج عن تناول بعض الأدوية."},
    7: {"n": "Hair Loss / Alopecia", "s": "🔍 حالة شعر", "c": "#5AC8FA", "d": "تساقط الشعر أو الثعلبة؛ حالات تؤثر على بصيلات الشعر."},
    8: {"n": "Herpes / HPV / STDs", "s": "🦠 عدوى فيروسية", "c": "#FF9500", "d": "عدوى فيروسية تشمل الهربس أو الثآليل؛ تتطلب فحصاً مخبرياً."},
    9: {"n": "Pigmentation Disorders", "s": "🔍 اضطراب صبغة", "c": "#AF52DE", "d": "اضطرابات التصبغ مثل البهاق أو الكلف؛ تتعلق بنشاط خلايا الميلانين."},
    10: {"n": "Lupus / Connective Tissue", "s": "🚨 مرض مناعي", "c": "#FF3B30", "d": "أمراض الأنسجة الضامة مثل الذئبة؛ تؤثر المناعة فيها على الجلد."},
    11: {"n": "Melanoma / Nevi / Moles", "s": "🚨 خبيث جداً / شامات", "c": "#FF2D55", "d": "الميلانوما أو الشامات؛ يجب مراقبة أي تغير في الحجم أو اللون فوراً."},
    12: {"n": "Nail Diseases", "s": "🔍 أمراض أظافر", "c": "#8E8E93", "d": "تشمل فطريات الأظافر أو تغيرات شكل الظفر الناتجة عن أمراض داخلية."},
    13: {"n": "Contact Dermatitis", "s": "🔍 تحسس تماسي", "c": "#5856D6", "d": "التهاب الجلد الناتج عن ملامسة مواد مهيجة مثل المنظفات."},
    14: {"n": "Psoriasis / Lichen Planus", "s": "🔍 حالة مزمنة", "c": "#AF52DE", "d": "الصدفية أو الحزاز؛ أمراض مناعية تسبب قشوراً فضية."},
    15: {"n": "Infestations and Bites", "s": "🦠 طفيليات", "c": "#FF9500", "d": "تشمل الجرب وعضات الحشرات؛ ناتجة عن كائنات خارجية."},
    16: {"n": "Seborrheic Keratoses / Benign", "s": "✅ حميد", "c": "#34C759", "d": "تقرن دهني؛ نمو جلدي حميد غير سرطاني يزداد مع العمر."},
    17: {"n": "Systemic Disease", "s": "⚠️ مؤشر جهازي", "c": "#FF9500", "d": "أعراض جلدية تعكس وجود مرض داخل أجهزة الجسم."},
    18: {"n": "Fungal Infections", "s": "🦠 عدوى فطريات", "c": "#FF9500", "d": "عدوى فطريات تشمل القوباء الحلقية؛ تتطلب مضادات فطريات."},
    19: {"n": "Urticaria Hives", "s": "✅ حالة شائعة", "c": "#34C759", "d": "الأرتيكاريا (الشرى)؛ طفح جلدي يسبب حكة ويظهر ويختفي فجأة."},
    20: {"n": "Vascular Tumors", "s": "✅ حميد غالباً", "c": "#34C759", "d": "أورام وعائية؛ تجمعات دموية تظهر كبقع حمراء بارزة."},
    21: {"n": "Vasculitis", "s": "🚨 خطير", "c": "#FF3B30", "d": "التهاب الأوعية الدموية؛ يسبب بقعاً أرجوانية نتيجة نزف تحت الجلد."},
    22: {"n": "Warts / Viral Infections", "s": "🦠 عدوى فيروسية", "c": "#FF9500", "d": "الثآليل الفيروسية؛ تظهر نتيجة نشاط فيروسي في الجلد."},
    23: {"n": "Archive / Other", "s": "🔍 غير محدد", "c": "#8E8E93", "d": "حالات مؤرشفة أو غير مصنفة حالياً."},
}

# --- 3. دالة فك الضغط وتحميل النموذج ---
@st.cache_resource
def load_system_model():
    zip_path = "skin_expert_hybrid_24ch.zip"
    extract_to = "temp_model_dir"
    
    try:
        # 1. فك ضغط الملف
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            st.error("❌ ملف الـ ZIP مفقود!")
            return None

        # 2. البحث عن ملف الأوزان (h5) داخل المجلد المفكوك
        h5_file = None
        for root, dirs, files in os.walk(extract_to):
            for file in files:
                if file.endswith(".h5"):
                    h5_file = os.path.join(root, file)
                    break
        
        if not h5_file:
            st.error("❌ لم يتم العثور على ملف أوزان .h5 داخل ملف الـ ZIP!")
            return None

        # 3. بناء هيكل النموذج
        base_1 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
        base_2 = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
        
        c = Concatenate()([GlobalAveragePooling2D()(base_1.output), GlobalAveragePooling2D()(base_2.output)])
        d = Dense(512, activation='relu')(c)
        o = Dense(24, activation='softmax')(Dropout(0.4)(d))
        
        full_model = Model(inputs=[base_1.input, base_2.input], outputs=o)
        
        # 4. تحميل الأوزان
        full_model.load_weights(h5_file)
        return full_model

    except Exception as e:
        st.error(f"⚠️ خطأ أثناء تحميل النموذج من ZIP: {e}")
        return None

model = load_system_model()

# --- 4. واجهة المستخدم (نفس المنطق السابق) ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>الذكاء الاصطناعي لفحص سلامة الجلد</h1>", unsafe_allow_html=True)
st.write("---")

file = st.file_uploader("📥 ارفع صورة الجلد (JPG, PNG)", type=["jpg", "png", "jpeg"])

if file and model:
    img = Image.open(file).convert('RGB')
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(img, caption="الصورة المرفوعة", use_container_width=True)
    
    if st.button("🔍 بدء التحليل الرقمي"):
        with st.spinner("⏳ جاري تحليل الأنماط..."):
            img_res = cv2.resize(np.array(img), (224, 224))
            inp = (img_res.astype(np.float32) / 255.0)[np.newaxis, ...]
            
            preds = model.predict([inp, inp])[0]
            idx = np.argmax(preds)
            conf = preds[idx]
            res = MEDICAL_INFO.get(idx, MEDICAL_INFO[23])
            
            is_serious = "🚨" in res['s']
            threshold = 0.40 if is_serious else 0.45
            
            if conf >= threshold:
                with col2:
                    st.markdown(f"""
                    <div style="padding:20px; border-radius:10px; border-right:15px solid {res['c']}; background-color:#f8f9fa; direction:rtl; text-align:right;">
                        <h2 style="color:{res['c']};">{res['n']}</h2>
                        <h4>التصنيف: {res['s']}</h4>
                        <p style="font-size:1.2em;"><b>دقة التنبؤ:</b> {conf:.2%}</p>
                        <hr>
                        <p>{res['d']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ دقة التنبؤ منخفضة. يرجى محاولة رفع صورة أوضح.")
