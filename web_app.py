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
st.set_page_config(
    page_title="Skin AI Expert System",
    page_icon="🧬",
    layout="wide"
)

# تحسين مظهر الواجهة
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #1E3A8A; color: white; font-weight: bold; }
    div[data-testid="stMarkdownContainer"] > p { text-align: right; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. الدليل المرجعي للأمراض (24 صنف) ---
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

# --- 3. دالة معالجة الـ ZIP وتحميل النموذج ---
@st.cache_resource
def load_expert_model():
    zip_path = "skin_expert_hybrid_24ch.zip"
    extract_dir = "extracted_model"
    
    try:
        # 1. فك ضغط الملف إذا وجد
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            st.error(f"❌ ملف {zip_path} غير موجود!")
            return None

        # 2. البحث التلقائي عن ملف الـ .h5
        h5_path = None
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if f.endswith(".h5"):
                    h5_path = os.path.join(root, f)
                    break
        
        if not h5_path:
            st.error("❌ لم يتم العثور على ملف الأوزان (.h5) داخل الـ ZIP.")
            return None

        # 3. بناء هيكل الشبكة الهجين (Hybrid Architecture)
        base1 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
        base2 = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
        
        merge = Concatenate()([GlobalAveragePooling2D()(base1.output), GlobalAveragePooling2D()(base2.output)])
        dense = Dense(512, activation='relu')(merge)
        drop = Dropout(0.4)(dense)
        output = Dense(24, activation='softmax')(drop)
        
        model = Model(inputs=[base1.input, base2.input], outputs=output)
        
        # 4. تحميل الأوزان من الملف المستخرج
        model.load_weights(h5_path)
        return model
    except Exception as e:
        st.error(f"⚠️ فشل تقني: {e}")
        return None

# تشغيل دالة التحميل
model = load_expert_model()

# --- 4. واجهة المستخدم ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>نظام خبير الجلد بالذكاء الاصطناعي 🧬</h1>", unsafe_allow_html=True)
st.write("---")

# رفع الصورة
uploaded_file = st.file_uploader("📥 ارفع صورة الجلد المصاب للتحليل (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    if model is None:
        st.error("المحرك الذكي غير جاهز. تأكد من سلامة ملف الـ ZIP.")
    else:
        # عرض الصورة المرفوعة
        img = Image.open(uploaded_file).convert('RGB')
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.image(img, caption="الصورة الأصلية", use_container_width=True)
        
        if st.button("🔍 تحليل الصورة الآن"):
            with st.spinner("⏳ يتم الآن مطابقة الأنماط الحيوية..."):
                # معالجة الصورة
                img_array = np.array(img)
                img_res = cv2.resize(img_array, (224, 224))
                img_tensor = (img_res.astype(np.float32) / 255.0)[np.newaxis, ...]
                
                # التنبؤ (مدخلان متطابقان للنموذج الهجين)
                predictions = model.predict([img_tensor, img_tensor])[0]
                idx = np.argmax(predictions)
                confidence = predictions[idx]
                
                # جلب معلومات الحالة
                info = MEDICAL_INFO.get(idx, MEDICAL_INFO[23])
                
                # عتبة الثقة (Threshold)
                is_serious = "🚨" in info['s']
                req_conf = 0.40 if is_serious else 0.45
                
                if confidence >= req_conf:
                    with c2:
                        st.markdown(f"""
                        <div style="padding:25px; border-radius:15px; border-right:12px solid {info['c']}; background-color:#ffffff; direction:rtl; text-align:right; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                            <h2 style="color:{info['c']};">{info['n']}</h2>
                            <h4 style="color:#444;">حالة التشخيص: {info['s']}</h4>
                            <p style="font-size:1.2em; color:#1E3A8A;"><b>نسبة الثقة:</b> {confidence:.2%}</p>
                            <hr>
                            <p style="line-height:1.6;">{info['d']}</p>
                            <p style="color:red; font-size:0.8em; margin-top:15px;">* هذا التحليل استرشادي فقط، يرجى استشارة طبيب مختص فوراً.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ النتيجة غير مؤكدة ({confidence:.1%}). يرجى التقاط صورة أوضح وتحت إضاءة أفضل.")

st.write("---")
st.caption("نظام Skin Safety System V2.0 - 2026")
