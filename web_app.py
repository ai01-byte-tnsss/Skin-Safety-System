import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os
import zipfile

# --- 1. إعدادات الصفحة والجماليات ---
st.set_page_config(
    page_title="Skin AI Expert System",
    page_icon="🧬",
    layout="wide"
)

# تخصيص المظهر (CSS) لجعل الواجهة تدعم اللغة العربية (RTL)
st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3.5em; background-color: #1E3A8A; color: white; font-weight: bold; font-size: 1.1em; }
    div[data-testid="stMarkdownContainer"] > p { text-align: right; font-family: 'Arial'; }
    .status-card { padding: 20px; border-radius: 15px; background-color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.1); direction: rtl; text-align: right; }
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

# --- 3. محرك تحميل النموذج (التعامل مع ZIP) ---
@st.cache_resource
def get_model():
    zip_fn = "skin_expert_hybrid_24ch.zip"
    tmp_dir = "model_files"
    
    try:
        # فك الضغط برمجياً
        if os.path.exists(zip_fn):
            with zipfile.ZipFile(zip_fn, 'r') as z:
                z.extractall(tmp_dir)
        else:
            st.error("❌ ملف ZIP غير موجود!")
            return None

        # البحث عن ملف الأوزان h5
        target_h5 = None
        for r, d, fs in os.walk(tmp_dir):
            for f in fs:
                if f.endswith(".h5"):
                    target_h5 = os.path.join(r, f)
                    break
        
        if not target_h5:
            st.error("❌ لم يتم العثور على ملف أوزان h5 داخل ZIP.")
            return None

        # بناء الهيكل الهجين (يجب أن يطابق طريقة تدريبك للنموذج)
        b1 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
        b2 = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
        
        merged = Concatenate()([GlobalAveragePooling2D()(b1.output), GlobalAveragePooling2D()(b2.output)])
        dense = Dense(512, activation='relu')(merged)
        out = Dense(24, activation='softmax')(Dropout(0.4)(dense))
        
        m = Model(inputs=[b1.input, base_2_input := b2.input], outputs=out)
        
        # تحميل الأوزان
        m.load_weights(target_h5)
        return m
    except Exception as e:
        st.error(f"⚠️ خطأ في المحرك: {e}")
        return None

# تهيئة النموذج
model = get_model()

# --- 4. واجهة المستخدم الرسومية ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>الذكاء الاصطناعي لفحص سلامة الجلد 🧬</h1>", unsafe_allow_html=True)
st.write("---")

# رفع الملف
up_file = st.file_uploader("📥 ارفع صورة المنطقة المصابة (بوضوح تام)", type=["jpg", "png", "jpeg"])

if up_file:
    if model is None:
        st.error("النظام غير جاهز، يرجى التأكد من ملفات الأوزان.")
    else:
        raw_img = Image.open(up_file).convert('RGB')
        left, right = st.columns([1, 1])
        
        with left:
            st.image(raw_img, caption="الصورة المرفوعة", use_container_width=True)
        
        if st.button("🔍 تحليل الأنماط الحيوية الآن"):
            with st.spinner("⏳ جاري تحليل الصورة بمحرك الذكاء الاصطناعي..."):
                # المعالجة المسبقة
                img_cv = cv2.resize(np.array(raw_img), (224, 224))
                tensor = (img_cv.astype(np.float32) / 255.0)[np.newaxis, ...]
                
                # التنبؤ
                pred_raw = model.predict([tensor, tensor])[0]
                best_idx = np.argmax(pred_raw)
                score = pred_raw[best_idx]
                
                # معلومات النتيجة
                data = MEDICAL_INFO.get(best_idx, MEDICAL_INFO[23])
                
                # تحديد عتبة القبول
                is_serious = "🚨" in data['s']
                threshold = 0.40 if is_serious else 0.45
                
                if score >= threshold:
                    with right:
                        st.markdown(f"""
                        <div class="status-card" style="border-right: 12px solid {data['c']};">
                            <h2 style="color:{data['c']}; margin-top:0;">{data['n']}</h2>
                            <h4 style="color:#555;">التصنيف السريري: {data['s']}</h4>
                            <p style="font-size:1.3em; color:#1E3A8A;"><b>دقة التنبؤ:</b> {score:.2%}</p>
                            <hr style="border-top: 1px solid #eee;">
                            <p style="color:#333; line-height:1.7; font-size:1.1em;">{data['d']}</p>
                            <p style="color:red; font-size:0.85em; margin-top:15px;">⚠️ ملاحظة: هذا التشخيص آلي استرشادي، يجب مراجعة الطبيب.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ لم يتمكن النظام من تحديد الحالة بدقة كافية ({score:.1%}). يرجى محاولة رفع صورة بتركيز وإضاءة أفضل.")

st.write("---")
st.caption("تم التطوير لصالح نظام Skin Safety System | جميع الحقوق محفوظة 2026")
