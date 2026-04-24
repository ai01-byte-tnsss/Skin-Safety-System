import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Skin AI Expert System", layout="wide")

# --- 2. الدليل المرجعي لـ 24 صنف (مطابق لترتيب التدريب) ---
MEDICAL_INFO = {
    0: {"n": "Acne and Rosacea", "s": "✅ حالة شائعة", "c": "#34C759", "d": "حب الشباب والوردية؛ حالات تتعلق بانسداد المسام والتهاب الغدد الدهنية."},
    1: {"n": "Actinic Keratosis / BCC", "s": "🚨 خبيث / ما قبل سرطاني", "c": "#FF3B30", "d": "تقرن ضوئي أو سرطان الخلايا القاعدية؛ يتطلب استشارة طبية فورية لاستئصاله."},
    2: {"n": "Atopic Dermatitis", "s": "🔍 حالة جلدية", "c": "#5856D6", "d": "التهاب الجلد التأتبي؛ نوع مزمن من الحساسية يسبب حكة وطفحاً جلدياً."},
    3: {"n": "Bullous Disease", "s": "🚨 حالة حرجة", "c": "#FF3B30", "d": "أمراض فقاعية؛ تسبب بثوراً مائية كبيرة وتطلب رعاية طبية متخصصة."},
    4: {"n": "Bacterial Infections", "s": "🦠 عدوى بكتيرية", "c": "#FF9500", "d": "التهابات بكتيرية مثل القوباء؛ تتطلب عادةً مضادات حيوية تحت إشراف طبي."},
    5: {"n": "Eczema", "s": "🔍 حالة جلدية", "c": "#5856D6", "d": "الإكزيما؛ تهيج جلدي يسبب احمراراً وجفافاً وحكة شديدة."},
    6: {"n": "Exanthems and Drug Eruptions", "s": "⚠️ طارئ طبي", "c": "#FF9500", "d": "طفح دوائي؛ رد فعل تحسسي مفاجئ ناتج عن تناول بعض الأدوية."},
    7: {"n": "Hair Loss / Alopecia", "s": "🔍 حالة شعر", "c": "#5AC8FA", "d": "تساقط الشعر أو الثعلبة؛ حالات تؤثر على بصيلات الشعر وفروة الرأس."},
    8: {"n": "Herpes / HPV / STDs", "s": "🦠 عدوى فيروسية", "c": "#FF9500", "d": "عدوى فيروسية تشمل الهربس أو الثآليل؛ تتطلب فحصاً مخبرياً دقيقاً."},
    9: {"n": "Pigmentation Disorders", "s": "🔍 اضطراب صبغة", "c": "#AF52DE", "d": "اضطرابات التصبغ مثل البهاق أو الكلف؛ تتعلق بنشاط خلايا الميلانين."},
    10: {"n": "Lupus / Connective Tissue", "s": "🚨 مرض مناعي", "c": "#FF3B30", "d": "أمراض الأنسجة الضامة مثل الذئبة؛ تؤثر المناعة فيها على الجلد."},
    11: {"n": "Melanoma / Nevi / Moles", "s": "🚨 خبيث جداً / شامات", "c": "#FF2D55", "d": "الميلانوما أو الشامات؛ يجب مراقبة أي تغير في الحجم أو اللون فوراً."},
    12: {"n": "Nail Diseases", "s": "🔍 أمراض أظافر", "c": "#8E8E93", "d": "تشمل فطريات الأظافر أو تغيرات شكل الظفر الناتجة عن أمراض داخلية."},
    13: {"n": "Contact Dermatitis", "s": "🔍 تحسس تماسي", "c": "#5856D6", "d": "التهاب الجلد الناتج عن ملامسة مواد مهيجة مثل المنظفات أو المعادن."},
    14: {"n": "Psoriasis / Lichen Planus", "s": "🔍 حالة مزمنة", "c": "#AF52DE", "d": "الصدفية أو الحزاز؛ أمراض مناعية تسبب قشوراً فضية أو بقعاً أرجوانية."},
    15: {"n": "Infestations and Bites", "s": "🦠 طفيليات", "c": "#FF9500", "d": "تشمل الجرب وعضات الحشرات؛ ناتجة عن كائنات تعيش على الجلد أو تلدغه."},
    16: {"n": "Seborrheic Keratoses / Benign", "s": "✅ حميد", "c": "#34C759", "d": "تقرن دهني؛ نمو جلدي حميد غير سرطاني يزداد مع التقدم في العمر."},
    17: {"n": "Systemic Disease", "s": "⚠️ مؤشر جهازي", "c": "#FF9500", "d": "أعراض جلدية تعكس وجود مرض داخل أجهزة الجسم (مثل الكبد أو الكلى)."},
    18: {"n": "Fungal Infections", "s": "🦠 عدوى فطريات", "c": "#FF9500", "d": "عدوى فطرية تشمل القوباء الحلقية؛ تتطلب كريمات مضادة للفطريات."},
    19: {"n": "Urticaria Hives", "s": "✅ حالة شائعة", "c": "#34C759", "d": "الأرتيكاريا (الشرى)؛ طفح جلدي يسبب حكة شديدة ويظهر ويختفي فجأة."},
    20: {"n": "Vascular Tumors", "s": "✅ حميد غالباً", "c": "#34C759", "d": "أورام وعائية؛ تجمعات دموية تظهر كبقع حمراء بارزة."},
    21: {"n": "Vasculitis", "s": "🚨 خطير", "c": "#FF3B30", "d": "التهاب الأوعية الدموية؛ يسبب بقعاً أرجوانية تحت الجلد نتيجة نزف بسيط."},
    22: {"n": "Warts / Viral Infections", "s": "🦠 عدوى فيروسية", "c": "#FF9500", "d": "الثآليل الفيروسية؛ تظهر نتيجة نشاط فيروسي في خلايا الجلد."},
    23: {"n": "Archive / Other", "s": "🔍 غير محدد", "c": "#8E8E93", "d": "حالات غير مصنفة حالياً أو تتبع فئة الأرشيف في قاعدة البيانات."},
}

# --- 3. تجميع وتحميل النموذج الهجين ---
@st.cache_resource
def load_system_model():
    # الأجزاء الثلاثة التي تم رفعها
    parts = [
        "skin_expert_hybrid_24ch.z01", 
        "skin_expert_hybrid_24ch.z02", 
        "skin_expert_hybrid_24ch.z03"
    ]
    temp_h5 = "full_model_assembled.h5"
    
    # تجميع الأجزاء في ملف واحد
    try:
        with open(temp_h5, "wb") as outfile:
            for part in parts:
                with open(part, "rb") as infile:
                    outfile.write(infile.read())
    except FileNotFoundError:
        st.error("⚠️ ملفات النموذج (.z01, .z02, .z03) غير موجودة في المستودع.")
        return None

    # بناء هيكل النموذج الهجين (EfficientNet + MobileNet)
    b1 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    b2 = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    comb = Concatenate()([GlobalAveragePooling2D()(b1.output), GlobalAveragePooling2D()(b2.output)])
    dense = Dense(512, activation='relu')(comb)
    out = Dense(24, activation='softmax')(Dropout(0.4)(dense))
    
    final_model = Model(inputs=[b1.input, b2.input], outputs=out)
    final_model.load_weights(temp_h5)
    return final_model

model = load_system_model()

# --- 4. واجهة المستخدم الرسومية ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>الذكاء الاصطناعي لفحص وسلامة الجلد</h1>", unsafe_allow_html=True)
st.info("⚠️ هذا النظام هو مشروع تخرج تعليمي، يرجى استشارة الطبيب المختص دائماً.")

uploaded_file = st.file_uploader("📥 ارفع صورة الجلد (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(img, caption="الصورة المرفوعة", use_container_width=True)
    
    if st.button("🔍 بدء التحليل الرقمي"):
        with st.spinner("⏳ جاري تحليل الأنماط الحيوية..."):
            # معالجة الصورة
            img_np = np.array(img)
            img_res = cv2.resize(img_np, (224, 224))
            
            # تحسين التباين CLAHE
            lab = cv2.cvtColor(img_res, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            img_proc = cv2.merge((clahe.apply(l), a, b))
            img_proc = cv2.cvtColor(img_proc, cv2.COLOR_LAB2RGB)
            
            # تجهيز المدخلات
            inp = img_proc.astype(np.float32) / 255.0
            inp = np.expand_dims(inp, axis=0)
            
            # التنبؤ
            predictions = model.predict([inp, inp])[0]
            idx = np.argmax(predictions)
            confidence = predictions[idx]
            
            res = MEDICAL_INFO[idx]
            
            # --- منطق الموازنة والنسب المطلوبة ---
            is_serious = "🚨" in res['s'] or "⚠️" in res['s']
            is_benign = "✅" in res['s']
            
            valid_result = False
            if is_serious and confidence >= 0.40: # شرط الخبيث/الخطير
                valid_result = True
            elif is_benign and confidence >= 0.45: # شرط الحميد
                valid_result = True
            elif confidence >= 0.35: # باقي الحالات
                valid_result = True
            
            if valid_result:
                with col2:
                    st.markdown(f"""
                    <div style="padding:20px; border-radius:10px; border-right:15px solid {res['c']}; background-color:#f8f9fa;">
                        <h2 style="color:{res['c']};">{res['n']}</h2>
                        <h4>التصنيف: {res['s']}</h4>
                        <p style="font-size:1.2em;"><b>دقة التنبؤ:</b> {confidence:.2%}</p>
                        <hr>
                        <p>{res['d']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ لم يتمكن النظام من تحديد الحالة بدقة كافية. يرجى محاولة رفع صورة أوضح.")

# --- 5. قسم الدليل المرجعي ---
st.write("---")
with st.expander("📖 عرض كافة الحالات الجلدية الـ 24 التي يدعمها النظام"):
    for k, v in MEDICAL_INFO.items():
        st.markdown(f"**{v['n']}**: {v['d']}")
