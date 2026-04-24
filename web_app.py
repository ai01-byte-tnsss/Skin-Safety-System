import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. إعدادات الصفحة ---
st.set_page_config(
    page_title="Skin AI Expert System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تخصيص التصميم عبر CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #1E3A8A; color: white; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    div[data-testid="stMarkdownContainer"] > p { text-align: right; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. الدليل المرجعي الكامل (24 صنف) ---
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

# --- 3. دالة تجميع وتحميل النموذج الهجين ---
@st.cache_resource
def load_system_model():
    parts = ["skin_expert_hybrid_24ch.z01", "skin_expert_hybrid_24ch.z02", "skin_expert_hybrid_24ch.z03"]
    temp_h5 = "model_assembled_final.h5"
    
    try:
        # دمج ملفات الأوزان
        with open(temp_h5, "wb") as outfile:
            for part in parts:
                if os.path.exists(part):
                    with open(part, "rb") as infile:
                        outfile.write(infile.read())
                else:
                    st.error(f"❌ الملف {part} مفقود من المستودع!")
                    return None
        
        # بناء الهيكل المتوافق مع الأوزان (Hybrid)
        base_model_1 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
        base_model_2 = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
        
        c = Concatenate()([
            GlobalAveragePooling2D()(base_model_1.output), 
            GlobalAveragePooling2D()(base_model_2.output)
        ])
        d = Dense(512, activation='relu')(c)
        drop = Dropout(0.4)(d)
        out = Dense(24, activation='softmax')(drop)
        
        full_model = Model(inputs=[base_model_1.input, base_model_2.input], outputs=out)
        
        # تحميل الأوزان إلى الهيكل
        full_model.load_weights(temp_h5)
        return full_model
    except Exception as e:
        st.error(f"⚠️ خطأ فني أثناء تحميل النموذج: {e}")
        return None

# تحميل النموذج لمرة واحدة وتخزينه في الذاكرة
model = load_system_model()

# --- 4. واجهة المستخدم ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>الذكاء الاصطناعي لفحص سلامة الجلد 🧬</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>نظام خبير لتحليل الصور الجلدية باستخدام الشبكات العصبية العميقة</p>", unsafe_allow_html=True)
st.write("---")

# الشريط الجانبي
with st.sidebar:
    st.header("حول النظام")
    st.info("هذا النظام يستخدم نموذجاً هجيناً (EfficientNet + MobileNet) تم تدريبه على آلاف الصور السريرية لتصنيف 24 نوعاً من الحالات الجلدية.")
    st.warning("⚠️ تنبيه: هذا النظام للاستخدام الاسترشادي فقط ولا يغني عن زيارة الطبيب المختص.")

# رفع الصورة
file = st.file_uploader("📥 ارفع صورة الجلد بوضوح (JPG, PNG)", type=["jpg", "png", "jpeg"])

if file:
    if model is None:
        st.error("فشل النظام في التحميل. يرجى مراجعة ملفات الأوزان في GitHub.")
    else:
        img = Image.open(file).convert('RGB')
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(img, caption="الصورة المرفوعة للتحليل", use_container_width=True)
        
        if st.button("🔍 بدء التحليل الرقمي"):
            with st.spinner("⏳ جاري فحص الأنماط الحيوية ومقارنتها بقاعدة البيانات..."):
                # معالجة الصورة للنموذج
                img_cv = np.array(img)
                img_res = cv2.resize(img_cv, (224, 224))
                img_tensor = (img_res.astype(np.float32) / 255.0)[np.newaxis, ...]
                
                # التنبؤ (يتم إرسال الصورة للمدخلين)
                preds = model.predict([img_tensor, img_tensor])[0]
                idx = np.argmax(preds)
                conf = preds[idx]
                
                # استرجاع معلومات الحالة
                res = MEDICAL_INFO.get(idx, MEDICAL_INFO[23])
                
                # تحديد العتبة (Threshold) بناءً على خطورة الحالة
                is_serious = "🚨" in res['s']
                threshold = 0.40 if is_serious else 0.45
                
                if conf >= threshold:
                    with col2:
                        st.markdown(f"""
                        <div style="padding:20px; border-radius:15px; border-right:15px solid {res['c']}; background-color:#ffffff; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); direction:rtl; text-align:right;">
                            <h2 style="color:{res['c']}; margin-top:0;">{res['n']}</h2>
                            <h4 style="color:#555;">التصنيف: {res['s']}</h4>
                            <p style="font-size:1.2em; color:#1E3A8A;"><b>دقة التنبؤ:</b> {conf:.2%}</p>
                            <hr style="border-top: 1px solid #eee;">
                            <p style="color:#333; line-height:1.6;">{res['d']}</p>
                            <p style="font-size:0.8em; color:#888; margin-top:10px;">* يوصى بمراجعة مختص للتأكيد السريري.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ دقة التنبؤ منخفضة ({conf:.1%}). لم يتمكن النظام من الجزم بالحالة. يرجى رفع صورة بتركيز أفضل وإضاءة طبيعية.")

# تذييل الصفحة
st.write("---")
st.caption("تم التطوير بواسطة: Skin AI Expert System Team | 2026")
