import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import cv2
import os
import zipfile
import time

# --- 1. إعدادات الصفحة والهوية البصرية ---
st.set_page_config(
    page_title="Skin AI Expert System",
    page_icon="🧬",
    layout="wide"
)

# تخصيص الواجهة لتدعم اللغة العربية
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: right; }
    .main { background-color: #f8fbfb; }
    .stButton>button { width: 100%; border-radius: 12px; height: 3.5em; background-color: #1E3A8A; color: white; font-weight: bold; }
    .report-card { padding: 25px; border-radius: 15px; background-color: white; border-right: 10px solid #1E3A8A; box-shadow: 0 4px 15px rgba(0,0,0,0.05); direction: rtl; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. القاموس الطبي المتكامل (24 فئة) ---
MEDICAL_INFO = {
    0: {"n": "Acne and Rosacea", "s": "✅ حالة شائعة", "c": "#34C759", "d": "حب الشباب والوردية؛ تتعلق بالتهاب الغدد الدهنية."},
    1: {"n": "Actinic Keratosis / BCC", "s": "🚨 ما قبل سرطاني", "c": "#FF3B30", "d": "تقرن ضوئي أو سرطان الخلايا القاعدية؛ يتطلب فحصاً طبياً."},
    2: {"n": "Atopic Dermatitis", "s": "🔍 حالة جلدية", "c": "#5856D6", "d": "التهاب الجلد التأتبي؛ نوع من الحساسية المزمنة."},
    3: {"n": "Bullous Disease", "s": "🚨 حالة حرجة", "c": "#FF3B30", "d": "أمراض فقاعية؛ تسبب بثوراً مائية وتتطلب رعاية طبية."},
    4: {"n": "Bacterial Infections", "s": "🦠 عدوى بكتيرية", "c": "#FF9500", "d": "التهابات بكتيرية مثل القوباء؛ تعالج عادةً بالمضادات."},
    5: {"n": "Eczema", "s": "🔍 حالة جلدية", "c": "#5856D6", "d": "الإكزيما؛ تهيج جلدي يسبب الحكة والجفاف."},
    6: {"n": "Exanthems and Drug Eruptions", "s": "⚠️ طارئ طبي", "c": "#FF9500", "d": "طفح دوائي نتيجة رد فعل تحسسي للأدوية."},
    7: {"n": "Hair Loss / Alopecia", "s": "🔍 حالة شعر", "c": "#5AC8FA", "d": "تساقط الشعر أو الثعلبة بمختلف أنواعها."},
    8: {"n": "Herpes / HPV / STDs", "s": "🦠 عدوى فيروسية", "c": "#FF9500", "d": "عدوى فيروسية تشمل الهربس أو الثآليل."},
    9: {"n": "Pigmentation Disorders", "s": "🔍 اضطراب صبغة", "c": "#AF52DE", "d": "اضطرابات التصبغ مثل البهاق أو الكلف."},
    10: {"n": "Lupus / Connective Tissue", "s": "🚨 مرض مناعي", "c": "#FF3B30", "d": "أمراض الأنسجة الضامة التي تؤثر على الجلد."},
    11: {"n": "Melanoma / Nevi / Moles", "s": "🚨 خبيث جداً", "c": "#FF2D55", "d": "الميلانوما أو الشامات غير الطبيعية؛ تتطلب رقابة فورية."},
    12: {"n": "Nail Diseases", "s": "🔍 أمراض أظافر", "c": "#8E8E93", "d": "فطريات أو تغيرات في بنية الأظافر."},
    13: {"n": "Contact Dermatitis", "s": "🔍 تحسس تماسي", "c": "#5856D6", "d": "التهاب ناتج عن ملامسة مواد كيميائية مهيجة."},
    14: {"n": "Psoriasis / Lichen Planus", "s": "🔍 حالة مزمنة", "c": "#AF52DE", "d": "الصدفية؛ تسبب قشوراً فضية وحكة."},
    15: {"n": "Infestations and Bites", "s": "🦠 طفيليات", "c": "#FF9500", "d": "عضات الحشرات أو الجرب."},
    16: {"n": "Seborrheic Keratoses / Benign", "s": "✅ حميد", "c": "#34C759", "d": "تقرن دهني؛ نمو جلدي حميد وشائع جداً."},
    17: {"n": "Systemic Disease", "s": "⚠️ مؤشر جهازي", "c": "#FF9500", "d": "أعراض جلدية مرتبطة بأمراض داخلية في الجسم."},
    18: {"n": "Fungal Infections", "s": "🦠 عدوى فطريات", "c": "#FF9500", "d": "عدوى فطرية سطحية تتطلب مضادات للفطريات."},
    19: {"n": "Urticaria Hives", "s": "✅ حالة شائعة", "c": "#34C759", "d": "الأرتيكاريا؛ طفح جلدي يظهر ويختفي بسرعة."},
    20: {"n": "Vascular Tumors", "s": "✅ حميد غالباً", "c": "#34C759", "d": "أورام وعائية دموية حميدة."},
    21: {"n": "Vasculitis", "s": "🚨 خطير", "c": "#FF3B30", "d": "التهاب الأوعية الدموية تحت الجلد."},
    22: {"n": "Warts / Viral Infections", "s": "🦠 عدوى فيروسية", "c": "#FF9500", "d": "الثآليل الفيروسية الناتجة عن عدوى موضعية."},
    23: {"n": "Archive / Other", "s": "🔍 غير محدد", "c": "#8E8E93", "d": "حالات غير مصنفة حالياً."},
}

# --- 3. محرك معالجة الملفات والنموذج الذكي ---
@st.cache_resource
def load_expert_ai():
    zip_path = "skin_expert_hybrid_24ch.zip"
    extract_to = "model_cache_dir"
    
    try:
        # فك الضغط بأمان
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            # انتظار بسيط لضمان اكتمال الكتابة على القرص ومنع خطأ Magic Number
            time.sleep(3) 
        else:
            st.error("❌ ملف الأوزان (ZIP) مفقود من المستودع!")
            return None

        # البحث عن ملف h5
        h5_file = None
        for root, dirs, files in os.walk(extract_to):
            for f in files:
                if f.endswith(".h5"):
                    h5_file = os.path.join(root, f)
                    break
        
        if not h5_file:
            st.error("❌ لم يتم العثور على ملف الأوزان داخل الـ ZIP.")
            return None

        # بناء الهيكل الهجين (EfficientNet + MobileNet)
        base1 = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
        base2 = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
        
        merged = Concatenate()([GlobalAveragePooling2D()(base1.output), GlobalAveragePooling2D()(base2.output)])
        dense = Dense(512, activation='relu')(merged)
        dropout = Dropout(0.4)(dense)
        output = Dense(24, activation='softmax')(dropout)
        
        full_model = Model(inputs=[base1.input, base2.input], outputs=output)
        
        # تحميل الأوزان النهائية
        full_model.load_weights(h5_file)
        return full_model
    except Exception as e:
        st.error(f"⚠️ خطأ تقني في المحرك: {e}")
        return None

# تهيئة المحرك
model = load_expert_ai()

# --- 4. واجهة المستخدم ---
st.markdown("<h1 style='text-align:center; color:#1E3A8A;'>الذكاء الاصطناعي لفحص سلامة الجلد 🧬</h1>", unsafe_allow_html=True)
st.write("---")

# رفع الصورة
uploaded_image = st.file_uploader("📥 ارفع صورة المنطقة المصابة (يرجى التأكد من وضوح الإضاءة)", type=["jpg", "png", "jpeg"])

if uploaded_image:
    if model:
        # عرض الصورة
        img = Image.open(uploaded_image).convert('RGB')
        col_img, col_res = st.columns([1, 1.2])
        
        with col_img:
            st.image(img, caption="الصورة التي تم رفعها", use_container_width=True)
        
        if st.button("🔍 بدء التحليل الرقمي"):
            with st.spinner("⏳ جاري تحليل الأنماط الحيوية ومطابقتها..."):
                # معالجة الصورة للمحرك
                img_cv = cv2.resize(np.array(img), (224, 224))
                tensor = (img_cv.astype(np.float32) / 255.0)[np.newaxis, ...]
                
                # التنبؤ (مدخلان للنموذج الهجين)
                prediction = model.predict([tensor, tensor])[0]
                best_class = np.argmax(prediction)
                confidence = prediction[best_class]
                
                # جلب البيانات الطبية
                res = MEDICAL_INFO.get(best_class, MEDICAL_INFO[23])
                
                # تحديد عتبة الثقة (Threshold)
                serious = "🚨" in res['s']
                req_conf = 0.40 if serious else 0.45
                
                if confidence >= req_conf:
                    with col_res:
                        st.markdown(f"""
                        <div class="report-card" style="border-right-color: {res['c']};">
                            <h2 style="color:{res['c']}; margin-top:0;">{res['n']}</h2>
                            <h4 style="color:#555;">نوع الحالة: {res['s']}</h4>
                            <p style="font-size:1.3em; color:#1E3A8A;"><b>نسبة الدقة:</b> {confidence:.2%}</p>
                            <hr style="opacity:0.2;">
                            <p style="line-height:1.7;">{res['d']}</p>
                            <p style="color:red; font-size:0.85em; margin-top:15px;">⚠️ إخلاء مسؤولية: هذا الفحص آلي لغرض الإرشاد فقط، لا يغني عن استشارة الطبيب المختص.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ دقة التحليل غير كافية ({confidence:.1%}). يرجى التقاط صورة بتركيز (Focus) أفضل وإضاءة طبيعية.")

st.write("---")
st.caption("نظام Skin Safety System V2.1 | تم التطوير باستخدام تقنيات التعلم العميق 2026")
