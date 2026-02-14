if st.button('إجراء التشخيص'):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
    dtype = input_details[0]['dtype']

    # Resize مناسب للتصنيف
    img = image.resize((w, h), Image.Resampling.BILINEAR)
    img_array = np.array(img)

    # تطبيع ذكي حسب نوع الإدخال
    if dtype == np.float32:
        img_array = img_array.astype(np.float32) / 255.0
    else:
        img_array = img_array.astype(dtype)

    img_array = np.expand_dims(img_array, axis=0)

    try:
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        probs = output_data[0]

        # 🔎 تأكد أن عدد الفئات يطابق النموذج
        if len(probs) != len(labels):
            st.error("⚠️ عدد الفئات في النموذج لا يطابق قائمة labels")
            st.stop()

        sorted_indices = np.argsort(probs)[::-1]

        top_idx = sorted_indices[0]
        top_label = labels[top_idx]
        top_conf = probs[top_idx] * 100

        st.write("---")
        st.write("### 🔍 نتيجة التشخيص:")
        st.write(f"**التشخيص الأعلى:** {top_label}")
        st.write(f"**درجة الثقة:** {top_conf:.2f}%")

        # --- تعريف الفئات السرطانية ---
        malignant_labels = ['Melanoma']
        premalignant_labels = ['Actinic Keratosis']
        suspicious_labels = ['Vascular Tumors']

        # حساب الاحتمال الكلي للسرطان
        cancer_probability = sum(
            [probs[labels.index(lbl)] 
             for lbl in malignant_labels + premalignant_labels 
             if lbl in labels]
        ) * 100

        st.write(f"🔬 إجمالي احتمال السرطان: {cancer_probability:.2f}%")

        st.write("---")
        st.write("### 🧬 التصنيف الطبي:")

        if top_label in malignant_labels:
            st.error("🔴 خبيث (سرطان جلدي)")
        
        elif top_label in premalignant_labels:
            st.warning("🟠 ما قبل سرطاني (يحتاج متابعة)")
        
        elif cancer_probability > 25:
            st.warning("🟠 توجد مؤشرات لاحتمال سرطاني — يُفضل مراجعة طبيب")

        else:
            st.success("🟢 حميد")

        # عرض أفضل 3 احتمالات
        st.write("---")
        st.write("### 📊 أفضل 3 احتمالات:")
        for idx in sorted_indices[:3]:
            st.write(f"{labels[idx]} — {probs[idx]*100:.2f}%")

    except Exception as e:
        st.error(f"خطأ تقني: {e}")
