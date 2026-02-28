import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# 1. احصل على التوقعات (Predictions) من النموذج
# استبدل y_pred و y_true بالمتغيرات الخاصة بك في كود التدريب
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1) # إذا كانت البيانات One-Hot Encoded

# 2. احسب مصفوفة الارتباك
cm = confusion_matrix(y_true, y_pred_classes)

# 3. ارسم المصفوفة
# استبدل labels بأسماء الأمراض الحقيقية لديك (akiec, bcc, ...)
labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# 4. تنسيق وحفظ الصورة
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
plt.title('Confusion Matrix: Skin Lesion Classification')

# احفظ الصورة في نفس مجلد ملف الـ web_app.py
plt.savefig('confusion_matrix.png', bbox_inches='tight')
print("تم حفظ الصورة بنجاح باسم confusion_matrix.png")
