
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. تحميل البيانات
iris = load_iris()
X, y = iris.data, iris.target

# 2. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. إنشاء وتدريب نموذج التصنيف
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# 4. التنبؤ والتقييم
y_pred = clf.predict(X_test)

print("--- Classification Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(
    "\nClassification Report:\n",
    classification_report(
        y_test, y_pred, target_names=iris.target_names
    )
)
