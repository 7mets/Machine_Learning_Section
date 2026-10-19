
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. إنشاء بيانات تجريبية (Data Generation)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 2. تقسيم البيانات إلى تدريب واختبار (Train/Test Split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. إنشاء نموذج الانحدار الخطي وتدريبه
model = LinearRegression()
model.fit(X_train, y_train)

# 4. التنبؤ وتقييم النموذج
y_pred = model.predict(X_test)

print("--- Linear Regression Results ---")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 5. رسم النتائج
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
