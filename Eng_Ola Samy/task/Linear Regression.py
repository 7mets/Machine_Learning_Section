
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data
X = np.array([[2], [4], [6], [8], [10]])
y = np.array([40, 50, 65, 80, 90])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Evaluation metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Predict a new grade
new_hours = np.array([[7]])
new_grade = model.predict(new_hours)[0]
print("Predicted grade:", new_grade)

# Plot 1: Regression Line
x_line = np.linspace(2, 10, 100).reshape(-1, 1)
y_line = model.predict(x_line)

plt.figure(figsize=(8, 5))
plt.scatter(X.ravel(), y, label="Actual Grades")
plt.plot(x_line.ravel(), y_line, color="red",
         label="Regression Line")
plt.scatter(7, new_grade, color="green", marker="*",
            s=150, label="New Prediction")
plt.xlabel("Study Hours")
plt.ylabel("Exam Grade")
plt.title("Linear Regression")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Actual vs Predicted
plt.figure(figsize=(6, 5))
plt.scatter(y, y_pred)
low = min(y.min(), y_pred.min()) - 5
high = max(y.max(), y_pred.max()) + 5
plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()

# Plot 3: Residuals
residuals = y - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Grade")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.grid(True)
plt.show()