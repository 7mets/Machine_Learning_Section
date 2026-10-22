import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n_samples = 365


temperature = np.random.uniform(15, 40, n_samples)


electricity_demand = 50 + 2.5 * (temperature - 25)**2 + np.random.normal(0, 15, n_samples)


data = pd.DataFrame({
    'Temperature': temperature,
    'Electricity_Demand': electricity_demand
})


print("--- Dataset Sample ---")
print(data.head())
print("-" * 30)


X = data[['Temperature']]
y = data['Electricity_Demand']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")


plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data (Test Set)', alpha=0.6)

sort_axis = np.argsort(X_test['Temperature'])
plt.plot(X_test['Temperature'].iloc[sort_axis], y_pred[sort_axis], color='red', linewidth=2, label='Model Prediction')

plt.title('Energy Demand Forecast based on Temperature', fontsize=14)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Electricity Demand (kWh)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()