import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# load Dataset from california
housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

print("Dataset Preview:")
print(X.head())

print("\nDataset Shape:", X.shape)


# split Dataset into Training and Testing =>  ( 20 test -> 80 train )
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# create and train the Model linear regression
model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")


# predictions
y_pred = model.predict(X_test)


# evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== Model Evaluation ==========")
print(f"MAE  : ${mae * 100000:,.2f}")
print(f"RMSE : ${rmse * 100000:,.2f}")
print(f"R2   : {r2:.4f}")


# compare Actual and Predicted Prices
results = pd.DataFrame({
    "Actual Price ($)": y_test.iloc[:10].values * 100000,
    "Predicted Price ($)": y_pred[:10] * 100000
})

print("\n========== Actual vs Predicted ==========")
print(results.round(2))


# Predict a New House Value
new_house = X_test.iloc[[0]]

predicted_price = model.predict(new_house)[0] * 100000

print("\n========== New House Prediction ==========")
print("House Features:")
print(new_house)

print(f"\nPredicted House Value: ${predicted_price:,.2f}")


# Display Model Coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\n========== Feature Coefficients ==========")
print(coefficients.round(4).to_string(index=False))