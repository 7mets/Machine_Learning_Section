import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
import matplotlib.pyplot as plt

try:
    data = pd.read_csv('student-mat.csv',sep=';')
    print("Data loaded succssfully")
except FileNotFoundError:
    print("Error: Dataset not found ")
    exit()

features = ['studytime','failures','absences','G1','G2']
target = 'G3'

x = data[features]
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)

r2 = r2_score(y_test,y_pred)

print("/n--- Model Evaluation ---")
print(f"Mean Squared Error (MSE) : {mse:.2f}")
print(f"R-squared Score (R2) :{r2:.2f}")

print("\n--- Feature Coffinients ---")
for feature , coef in zip(features, model.coef_):
    print(f"{features}: {coef:.2f}")

plt.scatter(y_test,y_pred,alpha=0.7, color='b')
plt.plot([0,20],[0,20],color='red',linestyle='--')
plt.xlabel('Actual Final Grade (G3)')
plt.ylabel('Predicted Final Grade (G3)')
plt.title('Actual vs Predicted Student Grade')
plt.show()

new_student_data = pd.DataFrame([[3,0,2,15,16]],columns=features)
predicted_grade = model.predict(new_student_data)
print(f"\n--- Example Prediction ---")
print(f"Predicted Final Grade for the new student : {predicted_grade[0]:.2f}(out of 20)")