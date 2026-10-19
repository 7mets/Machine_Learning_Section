import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)




print("=== 1. Binary Classification (2 Class) ===")


y_true_2 = [0, 1, 0, 1, 0, 0, 1, 1, 0, 1]

# Predicted labels = values predicted by the model
y_pred_2 = [0, 1, 0, 0, 0, 1, 1, 1, 0, 1]


# Calculate the four final evaluation metrics
accuracy_2 = accuracy_score(y_true_2, y_pred_2)
precision_2 = precision_score(y_true_2, y_pred_2)
recall_2 = recall_score(y_true_2, y_pred_2)
f1_2 = f1_score(y_true_2, y_pred_2)


# Print only the final results
print("Accuracy :", accuracy_2)
print("Precision:", precision_2)
print("Recall   :", recall_2)
print("F1-Score :", f1_2)


# Plot Confusion Matrix
cm_2 = confusion_matrix(y_true_2, y_pred_2)

plt.figure(figsize=(5, 4))

sns.heatmap(
    cm_2,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False
)

plt.title('Confusion Matrix - 2 Class')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.show()


print("\n" + "=" * 40 + "\n")