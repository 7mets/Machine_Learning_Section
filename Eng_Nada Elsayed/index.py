import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

y_true_2 = [0, 1, 0, 1, 0, 1, 1, 0, 1]
y_pred_2 = [0, 1, 0, 0, 0, 1, 1, 1, 0]

accuracy_2 = accuracy_score(y_true_2, y_pred_2)
precision_2 = precision_score(y_true_2, y_pred_2)
recall_2 = recall_score(y_true_2, y_pred_2)
f1_2 = f1_score(y_true_2, y_pred_2)

print("Accuracy :", accuracy_2)
print("Precision :", precision_2)
print("Recall :", recall_2)
print("F1-Score :", f1_2)

cm_2 = confusion_matrix(y_true_2, y_pred_2)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_2,
    annot=True,
    fmt='d',
    cmap='Greens',
    cbar=False
)

plt.title('confusion_matrix 0-2 class')
plt.xlabel('precision label')
plt.ylabel('true label')
plt.show()