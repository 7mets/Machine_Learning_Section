import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_blobs, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# PART 1: CLASSIFICATION ALGORITHM (Logistic Regression)

X_class, y_class = make_classification(
    n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=42
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

classifier = LogisticRegression()
classifier.fit(X_train_c, y_train_c)

y_pred_c = classifier.predict(X_test_c)
print("Classification Report:\n", classification_report(y_test_c, y_pred_c))


# PART 2: REGRESSION ALGORITHM (Linear Regression)

X_reg, y_reg = make_regression(
    n_samples=150, n_features=1, noise=15, random_state=42
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

regressor = LinearRegression()
regressor.fit(X_train_r, y_train_r)

y_pred_r = regressor.predict(X_test_r)
mse = mean_squared_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R²): {r2:.2f}")


# PART 3: K-MEANS CLUSTERING ALGORITHM (Unsupervised Learning)

X_cluster, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")

y_kmeans = kmeans.fit_predict(X_cluster)

print("Cluster Centers:\n", kmeans.cluster_centers_)
