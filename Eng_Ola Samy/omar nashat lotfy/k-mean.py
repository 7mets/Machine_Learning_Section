import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. إنشاء بيانات تجريبية (عشوائية) للتجربة
# سنقوم بتوليد 300 نقطة موزعة على 4 مجموعات (Centers = 4)
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 2. بناء نموذج K-Means وتحديد عدد المجموعات بـ 4 (n_clusters=4)
kmeans = KMeans(n_clusters=4, n_init=10, random_state=0)

# 3. تدريب النموذج على البيانات
kmeans.fit(X)

# 4. التنبؤ بالمجموعات التي تنتمي إليها كل نقطة
y_kmeans = kmeans.predict(X)

# 5. استخراج المراكز (Centroids) الخاصة بكل مجموعة
centers = kmeans.cluster_centers_

# 6. رسم النتائج لتوضيح شكل التجمعات والمراكز
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# رسم مراكز المجموعات باللون الأحمر وعلامة 'X' كبيرة
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')

plt.title("K-Means Clustering Example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()