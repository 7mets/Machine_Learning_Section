
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

ks = range(1, 10)
inertias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_
            for k in ks]


kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)

print("Silhouette Score:", silhouette_score(X, labels))
print("Centers:\n", kmeans.cluster_centers_)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(list(ks), inertias, "o-")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia")

axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30)
axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c="red", marker="X", s=200, label="Centers")
axes[1].set_title("K-Means Result (K=4)")
axes[1].legend()

plt.tight_layout()
plt.show()