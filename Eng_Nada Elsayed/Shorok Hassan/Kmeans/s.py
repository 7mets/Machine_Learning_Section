import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

img = cv2.imread('istockphoto-520700958-612x612.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_array = img.reshape((-1, 3))

kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(img_array)

compressed_img = kmeans.cluster_centers_[kmeans.labels_]
compressed_img = np.clip(compressed_img.astype('uint8'), 0, 255)
compressed_img = compressed_img.reshape(img.shape)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(compressed_img)
plt.axis('off')

plt.show()