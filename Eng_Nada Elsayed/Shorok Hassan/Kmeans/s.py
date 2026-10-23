import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Resolve image path dynamically relative to this script's directory
# This fixes the common error: cv2.error (-215:Assertion failed) !_src.empty()
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, 'istockphoto-520700958-612x612.jpg')

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found at path: {img_path}")

# 2. Read image and convert from BGR (OpenCV) to RGB (Matplotlib)
img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Could not decode image at: {img_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, c = img.shape
print(f"Original Image Loaded: {w}x{h} pixels, Total Pixels: {w*h:,}", flush=True)

# Flatten image into 2D array of (N, 3) RGB values
img_array = img.reshape((-1, 3))

# Count unique colors in original image
unique_original_colors = len(np.unique(img_array, axis=0))
print(f"Number of unique colors in original image: {unique_original_colors:,}", flush=True)

# 3. Apply K-Means Clustering for Color Quantization (Image Compression)
# Grouping hundreds of thousands of colors into only K=8 distinct palette colors
K = 8
print(f"Training K-Means (K={K}) for color quantization...", flush=True)
kmeans = KMeans(n_clusters=K, random_state=42, n_init=5)
kmeans.fit(img_array)

# 4. Map each pixel to its nearest cluster centroid color
compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
compressed_img = np.clip(compressed_pixels.astype(np.uint8), 0, 255)
compressed_img = compressed_img.reshape(img.shape)

print(f"Color Quantization Complete! Compressed from {unique_original_colors:,} colors down to {K} palette colors.", flush=True)

# 5. Save and Display Visual Comparison
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title(f"Original Image\n({unique_original_colors:,} Colors)", fontsize=13, fontweight='bold')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(compressed_img)
plt.title(f"Compressed Image (K-Means K={K})\n({K} Colors Palette)", fontsize=13, fontweight='bold')
plt.axis('off')

output_path = os.path.join(script_dir, 'compressed_output_comparison.png')
plt.tight_layout()
plt.savefig(output_path, dpi=200)
print(f"Comparison plot saved successfully at: {output_path}", flush=True)

plt.close()