import numpy as np
from skimage import io
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# Membaca citra
image = io.imread("gambar1.png")
height, width, _ = image.shape
image_2d = image.reshape(-1, 3)
num_segments = 4  
kmeans = KMeans(n_clusters=num_segments, random_state=0).fit(image_2d)
labels = kmeans.labels_
segmented_image = labels.reshape(height, width)
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image)
plt.axis('off')
plt.title('Gambar Citra Asli')

plt.subplot(122)
plt.imshow(segmented_image, cmap='nipy_spectral')
plt.axis('off')
plt.title('Hasil Segmentasi Normalized Cut')

plt.show()