import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

def isodata_segmentation(image, num_clusters, max_iterations, min_samples, max_variance):
    gray_image = color.rgb2gray(image)
    m, n = gray_image.shape
    data = gray_image.reshape(-1, 1)
    np.random.seed(0)
    centroids = np.random.rand(num_clusters, 1)
    
    for _ in range(max_iterations):
        distances = np.abs(data - centroids.T)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == i].mean() for i in range(num_clusters)])
        if np.all(np.isclose(new_centroids, centroids, atol=1e-2)):
            break
        
        centroids = new_centroids
    cluster_variances = np.array([data[labels == i].var() for i in range(num_clusters)])
    mean_variance = cluster_variances.mean()
    low_variance_clusters = np.where(cluster_variances < max_variance * mean_variance)[0]
    for cluster in low_variance_clusters:
        labels[labels == cluster] = labels[labels == cluster].min()
    segmented_image = labels.reshape(m, n)
    
    return segmented_image  

if __name__ == "__main__":
    image = io.imread('gambar1.png')
    num_clusters = 3 
    max_iterations = 100  
    min_samples = 0.1  
    max_variance = 0.5  
    
    segmented_image = isodata_segmentation(image, num_clusters, max_iterations, min_samples, max_variance)
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('Gambar Citra Asli')
    
    plt.subplot(122)
    plt.imshow(segmented_image, cmap='nipy_spectral')
    plt.axis('off')
    plt.title('Hasil Gambar Segmentasi ISODATA')
    
    plt.show()