import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def kmeans_segmentation(image_path, k):
    # Load the image
    image = cv2.imread(image_path)
    
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))
    
    # Convert the pixel values to float
    pixels = np.float32(pixels)
    
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8-bit values
    centers = np.uint8(centers)
    
    # Map the labels to the centers
    segmented_image = centers[labels.flatten()]
    
    # Reshape back to the original image shape
    segmented_image = segmented_image.reshape(image.shape)
    
    # Display the original and segmented images using matplotlib
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Segmented Image (K=' + str(k) + ')')
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.show()

# Example usage
image_path = 'gambar1.png'
k = 3  # Number of clusters (adjust as needed)
kmeans_segmentation(image_path, k)
