import numpy as np
import cv2
from scipy.sparse.linalg import eigsh

def build_similarity_graph(image, resize_factor=0.5):
    # Resize the image
    resized_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)

    # Build similarity graph based on color differences between neighbors
    rows, cols, _ = resized_image.shape
    num_pixels = rows * cols
    image_flat = resized_image.reshape((num_pixels, 3))

    # Compute the color difference matrix between each pair of pixels
    color_diff = np.linalg.norm(image_flat[:, None] - image_flat, axis=2)

    # Use the Gaussian function to compute similarity between pixels
    similarity_graph = np.exp(-color_diff / color_diff.std())

    return similarity_graph

def normalized_cut(image, num_segments):
    # Build the similarity graph
    similarity_graph = build_similarity_graph(image)

    # Degree matrix
    degree_matrix = np.diag(similarity_graph.sum(axis=1))

    # Laplacian matrix
    laplacian = degree_matrix - similarity_graph

    # Compute eigenvalues and eigenvectors for the normalized Laplacian matrix
    _, eigenvectors = eigsh(laplacian, k=num_segments, which='SM')

    # Use the eigenvectors to perform k-means clustering on the data
    _, labels, _ = cv2.kmeans(eigenvectors, num_segments, None, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.2), attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

    # Return the segmented image
    segmented_image = labels.reshape(image.shape[:2])

    return segmented_image

# Load the image
image_path = 'gambar1.png'
image = cv2.imread(image_path)

# Specify the number of segments
num_segments = 2

# Perform Normalized Cut for segmentation
segmented_image = normalized_cut(image, num_segments)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', (segmented_image * 255.0 / num_segments).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
