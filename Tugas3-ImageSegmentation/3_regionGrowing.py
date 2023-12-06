import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_growing(image, seed):
    tolerance = 10
    rows, cols = image.shape
    visited = np.zeros_like(image, dtype=np.uint8)
    stack = []

    result = np.zeros_like(image)
    def is_within_tolerance(p1, p2):
        return abs(int(p1) - int(p2)) < tolerance

    stack.append(seed)
    visited[seed] = 1
    while len(stack) > 0:
        current_point = stack.pop()
        result[current_point] = image[current_point]

        for i in range(-1, 2):
            for j in range(-1, 2):
                if current_point[0] + i >= 0 and current_point[1] + j >= 0 and current_point[0] + i < rows and current_point[1] + j < cols:
                    if visited[current_point[0] + i, current_point[1] + j] == 0 and is_within_tolerance(image[current_point], image[current_point[0] + i, current_point[1] + j]):
                        stack.append((current_point[0] + i, current_point[1] + j))
                        visited[current_point[0] + i, current_point[1] + j] = 1

    return result

if __name__ == "__main__":
    image = cv2.imread("gambar2.png", 0) 
    seed = (50, 50) 

    result = region_growing(image, seed)

    # Menampilkan gambar asli dan hasil region growing dalam satu window Matplotlib
    plt.figure(figsize=(10, 5))

    # Gambar Asli
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    # Hasil Region Growing
    plt.subplot(1, 2, 2)
    plt.imshow(result, cmap='gray')
    plt.title('Result Region Growing')

    plt.show()
