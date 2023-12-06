import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imshow, imread

def quantize_image(image_path, k_values):
    # Load the original image
    img = imread(image_path)

    # Create subplots for different k values
    fig, ax = plt.subplots(1, len(k_values), figsize=(12, 4))

    for i, k in enumerate(k_values):
        # Create k bins of equal width between 0 and the maximum intensity value
        bins = np.linspace(0, img.max(), k)

        # Map the pixel values of the original image to the nearest bin
        quantized_image = np.digitize(img, bins)

        # Convert the binned values back to the original range of intensity values
        reconstructed_image = (np.vectorize(bins.tolist().__getitem__)(quantized_image-1).astype(int))

        # Display the quantized image with title showing the number of bins (k)
        ax[i].imshow(reconstructed_image)
        ax[i].set_title(r'$k = %d$' % k)

    # Adjust the layout of the subplots to prevent overlap
    plt.tight_layout()

    # Show the final plot
    plt.show()

# Quantize the image
k_values = [2,  4, 16, 256]
quantize_image('gambar2.png', k_values)