import cv2
import numpy as np

def template_matching_ssd(image, template):
    # Convert images to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Get the dimensions of the images
    image_height, image_width = image_gray.shape
    template_height, template_width = template_gray.shape

    # Initialize variables for best match
    min_ssd = float('inf')
    best_match_position = (0, 0)

    # Iterate over possible positions
    for x in range(image_width - template_width + 1):
        for y in range(image_height - template_height + 1):
            # Extract the region from the larger image
            region = image_gray[y:y+template_height, x:x+template_width]
            
            # Calculate the Sum of Squared Differences (SSD)
            ssd = np.sum((region - template_gray)**2)

            # Update the minimum SAD and best match position
            if ssd < min_ssd:
                min_ssd = ssd
                best_match_position = (x, y)

    return best_match_position

if __name__ == "__main__":
    # Read the images
    image = cv2.imread("gambar1.png")
    template = cv2.imread("tmp.png")

    best_match_position = template_matching_ssd(image, template)
    
    print("Best match position:", best_match_position)

    # Get the dimensions of the template
    template_height, template_width, _ = template.shape
    
    # Draw a rectangle on the image to highlight the best match
    x, y = best_match_position
    cv2.rectangle(image, (x, y), (x + template_width, y + template_height), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Template', template)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()