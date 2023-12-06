import cv2 

image = cv2.imread('gambar1.png',0)
level = 3
gaussian = []

gaussian_layer= image.copy()
for i in range(level):
    gaussian_layer = cv2.pyrDown(gaussian_layer)
    gaussian.append(gaussian_layer)
    cv2.imshow('Gaussian Layer -{}'.format(i),gaussian_layer)
cv2.waitKey(0)
cv2.destroyAllWindows()

laplacian = [gaussian[-1]] 
for i in range(level-1,0,-1):
    size = (gaussian[i - 1].shape[1], gaussian[i - 1].shape[0])
    gaussian_expanded = cv2.pyrUp(gaussian[i], dstsize=size)
    laplacian_layer = cv2.subtract(gaussian[i-1], gaussian_expanded)
    laplacian.append(laplacian_layer)
    cv2.imshow('laplacian layer -{}'.format(i-1),laplacian_layer)
cv2.waitKey(0)
cv2.destroyAllWindows()