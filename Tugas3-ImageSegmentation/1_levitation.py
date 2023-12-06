import cv2
import matplotlib.pyplot as plt

# Baca gambar A, B, dan C
image_A = cv2.imread('imgA.jpg')
image_B = cv2.imread('imgB.jpg')
image_C = cv2.imread('imgA-B.jpg')

# Resize semua gambar menjadi dimensi yang lebih kecil
new_dimensions = (500, 500)  # Ganti dimensi ini sesuai kebutuhan

image_A_resized = cv2.resize(image_A, new_dimensions)
image_B_resized = cv2.resize(image_B, new_dimensions)
image_C_resized = cv2.resize(image_C, new_dimensions)

# Hitung X = (A - B) + C pada gambar yang sudah diresize
image_X = cv2.add(cv2.subtract(image_A_resized, image_B_resized), image_C_resized)

# Menampilkan gambar dalam subplot 2 baris 2 kolom
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image_A_resized, cv2.COLOR_BGR2RGB))
plt.title('Gambar A')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(image_B_resized, cv2.COLOR_BGR2RGB))
plt.title('Gambar B')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(image_C_resized, cv2.COLOR_BGR2RGB))
plt.title('Gambar C')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(image_X, cv2.COLOR_BGR2RGB))
plt.title('Gambar (A-B)+C')
plt.axis('off')

plt.show()