import cv2
import matplotlib.pyplot as plt
import os

# Calea catre un fisier de amprenta din dataset
image_path = "dataset/SOCOFing/Real/100__M_Left_index_finger.BMP"

# Incarca imaginea
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Verifica daca imaginea a fost incarcata corect
if image is None:
	print("Eroare: Nu am putut incarca imaginea.")
else:
	plt.imshow(image, cmap='gray')
	plt.title("Exemplu de amprenta")
	plt.axis("off")
	plt.show()
