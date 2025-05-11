import cv2
import numpy as np
import matplotlib.pyplot as plt

# Helper function: calculeaza numarul de tranzitii intr-o lista de vecini
def count_transitions(neighbors):
    transitions = 0
    for k in range(len(neighbors)):
        if neighbors[k] == 0 and neighbors[(k+1) % len(neighbors)] == 1:
            transitions += 1
    return transitions

# Algoritmul de thinning Zhang-Suen cu oprire dupa max_iter iteratii
def zhang_suen_thinning(binary_image, max_iter=5):
    # Convertim imaginea la valori 0/1 (0: background, 1: riduri)
    bin_img = (binary_image // 255).astype(np.uint8)
    prev = np.zeros(bin_img.shape, np.uint8)
    iteration = 0
    while True:
        mflag = np.zeros(bin_img.shape, np.uint8)
        rows, cols = bin_img.shape
        # Pasul 1
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                P2 = bin_img[i-1, j]
                P3 = bin_img[i-1, j+1]
                P4 = bin_img[i, j+1]
                P5 = bin_img[i+1, j+1]
                P6 = bin_img[i+1, j]
                P7 = bin_img[i+1, j-1]
                P8 = bin_img[i, j-1]
                P9 = bin_img[i-1, j-1]
                if (bin_img[i, j] == 1 and 
                    2 <= (P2+P3+P4+P5+P6+P7+P8+P9) <= 6):
                    neighbors = [P2, P3, P4, P5, P6, P7, P8, P9]
                    if count_transitions(neighbors) == 1:
                        if P2 * P4 * P6 == 0 and P4 * P6 * P8 == 0:
                            mflag[i, j] = 1
        bin_img = bin_img - mflag
        mflag = np.zeros(bin_img.shape, np.uint8)
        # Pasul 2
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                P2 = bin_img[i-1, j]
                P3 = bin_img[i-1, j+1]
                P4 = bin_img[i, j+1]
                P5 = bin_img[i+1, j+1]
                P6 = bin_img[i+1, j]
                P7 = bin_img[i+1, j-1]
                P8 = bin_img[i, j-1]
                P9 = bin_img[i-1, j-1]
                if (bin_img[i, j] == 1 and 
                    2 <= (P2+P3+P4+P5+P6+P7+P8+P9) <= 6):
                    neighbors = [P2, P3, P4, P5, P6, P7, P8, P9]
                    if count_transitions(neighbors) == 1:
                        if P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0:
                            mflag[i, j] = 1
        bin_img = bin_img - mflag
        diff = np.sum(np.abs(bin_img - prev))
        iteration += 1
        if diff == 0 or iteration >= max_iter:
            break
        prev = bin_img.copy()
    return (bin_img * 255).astype(np.uint8)

# Extrage minutiae folosind metoda crossing number
def extract_minutiae(thinned):
    thinned_bin = (thinned // 255).astype(np.uint8)
    padded = np.pad(thinned_bin, ((1,1),(1,1)), mode='constant', constant_values=0)
    minutiae = []
    rows, cols = thinned_bin.shape
    for i in range(1, rows+1):
        for j in range(1, cols+1):
            if padded[i, j] == 1:
                p2 = padded[i-1, j]
                p3 = padded[i-1, j+1]
                p4 = padded[i, j+1]
                p5 = padded[i+1, j+1]
                p6 = padded[i+1, j]
                p7 = padded[i+1, j-1]
                p8 = padded[i, j-1]
                p9 = padded[i-1, j-1]
                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                cn = count_transitions(neighbors)
                if cn == 1:
                    minutiae.append((i-1, j-1, 'ending'))
                elif cn == 3:
                    minutiae.append((i-1, j-1, 'bifurcation'))
    return minutiae

# ------------------- Pipeline Principal -------------------

# Step 1: Load original image in grayscale
image_path = "dataset/SOCOFing/Real/100__M_Left_index_finger.BMP"
orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if orig is None:
    print("Error: Cannot load image")
    exit()

# Step 2: CLAHE Enhanced
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(orig)

# Step 3: Adaptive Threshold (pastrand detaliile si curbele)
binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 3)
# Asiguram ca ridurile sunt albe; daca media e mare, inversam
if np.mean(binary) > 127:
    binary = cv2.bitwise_not(binary)

# Step 4: Thinning (Zhang-Suen) direct pe imaginea din Adaptive Threshold,
# oprim thinning-ul dupa 5 iteratii pentru a pastra detaliile
thinned = zhang_suen_thinning(binary, max_iter=5)

# Step 5: Extract minutiae points from thinned image
minutiae = extract_minutiae(thinned)
minutiae_img = cv2.cvtColor(thinned, cv2.COLOR_GRAY2BGR)
for (x, y, typ) in minutiae:
    if typ == 'ending':
        cv2.circle(minutiae_img, (y, x), 2, (0,255,0), -1)  # Green for ending
    elif typ == 'bifurcation':
        cv2.circle(minutiae_img, (y, x), 2, (0,0,255), -1)  # Red for bifurcation

# ------------------- Display Results -------------------

plt.figure(figsize=(18,12))

# First row: Original, CLAHE Enhanced, Adaptive Threshold
plt.subplot(2,3,1)
plt.imshow(orig, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(enhanced, cmap='gray')
plt.title("CLAHE Enhanced")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(binary, cmap='gray')
plt.title("Adaptive Threshold")
plt.axis("off")

# Second row: Thinned image, Minutiae Points, Legend
plt.subplot(2,3,4)
plt.imshow(thinned, cmap='gray')
plt.title("Thinned (Zhang-Suen)")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(cv2.cvtColor(minutiae_img, cv2.COLOR_BGR2RGB))
plt.title("Minutiae Points")
plt.axis("off")

plt.subplot(2,3,6)
plt.axis('off')
plt.text(0.1, 0.85, "Legend:", fontsize=14, weight='bold')
plt.text(0.1, 0.65, "Green = Ending", fontsize=12)
plt.text(0.1, 0.45, "Red = Bifurcation", fontsize=12)
plt.text(0.1, 0.25, "Adaptive Threshold preserves fine details", fontsize=12)
plt.text(0.1, 0.05, "Thinning creates a roadmap of ridges", fontsize=12)

# Ajustam spatiul: reducerea hspace la 0.5 si marginile figurii
plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.5, wspace=0.3)
plt.show()
