import cv2
import numpy as np
import os
import matplotlib.pyplot as plt  # Import pentru a putea afisa plot-uri
import time

# Helper function: calculeaza numarul de tranzitii intr-o lista de vecini
def count_transitions(neighbors):
    transitions = 0
    for k in range(len(neighbors)):
        if neighbors[k] == 0 and neighbors[(k+1) % len(neighbors)] == 1:
            transitions += 1
    return transitions

# Algoritmul de thinning Zhang-Suen cu oprire dupa max_iter iteratii
def zhang_suen_thinning(binary_image, max_iter=5):
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

# ------------------- Functie Principala -------------------

def process_fingerprint_image(image_path, save_path=None, show_plot=False, save_full_plot=False):
    """
    Proceseaza o imagine cu amprenta si extrage caracteristicile minutiae.
    
    Parametri:
    - image_path (str): Calea catre imaginea de intrare.
    - save_path (str, optional): Calea unde se va salva imaginea cu minutiae suprapuse.
    - show_plot (bool): Daca este True, afiseaza vizualizarile intr-o fereastra interactiva.
    - save_full_plot (bool): Daca este True, salveaza o figura completa cu toti pasii de preprocesare.
    """

    start_time = time.time()  # Timpul de start pentru a masura performanta

    # 1. incarca imaginea in tonuri de gri
    orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if orig is None:
        raise FileNotFoundError(f"Image could not be loaded: {image_path}")

    # 2. Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) pentru imbunatatirea contrastului
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(orig)

    # 3. Binarizeaza imaginea folosind Adaptive Threshold
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 3)
    # Inverseaza imaginea daca fundalul este dominant alb
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # 4. Subtiaza (thinning) liniile crestelor folosind algoritmul Zhang-Suen
    thinned = zhang_suen_thinning(binary, max_iter=5)

    # 5. Extrage punctele minutiae: terminatii si bifurcatii
    minutiae = extract_minutiae(thinned)

    # 6. Creeaza imagine color pentru a desena punctele minutiae (verde = terminatie, rosu = bifurcatie)
    minutiae_img = cv2.cvtColor(thinned, cv2.COLOR_GRAY2BGR)
    for (x, y, typ) in minutiae:
        color = (0, 255, 0) if typ == 'ending' else (0, 0, 255)
        cv2.circle(minutiae_img, (y, x), 2, color, -1)

    # 7. Salveaza imaginea cu punctele minutiae (daca se specifica un path)
    if save_path:
        cv2.imwrite(save_path, minutiae_img)

    # 8. Optional: Afiseaza sau salveaza figura completa cu toti pasii de procesare
    if show_plot or save_full_plot:
        fig = plt.figure(figsize=(18, 12))

        # Subplot 1: imagine originala
        plt.subplot(2,3,1)
        plt.imshow(orig, cmap='gray')
        plt.title("Original")
        plt.axis("off")

        # Subplot 2: imagine dupa CLAHE
        plt.subplot(2,3,2)
        plt.imshow(enhanced, cmap='gray')
        plt.title("CLAHE Enhanced")
        plt.axis("off")

        # Subplot 3: imagine binarizata
        plt.subplot(2,3,3)
        plt.imshow(binary, cmap='gray')
        plt.title("Adaptive Threshold")
        plt.axis("off")

        # Subplot 4: imagine subtiata
        plt.subplot(2,3,4)
        plt.imshow(thinned, cmap='gray')
        plt.title("Thinned (Zhang-Suen)")
        plt.axis("off")

        # Subplot 5: imagine cu puncte minutiae
        plt.subplot(2,3,5)
        plt.imshow(cv2.cvtColor(minutiae_img, cv2.COLOR_BGR2RGB))
        plt.title("Minutiae Points")
        plt.axis("off")

        # Subplot 6: legenda explicativa
        plt.subplot(2,3,6)
        plt.axis('off')
        plt.text(0.1, 0.85, "Legend:", fontsize=14, weight='bold')
        plt.text(0.1, 0.65, "Green = Ending", fontsize=12)
        plt.text(0.1, 0.45, "Red = Bifurcation", fontsize=12)
        plt.text(0.1, 0.25, "Adaptive Threshold preserves fine details", fontsize=12)
        plt.text(0.1, 0.05, "Thinning creates a roadmap of ridges", fontsize=12)

        # Ajusteaza spatierea intre subgrafice
        plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.5, wspace=0.3)

        # Afiseaza figura, daca este specificat
        if show_plot:
            plt.show()

        # Salveaza figura completa in folderul `full_plots`, daca este necesar
        if save_full_plot and save_path:
            full_plot_dir = os.path.join(os.path.dirname(save_path), "full_plots")
            os.makedirs(full_plot_dir, exist_ok=True)
            filename = os.path.splitext(os.path.basename(save_path))[0] + "_full_plot.png"
            full_plot_path = os.path.join(full_plot_dir, filename)
            fig.savefig(full_plot_path)

        # inchide figura pentru a elibera memoria
        plt.close(fig)

    # Afiseaza durata totala a procesarii
    print(f"Processed {os.path.basename(image_path)} in {time.time() - start_time:.2f} seconds.")
