import os
import time
from updated_dataset_image_preproccesing import process_fingerprint_image

# Folderul sursa unde se afla imaginile de procesat
image_folder = '/home/user/PDS/dataset/SOCOFing/Real'

# Folderul de destinatie unde vor fi salvate imaginile procesate
output_folder = '/home/user/PDS/Result_Images'

# Verificam daca folderul de destinatie exista; daca nu, il cream
os.makedirs(output_folder, exist_ok=True)

# Seteaza daca vrei sa afisezi plot-ul pentru fiecare imagine (True sau False)
display_plot = False

# Seteaza daca vrei sa salvezi plot-ul complet pentru fiecare imagine
save_full_plot = True  # parametru pentru salvare plot complet

# Start timer pentru masurarea timpului de procesare
start_time = time.time()

# Parcurgem primele 20 de imagini pentru testare
for i, image_name in enumerate(os.listdir(image_folder)):
    if i >= 20:  # pana la 20 de imagini
        break

    image_path = os.path.join(image_folder, image_name)

    # Procesam fisierele de tip .bmp
    if image_name.lower().endswith('.bmp'):
        save_path = os.path.join(output_folder, f"processed_{i+1}.png")

        # Proceseaza imaginea si salveaza rezultatul + plotul complet
        process_fingerprint_image(
            image_path,
            save_path=save_path,
            show_plot=display_plot,
            save_full_plot=save_full_plot
        )

        print(f"Imaginea {image_name} a fost procesata si salvata ca {save_path}")

# Calculam timpul total de procesare
end_time = time.time()
elapsed_time = end_time - start_time

# Afisam timpul total in consola
minutes, seconds = divmod(elapsed_time, 60)
seconds, milliseconds = divmod(seconds, 1)
milliseconds = round(milliseconds * 1000)

print(f"Procesarea imaginilor a durat: {int(minutes)} minute, {int(seconds)} secunde si {milliseconds} milisecunde.")

# Salvam timpul total intr-un fisier text
with open('/home/user/PDS/process_time.txt', 'w') as f:
    f.write(f"Procesarea imaginilor a durat: {int(minutes)} minute, {int(seconds)} secunde si {milliseconds} milisecunde.\n")

print("Timpul de procesare a fost salvat in fisierul process_time.txt.")
print("Procesarea imaginilor s-a finalizat!")
