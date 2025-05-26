# Kernel optimizat pentru extragerea punctelor caracteristice dintr-o amprentă

### Descrierea proiectului

Acest proiect are ca scop dezvoltarea unei aplicații de procesare digitală a imaginilor, concentrată pe extragerea trasăturilor esențiale din amprente. Implementarea a fost realizată pe un Raspberry Pi 4B, care permite accelerarea algoritmilor cu instrucțiuni SIMD (Neon) disponibile pe arhitectura ARM.

### Pipeline-ul de prelucrare

1. **Preprocesare**: CLAHE și binarizare adaptivă.
2. **Skeletonizare**: Algoritmul Zhang-Suen pentru subțiere.
3. **Extragere minutiae**: Detectarea terminațiilor și bifurcațiilor prin metoda crossing number.
4. **Vizualizare**: Imagine finală cu marcaje colorate pentru minutiae.

### Dataset

- Sursă: https://www.kaggle.com/datasets/ruizgara/socofing  
- Dataset cu amprente reale și sintetice, utilizat pentru testare și validare.

### Rezumatul optimizărilor și rezultatelor de performanță

Pentru accelerarea procesării, algoritmii de skeletonizare și extragere a caracteristicilor au fost optimizați folosind instrucțiuni SIMD NEON, care permit procesarea paralelă a pixelilor. Această optimizare a fost activată prin compilare cu suport NEON și optimizare maximă (`-mfpu=neon -O3`).

Testele au fost realizate pe același input, cu 500 de execuții consecutive pentru fiecare variantă:

| Variantă            | Timp minim | Timp maxim |
|---------------------|------------|------------|
| Optimizată NEON     | 6.72 s     | 7.18 s     |
| Neoptimizată        | 43.87 s    | 44.35 s    |

Accelerarea a rezultat într-o reducere de peste 6 ori a timpului de execuție.

### Output final

- Imagine finală cu amprenta skeletonizată și marcajele minutiae
- Fișiere de ieșire: `output_images/result.png` și `logs/result.txt`

### Bibliografie

- [SOCOFing Dataset - Kaggle](https://www.kaggle.com/datasets/ruizgara/socofing)  
- [Zhang, T. Y., & Suen, C. Y. (1984). A fast parallel algorithm for thinning digital patterns. *Communications of the ACM*, 27(3), 236–239.](https://doi.org/10.1145/357994.358023)
- [OpenCV Documentation](https://docs.opencv.org/)
