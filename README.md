# Kernel optimizat pentru extragerea punctelor caracteristice dintr-o amprentă

### Descrierea proiectului

Acest proiect are ca scop dezvoltarea unei aplicații practice de procesare digitală a semnalelor, concentrată pe analiza și extragerea trasăturilor esențiale din imagini cu amprente. 

Folosind un Raspberry Pi 4B – un CPU ARM echipat cu extensii Neon pentru DSP – se demonstrează utilizarea tehnicilor de optimizare specifice prelucrărilor de semnale prin accelerarea algoritmilor cu instrucțiuni SIMD (Neon).

#### Pipeline-ul de prelucrare implementat include următoarele etape:

1. **Preprocesare**: Îmbunătățirea contrastului folosind Contrast Limited Adaptive Histogram Equalization (CLAHE) și aplicarea unui Adaptive Threshold care păstrează detaliile fine și curbele amprentei.
2. **Skeletonizare**: Reducerea imaginii binare la un "roadmap" al ridurilor folosind algoritmul de thinning Zhang-Suen, astfel încât forma de bază a amprentei să fie păstrată.
3. **Extragerea minutiae-lor**: Identificarea punctelor caracteristice (terminații și bifurcații) folosind metoda crossing number, cu rezultate vizuale evidențiate prin marcaje colorate (verde pentru ending și roșu pentru bifurcație).

### Dataset folosit pentru testare

- Sursă: https://www.kaggle.com/datasets/ruizgara/socofing

Pentru evaluarea performanței algoritmului, a fost utilizat un set de date extins, alcătuit din imagini cu amprente, provenind dintr-o sursă web pentru cercetare. Dataset-ul include amprente de o diversitate ridicată. Acest lucru ajută la validarea procesului de prelucrare și la comparațiile de performanță între implementarea de bază și cea optimizată.

### Accelerarea performanței

Un aspect important al proiectului este măsurarea și compararea timpilor de execuție a întregului proces pe acest set de amprente, atât înainte, cât și după aplicarea optimizărilor bazate pe extensiile Neon. Această analiză de performanță demonstrează beneficiile accelerării algoritmilor de prelucrare a semnalelor pe un CPU ARM cu suport DSP.
