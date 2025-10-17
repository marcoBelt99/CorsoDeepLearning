'''
Calcolo e visualizzazione dell’istogramma
Obiettivo: Calcolare e visualizzare l'istogramma di ciascun canale di un'immagine a colori
Carica un’immagine a colori con OpenCV
Calcola l’istogramma per ogni canale usando usando sempre OpenCV
Visualizzarli gli istogrammi tramite la libreria matplotlib

*: Usare immagine Lena.png
'''

import cv2
import matplotlib.pyplot as plt

# Carico l'immagine
immagine = cv2.imread('Lenna.png')

# Calcolo gli istogrammi per ciascun canale
hist_blu = cv2.calcHist([immagine], [0], None, [256], [0, 256])
hist_verde = cv2.calcHist([immagine], [1], None, [256], [0, 256])
hist_rosso = cv2.calcHist([immagine], [2], None, [256], [0, 256])

# Visualizzo gli istogrammi
plt.figure(figsize=(8, 6))

plt.plot(hist_blu, color='blue', label='Blu')
plt.plot(hist_verde, color='green', label='Verde')
plt.plot(hist_rosso, color='red', label='Rosso')

plt.title('Istogrammi dei Canali RGB')
plt.xlabel('Intensità')
plt.ylabel('Numero di Pixel')
plt.legend()

plt.show()