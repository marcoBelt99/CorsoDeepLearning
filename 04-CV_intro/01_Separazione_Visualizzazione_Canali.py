'''
- Separazione e Visualizzazione dei Canali RGB
- Obiettivo: Dividere un'immagine nei suoi tre canali RGB e visualizzarli separatamente
    1. Carica un’immagine a colori con OpenCV
    2. Dividere l’immagine nei 3 canali
    3. Visualizzarli separatamente tramite la libreria matplotlib

*: Usare immagine Lena.png

'''
import cv2
import matplotlib.pyplot as plt

# Carico l'immagine
immagine = cv2.imread('Lenna.png')

# Divido i canali
canale_blu, canale_verde, canale_rosso = cv2.split(immagine)

# Visualizzo i canali
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(canale_blu)
plt.title('Canale Blu')


plt.subplot(1, 3, 2)
plt.imshow(canale_verde)
plt.title('Canale Verde')


plt.subplot(1, 3, 3)
plt.imshow(canale_rosso)
plt.title('Canale Rosso')

plt.show()