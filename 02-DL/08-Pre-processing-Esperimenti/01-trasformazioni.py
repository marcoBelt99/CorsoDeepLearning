import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os

# Mostra la directory corrente
print("Mi trovo in:")
print(os.getcwd())
print('\n')

# Percorso corretto dell'immagine
base_dir = os.path.dirname(__file__)
img_path = os.path.join(base_dir, 'radiografia_originale.jpg')

# Leggi l'immagine con OpenCV
immagine_originale = cv2.imread(img_path)

if immagine_originale is None:
    raise FileNotFoundError(f"❌ Impossibile leggere l'immagine: {img_path}")
else:
    print("Immagine caricata correttamente.")

# Conversione da BGR (OpenCV) a RGB (PIL/torchvision)
immagine_rgb = cv2.cvtColor(immagine_originale, cv2.COLOR_BGR2RGB)

# Conversione da NumPy a immagine PIL
immagine_pil = Image.fromarray(immagine_rgb)


# Trasformazioni torchvision
transform = transforms.Compose([
    transforms.Resize((224, 224)), # ridimensiono l'immagine ad una dimensione 224x224 px
    transforms.ToTensor(), # converto l'immagine in un tensore Pytorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # normalizzo i valori dei pixel in base a media e deviazione standard
                         std=[0.229, 0.224, 0.225])
])


# questa qui commentata è problematica hahaha 
""" # Trasformazioni torchvision
transform = transforms.Compose([
    transforms.Resize((224, 224)), # ridimensiono l'immagine ad una dimensione 224x224 px
    transforms.ToTensor(), # converto l'immagine in un tensore Pytorch
    transforms.Normalize(mean=[0.5, 0.5, 0.5], # normalizzo i valori dei pixel in base a media e deviazione standard
                         std=[0.5, 0.5, 0.5])
])
 """

immagine_trasformata = transform(immagine_pil)


print("Trasformazione completata. Dimensioni tensore:", immagine_trasformata.shape)

# !! Funzione per denormalizzare (ricostruire i colori originali)
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Ricostruisci immagine per la visualizzazione
img_denorm = denormalize(immagine_trasformata.clone(),
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

# Converti da [C, H, W] → [H, W, C] per matplotlib
img_denorm = img_denorm.permute(1, 2, 0)

# Assicura che i valori siano tra 0 e 1
img_denorm = torch.clamp(img_denorm, 0, 1)

# Mostra le immagini
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(immagine_pil)
plt.title("Immagine Originale")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_denorm)
plt.title("Immagine Trasformata (denormalizzata)")
plt.axis("off")

plt.tight_layout()
plt.show()