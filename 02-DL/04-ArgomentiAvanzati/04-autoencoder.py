## Esercizio: comprimo e ricostruisco immagini dal DS MNIST

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleAutoencoder(nn.Module):

    def __init__(self):
        '''
        Configuro i moduli per l'encoder e per il decoder.
        '''
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential( # ha una serie di layer lineari e di ReLU, per ridurre la dimensione dell'input da 784 a 3
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3) # Codifica in uno spazio a 3 dimensioni
        )
        
        self.decoder = nn.Sequential( #ha una serie di layer che fanno l'operazione inversa del decoder, quindi ricostruisce l'immagine originaria partendo da 3 dimensioni e arrivando a 784
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid() # Uscita tra 0 ed 1. Questo è coerente con i valori normalizzati delle immagini in input.
        )
    
    def forward(self, x):
        '''
        Definisco come i dati passano attraverso la rete: prima attraverso l'encoder, e poi il decoder.
        '''
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

transform = transforms.Compose([transforms.ToTensor(), # converte l'immagine in tensori PyTorch
                                transforms.Normalize( (0.5), (0.5) ), # normalizza l'immagine per avere media 0.5 e std dev. 0.5
                                transforms.Lambda(lambda x : torch.flatten(x))]) # Lambda appiattisce l'immagine da una matrice 28x28 in un vettore unidimensionale

dataset = datasets.MNIST('./data', download=True, transform=transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True) #carico i dati in batch di 32, mescolando i dati ad ogni epoch

## Inizializzo il modello, la funzione di perdita e l'ottimizzatore
model = SimpleAutoencoder() # creo l'istanza del modello
criterion = nn.MSELoss() # come loss function uso la MSE, che è adatta per problemi di regressione, come la ricostruzione delle immagini
optimizer = optim.Adam(model.parameters(), lr=0.001) # come ottimizzatore uso Adam, configurato con un learning rate di 0.001

## Ciclo di addestramento
epochs = 20

# Si itera per un numero definito di epoche
for epoch in range(epochs):
    # Per ogni batch del dataloader si esegue un passo del training
    for data in dataloader:
        img, _ = data 
        output = model(img) # viene passata l'immagine attraverso l'autoencoder per ottenere la ricostruzione
        loss = criterion(output, img) # viene calcolata la perdita comparando l'immagine ricostruita con l'originale
        optimizer.zero_grad() # azzero i gradienti degli ottimizzatori prima di iniziare la backpropagation
        loss.backward() # eseguo la backpropagation per calcolare i gradienti
        optimizer.step() # aggiorno i pesi del modello

    # Stampo la loss alla fine di ogni epoch, per monitorare il progresso del training
    print(f'Epoca {epoch+1}, Loss: {loss.item()}')



